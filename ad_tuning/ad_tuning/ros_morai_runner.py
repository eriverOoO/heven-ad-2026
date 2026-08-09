from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import math
import threading
import time
from typing import Callable, Protocol, runtime_checkable

from ad_interfaces.msg import PlannerStatus, PredictedObjectArray
from ad_morai_interfaces.msg import CollisionArray, CtrlCmd, EgoVehicleStatus
from ad_morai_interfaces_dev.msg import MultiEgoSetting, MultiEgoVehicle
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rcl_interfaces.msg import ParameterType
from rcl_interfaces.srv import GetParameters, SetParametersAtomically
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Empty, Float32
from std_srvs.srv import SetBool, Trigger

from ad_tuning.objective import RunMetrics
from ad_tuning.perception_epoch import perception_epoch_is_ready
from ad_tuning.route import (
    cumulative_lengths,
    pose_at_progress,
    project_to_route,
    start_yaw_deg,
)


@runtime_checkable
class TrialRunner(Protocol):
    def run_trial(
        self, parameters: Mapping[str, float]
    ) -> tuple[RunMetrics, list[tuple[object, ...]]]:
        """Reset, arm, and drive once with one total outcome."""


@dataclass(frozen=True)
class RouteSnapshot:
    xyz: tuple[tuple[float, float, float], ...]
    xy: tuple[tuple[float, float], ...]
    lengths: tuple[float, ...]
    digest: str
    yaw_deg: float

    @property
    def length_m(self) -> float:
        return self.lengths[-1]


def stale_input_names(
    *,
    now_s: float,
    timeout_s: float,
    samples: Mapping[str, tuple[object | None, float]],
) -> tuple[str, ...]:
    """Return exact missing or stale live-input channel names."""
    return tuple(
        name
        for name, (value, received_s) in samples.items()
        if value is None or now_s - received_s > timeout_s
    )


def dwa_failure_is_controller_infeasible(
    reason: str, stale_inputs: tuple[str, ...]
) -> bool:
    """Separate expected DWA dead ends from lost ROS/MORAI infrastructure."""
    return reason in {
        "initial footprint is unsafe",
        "no safe DWA candidate",
    } and set(stale_inputs).issubset({"target_speed"})


class RosMoraiGlobalPathRunner:
    def __init__(self, node: Node) -> None:
        self.node = node
        self._condition = threading.Condition()
        self._route: RouteSnapshot | None = None
        self._odom: tuple[float, float, float] | None = None
        self._odom_maximum_xy_variance_m2: float | None = None
        self._odom_received_s = 0.0
        self._odom_sequence = 0
        self._speed_mps: float | None = None
        self._status_received_s = 0.0
        self._status_sequence = 0
        self._status_sim_s: float | None = None
        self._status_header_s: float | None = None
        self._status_steering_rad = 0.0
        self._status_lateral_acceleration_mps2 = 0.0
        self._target_speed_mps: float | None = None
        self._target_received_s = 0.0
        self._throttle_command: float | None = None
        self._brake_command: float | None = None
        self._steering_command: float | None = None
        self._command_received_s = 0.0
        self._planner_ready = False
        self._planner_received_s = 0.0
        self._planner_active_behavior = ""
        self._planner_failsafe_reason = ""
        self._planner_dwa_failure_reason = ""
        self._collision_received_s = 0.0
        self._collision_episode_sequence = 0
        self._collision_present = False
        self._occupancy_grid_received_s = 0.0
        self._occupancy_grid_valid = False
        self._prediction_received_s = 0.0
        self._prediction_sequence = 0
        self._shutdown = False
        self._external_abort_reason: Callable[[], str] = lambda: ""

        self.startup_timeout_s = float(
            node.declare_parameter("startup_timeout_sec", 60.0).value
        )
        self.trial_timeout_s = float(
            node.declare_parameter("trial_timeout_sec", 420.0).value
        )
        self.trial_wall_timeout_s = float(
            node.declare_parameter("trial_wall_timeout_sec", 600.0).value
        )
        self.sim_clock_stale_timeout_s = float(
            node.declare_parameter(
                "sim_clock_stale_timeout_sec", 5.0
            ).value
        )
        self.maximum_sim_step_s = float(
            node.declare_parameter(
                "maximum_sim_step_sec", 0.5
            ).value
        )
        self.require_device_stamp = bool(
            node.declare_parameter(
                "timing.require_device_stamp", True
            ).value
        )
        self.require_occupancy_grid = bool(
            node.declare_parameter(
                "timing.require_occupancy_grid", False
            ).value
        )
        self.require_scenario_reset = bool(
            node.declare_parameter(
                "scenario_reset.required", False
            ).value
        )
        self.require_perception_epoch = bool(
            node.declare_parameter(
                "perception_reset.required", False
            ).value
        )
        self.perception_epoch_settle_sim_s = float(
            node.declare_parameter(
                "perception_reset.settle_sim_sec", 1.05
            ).value
        )
        self.perception_epoch_minimum_prediction_samples = int(
            node.declare_parameter(
                "perception_reset.minimum_prediction_samples", 5
            ).value
        )
        self.prediction_frame_id = str(
            node.declare_parameter(
                "perception_reset.prediction_frame_id", "odom"
            ).value
        ).strip()
        scenario_reset_service = str(
            node.declare_parameter(
                "scenario_reset.service", "/ad/dev/scenario/reset"
            ).value
        ).strip()
        if self.require_scenario_reset and not scenario_reset_service:
            raise ValueError(
                "scenario_reset.service must be set when scenario reset "
                "is required"
            )
        if self.require_perception_epoch and not self.require_device_stamp:
            raise ValueError(
                "perception reset requires MORAI device timestamps"
            )
        if (
            not math.isfinite(self.perception_epoch_settle_sim_s)
            or self.perception_epoch_settle_sim_s < 1.0
        ):
            raise ValueError(
                "perception_reset.settle_sim_sec must be at least 1.0"
            )
        if self.perception_epoch_minimum_prediction_samples <= 0:
            raise ValueError(
                "perception_reset.minimum_prediction_samples must be "
                "positive"
            )
        if self.require_perception_epoch and not self.prediction_frame_id:
            raise ValueError(
                "perception_reset.prediction_frame_id must be set"
            )
        self.course_length_m = float(
            node.declare_parameter("course_length_m", 0.0).value
        )
        self.completion_margin_m = float(
            node.declare_parameter("completion_margin_m", 5.0).value
        )
        self.overspeed_mps = float(
            node.declare_parameter("overspeed_kph", 50.0).value
        ) / 3.6
        self.brake_metric_speed_deadband_mps = float(
            node.declare_parameter(
                "brake_metric_speed_deadband_kph", 1.0
            ).value
        ) / 3.6
        self.brake_saturation_threshold = float(
            node.declare_parameter(
                "brake_saturation_threshold", 0.95
            ).value
        )
        if (
            not math.isfinite(self.brake_metric_speed_deadband_mps)
            or self.brake_metric_speed_deadband_mps < 0.0
        ):
            raise ValueError(
                "brake_metric_speed_deadband_kph must be finite "
                "and nonnegative"
            )
        if (
            not math.isfinite(self.brake_saturation_threshold)
            or not 0.0 <= self.brake_saturation_threshold <= 1.0
        ):
            raise ValueError(
                "brake_saturation_threshold must be between 0 and 1"
            )
        self.divergence_cte_m = float(
            node.declare_parameter("divergence_cte_m", 12.0).value
        )
        self.maximum_pose_jump_m = float(
            node.declare_parameter("maximum_pose_jump_m", 20.0).value
        )
        self.wrong_way_heading_rad = math.radians(
            float(node.declare_parameter("wrong_way_heading_deg", 120.0).value)
        )
        self.stale_timeout_s = float(
            node.declare_parameter("input_stale_timeout_sec", 2.0).value
        )
        self.reset_timeout_s = float(
            node.declare_parameter("reset_timeout_sec", 12.0).value
        )
        self.stop_timeout_s = float(
            node.declare_parameter("stop_timeout_sec", 8.0).value
        )
        self.start_tolerance_m = float(
            node.declare_parameter("start_tolerance_m", 3.0).value
        )
        self.start_heading_tolerance_rad = math.radians(
            float(
                node.declare_parameter(
                    "start_heading_tolerance_deg", 3.0
                ).value
            )
        )
        self.start_speed_tolerance_mps = float(
            node.declare_parameter(
                "start_speed_tolerance_mps", 0.2
            ).value
        )
        self.start_stable_sample_count = int(
            node.declare_parameter(
                "start_stable_sample_count", 5
            ).value
        )
        self.start_maximum_xy_variance_m2 = float(
            node.declare_parameter(
                "start_maximum_xy_variance_m2", 1.0
            ).value
        )
        self.completion_position_tolerance_m = float(
            node.declare_parameter(
                "completion_position_tolerance_m", 8.0
            ).value
        )
        self.completion_heading_tolerance_rad = math.radians(
            float(
                node.declare_parameter(
                    "completion_heading_tolerance_deg", 30.0
                ).value
            )
        )
        self.maximum_progress_jump_m = float(
            node.declare_parameter(
                "maximum_progress_jump_m", 20.0
            ).value
        )
        self.metric_control_point_x_m = float(
            node.declare_parameter("metric_control_point_x_m", 3.0).value
        )
        positive_parameters = {
            "startup_timeout_sec": self.startup_timeout_s,
            "trial_timeout_sec": self.trial_timeout_s,
            "trial_wall_timeout_sec": self.trial_wall_timeout_s,
            "sim_clock_stale_timeout_sec": self.sim_clock_stale_timeout_s,
            "maximum_sim_step_sec": self.maximum_sim_step_s,
            "start_tolerance_m": self.start_tolerance_m,
            "start_heading_tolerance_deg": self.start_heading_tolerance_rad,
            "start_speed_tolerance_mps": self.start_speed_tolerance_mps,
            "start_maximum_xy_variance_m2": (
                self.start_maximum_xy_variance_m2
            ),
            "completion_position_tolerance_m": (
                self.completion_position_tolerance_m
            ),
            "completion_heading_tolerance_deg": (
                self.completion_heading_tolerance_rad
            ),
            "maximum_progress_jump_m": self.maximum_progress_jump_m,
        }
        if any(
            not math.isfinite(value) or value <= 0.0
            for value in positive_parameters.values()
        ):
            raise ValueError(
                "timing, reset, and completion tolerances must be positive"
            )
        if self.start_stable_sample_count <= 0:
            raise ValueError("start_stable_sample_count must be positive")

        sensor_qos = QoSProfile(depth=10)
        sensor_qos.reliability = ReliabilityPolicy.BEST_EFFORT
        path_qos = QoSProfile(depth=1)
        path_qos.reliability = ReliabilityPolicy.RELIABLE
        path_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        node.create_subscription(
            Path, "/ad/planner/path", self._on_path, path_qos
        )
        node.create_subscription(
            Odometry,
            "/ad/localization/odometry",
            self._on_odometry,
            sensor_qos,
        )
        node.create_subscription(
            EgoVehicleStatus,
            "/ad/vehicle/status",
            self._on_status,
            sensor_qos,
        )
        node.create_subscription(
            PlannerStatus,
            "/ad/planner/status",
            self._on_planner_status,
            QoSProfile(depth=10),
        )
        node.create_subscription(
            CollisionArray,
            "/ad/safety/collisions",
            self._on_collisions,
            sensor_qos,
        )
        node.create_subscription(
            OccupancyGrid,
            "/ad/perception/occupancy_grid",
            self._on_occupancy_grid,
            sensor_qos,
        )
        node.create_subscription(
            PredictedObjectArray,
            "/ad/perception/objects/predicted",
            self._on_predicted_objects,
            QoSProfile(depth=10),
        )
        node.create_subscription(
            Float32,
            "/ad/planner/target_speed",
            self._on_target_speed,
            QoSProfile(depth=10),
        )
        node.create_subscription(
            CtrlCmd,
            "/ad/control/command",
            self._on_control_command,
            QoSProfile(depth=10),
        )

        self._reset_publisher = node.create_publisher(
            MultiEgoSetting,
            "/ad/dev/command/multi_ego",
            QoSProfile(depth=1),
        )
        self._lease_publisher = node.create_publisher(
            Empty,
            "/ad/tuning/lease",
            QoSProfile(depth=1),
        )
        self._lease_timer = node.create_timer(0.1, self._publish_lease)
        # A valid lease permits planner control. Keep it silent until the
        # planner has acknowledged the explicit tuning hold.
        self._lease_timer.cancel()
        self._hold_client = node.create_client(
            SetBool, "/ad/planner/hold_control"
        )
        self._tracker_reset_client = node.create_client(
            Trigger, "/ad/planner/reset_path_tracking"
        )
        self._adapter_reset_client = node.create_client(
            Trigger, "/ad/localization/reset_adapter"
        )
        self._localizer_reset_client = node.create_client(
            Trigger, "/ad/localization/reset_gnss_imu"
        )
        self._parameter_client = node.create_client(
            SetParametersAtomically,
            "/ad_planner/set_parameters_atomically",
        )
        self._get_parameter_client = node.create_client(
            GetParameters,
            "/ad_planner/get_parameters",
        )
        self._scenario_reset_client = (
            node.create_client(Trigger, scenario_reset_service)
            if self.require_scenario_reset
            else None
        )

    @property
    def route(self) -> RouteSnapshot | None:
        with self._condition:
            return self._route

    def close(self) -> None:
        self._lease_timer.cancel()
        with self._condition:
            self._shutdown = True
            self._condition.notify_all()

    def set_external_abort_checker(
        self, checker: Callable[[], str]
    ) -> None:
        self._external_abort_reason = checker

    def _publish_lease(self) -> None:
        if not self._shutdown:
            self._lease_publisher.publish(Empty())

    def _on_path(self, message: Path) -> None:
        xyz = tuple(
            (
                float(pose.pose.position.x),
                float(pose.pose.position.y),
                float(pose.pose.position.z),
            )
            for pose in message.poses
        )
        if len(xyz) < 2:
            return
        xy = tuple((point[0], point[1]) for point in xyz)
        try:
            lengths = tuple(cumulative_lengths(xy))
            yaw = start_yaw_deg(xy)
        except ValueError as error:
            self.node.get_logger().error(f"rejected global path: {error}")
            return
        digest_source = "\n".join(
            f"{x:.6f},{y:.6f},{z:.6f}" for x, y, z in xyz
        )
        route = RouteSnapshot(
            xyz=xyz,
            xy=xy,
            lengths=lengths,
            digest=hashlib.sha256(digest_source.encode()).hexdigest()[:16],
            yaw_deg=yaw,
        )
        with self._condition:
            self._route = route
            self._condition.notify_all()

    def _on_odometry(self, message: Odometry) -> None:
        position = message.pose.pose.position
        orientation = message.pose.pose.orientation
        yaw = math.atan2(
            2.0
            * (
                orientation.w * orientation.z
                + orientation.x * orientation.y
            ),
            1.0
            - 2.0
            * (
                orientation.y * orientation.y
                + orientation.z * orientation.z
            ),
        )
        if (
            not math.isfinite(position.x)
            or not math.isfinite(position.y)
            or not math.isfinite(yaw)
        ):
            return
        with self._condition:
            self._odom = (float(position.x), float(position.y), yaw)
            xy_variances = (
                float(message.pose.covariance[0]),
                float(message.pose.covariance[7]),
            )
            self._odom_maximum_xy_variance_m2 = (
                max(xy_variances)
                if all(
                    math.isfinite(value) and value >= 0.0
                    for value in xy_variances
                )
                else None
            )
            self._odom_received_s = time.monotonic()
            self._odom_sequence += 1
            self._condition.notify_all()

    def _on_status(self, message: EgoVehicleStatus) -> None:
        speed = abs(float(message.velocity.x))
        steering = float(message.steering)
        lateral_acceleration = float(message.acceleration.y)
        if not all(
            math.isfinite(value)
            for value in (speed, steering, lateral_acceleration)
        ):
            return
        sim_s = None
        if message.has_device_stamp:
            sim_s = (
                float(message.device_stamp.sec)
                + float(message.device_stamp.nanosec) * 1.0e-9
            )
            if not math.isfinite(sim_s):
                sim_s = None
        header_s = (
            float(message.header.stamp.sec)
            + float(message.header.stamp.nanosec) * 1.0e-9
        )
        with self._condition:
            self._speed_mps = speed
            self._status_received_s = time.monotonic()
            self._status_sequence += 1
            self._status_sim_s = sim_s
            self._status_header_s = header_s
            self._status_steering_rad = steering
            self._status_lateral_acceleration_mps2 = lateral_acceleration
            self._condition.notify_all()

    def _on_planner_status(self, message: PlannerStatus) -> None:
        with self._condition:
            self._planner_ready = bool(message.inputs_ready)
            self._planner_received_s = time.monotonic()
            self._planner_active_behavior = str(message.active_behavior)
            self._planner_failsafe_reason = str(message.failsafe_reason)
            self._planner_dwa_failure_reason = str(
                message.dwa_failure_reason
            )
            self._condition.notify_all()

    def _on_collisions(self, message: CollisionArray) -> None:
        present = bool(message.collisions)
        with self._condition:
            if present and not self._collision_present:
                self._collision_episode_sequence += 1
            self._collision_present = present
            self._collision_received_s = time.monotonic()
            self._condition.notify_all()

    def _on_occupancy_grid(self, message: OccupancyGrid) -> None:
        width = int(message.info.width)
        height = int(message.info.height)
        resolution = float(message.info.resolution)
        valid = (
            bool(message.header.frame_id)
            and width > 0
            and height > 0
            and math.isfinite(resolution)
            and resolution > 0.0
            and len(message.data) == width * height
        )
        with self._condition:
            self._occupancy_grid_valid = valid
            self._occupancy_grid_received_s = time.monotonic()
            self._condition.notify_all()

    def _on_predicted_objects(
        self, message: PredictedObjectArray
    ) -> None:
        stamp = message.header.stamp
        valid = (
            message.header.frame_id == self.prediction_frame_id
            and stamp.sec >= 0
            and stamp.nanosec < 1_000_000_000
            and (stamp.sec > 0 or stamp.nanosec > 0)
        )
        if not valid:
            return
        with self._condition:
            self._prediction_received_s = time.monotonic()
            self._prediction_sequence += 1
            self._condition.notify_all()

    def _on_target_speed(self, message: Float32) -> None:
        target_speed = float(message.data)
        if not math.isfinite(target_speed) or target_speed < 0.0:
            return
        with self._condition:
            self._target_speed_mps = target_speed
            self._target_received_s = time.monotonic()
            self._condition.notify_all()

    def _on_control_command(self, message: CtrlCmd) -> None:
        throttle = float(message.accel)
        brake = float(message.brake)
        steering = float(message.steering)
        if not all(
            math.isfinite(value) for value in (throttle, brake, steering)
        ):
            return
        with self._condition:
            self._throttle_command = min(1.0, max(0.0, throttle))
            self._brake_command = min(1.0, max(0.0, brake))
            self._steering_command = min(1.0, max(-1.0, steering))
            self._command_received_s = time.monotonic()
            self._condition.notify_all()

    def _wait_future(self, future, timeout_s: float):
        deadline = time.monotonic() + timeout_s
        while not future.done() and time.monotonic() < deadline:
            if self._shutdown:
                raise RuntimeError("tuner is shutting down")
            time.sleep(0.02)
        if not future.done():
            raise TimeoutError("ROS service response timed out")
        exception = future.exception()
        if exception is not None:
            raise RuntimeError(str(exception))
        return future.result()

    def _startup_remaining_s(
        self, deadline_s: float, dependency: str
    ) -> float:
        remaining_s = deadline_s - time.monotonic()
        if remaining_s <= 0.0:
            raise TimeoutError(
                f"startup timeout while waiting for {dependency}"
            )
        return remaining_s

    def wait_until_ready(self) -> RouteSnapshot:
        deadline_s = time.monotonic() + self.startup_timeout_s
        hold_dependency = "planner hold service"
        while not self._hold_client.wait_for_service(
            timeout_sec=min(
                1.0,
                self._startup_remaining_s(
                    deadline_s, hold_dependency
                ),
            )
        ):
            self._startup_remaining_s(deadline_s, hold_dependency)
            if self._shutdown:
                raise RuntimeError("tuner is shutting down")
            self.node.get_logger().info(
                "waiting for planner hold service"
            )
        self.hold_control(
            True,
            timeout_s=min(
                5.0,
                self._startup_remaining_s(
                    deadline_s, "planner hold response"
                ),
            ),
        )
        self._startup_remaining_s(deadline_s, "planner hold response")
        self._lease_timer.reset()

        services = [
            (self._tracker_reset_client, "path tracker reset"),
            (self._adapter_reset_client, "localization adapter reset"),
            (self._localizer_reset_client, "GNSS/IMU localizer reset"),
            (self._parameter_client, "planner parameter"),
            (self._get_parameter_client, "planner parameter read-back"),
        ]
        if self._scenario_reset_client is not None:
            services.append(
                (self._scenario_reset_client, "MORAI scenario actor reset")
            )
        for client, name in services:
            dependency = f"{name} service"
            while not client.wait_for_service(
                timeout_sec=min(
                    1.0,
                    self._startup_remaining_s(
                        deadline_s, dependency
                    ),
                )
            ):
                self._startup_remaining_s(deadline_s, dependency)
                if self._shutdown:
                    raise RuntimeError("tuner is shutting down")
                self.node.get_logger().info(f"waiting for {name} service")
        while self._reset_publisher.get_subscription_count() == 0:
            remaining_s = self._startup_remaining_s(
                deadline_s, "MORAI multi-ego reset bridge"
            )
            if self._shutdown:
                raise RuntimeError("tuner is shutting down")
            self.node.get_logger().info(
                "waiting for MORAI multi-ego reset bridge"
            )
            time.sleep(min(1.0, remaining_s))

        waiting_logged = False
        with self._condition:
            while True:
                remaining_s = self._startup_remaining_s(
                    deadline_s, "fresh planner inputs"
                )
                if self._shutdown:
                    raise RuntimeError("tuner is shutting down")
                now_s = time.monotonic()
                live = (
                    self._route is not None
                    and self._odom is not None
                    and self._speed_mps is not None
                    and (
                        not self.require_device_stamp
                        or self._status_sim_s is not None
                    )
                    and self._planner_ready
                    and self._collision_received_s > 0.0
                    and now_s - self._odom_received_s <= self.stale_timeout_s
                    and now_s - self._status_received_s <= self.stale_timeout_s
                    and (
                        now_s - self._planner_received_s
                        <= self.stale_timeout_s
                    )
                    and (
                        now_s - self._collision_received_s
                        <= self.stale_timeout_s
                    )
                    and (
                        not self.require_occupancy_grid
                        or (
                            self._occupancy_grid_valid
                            and self._occupancy_grid_received_s > 0.0
                            and now_s - self._occupancy_grid_received_s
                            <= self.stale_timeout_s
                        )
                    )
                )
                if live:
                    return self._route
                if not waiting_logged:
                    self.node.get_logger().info(
                        "waiting for fresh path, odometry, vehicle status, "
                        "collision input, required occupancy grid, and "
                        "planner readiness"
                    )
                    waiting_logged = True
                self._condition.wait(timeout=min(1.0, remaining_s))

    def hold_control(
        self, hold: bool, *, timeout_s: float = 5.0
    ) -> None:
        request = SetBool.Request()
        request.data = hold
        response = self._wait_future(
            self._hold_client.call_async(request), timeout_s
        )
        if not response.success:
            raise RuntimeError(response.message)

    def apply_parameters(self, parameters: Mapping[str, float]) -> None:
        request = SetParametersAtomically.Request()
        request.parameters = [
            Parameter(name=name, value=float(value)).to_parameter_msg()
            for name, value in parameters.items()
        ]
        response = self._wait_future(
            self._parameter_client.call_async(request), 8.0
        )
        if not response.result.successful:
            raise RuntimeError(
                response.result.reason or "planner rejected tuning parameters"
            )
        self.verify_parameters(parameters)

    def verify_parameters(
        self, parameters: Mapping[str, object]
    ) -> None:
        read_request = GetParameters.Request()
        read_request.names = list(parameters)
        read_response = self._wait_future(
            self._get_parameter_client.call_async(read_request), 8.0
        )
        if len(read_response.values) != len(parameters):
            raise RuntimeError("planner parameter read-back count mismatch")
        mismatches = []
        for name, expected, value in zip(
            parameters,
            parameters.values(),
            read_response.values,
        ):
            if isinstance(expected, bool):
                expected_type = ParameterType.PARAMETER_BOOL
                expected_label = "bool"
                actual = bool(value.bool_value)
                matches = value.type == expected_type and actual is expected
            elif isinstance(expected, str):
                expected_type = ParameterType.PARAMETER_STRING
                expected_label = "string"
                actual = str(value.string_value)
                matches = value.type == expected_type and actual == expected
            elif isinstance(expected, int):
                expected_type = ParameterType.PARAMETER_INTEGER
                expected_label = "integer"
                actual = int(value.integer_value)
                matches = value.type == expected_type and actual == expected
            elif isinstance(expected, float):
                expected_type = ParameterType.PARAMETER_DOUBLE
                expected_label = "double"
                actual = float(value.double_value)
                matches = value.type == expected_type and math.isclose(
                    actual,
                    expected,
                    rel_tol=1.0e-9,
                    abs_tol=1.0e-9,
                )
            else:
                raise TypeError(
                    f"unsupported expected ROS parameter type for {name}: "
                    f"{type(expected).__name__}"
                )
            if not matches:
                type_labels = {
                    ParameterType.PARAMETER_NOT_SET: "not set",
                    ParameterType.PARAMETER_BOOL: "bool",
                    ParameterType.PARAMETER_INTEGER: "integer",
                    ParameterType.PARAMETER_DOUBLE: "double",
                    ParameterType.PARAMETER_STRING: "string",
                    ParameterType.PARAMETER_BYTE_ARRAY: "byte array",
                    ParameterType.PARAMETER_BOOL_ARRAY: "bool array",
                    ParameterType.PARAMETER_INTEGER_ARRAY: "integer array",
                    ParameterType.PARAMETER_DOUBLE_ARRAY: "double array",
                    ParameterType.PARAMETER_STRING_ARRAY: "string array",
                }
                actual_label = type_labels.get(
                    value.type, f"unknown type {value.type}"
                )
                mismatches.append(
                    f"{name}: expected {expected_label} {expected!r}, "
                    f"got {actual_label} {actual!r}"
                )
        if mismatches:
            raise RuntimeError(
                "planner parameter read-back mismatch: "
                + "; ".join(mismatches)
            )

    def reset_tracker(self) -> None:
        response = self._wait_future(
            self._tracker_reset_client.call_async(Trigger.Request()), 8.0
        )
        if not response.success:
            raise RuntimeError(response.message)

    def reset_scenario(self) -> float | None:
        if self._scenario_reset_client is None:
            return None
        response = self._wait_future(
            self._scenario_reset_client.call_async(Trigger.Request()),
            max(8.0, self.reset_timeout_s),
        )
        if not response.success:
            raise RuntimeError(
                "MORAI scenario actor reset failed: "
                + (response.message or "unknown error")
            )
        return time.monotonic()

    def reset_localization(self) -> None:
        for client, name in (
            (self._adapter_reset_client, "localization adapter"),
            (self._localizer_reset_client, "GNSS/IMU localizer"),
        ):
            response = self._wait_future(
                client.call_async(Trigger.Request()), 8.0
            )
            if not response.success:
                raise RuntimeError(
                    f"{name} reset failed: {response.message}"
                )

    def _publish_ego_reset(self, route: RouteSnapshot) -> float:
        command = MultiEgoSetting()
        command.header.stamp = self.node.get_clock().now().to_msg()
        command.header.frame_id = "map"
        command.camera_index = 0
        vehicle = MultiEgoVehicle()
        vehicle.ego_index = 0
        vehicle.position.x = route.xyz[0][0]
        vehicle.position.y = route.xyz[0][1]
        vehicle.position.z = route.xyz[0][2]
        vehicle.rpy.x = 0.0
        vehicle.rpy.y = 0.0
        vehicle.rpy.z = route.yaw_deg
        vehicle.velocity = 0.0
        vehicle.gear = 4
        # S4.251001's UDP MultiEgoSetting protocol uses 2 for automode.
        # The ROS bridge message documentation's value 16 is not accepted by
        # this simulator build even though the remaining wire layout matches.
        vehicle.ctrl_mode = 2
        command.vehicles = [vehicle]
        published_s = time.monotonic()
        for _ in range(8):
            self._reset_publisher.publish(command)
            time.sleep(0.1)
        return published_s

    def _wait_for_reset_pose(
        self, route: RouteSnapshot, published_s: float
    ) -> bool:
        deadline = time.monotonic() + self.reset_timeout_s
        stable_samples = 0
        last_odom_sequence = -1
        last_status_sequence = -1
        route_yaw = math.radians(route.yaw_deg)
        with self._condition:
            while time.monotonic() < deadline:
                fresh = (
                    self._odom is not None
                    and self._speed_mps is not None
                    and self._odom_maximum_xy_variance_m2 is not None
                    and self._odom_received_s > published_s
                    and self._status_received_s > published_s
                    and self._odom_sequence > last_odom_sequence
                    and self._status_sequence > last_status_sequence
                )
                if fresh:
                    last_odom_sequence = self._odom_sequence
                    last_status_sequence = self._status_sequence
                    distance = math.hypot(
                        self._odom[0] - route.xy[0][0],
                        self._odom[1] - route.xy[0][1],
                    )
                    heading_error = abs(
                        math.atan2(
                            math.sin(self._odom[2] - route_yaw),
                            math.cos(self._odom[2] - route_yaw),
                        )
                    )
                    stable = (
                        distance <= self.start_tolerance_m
                        and heading_error <= self.start_heading_tolerance_rad
                        and self._speed_mps
                        <= self.start_speed_tolerance_mps
                        and self._odom_maximum_xy_variance_m2
                        <= self.start_maximum_xy_variance_m2
                        and (
                            not self.require_device_stamp
                            or self._status_sim_s is not None
                        )
                    )
                    stable_samples = stable_samples + 1 if stable else 0
                    if stable_samples >= self.start_stable_sample_count:
                        return True
                self._condition.wait(timeout=0.1)
        return False

    def _wait_vehicle_stopped(self) -> bool:
        deadline = time.monotonic() + self.stop_timeout_s
        with self._condition:
            while time.monotonic() < deadline:
                if (
                    self._speed_mps is not None
                    and time.monotonic() - self._status_received_s
                    <= self.stale_timeout_s
                    and self._speed_mps <= 0.5
                ):
                    return True
                self._condition.wait(timeout=0.1)
        return False

    def _wait_for_perception_epoch(
        self,
        newer_than_s: float,
        baseline_prediction_sequence: int,
    ) -> bool:
        if not self.require_perception_epoch:
            return True
        deadline = time.monotonic() + self.reset_timeout_s
        first_post_reset_sim_s: float | None = None
        previous_sim_s: float | None = None
        with self._condition:
            while time.monotonic() < deadline:
                if self._shutdown:
                    return False
                current_sim_s = (
                    self._status_sim_s
                    if self._status_received_s > newer_than_s
                    else None
                )
                if current_sim_s is not None:
                    if (
                        first_post_reset_sim_s is None
                        or (
                            previous_sim_s is not None
                            and current_sim_s < previous_sim_s
                        )
                    ):
                        # A rollback starts a new simulator epoch. Requiring
                        # another full settle period prevents old tracker
                        # state from crossing the reset boundary.
                        first_post_reset_sim_s = current_sim_s
                    previous_sim_s = current_sim_s
                now_s = time.monotonic()
                if perception_epoch_is_ready(
                    reset_wall_s=newer_than_s,
                    now_wall_s=now_s,
                    prediction_received_wall_s=(
                        self._prediction_received_s
                    ),
                    occupancy_received_wall_s=(
                        self._occupancy_grid_received_s
                    ),
                    baseline_prediction_sequence=(
                        baseline_prediction_sequence
                    ),
                    prediction_sequence=self._prediction_sequence,
                    minimum_prediction_samples=(
                        self.perception_epoch_minimum_prediction_samples
                    ),
                    first_post_reset_sim_s=first_post_reset_sim_s,
                    current_sim_s=current_sim_s,
                    settle_sim_s=self.perception_epoch_settle_sim_s,
                    stale_timeout_s=self.stale_timeout_s,
                    require_occupancy_grid=self.require_occupancy_grid,
                ):
                    return True
                self._condition.wait(timeout=0.1)
        return False

    def _wait_planner_ready(self, newer_than_s: float = 0.0) -> bool:
        deadline = time.monotonic() + self.reset_timeout_s
        with self._condition:
            while time.monotonic() < deadline:
                now_s = time.monotonic()
                if (
                    self._planner_ready
                    and self._planner_received_s > newer_than_s
                    and now_s - self._planner_received_s
                    <= self.stale_timeout_s
                    and self._collision_received_s > 0.0
                    and self._collision_received_s > newer_than_s
                    and now_s - self._collision_received_s
                    <= self.stale_timeout_s
                    and not self._collision_present
                    and (
                        not self.require_occupancy_grid
                        or (
                            self._occupancy_grid_valid
                            and self._occupancy_grid_received_s > newer_than_s
                            and now_s - self._occupancy_grid_received_s
                            <= self.stale_timeout_s
                        )
                    )
                ):
                    return True
                self._condition.wait(timeout=0.1)
        return False

    def _failure(
        self,
        reason: str,
        *,
        reset_failed: bool = False,
        disconnected: bool = False,
        aborted: bool = False,
    ) -> tuple[RunMetrics, list[tuple[object, ...]]]:
        return (
            RunMetrics(
                completed=False,
                elapsed_s=0.0,
                progress_m=0.0,
                mean_cte_sq_m2=0.0,
                max_cte_m=0.0,
                rear_mean_cte_sq_m2=0.0,
                rear_max_cte_m=0.0,
                overspeed_s=0.0,
                target_overspeed_sq_integral=0.0,
                unnecessary_brake_sq_integral=0.0,
                brake_saturation_s=0.0,
                reset_failed=reset_failed,
                disconnected=disconnected,
                aborted=aborted,
                reason=reason,
            ),
            [],
        )

    def run_trial(
        self, parameters: Mapping[str, float]
    ) -> tuple[RunMetrics, list[tuple[object, ...]]]:
        route = self.wait_until_ready()
        try:
            self.hold_control(True)
            if not self._wait_vehicle_stopped():
                return self._failure(
                    "vehicle did not stop before reset", reset_failed=True
                )
            self.apply_parameters(parameters)
            reset_ok = False
            pose_reset_seen = False
            for _ in range(2):
                with self._condition:
                    prediction_sequence_before_reset = (
                        self._prediction_sequence
                    )
                scenario_reset_s = self.reset_scenario()
                published_s = (
                    scenario_reset_s
                    if scenario_reset_s is not None
                    else self._publish_ego_reset(route)
                )
                self.reset_localization()
                if not self._wait_for_reset_pose(route, published_s):
                    continue
                pose_reset_seen = True
                if not self._wait_for_perception_epoch(
                    published_s,
                    prediction_sequence_before_reset,
                ):
                    continue
                reset_ok = True
                break
            if not reset_ok:
                if pose_reset_seen and self.require_perception_epoch:
                    return self._failure(
                        "post-reset perception epoch did not settle",
                        reset_failed=True,
                        disconnected=True,
                    )
                return self._failure(
                    "MORAI did not reset Ego to global-path start",
                    reset_failed=True,
                )
            self.reset_tracker()
            if not self._wait_planner_ready(published_s):
                return self._failure(
                    "planner inputs or post-reset occupancy grid did not "
                    "become ready",
                    disconnected=True,
                )
            with self._condition:
                self._target_speed_mps = None
                self._target_received_s = 0.0
                self._throttle_command = None
                self._brake_command = None
                self._steering_command = None
                self._command_received_s = 0.0
                release_status_sequence = self._status_sequence
                release_collision_sequence = (
                    self._collision_episode_sequence
                )
            self.hold_control(False)
        except Exception as error:
            return self._failure(str(error), reset_failed=True)

        target_m = (
            route.length_m - self.completion_margin_m
            if self.course_length_m <= 0.0
            else min(self.course_length_m, route.length_m)
        )
        target_m = max(1.0, target_m)
        target_x, target_y, target_heading = pose_at_progress(
            route.xy, route.lengths, target_m
        )
        started_wall_s = time.monotonic()
        started_sim_s: float | None = None
        previous_sim_s: float | None = None
        elapsed_sim_s = 0.0
        last_sim_advance_wall_s = started_wall_s
        stale_started_s: float | None = None
        stalled_started_sim_s: float | None = None
        last_status_sequence = release_status_sequence
        hint = 0
        previous_odom: tuple[float, float, float] | None = None
        max_progress = 0.0
        max_cte = 0.0
        cte_sq_sum = 0.0
        distance_cte_sq_integral = 0.0
        rear_max_cte = 0.0
        rear_cte_sq_sum = 0.0
        rear_distance_cte_sq_integral = 0.0
        distance_integrated_m = 0.0
        time_cte_sq_integral = 0.0
        rear_time_cte_sq_integral = 0.0
        time_integrated_s = 0.0
        previous_cte: float | None = None
        previous_rear_cte: float | None = None
        sample_count = 0
        overspeed_s = 0.0
        target_overspeed_sq_integral = 0.0
        unnecessary_brake_sq_integral = 0.0
        brake_saturation_s = 0.0
        throttle_saturation_s = 0.0
        local_planner_active_s = 0.0
        local_planner_failure_s = 0.0
        stopped_s = 0.0
        samples: list[tuple[object, ...]] = []
        completed = False
        disconnected = False
        reason = "trial timeout"

        while not self._shutdown:
            now_wall_s = time.monotonic()
            wall_elapsed_s = now_wall_s - started_wall_s
            external_abort = self._external_abort_reason()
            if external_abort:
                reason = external_abort
                disconnected = True
                break
            if wall_elapsed_s >= self.trial_wall_timeout_s:
                reason = (
                    "wall-clock safety timeout while simulator time lagged"
                )
                disconnected = True
                break

            with self._condition:
                odom = self._odom
                speed = self._speed_mps
                target_speed = self._target_speed_mps
                throttle = self._throttle_command
                brake = self._brake_command
                steering_command = self._steering_command
                status_sequence = self._status_sequence
                sim_s = self._status_sim_s
                header_s = self._status_header_s
                actual_steering = self._status_steering_rad
                lateral_acceleration = (
                    self._status_lateral_acceleration_mps2
                )
                collision_sequence = self._collision_episode_sequence
                collision_received_s = self._collision_received_s
                live_samples = {
                    "odometry": (odom, self._odom_received_s),
                    "vehicle_status": (
                        speed,
                        self._status_received_s,
                    ),
                    "target_speed": (
                        target_speed,
                        self._target_received_s,
                    ),
                    "control_command": (
                        (
                            throttle,
                            brake,
                            steering_command,
                        )
                        if throttle is not None
                        and brake is not None
                        and steering_command is not None
                        else None,
                        self._command_received_s,
                    ),
                    "collisions": (
                        collision_sequence,
                        collision_received_s,
                    ),
                }
                if self.require_occupancy_grid:
                    live_samples["occupancy_grid"] = (
                        True if self._occupancy_grid_valid else None,
                        self._occupancy_grid_received_s,
                    )
                stale_inputs = stale_input_names(
                    now_s=now_wall_s,
                    timeout_s=self.stale_timeout_s,
                    samples=live_samples,
                )
                planner_behavior = self._planner_active_behavior
                planner_failsafe = self._planner_failsafe_reason
                planner_dwa_failure = self._planner_dwa_failure_reason
            if stale_inputs:
                stale_started_s = stale_started_s or now_wall_s
                if (
                    now_wall_s - stale_started_s
                    >= self.stale_timeout_s
                ):
                    if dwa_failure_is_controller_infeasible(
                        planner_dwa_failure, stale_inputs
                    ):
                        reason = "DWA failure: " + planner_dwa_failure
                        break
                    details = [
                        "stale input(s): " + ", ".join(stale_inputs),
                        "planner behavior: "
                        + (planner_behavior or "<empty>"),
                    ]
                    if planner_failsafe:
                        details.append(
                            "planner failsafe: " + planner_failsafe
                        )
                    if planner_dwa_failure:
                        details.append(
                            "DWA failure: " + planner_dwa_failure
                        )
                    reason = "; ".join(details)
                    disconnected = True
                    break
                time.sleep(0.05)
                continue
            stale_started_s = None
            if status_sequence <= last_status_sequence:
                time.sleep(0.01)
                continue
            last_status_sequence = status_sequence
            if self.require_device_stamp and sim_s is None:
                reason = "MORAI vehicle status has no device timestamp"
                disconnected = True
                break
            if sim_s is None:
                sim_s = now_wall_s
            if started_sim_s is None:
                started_sim_s = sim_s
                previous_sim_s = sim_s
            if sim_s < previous_sim_s - 1.0e-6:
                reason = "MORAI device timestamp regressed"
                disconnected = True
                break
            if sim_s <= previous_sim_s + 1.0e-9:
                if (
                    now_wall_s - last_sim_advance_wall_s
                    >= self.sim_clock_stale_timeout_s
                ):
                    reason = "MORAI device timestamp stopped advancing"
                    disconnected = True
                    break
                time.sleep(0.01)
                continue
            dt = sim_s - previous_sim_s
            if dt > self.maximum_sim_step_s:
                reason = (
                    "MORAI device timestamp advanced by an excessive step"
                )
                disconnected = True
                break
            previous_sim_s = sim_s
            last_sim_advance_wall_s = now_wall_s
            elapsed_sim_s = sim_s - started_sim_s
            if elapsed_sim_s >= self.trial_timeout_s:
                break
            if previous_odom is not None and math.hypot(
                odom[0] - previous_odom[0], odom[1] - previous_odom[1]
            ) > self.maximum_pose_jump_m:
                reason = "localization pose jumped"
                break
            previous_odom = odom

            control_x = (
                odom[0]
                + self.metric_control_point_x_m * math.cos(odom[2])
            )
            control_y = (
                odom[1]
                + self.metric_control_point_x_m * math.sin(odom[2])
            )
            projection = project_to_route(
                route.xy,
                route.lengths,
                control_x,
                control_y,
                hint=hint,
                yaw_rad=odom[2],
            )
            rear_projection = project_to_route(
                route.xy,
                route.lengths,
                odom[0],
                odom[1],
                hint=projection.segment_index,
                yaw_rad=odom[2],
            )
            progress_jump = projection.progress_m - max_progress
            if (
                sample_count > 0
                and progress_jump > self.maximum_progress_jump_m
            ):
                reason = "route progress jumped to a non-contiguous branch"
                break
            hint = max(hint, projection.segment_index)
            previous_progress = max_progress
            max_progress = max(previous_progress, projection.progress_m)
            max_cte = max(max_cte, projection.cte_m)
            cte_sq = projection.cte_m * projection.cte_m
            rear_max_cte = max(rear_max_cte, rear_projection.cte_m)
            rear_cte_sq = rear_projection.cte_m * rear_projection.cte_m
            cte_sq_sum += cte_sq
            rear_cte_sq_sum += rear_cte_sq
            sample_count += 1
            progress_delta = max_progress - previous_progress
            if progress_delta > 0.0:
                previous_cte_sq = (
                    cte_sq
                    if previous_cte is None
                    else previous_cte * previous_cte
                )
                distance_cte_sq_integral += (
                    0.5 * (previous_cte_sq + cte_sq) * progress_delta
                )
                previous_rear_cte_sq = (
                    rear_cte_sq
                    if previous_rear_cte is None
                    else previous_rear_cte * previous_rear_cte
                )
                rear_distance_cte_sq_integral += (
                    0.5
                    * (previous_rear_cte_sq + rear_cte_sq)
                    * progress_delta
                )
                distance_integrated_m += progress_delta
            if dt > 0.0:
                previous_cte_sq = (
                    cte_sq
                    if previous_cte is None
                    else previous_cte * previous_cte
                )
                time_cte_sq_integral += (
                    0.5 * (previous_cte_sq + cte_sq) * dt
                )
                previous_rear_cte_sq = (
                    rear_cte_sq
                    if previous_rear_cte is None
                    else previous_rear_cte * previous_rear_cte
                )
                rear_time_cte_sq_integral += (
                    0.5 * (previous_rear_cte_sq + rear_cte_sq) * dt
                )
                time_integrated_s += dt
            previous_cte = projection.cte_m
            previous_rear_cte = rear_projection.cte_m
            if speed > self.overspeed_mps:
                overspeed_s += dt
            target_overspeed = max(
                0.0,
                speed
                - target_speed
                - self.brake_metric_speed_deadband_mps,
            )
            target_overspeed_sq_integral += (
                target_overspeed * target_overspeed * dt
            )
            if (
                speed
                <= target_speed + self.brake_metric_speed_deadband_mps
            ):
                unnecessary_brake_sq_integral += brake * brake * dt
            if brake >= self.brake_saturation_threshold:
                brake_saturation_s += dt
            if throttle >= self.brake_saturation_threshold:
                throttle_saturation_s += dt
            local_planner_active = planner_behavior == "perception_mission"
            if local_planner_active:
                local_planner_active_s += dt
            if planner_dwa_failure:
                local_planner_failure_s += dt
            if speed < 1.0 / 3.6:
                stopped_s += dt
            collision_count = max(
                0, collision_sequence - release_collision_sequence
            )
            samples.append(
                (
                    round(elapsed_sim_s, 3),
                    round(wall_elapsed_s, 3),
                    round(sim_s, 6),
                    round(header_s or 0.0, 6),
                    round(odom[0], 4),
                    round(odom[1], 4),
                    round(control_x, 4),
                    round(control_y, 4),
                    round(speed, 4),
                    round(target_speed, 4),
                    round(throttle, 4),
                    round(brake, 4),
                    round(steering_command, 4),
                    round(actual_steering, 5),
                    round(lateral_acceleration, 5),
                    round(projection.cte_m, 4),
                    round(rear_projection.cte_m, 4),
                    round(max_progress, 3),
                    hint,
                    int(local_planner_active),
                    int(bool(planner_dwa_failure)),
                    planner_dwa_failure,
                    collision_count,
                )
            )

            if collision_count > 0:
                reason = "collision detected during trial"
                break

            if max_progress >= target_m:
                terminal_distance = math.hypot(
                    control_x - target_x,
                    control_y - target_y,
                )
                terminal_heading_error = abs(
                    math.atan2(
                        math.sin(odom[2] - target_heading),
                        math.cos(odom[2] - target_heading),
                    )
                )
                if (
                    terminal_distance
                    <= self.completion_position_tolerance_m
                    and terminal_heading_error
                    <= self.completion_heading_tolerance_rad
                ):
                    completed = True
                    reason = "completed"
                    break
            if max(
                projection.cte_m, rear_projection.cte_m
            ) > self.divergence_cte_m:
                reason = "cross-track error diverged"
                break
            if abs(projection.heading_error_rad) > self.wrong_way_heading_rad:
                reason = "vehicle heading reversed relative to route"
                break
            if elapsed_sim_s > 8.0 and speed < 1.0 / 3.6:
                stalled_started_sim_s = (
                    stalled_started_sim_s or elapsed_sim_s
                )
                if elapsed_sim_s - stalled_started_sim_s > 4.0:
                    reason = "vehicle stalled"
                    break
            else:
                stalled_started_sim_s = None
            time.sleep(0.01)

        wall_elapsed_s = time.monotonic() - started_wall_s
        if self._shutdown:
            reason = "tuner shutdown"
        try:
            self.hold_control(True)
        except Exception as error:
            self.node.get_logger().error(
                f"failed to hold after trial: {error}"
            )
        mean_cte_sq = cte_sq_sum / max(1, sample_count)
        rear_mean_cte_sq = rear_cte_sq_sum / max(1, sample_count)
        distance_cte_mse = (
            distance_cte_sq_integral / distance_integrated_m
            if distance_integrated_m > 0.0
            else mean_cte_sq
        )
        rear_distance_cte_mse = (
            rear_distance_cte_sq_integral / distance_integrated_m
            if distance_integrated_m > 0.0
            else rear_mean_cte_sq
        )
        time_cte_mse = (
            time_cte_sq_integral / time_integrated_s
            if time_integrated_s > 0.0
            else mean_cte_sq
        )
        rear_time_cte_mse = (
            rear_time_cte_sq_integral / time_integrated_s
            if time_integrated_s > 0.0
            else rear_mean_cte_sq
        )
        metrics = RunMetrics(
            completed=completed,
            elapsed_s=round(elapsed_sim_s, 3),
            progress_m=round(max_progress, 3),
            mean_cte_sq_m2=round(mean_cte_sq, 6),
            max_cte_m=round(max_cte, 4),
            rear_mean_cte_sq_m2=round(rear_mean_cte_sq, 6),
            rear_max_cte_m=round(rear_max_cte, 4),
            overspeed_s=round(overspeed_s, 3),
            distance_cte_mse_m2=round(distance_cte_mse, 6),
            time_cte_mse_m2=round(time_cte_mse, 6),
            rear_distance_cte_mse_m2=round(
                rear_distance_cte_mse, 6
            ),
            rear_time_cte_mse_m2=round(rear_time_cte_mse, 6),
            wall_elapsed_s=round(wall_elapsed_s, 3),
            real_time_factor=round(
                elapsed_sim_s / wall_elapsed_s
                if wall_elapsed_s > 0.0
                else 0.0,
                6,
            ),
            target_overspeed_sq_integral=round(
                target_overspeed_sq_integral, 6
            ),
            unnecessary_brake_sq_integral=round(
                unnecessary_brake_sq_integral, 6
            ),
            brake_saturation_s=round(brake_saturation_s, 3),
            throttle_saturation_s=round(throttle_saturation_s, 3),
            collision_count=max(
                0,
                self._collision_episode_sequence
                - release_collision_sequence,
            ),
            local_planner_active_s=round(local_planner_active_s, 3),
            local_planner_failure_s=round(local_planner_failure_s, 3),
            stopped_s=round(stopped_s, 3),
            disconnected=disconnected,
            aborted=self._shutdown,
            reason=reason,
        )
        return metrics, samples
