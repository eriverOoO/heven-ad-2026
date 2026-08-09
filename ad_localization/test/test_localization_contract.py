"""Static and opt-in runtime contracts for the localization adapter node."""

from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
RUN_INTEGRATION = os.environ.get("AD_RUN_ROS_INTEGRATION") == "1"


def _spin_until(node, predicate, timeout_sec: float = 5.0):
    import rclpy

    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
        if predicate():
            return True
    return False


def _change_state(node, client, transition_id: int, expected: bool = True):
    from lifecycle_msgs.srv import ChangeState

    request = ChangeState.Request()
    request.transition.id = transition_id
    future = client.call_async(request)
    assert _spin_until(node, future.done), "lifecycle transition timed out"
    assert future.result() is not None
    assert future.result().success is expected


def _stop_process(process: subprocess.Popen[str]):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGINT)
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)


@pytest.mark.skipif(not RUN_INTEGRATION, reason="set AD_RUN_ROS_INTEGRATION=1")
def test_lifecycle_adapts_inputs_retries_initial_pose_and_never_publishes_dynamic_tf():
    import rclpy
    from ad_morai_interfaces.msg import EgoVehicleStatus
    from diagnostic_msgs.msg import DiagnosticArray
    from geometry_msgs.msg import Pose2D, PoseStamped, TwistWithCovarianceStamped
    from lifecycle_msgs.msg import Transition
    from lifecycle_msgs.srv import ChangeState
    from nav_msgs.msg import Odometry
    from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
    from sensor_msgs.msg import Imu, NavSatFix, NavSatStatus
    from tf2_msgs.msg import TFMessage

    process_log = tempfile.TemporaryFile(mode="w+")
    process = subprocess.Popen(
        ["ros2", "run", "ad_localization", "ad_localization_node"],
        stdout=process_log,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    rclpy.init()
    probe = rclpy.create_node("ad_localization_adapter_contract_probe")
    lifecycle = probe.create_client(ChangeState, "/ad_localization/change_state")
    reliable = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
    initial_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)

    gnss_poses: list[PoseStamped] = []
    initial_poses: list[PoseStamped] = []
    wheel_speeds: list[TwistWithCovarianceStamped] = []
    poses: list[Pose2D] = []
    diagnostics: list[DiagnosticArray] = []
    dynamic_tf: list[TFMessage] = []
    static_tf: list[TFMessage] = []
    probe.create_subscription(
        PoseStamped, "/ad/localization/input/gnss_pose", gnss_poses.append, reliable
    )
    probe.create_subscription(
        PoseStamped,
        "/ad/localization/input/initial_pose",
        initial_poses.append,
        initial_qos,
    )
    probe.create_subscription(
        TwistWithCovarianceStamped,
        "/ad/localization/input/wheel_speed",
        wheel_speeds.append,
        reliable,
    )
    probe.create_subscription(Pose2D, "/ad/localization/pose2d", poses.append, reliable)
    probe.create_subscription(DiagnosticArray, "/diagnostics", diagnostics.append, reliable)
    probe.create_subscription(TFMessage, "/tf", dynamic_tf.append, reliable)
    probe.create_subscription(TFMessage, "/tf_static", static_tf.append, reliable)

    gps_publisher = probe.create_publisher(
        NavSatFix, "/ad/sensors/gps/fix", qos_profile_sensor_data
    )
    imu_publisher = probe.create_publisher(
        Imu, "/ad/sensors/imu/data", qos_profile_sensor_data
    )
    status_publisher = probe.create_publisher(
        EgoVehicleStatus, "/ad/vehicle/status", qos_profile_sensor_data
    )
    odometry_publisher = probe.create_publisher(
        Odometry, "/ad/localization/odometry", reliable
    )

    try:
        assert lifecycle.wait_for_service(timeout_sec=8.0)
        _change_state(probe, lifecycle, Transition.TRANSITION_CONFIGURE)
        _change_state(probe, lifecycle, Transition.TRANSITION_ACTIVATE)
        assert _spin_until(
            probe,
            lambda: (
                gps_publisher.get_subscription_count() == 1
                and imu_publisher.get_subscription_count() == 1
                and status_publisher.get_subscription_count() == 1
                and odometry_publisher.get_subscription_count() == 1
            ),
            3.0,
        )

        invalid_imu = Imu()
        invalid_imu.header.frame_id = "imu_link"
        invalid_imu.orientation.w = 0.0
        invalid_imu_observed = False
        deadline = time.monotonic() + 2.5
        while time.monotonic() < deadline and not invalid_imu_observed:
            invalid_imu.header.stamp = probe.get_clock().now().to_msg()
            imu_publisher.publish(invalid_imu)
            invalid_imu_observed = _spin_until(
                probe,
                lambda: bool(diagnostics)
                and int(
                    next(
                        value.value
                        for value in diagnostics[-1].status[0].values
                        if value.key == "imu_dropped"
                    )
                )
                >= 1,
                0.25,
            )
        if not invalid_imu_observed:
            _stop_process(process)
            process_log.seek(0)
            pytest.fail("invalid IMU was not counted:\n" + process_log.read())
        diagnostic_values = {
            value.key: int(value.value)
            for value in diagnostics[-1].status[0].values
            if value.key in {"imu_dropped", "imu_accepted"}
        }
        assert diagnostic_values["imu_dropped"] >= 1
        assert diagnostic_values["imu_accepted"] == 0
        diagnostics.clear()

        now = probe.get_clock().now().to_msg()
        fix = NavSatFix()
        fix.header.stamp = now
        fix.header.frame_id = "gps_link"
        fix.status.status = NavSatStatus.STATUS_FIX
        fix.latitude = 37.2390904486269
        fix.longitude = 126.773066537479
        fix.altitude = 29.5769634246826
        imu = Imu()
        imu.header.stamp = now
        imu.header.frame_id = "imu_link"
        imu.orientation.w = 1.0
        reverse = EgoVehicleStatus()
        reverse.header.stamp = now
        reverse.gear = 2
        reverse.velocity.x = 3.0

        for _ in range(8):
            gps_publisher.publish(fix)
            imu_publisher.publish(imu)
            status_publisher.publish(reverse)
            if _spin_until(
                probe,
                lambda: bool(gnss_poses and initial_poses and wheel_speeds),
                timeout_sec=0.25,
            ):
                break
        assert gnss_poses
        assert len(initial_poses) >= 1
        assert wheel_speeds[-1].twist.twist.linear.x == -3.0
        initial_count = len(initial_poses)
        assert _spin_until(probe, lambda: len(initial_poses) > initial_count, 1.0)

        odometry = Odometry()
        odometry.header.stamp = probe.get_clock().now().to_msg()
        odometry.header.frame_id = "odom"
        odometry.child_frame_id = "base_link"
        odometry.pose.pose.position.x = 8.0
        odometry.pose.pose.position.y = -2.0
        odometry.pose.pose.orientation.w = 1.0
        odometry_publisher.publish(odometry)
        assert _spin_until(probe, lambda: bool(poses), 1.0)
        assert poses[-1].x == 8.0
        acknowledged_count = len(initial_poses)
        _spin_until(probe, lambda: False, 0.5)
        assert len(initial_poses) == acknowledged_count

        no_fix = NavSatFix()
        no_fix.header.stamp = probe.get_clock().now().to_msg()
        no_fix.status.status = NavSatStatus.STATUS_NO_FIX
        prior_gnss_count = len(gnss_poses)
        gps_publisher.publish(no_fix)
        _spin_until(probe, lambda: bool(diagnostics), 1.2)
        assert len(gnss_poses) == prior_gnss_count
        assert diagnostics

        transforms = [item for message in static_tf for item in message.transforms]
        assert any(
            item.header.frame_id == "map" and item.child_frame_id == "odom"
            for item in transforms
        )
        assert not [item for message in dynamic_tf for item in message.transforms]

        _change_state(probe, lifecycle, Transition.TRANSITION_DEACTIVATE)
        prior_wheel_count = len(wheel_speeds)
        reverse.header.stamp = probe.get_clock().now().to_msg()
        status_publisher.publish(reverse)
        _spin_until(probe, lambda: False, 0.3)
        assert len(wheel_speeds) == prior_wheel_count
        _change_state(probe, lifecycle, Transition.TRANSITION_CLEANUP)
        _change_state(probe, lifecycle, Transition.TRANSITION_UNCONFIGURED_SHUTDOWN)
    finally:
        probe.destroy_node()
        rclpy.shutdown()
        _stop_process(process)
        process_log.close()

    assert process.returncode == 0


def _run_rejected_configuration(parameters: list[str]) -> str:
    import rclpy
    from lifecycle_msgs.msg import Transition
    from lifecycle_msgs.srv import ChangeState

    process = subprocess.Popen(
        [
            "ros2",
            "run",
            "ad_localization",
            "ad_localization_node",
            "--ros-args",
            *parameters,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    rclpy.init()
    probe = rclpy.create_node("ad_localization_rejection_probe")
    lifecycle = probe.create_client(ChangeState, "/ad_localization/change_state")
    try:
        assert lifecycle.wait_for_service(timeout_sec=8.0)
        _change_state(
            probe,
            lifecycle,
            Transition.TRANSITION_CONFIGURE,
            expected=False,
        )
    finally:
        probe.destroy_node()
        rclpy.shutdown()
        _stop_process(process)
    assert process.returncode == 0
    assert process.stdout is not None
    return process.stdout.read()


@pytest.mark.skipif(not RUN_INTEGRATION, reason="set AD_RUN_ROS_INTEGRATION=1")
def test_invalid_configuration_fails_closed():
    rejected = (
        (["-p", "gnss_pose_topic:=/localization/gnss_pose"], "under /ad/"),
        (["-p", "map_frame:=/map"], "relative TF frame"),
        (["-p", "input_timeout_sec:=0.0"], "positive"),
        (["-p", "input_future_tolerance_sec:=-0.1"], "nonnegative"),
        (["-p", "wheel_speed_variance:=0.0"], "positive"),
        (["-p", "utm_epsg:=32651"], "32652"),
        (["-p", "gnss_lever_arm_m:=[0.0,0.0]"], "exactly three"),
    )
    for parameters, expected_reason in rejected:
        assert expected_reason in _run_rejected_configuration(parameters)
