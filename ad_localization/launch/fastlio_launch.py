"""Shared construction for the mutually-exclusive FastLIO leaf launches."""

from __future__ import annotations

import math
from pathlib import Path

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessStart
from launch.events import matches_action
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode, Node
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState
from lifecycle_msgs.msg import Transition
import yaml


def _rotation_from_rpy(rpy: dict[str, object]) -> list[list[float]]:
    """Rz(yaw) @ Ry(pitch) @ Rx(roll) rotation matrix."""
    roll, pitch, yaw = (float(rpy[axis]) for axis in ("roll", "pitch", "yaw"))
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def _transpose(matrix: list[list[float]]) -> list[list[float]]:
    return [[matrix[column][row] for column in range(3)] for row in range(3)]


def _multiply(lhs: list[list[float]], rhs: list[list[float]]) -> list[list[float]]:
    return [
        [sum(lhs[row][index] * rhs[index][column] for index in range(3)) for column in range(3)]
        for row in range(3)
    ]


def _multiply_vector(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [sum(matrix[row][column] * vector[column] for column in range(3)) for row in range(3)]


def _flatten(matrix: list[list[float]]) -> list[float]:
    return [matrix[row][column] for row in range(3) for column in range(3)]


def _position(sensor: dict[str, object]) -> list[float]:
    return [float(sensor["position_m"][axis]) for axis in ("x", "y", "z")]


def _sensor_parent_position(document: dict[str, object]) -> list[float]:
    convention = document["coordinate_convention"]
    static_frames = convention.get("static_frames")
    if static_frames is None:
        return [0.0, 0.0, 0.0]
    rear_axle = static_frames["rear_axle"]
    if convention["sensor_parent_frame"] != rear_axle["frame_id"]:
        raise RuntimeError("sensor_parent_frame must match the rear axle frame")
    return [float(rear_axle["position_m"][axis]) for axis in ("x", "y", "z")]


def _add_position(lhs: list[float], rhs: list[float]) -> list[float]:
    return [lhs[index] + rhs[index] for index in range(3)]


def _load_fastlio_geometry(sensor_config: Path, requested_profile: str) -> dict[str, object]:
    document = yaml.safe_load(sensor_config.read_text(encoding="utf-8"))
    profile_name = requested_profile or document["active_profile"]
    sensors = document["profiles"][profile_name]["sensors"]
    base_frame = str(document["coordinate_convention"]["base_frame"])
    gps = sensors["gps"]
    imu = sensors["imu"]
    lidar = sensors["lidar"]

    sensor_parent_position = _sensor_parent_position(document)
    gps_position = _add_position(sensor_parent_position, _position(gps))
    imu_position = _add_position(sensor_parent_position, _position(imu))
    lidar_position = _add_position(sensor_parent_position, _position(lidar))
    base_rotation_imu = _rotation_from_rpy(imu["rpy_rad"])
    base_rotation_lidar = _rotation_from_rpy(lidar["rpy_rad"])
    imu_rotation_base = _transpose(base_rotation_imu)
    imu_rotation_lidar = _multiply(imu_rotation_base, base_rotation_lidar)
    imu_translation_lidar = _multiply_vector(
        imu_rotation_base,
        [lidar_position[index] - imu_position[index] for index in range(3)],
    )

    return {
        "base_frame": base_frame,
        "gnss_lever_arm_m": gps_position,
        "imu_frame": str(imu["frame_id"]),
        "imu_mount_rpy_rad": [
            float(imu["rpy_rad"][axis]) for axis in ("roll", "pitch", "yaw")
        ],
        "lidar_frame": str(lidar["frame_id"]),
        "base_to_imu_T": imu_position,
        "base_to_imu_R": _flatten(base_rotation_imu),
        "extrinsic_T": imu_translation_lidar,
        "extrinsic_R": _flatten(imu_rotation_lidar),
    }


def _parse_optional_xy(value: str) -> list[float]:
    stripped = value.strip()
    if not stripped:
        return []
    fields = [field.strip() for field in stripped.split(",")]
    if len(fields) != 2:
        raise RuntimeError(
            "initial_position_override_xy_m must contain two comma-separated values"
        )
    try:
        xy = [float(field) for field in fields]
    except ValueError as error:
        raise RuntimeError(
            "initial_position_override_xy_m must contain two comma-separated numbers"
        ) from error
    if not all(math.isfinite(coordinate) for coordinate in xy):
        raise RuntimeError("initial_position_override_xy_m values must be finite")
    return xy


def _parse_platform_profile(value: str) -> str:
    profile = str(value).strip()
    if profile not in {"morai", "real_hardware"}:
        raise RuntimeError(
            "platform_profile must be exactly 'morai' or 'real_hardware'"
        )
    return profile


# Shared adapter behavior for MORAI; also used by hybrid_localization.launch.py
# so these tuned values live in exactly one place.
MORAI_ADAPTER_DEFAULTS = {
    "initial_orientation_source": "vehicle_status",
    # MORAI reports true-north heading while this stack projects GNSS into
    # the UTM 52N grid. At K-City CP14, meridian convergence is
    # -1.346866 degrees, so rotate the initial body attitude into grid yaw.
    "initial_orientation_yaw_offset_rad": -0.02350724531030645,
    "initial_position_sample_count": 30,
    # Consume the bridge-normalized receipt-time header. Do not bypass that
    # shared clock-domain gate by substituting the raw device field.
    "wheel_use_device_timestamp": False,
    # MolitComp03 constructs RMC track from sensor/vehicle heading rather
    # than Rigidbody velocity, so it is not a valid course-over-ground
    # measurement for sideslip estimation.
    "gps_course_enabled": False,
    "gps_course_yaw_offset_rad": 0.0,
}


def _adapter_overrides(
    sensor_geometry, initial_position_override_xy_m, imu_topic
):
    overrides = {
        "status_topic": LaunchConfiguration("status_topic"),
        "gnss_lever_arm_m": sensor_geometry["gnss_lever_arm_m"],
        "localization_backend": "fastlio",
        "imu_topic": imu_topic,
        **MORAI_ADAPTER_DEFAULTS,
    }
    if initial_position_override_xy_m:
        overrides["initial_position_override_xy_m"] = initial_position_override_xy_m
    return overrides


def _make_adapter(sensor_geometry, initial_position_override_xy_m, imu_topic):
    return LifecycleNode(
        package="ad_localization",
        executable="ad_localization_node",
        name="ad_localization",
        namespace="",
        output="screen",
        parameters=[
            LaunchConfiguration("adapter_config"),
            _adapter_overrides(
                sensor_geometry, initial_position_override_xy_m, imu_topic
            ),
        ],
    )


def _make_autostart_handlers(adapter):
    autostart = LaunchConfiguration("autostart")
    configure = RegisterEventHandler(
        OnProcessStart(
            target_action=adapter,
            on_start=[
                EmitEvent(
                    event=ChangeState(
                        lifecycle_node_matcher=matches_action(adapter),
                        transition_id=Transition.TRANSITION_CONFIGURE,
                    ),
                    condition=IfCondition(autostart),
                )
            ],
        )
    )
    activate = RegisterEventHandler(
        OnStateTransition(
            target_lifecycle_node=adapter,
            goal_state="inactive",
            entities=[
                EmitEvent(
                    event=ChangeState(
                        lifecycle_node_matcher=matches_action(adapter),
                        transition_id=Transition.TRANSITION_ACTIVATE,
                    ),
                    condition=IfCondition(autostart),
                )
            ],
        )
    )
    return [configure, activate]


def _fastlio_parameters(
    mode: str,
    sensor_geometry: dict[str, object],
    platform_profile: str,
    map_path=None,
    imu_topic="/ad/sensors/imu/data",
):
    platform_profile = _parse_platform_profile(platform_profile)
    parameters: dict[str, object] = {
        "mode": mode,
        "base_frame": sensor_geometry["base_frame"],
        "imu_frame": sensor_geometry["imu_frame"],
        "lidar_frame": sensor_geometry["lidar_frame"],
        "base_to_imu_T": sensor_geometry["base_to_imu_T"],
        "base_to_imu_R": sensor_geometry["base_to_imu_R"],
        "common.imu_topic": imu_topic,
        "extrinsic_T": sensor_geometry["extrinsic_T"],
        "extrinsic_R": sensor_geometry["extrinsic_R"],
        # The MORAI tunnel is longitudinally degenerate with one VLP-16, so
        # retain the competition-provided forward speed in both modes.
        "wheel_velocity.enabled": True,
        # Adapter headers preserve the bridge-normalized measurement stamp.
        # Use it for wheel age checks instead of callback-arrival time.
        "wheel_velocity.header_stamp_is_device_time": False,
        # The competition status packet exposes longitudinal speed only; the
        # vehicle model is nonholonomic and MORAI RMC is not a true COG source.
        "wheel_velocity.use_nonholonomic_constraints": True,
        # With a canonical measurement-time header, integrate the position
        # guard over the LiDAR scan interval; do not invent a raw-device-time
        # distance outside the normalized clock domain.
        "wheel_velocity.preserve_predicted_longitudinal_position": True,
        "wheel_velocity.position_integration_period_sec": 0.0,
        # Flattening and publishing the growing ikd-tree on a single executor
        # stalls mapping odometry. The mapping save service still snapshots it
        # atomically; fixed localization publishes its immutable map once.
        "publish.map_en": mode == "localization",
        # The planner consumes odometry, not FastLIO's unbounded Path.  Keep a
        # downsampled registered scan for RViz while avoiding two full-cloud
        # transforms/serializations on FastLIO's single executor.
        "publish.path_en": False,
        "publish.dense_publish_en": False,
        "publish.scan_bodyframe_pub_en": False,
        # Online mapping must retain LiDAR's observable lateral correction.
        # Fixed-map localization also needs the correction to register to the
        # immutable map, so the planar mapping clamp stays disabled in both.
        "wheel_velocity.preserve_predicted_lateral_position_in_mapping": False,
        # The 2025 Velodyne profile rejected near-field body/ground clutter.
        "preprocess.blind": 2.0,
        # MORAI renders a snapshot cloud; real spinning LiDAR retains rolling
        # point timing and FAST-LIO's internal undistortion path.
        "preprocess.scan_timing_mode": (
            "instantaneous"
            if platform_profile == "morai"
            else "rolling"
        ),
        "map_path": map_path,
    }
    return parameters


def _launch_setup(context, mode: str):
    platform_profile = _parse_platform_profile(
        LaunchConfiguration("platform_profile").perform(context)
    )
    sensor_config = Path(LaunchConfiguration("sensor_config").perform(context))
    sensor_profile = LaunchConfiguration("sensor_profile").perform(context)
    sensor_geometry = _load_fastlio_geometry(sensor_config, sensor_profile)
    initial_position_override_xy_m = _parse_optional_xy(
        LaunchConfiguration("initial_position_override_xy_m").perform(context)
    )
    imu_topic = LaunchConfiguration("imu_topic")
    adapter = _make_adapter(
        sensor_geometry, initial_position_override_xy_m, imu_topic
    )
    parameters = _fastlio_parameters(
        mode,
        sensor_geometry,
        platform_profile,
        LaunchConfiguration("map_output_path")
        if mode == "mapping"
        else LaunchConfiguration("map_path"),
        imu_topic,
    )

    fastlio = Node(
        package="fast_lio",
        executable="fastlio_mapping",
        name="fastlio",
        output="screen",
        parameters=[LaunchConfiguration("fastlio_config"), parameters],
    )
    return [adapter, fastlio, *_make_autostart_handlers(adapter)]


def generate_fastlio_launch_description(mode: str, get_package_share_directory):
    share = Path(get_package_share_directory("ad_localization"))
    description_share = Path(get_package_share_directory("ad_description"))
    entities = [
        DeclareLaunchArgument("platform_profile", default_value="morai"),
        DeclareLaunchArgument(
            "imu_topic", default_value="/ad/sensors/imu/data"
        ),
        DeclareLaunchArgument(
            "adapter_config", default_value=str(share / "config" / "localization.yaml")
        ),
        DeclareLaunchArgument(
            "fastlio_config", default_value=str(share / "config" / "fastlio.yaml")
        ),
        DeclareLaunchArgument("status_topic", default_value="/ad/vehicle/status"),
        DeclareLaunchArgument("autostart", default_value="true"),
        DeclareLaunchArgument(
            "initial_position_override_xy_m",
            default_value="",
            description=(
                "Optional known base_link x,y in odom, used to avoid GNSS "
                "quantization at a surveyed start pose."
            ),
        ),
        DeclareLaunchArgument(
            "sensor_config",
            default_value=str(description_share / "config" / "sensor_mounts.yaml"),
        ),
        DeclareLaunchArgument(
            "sensor_profile",
            default_value="",
            description="Blank selects active_profile from sensor_mounts.yaml.",
        ),
    ]
    if mode == "mapping":
        entities.append(
            DeclareLaunchArgument(
                "map_output_path",
                default_value=str(share / "maps" / "cp14_to_cp15.pcd"),
                description="Destination PCD path used by the FastLIO map-save service.",
            )
        )
    else:
        entities.append(
            DeclareLaunchArgument(
                "map_path",
                default_value=str(share / "maps" / "cp14_to_cp15.pcd"),
                description="Readable fixed PCD map; FastLIO rejects an empty path.",
            )
        )
    entities.append(OpaqueFunction(function=lambda context: _launch_setup(context, mode)))
    return LaunchDescription(entities)
