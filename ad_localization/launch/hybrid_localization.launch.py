"""GNSS approach, fixed-map FastLIO tunnel, and GNSS finish localization."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode, Node


# launch/ is not an installed Python package, so load the shared module by path.
_spec = importlib.util.spec_from_file_location(
    "ad_localization_hybrid_fastlio_common",
    Path(__file__).with_name("fastlio_launch.py"),
)
_common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_common)


def _adapter_overrides(sensor_geometry, imu_topic):
    return {
        "status_topic": LaunchConfiguration("status_topic"),
        "gnss_lever_arm_m": sensor_geometry["gnss_lever_arm_m"],
        "localization_backend": "hybrid",
        "imu_topic": imu_topic,
        **_common.MORAI_ADAPTER_DEFAULTS,
        "initial_pose_topic": "/ad/localization/input/adapter_initial_pose",
    }


def _gnss_overrides(sensor_geometry, imu_topic):
    return {
        "gnss_lever_arm_m": sensor_geometry["gnss_lever_arm_m"],
        "imu_frame": sensor_geometry["imu_frame"],
        "imu_mount_rpy_rad": sensor_geometry["imu_mount_rpy_rad"],
        "imu_topic": imu_topic,
        "output_odometry_topic": "/ad/localization/backends/gnss_imu/odometry",
        "publish_tf": False,
    }


def _hybrid_fastlio_parameters(
    sensor_geometry, platform_profile, map_path, imu_topic
):
    parameters = _common._fastlio_parameters(
        "localization", sensor_geometry, platform_profile, map_path, imu_topic
    )
    parameters.update(
        {
            "odometry_topic": "/ad/localization/backends/fastlio/odometry",
            "publish_tf": False,
        }
    )
    return parameters


def _launch_setup(context):
    platform_profile = _common._parse_platform_profile(
        LaunchConfiguration("platform_profile").perform(context)
    )
    sensor_config = Path(LaunchConfiguration("sensor_config").perform(context))
    sensor_profile = LaunchConfiguration("sensor_profile").perform(context)
    sensor_geometry = _common._load_fastlio_geometry(sensor_config, sensor_profile)
    imu_topic = LaunchConfiguration("imu_topic")
    adapter = LifecycleNode(
        package="ad_localization",
        executable="ad_localization_node",
        name="ad_localization",
        namespace="",
        output="screen",
        parameters=[
            LaunchConfiguration("adapter_config"),
            _adapter_overrides(sensor_geometry, imu_topic),
        ],
    )
    gnss = Node(
        package="ad_localization",
        executable="gnss_imu_localization_node",
        name="gnss_imu_localization",
        output="screen",
        parameters=[
            LaunchConfiguration("gnss_imu_config"),
            _gnss_overrides(sensor_geometry, imu_topic),
        ],
    )
    fastlio = Node(
        package="fast_lio",
        executable="fastlio_mapping",
        name="fastlio",
        output="screen",
        parameters=[
            LaunchConfiguration("fastlio_config"),
            _hybrid_fastlio_parameters(
                sensor_geometry,
                platform_profile,
                LaunchConfiguration("map_path"),
                imu_topic,
            ),
        ],
    )
    handoff = Node(
        package="ad_localization",
        executable="localization_handoff_node",
        name="localization_handoff",
        output="screen",
        parameters=[LaunchConfiguration("hybrid_config")],
    )
    return [
        adapter,
        gnss,
        fastlio,
        handoff,
        *_common._make_autostart_handlers(adapter),
    ]


def generate_launch_description():
    share = Path(get_package_share_directory("ad_localization"))
    description_share = Path(get_package_share_directory("ad_description"))
    return LaunchDescription(
        [
            DeclareLaunchArgument("platform_profile", default_value="morai"),
            DeclareLaunchArgument(
                "imu_topic", default_value="/ad/sensors/imu/data"
            ),
            DeclareLaunchArgument(
                "adapter_config",
                default_value=str(share / "config" / "localization.yaml"),
            ),
            DeclareLaunchArgument(
                "gnss_imu_config",
                default_value=str(share / "config" / "gnss_imu.yaml"),
            ),
            DeclareLaunchArgument(
                "fastlio_config",
                default_value=str(share / "config" / "fastlio.yaml"),
            ),
            DeclareLaunchArgument(
                "hybrid_config",
                default_value=str(share / "config" / "hybrid.yaml"),
            ),
            DeclareLaunchArgument(
                "map_path",
                default_value=str(share / "maps" / "cp14_to_cp15.pcd"),
                description="Readable CP14-to-CP15 fixed-map PCD.",
            ),
            DeclareLaunchArgument("status_topic", default_value="/ad/vehicle/status"),
            DeclareLaunchArgument("autostart", default_value="true"),
            DeclareLaunchArgument(
                "sensor_config",
                default_value=str(
                    description_share / "config" / "sensor_mounts.yaml"
                ),
            ),
            DeclareLaunchArgument(
                "sensor_profile",
                default_value="",
                description="Blank selects active_profile from sensor_mounts.yaml.",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
