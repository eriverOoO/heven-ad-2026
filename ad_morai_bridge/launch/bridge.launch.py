from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
import yaml


SUPPORTED_PLATFORM_PROFILES = {"morai", "real_hardware"}
SUPPORTED_POINT_TIMING_REQUESTS = {"auto", "azimuth", "zero"}
PUBLIC_POINTS_TOPIC = "/ad/sensors/lidar/points"
MORAI_SYNTHETIC_TIME_TOPIC = (
    "/ad/sensors/lidar/points_with_synthetic_time"
)


def _resolve_point_timing_mode(
    platform_profile: str, requested_mode: str
) -> str:
    if platform_profile not in SUPPORTED_PLATFORM_PROFILES:
        raise ValueError(
            "platform_profile must be exactly 'morai' or 'real_hardware'"
        )
    if requested_mode not in SUPPORTED_POINT_TIMING_REQUESTS:
        raise ValueError(
            "velodyne_point_timing_mode must be 'auto', 'azimuth', or 'zero'"
        )
    expected = "zero" if platform_profile == "morai" else "azimuth"
    if requested_mode != "auto" and requested_mode != expected:
        raise ValueError(
            f"velodyne_point_timing_mode={requested_mode!r} is incompatible "
            f"with platform_profile={platform_profile!r}"
        )
    return expected


def _load_vlp16_parameters(package_share: Path) -> dict:
    """The stock VLP16 transform parameters shipped by velodyne_pointcloud."""
    config_path = (
        package_share
        / "config"
        / "VLP16-velodyne_transform_node-params.yaml"
    )
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    parameters = dict(document["velodyne_transform_node"]["ros__parameters"])
    parameters["calibration"] = str(package_share / "params" / "VLP16db.yaml")
    return parameters


def _launch_measurement_compatibility(context):
    platform_profile = LaunchConfiguration("platform_profile").perform(context)
    if platform_profile not in SUPPORTED_PLATFORM_PROFILES:
        raise ValueError(
            "platform_profile must be exactly 'morai' or 'real_hardware'"
        )
    enabled = IfCondition(
        LaunchConfiguration("measurement_compatibility_enabled")
    ).evaluate(context)
    if not enabled:
        return []
    if platform_profile != "morai":
        raise ValueError(
            "measurement compatibility is MORAI-only and cannot be enabled "
            "for real_hardware"
        )
    return [
        Node(
            package="ad_morai_bridge",
            executable="ad_measurement_compatibility",
            name="ad_measurement_compatibility",
            output="screen",
            parameters=[
                LaunchConfiguration("measurement_compatibility_config")
            ],
        )
    ]


def _launch_velodyne_chain(context):
    platform_profile = LaunchConfiguration("platform_profile").perform(context)
    requested_timing_mode = LaunchConfiguration(
        "velodyne_point_timing_mode"
    ).perform(context)
    point_timing_mode = _resolve_point_timing_mode(
        platform_profile, requested_timing_mode
    )
    enabled = IfCondition(
        LaunchConfiguration("enable_velodyne_points")
    ).evaluate(context)
    if not enabled:
        return []

    pointcloud_share = Path(get_package_share_directory("velodyne_pointcloud"))
    transform_parameters = _load_vlp16_parameters(pointcloud_share)

    organize_cloud = IfCondition(
        LaunchConfiguration("velodyne_organize_cloud")
    ).evaluate(context)
    transform_parameters["organize_cloud"] = organize_cloud
    adapter = Node(
        package="ad_morai_bridge",
        executable="ad_velodyne_adapter",
        name="ad_velodyne_adapter",
        output="screen",
        parameters=[{"point_timing_mode": point_timing_mode}],
    )
    transform_output_topic = (
        MORAI_SYNTHETIC_TIME_TOPIC
        if platform_profile == "morai"
        else PUBLIC_POINTS_TOPIC
    )
    transform = Node(
        package="velodyne_pointcloud",
        executable="velodyne_transform_node",
        name="ad_velodyne_transform",
        output="screen",
        parameters=[transform_parameters],
        remappings=[
            ("velodyne_packets", "/ad/sensors/lidar/packets"),
            ("velodyne_points", transform_output_topic),
        ],
    )
    actions = [adapter, transform]
    if platform_profile == "morai":
        actions.append(
            Node(
                package="ad_morai_bridge",
                executable="ad_point_time_zero_boundary",
                name="ad_point_time_zero_boundary",
                output="screen",
            )
        )
    return actions


def generate_launch_description():
    default_config = str(
        Path(get_package_share_directory("ad_morai_bridge"))
        / "config"
        / "competition.yaml"
    )
    config = LaunchConfiguration("config")
    control_enabled = LaunchConfiguration("control_enabled")
    traffic_light_camera_enabled = LaunchConfiguration(
        "enable_traffic_light_camera"
    )

    bridge = Node(
        package="ad_morai_bridge",
        executable="ad_morai_bridge_node",
        name="ad_morai_bridge",
        output="screen",
        parameters=[
            config,
            {
                "control.enabled": ParameterValue(
                    control_enabled,
                    value_type=bool,
                ),
                "camera_traffic_light.enabled": ParameterValue(
                    traffic_light_camera_enabled,
                    value_type=bool,
                ),
            },
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("config", default_value=default_config),
            DeclareLaunchArgument("control_enabled", default_value="false"),
            DeclareLaunchArgument(
                "enable_traffic_light_camera", default_value="false"
            ),
            DeclareLaunchArgument(
                "enable_velodyne_points",
                default_value="false",
            ),
            DeclareLaunchArgument(
                "platform_profile",
                default_value="morai",
                description="exactly morai or real_hardware",
            ),
            DeclareLaunchArgument(
                "velodyne_point_timing_mode",
                default_value="auto",
                description=(
                    "auto derives zero for MORAI snapshots and "
                    "azimuth/rolling timing for real hardware; explicit "
                    "values must match the profile"
                ),
            ),
            DeclareLaunchArgument(
                "velodyne_organize_cloud",
                default_value="true",
            ),
            DeclareLaunchArgument(
                "measurement_compatibility_enabled",
                default_value="false",
                description=(
                    "Enable the optional exact-repeat IMU boundary; MORAI-only."
                ),
            ),
            DeclareLaunchArgument(
                "measurement_compatibility_config",
                default_value=str(
                    Path(get_package_share_directory("ad_morai_bridge"))
                    / "config"
                    / "measurement_compatibility.yaml"
                ),
            ),
            bridge,
            OpaqueFunction(function=_launch_velodyne_chain),
            OpaqueFunction(function=_launch_measurement_compatibility),
        ]
    )
