import math
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from rclpy.exceptions import InvalidTopicNameException
from rclpy.validate_full_topic_name import validate_full_topic_name
import yaml


def _required_number(mapping, key, path):
    try:
        value = mapping[key]
    except (KeyError, TypeError) as error:
        raise RuntimeError(
            f"invalid vehicle configuration: missing {path}"
        ) from error
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{path} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"{path} must be finite")
    return result


def _load_crop_bounds(vehicle_config: Path, clearance_m: float):
    try:
        document = yaml.safe_load(vehicle_config.read_text(encoding="utf-8"))
        vehicle = document["vehicle"]
        convention = vehicle["coordinate_convention"]
        geometry = vehicle["geometry"]
        base_frame = convention["base_frame"]
    except (KeyError, OSError, TypeError, yaml.YAMLError) as error:
        raise RuntimeError(f"invalid vehicle configuration: {error}") from error

    if base_frame != "base_link":
        raise RuntimeError("vehicle base frame must be base_link")
    front = _required_number(
        geometry, "front_bumper_x_m", "vehicle.geometry.front_bumper_x_m"
    )
    rear = _required_number(
        geometry, "rear_bumper_x_m", "vehicle.geometry.rear_bumper_x_m"
    )
    width = _required_number(geometry, "width_m", "vehicle.geometry.width_m")
    height = _required_number(
        geometry, "height_m", "vehicle.geometry.height_m"
    )
    if front <= rear:
        raise RuntimeError("vehicle bumper coordinates require rear < front")
    if width <= 0.0:
        raise RuntimeError("vehicle width must be positive")
    if height <= 0.0:
        raise RuntimeError("vehicle height must be positive")

    return {
        "base_frame": base_frame,
        "bounds.min_x_m": rear - clearance_m,
        "bounds.max_x_m": front + clearance_m,
        "bounds.min_y_m": -width / 2.0 - clearance_m,
        "bounds.max_y_m": width / 2.0 + clearance_m,
        "bounds.min_z_m": -clearance_m,
        "bounds.max_z_m": height + clearance_m,
    }


def _parse_clearance(value):
    try:
        clearance = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError("crop clearance must be numeric") from error
    if not math.isfinite(clearance):
        raise RuntimeError("crop clearance must be finite")
    if clearance < 0.0:
        raise RuntimeError("crop clearance must be nonnegative")
    return clearance


def _parse_enabled(name, value):
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise RuntimeError(f"{name} must be 'true' or 'false'")


def _parse_platform_profile(value):
    profile = str(value).strip()
    if profile not in {"morai", "real_hardware"}:
        raise RuntimeError(
            "platform_profile must be exactly 'morai' or 'real_hardware'"
        )
    return profile


def _perform(context, name):
    return LaunchConfiguration(name).perform(context)


def _validate_active_topics(active_topics):
    seen = {}
    for argument_name, topic in active_topics:
        try:
            validate_full_topic_name(topic)
        except InvalidTopicNameException as error:
            raise RuntimeError(
                f"{argument_name} must be a valid full ROS topic: {error}"
            ) from error
        if topic in seen:
            raise RuntimeError(
                "duplicate active topic "
                f"{topic!r} for {seen[topic]} and {argument_name}"
            )
        seen[topic] = argument_name


def _launch_setup(context):
    platform_profile = _parse_platform_profile(
        _perform(context, "platform_profile")
    )
    deskew_enabled = _parse_enabled(
        "deskew_enabled", _perform(context, "deskew_enabled")
    )
    if platform_profile == "morai" and deskew_enabled:
        raise RuntimeError(
            "MORAI instantaneous-scan profile: motion deskew is prohibited"
        )
    self_crop_enabled = _parse_enabled(
        "self_crop_enabled", _perform(context, "self_crop_enabled")
    )
    self_crop_input_reliable = _parse_enabled(
        "self_crop_input_reliable",
        _perform(context, "self_crop_input_reliable"),
    )
    point_layout_adapter_enabled = _parse_enabled(
        "point_layout_adapter_enabled",
        _perform(context, "point_layout_adapter_enabled"),
    )
    deskew_mode = _perform(context, "deskew_mode")
    if deskew_mode not in {"2d", "3d"}:
        raise RuntimeError("deskew_mode must be '2d' or '3d'")

    raw_input_topic = _perform(context, "raw_input_topic")
    deskew_output_topic = _perform(context, "deskew_output_topic")
    self_crop_output_topic = _perform(context, "self_crop_output_topic")
    adapter_output_topic = _perform(context, "adapter_output_topic")
    active_topics = [("raw_input_topic", raw_input_topic)]
    if deskew_enabled:
        active_topics.append(("deskew_output_topic", deskew_output_topic))
    if self_crop_enabled:
        active_topics.append(
            ("self_crop_output_topic", self_crop_output_topic)
        )
    if point_layout_adapter_enabled:
        active_topics.append(("adapter_output_topic", adapter_output_topic))
    _validate_active_topics(active_topics)

    actions = []
    upstream_topic = raw_input_topic
    if deskew_enabled:
        actions.append(
            Node(
                package="ad_lidar_perception",
                executable="ad_motion_deskew_node",
                name="ad_motion_deskew",
                output="screen",
                parameters=[
                    LaunchConfiguration("motion_deskew_config"),
                    {
                        "topics.input": upstream_topic,
                        "topics.output": deskew_output_topic,
                        "deskew_mode": deskew_mode,
                    },
                ],
            )
        )
        upstream_topic = deskew_output_topic

    if self_crop_enabled:
        clearance = _parse_clearance(_perform(context, "crop_clearance_m"))
        crop_parameters = _load_crop_bounds(
            Path(_perform(context, "vehicle_config")), clearance
        )
        actions.append(
            Node(
                package="ad_lidar_perception",
                executable="ad_self_crop_filter_node",
                name="ad_self_crop_filter",
                output="screen",
                parameters=[
                    LaunchConfiguration("self_crop_config"),
                    crop_parameters,
                    {
                        "topics.input": upstream_topic,
                        "topics.output": self_crop_output_topic,
                        "input_reliable": self_crop_input_reliable,
                    },
                ],
            )
        )
        upstream_topic = self_crop_output_topic

    if point_layout_adapter_enabled:
        actions.append(
            Node(
                package="ad_lidar_perception",
                executable="ad_point_layout_adapter_node",
                name="ad_point_layout_adapter",
                output="screen",
                parameters=[
                    LaunchConfiguration("point_layout_adapter_config"),
                    {
                        "topics.input": upstream_topic,
                        "topics.output": adapter_output_topic,
                    },
                ],
            )
        )
    return actions


def generate_launch_description():
    perception_share = Path(
        get_package_share_directory("ad_lidar_perception")
    )
    description_share = Path(get_package_share_directory("ad_description"))
    preprocessing_config = perception_share / "config" / "preprocessing"

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "platform_profile", default_value="morai"
            ),
            DeclareLaunchArgument("deskew_enabled", default_value="false"),
            DeclareLaunchArgument("deskew_mode", default_value="3d"),
            DeclareLaunchArgument("self_crop_enabled", default_value="true"),
            DeclareLaunchArgument(
                "self_crop_input_reliable", default_value="false"
            ),
            DeclareLaunchArgument(
                "point_layout_adapter_enabled", default_value="true"
            ),
            DeclareLaunchArgument(
                "raw_input_topic", default_value="/ad/sensors/lidar/points"
            ),
            DeclareLaunchArgument(
                "deskew_output_topic",
                default_value="/ad/perception/lidar/deskewed",
            ),
            DeclareLaunchArgument(
                "self_crop_output_topic",
                default_value="/ad/perception/lidar/cropped",
            ),
            DeclareLaunchArgument(
                "adapter_output_topic",
                default_value="/ad/perception/lidar/points_xyzirc",
            ),
            DeclareLaunchArgument(
                "motion_deskew_config",
                default_value=str(preprocessing_config / "motion_deskew.yaml"),
            ),
            DeclareLaunchArgument(
                "self_crop_config",
                default_value=str(preprocessing_config / "self_crop.yaml"),
            ),
            DeclareLaunchArgument(
                "point_layout_adapter_config",
                default_value=str(
                    preprocessing_config / "point_layout_adapter.yaml"
                ),
            ),
            DeclareLaunchArgument(
                "vehicle_config",
                default_value=str(
                    description_share / "config" / "vehicle_parameters.yaml"
                ),
            ),
            DeclareLaunchArgument("crop_clearance_m", default_value="0.20"),
            OpaqueFunction(function=_launch_setup),
        ]
    )
