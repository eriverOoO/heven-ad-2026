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


def _valid_relative_frame(frame):
    return bool(frame) and not frame.startswith("/") and not any(
        character.isspace() for character in frame
    )


def _required_finite(mapping, key, path):
    try:
        value = mapping[key]
    except (KeyError, TypeError) as error:
        raise RuntimeError(
            f"invalid Patchwork++ sensor profile: missing {path}"
        ) from error
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{path} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"{path} must be finite")
    return result


def _load_lidar_mount(sensor_config: Path, requested_profile: str):
    try:
        document = yaml.safe_load(sensor_config.read_text(encoding="utf-8"))
        profile_name = requested_profile or document["active_profile"]
        lidar = document["profiles"][profile_name]["sensors"]["lidar"]
        convention = document["coordinate_convention"]
        profile_base_frame = convention["base_frame"]
        frame_id = lidar["frame_id"]
        position = lidar["position_m"]
        rpy = lidar["rpy_rad"]
    except (KeyError, OSError, TypeError, ValueError, yaml.YAMLError) as error:
        raise RuntimeError(
            f"invalid Patchwork++ sensor profile {requested_profile!r}: {error}"
        ) from error

    if not isinstance(frame_id, str) or not _valid_relative_frame(frame_id):
        raise RuntimeError(
            f"LiDAR frame in sensor profile {profile_name!r} must be relative"
        )
    if not isinstance(profile_base_frame, str) or not _valid_relative_frame(
        profile_base_frame
    ):
        raise RuntimeError(
            f"base frame in sensor profile {profile_name!r} must be relative"
        )
    position_values = {
        axis: _required_finite(position, axis, f"lidar.position_m.{axis}")
        for axis in ("x", "y", "z")
    }
    for axis in ("roll", "pitch", "yaw"):
        _required_finite(rpy, axis, f"lidar.rpy_rad.{axis}")
    static_frames = convention.get("static_frames")
    if static_frames is None:
        sensor_parent_height = 0.0
    else:
        rear_axle = static_frames["rear_axle"]
        if convention["sensor_parent_frame"] != rear_axle["frame_id"]:
            raise RuntimeError("sensor_parent_frame must match the rear axle frame")
        sensor_parent_height = _required_finite(
            rear_axle["position_m"], "z", "rear_axle.position_m.z"
        )
    sensor_height = sensor_parent_height + position_values["z"]
    if sensor_height <= 0.0:
        raise RuntimeError(
            f"LiDAR height in sensor profile {profile_name!r} must be positive finite"
        )
    return frame_id, sensor_height, profile_base_frame


def _parse_enabled(name, value):
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise RuntimeError(f"{name} must be 'true' or 'false'")


def _perform(context, name):
    return LaunchConfiguration(name).perform(context)


def _leveled_frame(sensor_frame):
    if sensor_frame.endswith("_link"):
        return f"{sensor_frame[:-len('_link')]}_leveled_frame"
    return f"{sensor_frame}_leveled_frame"


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


BACKENDS = ("patchwork", "ransac")


def _selected_backend(context):
    backend = _perform(context, "backend")
    if backend not in BACKENDS:
        raise RuntimeError(
            f"backend must be one of {', '.join(BACKENDS)}, not {backend!r}"
        )
    return backend


def _launch_setup(context):
    backend = _selected_backend(context)
    leveling_enabled = _parse_enabled(
        "patchwork_leveling_enabled",
        _perform(context, "patchwork_leveling_enabled"),
    )
    cropped_input_topic = _perform(context, "cropped_input_topic")
    leveled_output_topic = _perform(context, "leveled_output_topic")
    active_topics = [
        ("cropped_input_topic", cropped_input_topic),
        ("ground_output", "/ad/perception/lidar/ground"),
        ("nonground_output", "/ad/perception/lidar/nonground"),
    ]
    if backend == "patchwork":
        # Only Patchwork republishes the whole input alongside the split.
        active_topics.append(
            ("patchwork_cloud_output", "/ad/perception/lidar/cloud")
        )
    if leveling_enabled:
        active_topics.append(("leveled_output_topic", leveled_output_topic))
    _validate_active_topics(active_topics)

    sensor_config = Path(_perform(context, "sensor_config"))
    sensor_profile = _perform(context, "sensor_profile")
    sensor_frame, sensor_height, profile_base_frame = _load_lidar_mount(
        sensor_config, sensor_profile
    )
    leveled_frame = _leveled_frame(sensor_frame)

    actions = []
    segmentation_input_topic = cropped_input_topic
    segmentation_frame = sensor_frame
    if leveling_enabled:
        odom_frame = _perform(context, "odom_frame")
        base_frame = _perform(context, "base_frame")
        if not _valid_relative_frame(odom_frame) or not _valid_relative_frame(
            base_frame
        ):
            raise RuntimeError(
                "odom_frame and base_frame must each be a relative frame"
            )
        if base_frame != profile_base_frame:
            raise RuntimeError(
                "base_frame must match the selected sensor profile base frame"
            )
        actions.append(
            Node(
                package="ad_lidar_perception",
                executable="ad_gravity_leveler_node",
                name="ad_gravity_leveler",
                output="screen",
                parameters=[
                    LaunchConfiguration("gravity_leveler_config"),
                    {
                        "topics.input": cropped_input_topic,
                        "topics.output": leveled_output_topic,
                        "expected_input_frame": sensor_frame,
                        "output_frame": leveled_frame,
                        "odom_frame": odom_frame,
                        "base_frame": base_frame,
                    },
                ],
            )
        )
        segmentation_input_topic = leveled_output_topic
        segmentation_frame = leveled_frame

    if backend == "patchwork":
        actions.append(
            Node(
                package="patchworkpp",
                executable="patchworkpp_node",
                name="ad_ground_segmentation",
                output="screen",
                remappings=[
                    ("pointcloud_topic", segmentation_input_topic),
                    ("/patchworkpp/cloud", "/ad/perception/lidar/cloud"),
                    ("/patchworkpp/ground", "/ad/perception/lidar/ground"),
                    ("/patchworkpp/nonground", "/ad/perception/lidar/nonground"),
                ],
                parameters=[
                    LaunchConfiguration("ground_config"),
                    {
                        "base_frame": segmentation_frame,
                        "sensor_height": sensor_height,
                    },
                ],
            )
        )
    else:
        actions.append(
            Node(
                package="autoware_ground_segmentation",
                executable="ransac_ground_filter_node",
                name="ad_ground_segmentation",
                output="screen",
                remappings=[
                    ("input", segmentation_input_topic),
                    ("output", "/ad/perception/lidar/nonground"),
                    (
                        "debug/ground/pointcloud",
                        "/ad/perception/lidar/ground",
                    ),
                ],
                parameters=[
                    LaunchConfiguration("ransac_config"),
                    # Autoware fits in base_frame and emits its output there.
                    # Matching the actual input frame avoids an unnecessary
                    # transform and keeps both leveled and bypass routes exact.
                    {"base_frame": segmentation_frame},
                ],
            )
        )
    return actions


def generate_launch_description():
    perception_share = Path(get_package_share_directory("ad_lidar_perception"))
    description_share = Path(get_package_share_directory("ad_description"))
    preprocessing_config = perception_share / "config" / "preprocessing"

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "backend",
                default_value="patchwork",
                description=(
                    "patchwork runs the Patchwork/Patchwork++ node and picks "
                    "between them with algorithm: in ground_segmentation.yaml. "
                    "ransac runs Autoware's plane-fitting filter, which does "
                    "not republish /ad/perception/lidar/cloud."
                ),
            ),
            DeclareLaunchArgument(
                "patchwork_leveling_enabled", default_value="false"
            ),
            DeclareLaunchArgument(
                "cropped_input_topic",
                default_value="/ad/perception/lidar/cropped",
            ),
            DeclareLaunchArgument(
                "leveled_output_topic",
                default_value="/ad/perception/lidar/leveled",
            ),
            DeclareLaunchArgument(
                "gravity_leveler_config",
                default_value=str(preprocessing_config / "gravity_leveler.yaml"),
            ),
            DeclareLaunchArgument(
                "ground_config",
                default_value=str(preprocessing_config / "ground_segmentation.yaml"),
            ),
            DeclareLaunchArgument(
                "ransac_config",
                default_value=str(
                    preprocessing_config / "ransac_ground_filter.yaml"
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
            DeclareLaunchArgument("odom_frame", default_value="odom"),
            DeclareLaunchArgument("base_frame", default_value="base_link"),
            OpaqueFunction(function=_launch_setup),
        ]
    )
