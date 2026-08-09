from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from rclpy.exceptions import InvalidTopicNameException
from rclpy.validate_full_topic_name import validate_full_topic_name


def _perform(context, name):
    return LaunchConfiguration(name).perform(context)


def _parse_enabled(name, value):
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise RuntimeError(f"{name} must be 'true' or 'false'")


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
    finite_filter_enabled = _parse_enabled(
        "finite_filter_enabled", _perform(context, "finite_filter_enabled")
    )
    densifier_enabled = _parse_enabled(
        "densifier_enabled", _perform(context, "densifier_enabled")
    )
    finite_input_topic = _perform(context, "finite_input_topic")
    finite_output_topic = _perform(context, "finite_output_topic")
    densified_output_topic = _perform(context, "densified_output_topic")
    active_topics = [
        ("finite_input_topic", finite_input_topic),
        ("objects_topic", "/ad/perception/objects/detected"),
        ("clusters_topic", "/ad/perception/lidar/clusters"),
    ]
    if finite_filter_enabled:
        active_topics.append(("finite_output_topic", finite_output_topic))
    if densifier_enabled:
        active_topics.append(
            ("densified_output_topic", densified_output_topic)
        )
    _validate_active_topics(active_topics)

    package_share = Path(get_package_share_directory("ad_lidar_perception"))
    filter_config = str(
        package_share / "config" / "preprocessing" / "finite_point_filter.yaml"
    )
    actions = []
    cluster_input_topic = finite_input_topic
    if finite_filter_enabled:
        actions.append(
            Node(
                package="ad_lidar_perception",
                executable="ad_finite_point_filter_node",
                name="ad_finite_point_filter",
                output="screen",
                parameters=[
                    filter_config,
                    {
                        "topics.input": cluster_input_topic,
                        "topics.output": finite_output_topic,
                    },
                ],
            )
        )
        cluster_input_topic = finite_output_topic
    if densifier_enabled:
        actions.append(
            Node(
                package="ad_lidar_perception",
                executable="ad_pointcloud_densifier_node",
                name="ad_pointcloud_densifier",
                output="screen",
                parameters=[
                    LaunchConfiguration("densifier_config"),
                    {
                        "topics.input": cluster_input_topic,
                        "topics.output": densified_output_topic,
                    },
                ],
            )
        )
        cluster_input_topic = densified_output_topic
    actions.append(
        Node(
            package="ad_lidar_perception",
            executable="ad_adaptive_euclidean_cluster_node",
            name="ad_adaptive_euclidean_cluster",
            output="screen",
            parameters=[
                LaunchConfiguration("cluster_config"),
                {"input_topic": cluster_input_topic},
            ],
        )
    )
    return actions


def generate_launch_description():
    package_share = Path(get_package_share_directory("ad_lidar_perception"))
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "finite_filter_enabled", default_value="true"
            ),
            DeclareLaunchArgument("densifier_enabled", default_value="false"),
            DeclareLaunchArgument(
                "finite_input_topic",
                default_value="/ad/perception/lidar/nonground",
            ),
            DeclareLaunchArgument(
                "finite_output_topic",
                default_value="/ad/perception/lidar/nonground_finite",
            ),
            DeclareLaunchArgument(
                "densified_output_topic",
                default_value="/ad/perception/lidar/nonground_densified",
            ),
            DeclareLaunchArgument(
                "densifier_config",
                default_value=str(
                    package_share / "config" / "preprocessing" / "densifier.yaml"
                ),
            ),
            DeclareLaunchArgument(
                "cluster_config",
                default_value=str(
                    package_share
                    / "config"
                    / "clustering"
                    / "adaptive_euclidean_cluster.yaml"
                ),
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
