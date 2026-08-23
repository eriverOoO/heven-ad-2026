"""One-command MORAI camera + LiDAR tracking replay.

This launch keeps the production Euclidean/Autoware path unchanged and adds
the frozen AB3DMOT baseline as an explicitly experimental comparison arm.
The identity odom anchor is replay-only: camera bags without localization TF
cannot support global-frame accuracy claims.
"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetParameter


def _launch_file(name: str) -> PythonLaunchDescriptionSource:
    package_share = Path(
        get_package_share_directory("ad_lidar_perception")
    )
    return PythonLaunchDescriptionSource(
        str(package_share / "launch" / name)
    )


def generate_launch_description() -> LaunchDescription:
    package_share = Path(
        get_package_share_directory("ad_lidar_perception")
    )
    rviz_config = package_share / "rviz" / "heven_camera_tracking.rviz"

    replay = IncludeLaunchDescription(
        _launch_file("lidar_bag_replay.launch.py"),
        launch_arguments={
            "bag_path": LaunchConfiguration("bag_path"),
            "rate": LaunchConfiguration("rate"),
            "startup_delay_sec": LaunchConfiguration(
                "startup_delay_sec"
            ),
            "start_paused": LaunchConfiguration("start_paused"),
            "include_front_camera": "true",
        }.items(),
    )
    experimental_tracker = IncludeLaunchDescription(
        _launch_file("ab3dmot_tracker.launch.py"),
        launch_arguments={
            "enabled": "true",
            "matcher": "hungarian",
            "association_metric": "euclidean",
            "euclidean_gate_m": "3.0",
            "state_estimator": "linear_kf",
            "yaw_measurement_mode": "unobserved",
        }.items(),
    )
    visualization = IncludeLaunchDescription(
        _launch_file("perception_visualization.launch.py"),
        launch_arguments={
            "start_rviz": "true",
            "use_sim_time": "true",
            "rviz_config": str(rviz_config),
        }.items(),
    )
    replay_odom_anchor = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="camera_bag_replay_odom_anchor",
        output="screen",
        arguments=[
            "--x",
            "0",
            "--y",
            "0",
            "--z",
            "0",
            "--roll",
            "0",
            "--pitch",
            "0",
            "--yaw",
            "0",
            "--frame-id",
            "odom",
            "--child-frame-id",
            "base_link",
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "bag_path",
                description="Absolute path to the extracted MCAP bag folder",
            ),
            DeclareLaunchArgument(
                "rate",
                default_value="0.5",
                description="Positive finite playback rate",
            ),
            DeclareLaunchArgument(
                "startup_delay_sec",
                default_value="4.0",
                description="Delay before playback while the graph starts",
            ),
            DeclareLaunchArgument(
                "start_paused",
                default_value="false",
                description="Start playback paused",
            ),
            SetParameter(name="use_sim_time", value=True),
            replay_odom_anchor,
            experimental_tracker,
            visualization,
            replay,
        ]
    )
