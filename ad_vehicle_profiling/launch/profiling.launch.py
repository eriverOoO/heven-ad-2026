from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    IncludeLaunchDescription,
    RegisterEventHandler,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    EnvironmentVariable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node


def generate_launch_description():
    profiler_share = Path(
        get_package_share_directory("ad_vehicle_profiling")
    )
    bridge_share = Path(get_package_share_directory("ad_morai_bridge"))

    output_root = LaunchConfiguration("output_root")
    run_id = LaunchConfiguration("run_id")
    resume = LaunchConfiguration("resume")
    profiler_config = LaunchConfiguration("profiler_config")
    bridge_config = LaunchConfiguration("bridge_config")
    loop_guard_enabled = LaunchConfiguration("loop_guard_enabled")

    bridge = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            str(bridge_share / "launch" / "bridge.launch.py")
        ),
        launch_arguments={
            "config": bridge_config,
            "control_enabled": "true",
            "enable_velodyne_points": "false",
        }.items(),
    )
    profiler = Node(
        package="ad_vehicle_profiling",
        executable="ad_vehicle_profiler",
        name="ad_vehicle_profiler",
        output="screen",
        parameters=[
            profiler_config,
            {
                "output_root": output_root,
                "run_id": run_id,
                "resume": resume,
            },
        ],
    )
    loop_guard = Node(
        package="ad_vehicle_profiling",
        executable="ad_vehicle_profile_loop_guard",
        name="ad_vehicle_profile_loop_guard",
        output="screen",
        condition=IfCondition(loop_guard_enabled),
        parameters=[
            {
                "run_directory": PathJoinSubstitution(
                    [output_root, "vehicle_dynamics", run_id]
                ),
                "grpc.target": "127.0.0.1:7789",
                "maximum_abs_x_m": 0.8,
                "minimum_y_m": -1000.0,
                "minimum_z_m": -0.5,
                "target_location_m": [0.0, 0.0, 0.36],
                "target_rotation_deg": [0.0, 0.0, -90.0],
            }
        ],
    )
    stop_launch_when_profiler_exits = RegisterEventHandler(
        OnProcessExit(
            target_action=profiler,
            on_exit=[
                EmitEvent(
                    event=Shutdown(
                        reason="vehicle profiling process exited"
                    )
                )
            ],
        )
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "profiler_config",
                default_value=str(
                    profiler_share / "config" / "profiling.yaml"
                ),
            ),
            DeclareLaunchArgument(
                "bridge_config",
                default_value=str(
                    profiler_share / "config" / "bridge_profiling.yaml"
                ),
            ),
            DeclareLaunchArgument(
                "output_root",
                default_value=EnvironmentVariable(
                    "AD_DATA_DIR", default_value=""
                ),
            ),
            DeclareLaunchArgument("run_id", default_value=""),
            DeclareLaunchArgument("resume", default_value="true"),
            DeclareLaunchArgument(
                "loop_guard_enabled", default_value="true"
            ),
            bridge,
            profiler,
            loop_guard,
            stop_launch_when_profiler_exits,
        ]
    )
