from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    IncludeLaunchDescription,
    LogInfo,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node

from ad_bringup.bringup_stack import (
    build_bringup_stack,
    registered_component_names,
)
from ad_bringup.component_config import load_components


DEFAULT_SCENARIO_FILE = "2026_molit_comp_sample_scene.json"


def _launch_file(package, name):
    package_share = Path(get_package_share_directory(package))
    return PythonLaunchDescriptionSource(str(package_share / "launch" / name))


def _after_scenario_setup(event, _context, runtime_actions):
    if event.returncode == 0:
        return [
            LogInfo(msg="MORAI scenario ready; starting global path trial"),
            *runtime_actions,
        ]
    return [
        LogInfo(
            msg=f"scenario setup failed with exit code {event.returncode}"
        ),
        EmitEvent(event=Shutdown(reason="scenario setup failed")),
    ]


def generate_launch_description():
    bringup_share = Path(get_package_share_directory("ad_bringup"))
    components_file = (
        bringup_share / "config" / "morai_global_path_components.yaml"
    )
    components = load_components(
        components_file,
        registered_component_names(),
    )
    scenario_file = LaunchConfiguration("scenario_file")
    grpc_target = LaunchConfiguration("grpc_target")
    control_enabled = LaunchConfiguration("control_enabled")

    development = IncludeLaunchDescription(
        _launch_file("ad_morai_bridge_dev", "bridge_dev.launch.py"),
        launch_arguments={
            "control_enabled": control_enabled,
            "enable_velodyne_points": str(
                components["lidar_perception"]
            ).lower(),
            "grpc_target": grpc_target,
        }.items(),
    )
    scenario_setup = Node(
        package="ad_morai_bridge_dev",
        executable="ad_morai_scenario_setup",
        name="ad_morai_scenario_setup",
        output="screen",
        parameters=[
            {
                "scenario_file": scenario_file,
                "grpc.target": grpc_target,
                "grpc.timeout_sec": 8.0,
            }
        ],
    )
    bringup = build_bringup_stack(
        components_file=str(components_file),
        status_topic="/ad/dev/vehicle/ego_status",
    )
    start_runtime = RegisterEventHandler(
        OnProcessExit(
            target_action=scenario_setup,
            on_exit=lambda event, context: _after_scenario_setup(
                event, context, [development, *bringup]
            ),
        )
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "data_dir",
                default_value=EnvironmentVariable(
                    "AD_DATA_DIR", default_value=""
                ),
            ),
            DeclareLaunchArgument(
                "scenario_file",
                default_value=EnvironmentVariable(
                    "MORAI_SCENARIO_FILE", default_value=DEFAULT_SCENARIO_FILE
                ),
            ),
            DeclareLaunchArgument(
                "grpc_target", default_value="127.0.0.1:7789"
            ),
            DeclareLaunchArgument("control_enabled", default_value="true"),
            DeclareLaunchArgument(
                "localization_backend", default_value="gnss_imu"
            ),
            DeclareLaunchArgument("path_file", default_value=""),
            DeclareLaunchArgument(
                "local_motion_prediction_mode", default_value="required"
            ),
            scenario_setup,
            start_runtime,
        ]
    )
