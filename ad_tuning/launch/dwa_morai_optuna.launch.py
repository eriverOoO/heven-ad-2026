from pathlib import Path

import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    OpaqueFunction,
    RegisterEventHandler,
    TimerAction,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from ad_bringup.bringup_stack import build_bringup_stack
from ad_tuning.scenario_identity import (
    default_morai_save_root,
    validate_scenario_identity,
)


def _load_control_point_x_m():
    vehicle_path = (
        Path(get_package_share_directory("ad_description"))
        / "config"
        / "vehicle_parameters.yaml"
    )
    with vehicle_path.open(encoding="utf-8") as stream:
        vehicle = yaml.safe_load(stream)["vehicle"]
    return float(vehicle["control"]["lateral_control_point_x_m"])


def _validate_scenario_configuration(context):
    validate_scenario_identity(
        scenario_file=LaunchConfiguration("scenario_file").perform(context),
        experiment_scenario=LaunchConfiguration(
            "experiment_scenario"
        ).perform(context),
        save_root=default_morai_save_root(),
    )
    return []


def generate_launch_description():
    tuning_share = Path(get_package_share_directory("ad_tuning"))
    bringup_actions = build_bringup_stack(
        components_file=str(
            tuning_share / "config" / "dwa_components.yaml"
        ),
        status_topic="/ad/vehicle/status",
        bridge_config=(
            tuning_share / "config" / "morai_dwa_tuning_bridge.yaml"
        ),
    )
    reset_bridge = Node(
        package="ad_morai_bridge_dev",
        executable="ad_morai_bridge_dev_node",
        name="ad_tuning_morai_reset_bridge",
        output="screen",
        parameters=[
            str(tuning_share / "config" / "morai_reset.yaml"),
            {"multi_ego.target_ip": LaunchConfiguration("morai_ip")},
        ],
    )
    scenario_reset = Node(
        package="ad_morai_bridge_dev",
        executable="ad_morai_scenario_reset",
        name="ad_tuning_scenario_reset",
        output="screen",
        parameters=[
            {
                "scenario_file": LaunchConfiguration("scenario_file"),
                "grpc.target": LaunchConfiguration("grpc_target"),
                "grpc.timeout_sec": 8.0,
            }
        ],
    )
    tuner = Node(
        package="ad_tuning",
        executable="dwa_tuner",
        name="ad_dwa_tuner",
        output="screen",
        parameters=[
            str(tuning_share / "config" / "dwa_tuning.yaml"),
            {
                "maximum_trials": ParameterValue(
                    LaunchConfiguration("maximum_trials"), value_type=int
                ),
                "maximum_total_trials": ParameterValue(
                    LaunchConfiguration("maximum_total_trials"),
                    value_type=int,
                ),
                "minimum_feasible_trials": ParameterValue(
                    LaunchConfiguration("minimum_feasible_trials"),
                    value_type=int,
                ),
                "warm_start.inherit_minimum_complete_trials": ParameterValue(
                    LaunchConfiguration(
                        "inherit_minimum_complete_trials"
                    ),
                    value_type=int,
                ),
                "warm_start.inherit_top_k": ParameterValue(
                    LaunchConfiguration("inherit_top_k"),
                    value_type=int,
                ),
                "maximum_consecutive_infrastructure_failures": ParameterValue(
                    LaunchConfiguration(
                        "maximum_consecutive_infrastructure_failures"
                    ),
                    value_type=int,
                ),
                "maximum_worker_wall_time_sec": ParameterValue(
                    LaunchConfiguration("maximum_worker_wall_time_sec"),
                    value_type=float,
                ),
                "startup_timeout_sec": ParameterValue(
                    LaunchConfiguration("startup_timeout_sec"),
                    value_type=float,
                ),
                "trial_timeout_sec": ParameterValue(
                    LaunchConfiguration("trial_timeout_sec"),
                    value_type=float,
                ),
                "course_length_m": ParameterValue(
                    LaunchConfiguration("course_length_m"),
                    value_type=float,
                ),
                "output_dir": LaunchConfiguration("output_dir"),
                "worker_id": LaunchConfiguration("worker_id"),
                "experiment.scenario": LaunchConfiguration(
                    "experiment_scenario"
                ),
                "experiment.scenario_file": LaunchConfiguration(
                    "scenario_file"
                ),
                "experiment.weather": LaunchConfiguration(
                    "experiment_weather"
                ),
                "experiment.morai_version": LaunchConfiguration(
                    "morai_version"
                ),
                "experiment.code_revision": LaunchConfiguration(
                    "code_revision"
                ),
                "experiment.vehicle_profile_id": LaunchConfiguration(
                    "vehicle_profile_id"
                ),
                "experiment.data_dir": LaunchConfiguration("data_dir"),
                "experiment.route_corridor_file": LaunchConfiguration(
                    "route_corridor_file"
                ),
                "metric_control_point_x_m": _load_control_point_x_m(),
                "scenario_reset.required": True,
            },
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "data_dir",
                default_value=EnvironmentVariable(
                    "AD_DATA_DIR", default_value=""
                ),
            ),
            DeclareLaunchArgument("morai_ip", default_value="127.0.0.1"),
            DeclareLaunchArgument(
                "scenario_file",
                default_value=EnvironmentVariable(
                    "AD_TUNING_SCENARIO_FILE", default_value=""
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
            DeclareLaunchArgument("route_corridor_file", default_value=""),
            DeclareLaunchArgument(
                "path_tracking_backend", default_value="profile_stanley"
            ),
            DeclareLaunchArgument(
                "perception_enabled", default_value="true"
            ),
            DeclareLaunchArgument(
                "local_motion_prediction_mode", default_value="required"
            ),
            DeclareLaunchArgument(
                "tuning_lease_required", default_value="true"
            ),
            DeclareLaunchArgument("maximum_trials", default_value="120"),
            DeclareLaunchArgument(
                "maximum_total_trials", default_value="150"
            ),
            DeclareLaunchArgument(
                "minimum_feasible_trials", default_value="30"
            ),
            DeclareLaunchArgument(
                "inherit_minimum_complete_trials", default_value="40"
            ),
            DeclareLaunchArgument("inherit_top_k", default_value="5"),
            DeclareLaunchArgument(
                "maximum_consecutive_infrastructure_failures",
                default_value="3",
            ),
            DeclareLaunchArgument(
                "maximum_worker_wall_time_sec", default_value="14400.0"
            ),
            DeclareLaunchArgument(
                "startup_timeout_sec", default_value="60.0"
            ),
            DeclareLaunchArgument("trial_timeout_sec", default_value="420.0"),
            DeclareLaunchArgument("course_length_m", default_value="0.0"),
            DeclareLaunchArgument("output_dir", default_value=""),
            DeclareLaunchArgument(
                "worker_id",
                default_value=EnvironmentVariable(
                    "AD_TUNING_WORKER_ID", default_value=""
                ),
            ),
            DeclareLaunchArgument(
                "experiment_scenario",
                default_value=EnvironmentVariable(
                    "AD_TUNING_SCENARIO", default_value="unspecified"
                ),
            ),
            DeclareLaunchArgument(
                "experiment_weather",
                default_value=EnvironmentVariable(
                    "AD_TUNING_WEATHER", default_value="unspecified"
                ),
            ),
            DeclareLaunchArgument(
                "morai_version",
                default_value=EnvironmentVariable(
                    "AD_TUNING_MORAI_VERSION",
                    default_value="S4.251001",
                ),
            ),
            DeclareLaunchArgument(
                "code_revision",
                default_value=EnvironmentVariable(
                    "AD_TUNING_CODE_REVISION",
                    default_value="unspecified",
                ),
            ),
            DeclareLaunchArgument(
                "vehicle_profile_id",
                default_value=EnvironmentVariable(
                    "AD_TUNING_VEHICLE_PROFILE_ID",
                    default_value=(
                        "20260727-ioniq5-accelerator40-brake20-v1"
                    ),
                ),
            ),
            OpaqueFunction(function=_validate_scenario_configuration),
            *bringup_actions,
            reset_bridge,
            scenario_reset,
            tuner,
            RegisterEventHandler(
                OnProcessExit(
                    target_action=tuner,
                    on_exit=[
                        TimerAction(
                            period=2.0,
                            actions=[
                                EmitEvent(
                                    event=Shutdown(
                                        reason="DWA tuning worker exited safely"
                                    )
                                )
                            ],
                        )
                    ],
                )
            ),
        ]
    )
