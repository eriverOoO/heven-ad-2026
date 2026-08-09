from pathlib import Path

import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.actions import TimerAction
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from ad_bringup.bringup_stack import build_bringup_stack


def _load_control_point_x_m():
    vehicle_path = (
        Path(get_package_share_directory("ad_description"))
        / "config"
        / "vehicle_parameters.yaml"
    )
    with vehicle_path.open(encoding="utf-8") as stream:
        vehicle = yaml.safe_load(stream)["vehicle"]
    return float(vehicle["control"]["lateral_control_point_x_m"])


def generate_launch_description():
    tuning_share = Path(get_package_share_directory("ad_tuning"))

    bringup_actions = build_bringup_stack(
        components_file=str(
            tuning_share / "config" / "profile_stanley_components.yaml"
        ),
        status_topic="/ad/vehicle/status",
        bridge_config=(
            tuning_share / "config" / "morai_tuning_bridge.yaml"
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
    tuner = Node(
        package="ad_tuning",
        executable="global_path_tuner",
        name="ad_global_path_tuner",
        output="screen",
        parameters=[
            str(tuning_share / "config" / "tuning.yaml"),
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
                "metric_control_point_x_m": _load_control_point_x_m(),
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
            DeclareLaunchArgument("control_enabled", default_value="true"),
            DeclareLaunchArgument(
                "localization_backend", default_value="gnss_imu"
            ),
            DeclareLaunchArgument("path_file", default_value=""),
            DeclareLaunchArgument(
                "path_tracking_backend", default_value="profile_stanley"
            ),
            # Keep production DWA configured, but exclude OGM/DWA
            # interventions from path-controller parameter comparison.
            DeclareLaunchArgument(
                "perception_enabled", default_value="false"
            ),
            DeclareLaunchArgument(
                "local_motion_prediction_mode", default_value="disabled"
            ),
            DeclareLaunchArgument(
                "tuning_lease_required", default_value="true"
            ),
            DeclareLaunchArgument("maximum_trials", default_value="30"),
            DeclareLaunchArgument(
                "maximum_total_trials", default_value="45"
            ),
            DeclareLaunchArgument(
                "minimum_feasible_trials", default_value="6"
            ),
            DeclareLaunchArgument(
                "inherit_minimum_complete_trials", default_value="36"
            ),
            DeclareLaunchArgument("inherit_top_k", default_value="5"),
            DeclareLaunchArgument(
                "maximum_consecutive_infrastructure_failures",
                default_value="3",
            ),
            DeclareLaunchArgument(
                "maximum_worker_wall_time_sec", default_value="7200.0"
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
            *bringup_actions,
            reset_bridge,
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
                                        reason="tuning worker exited safely"
                                    )
                                )
                            ],
                        )
                    ],
                )
            ),
        ]
    )
