from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    package_share = Path(get_package_share_directory("ad_lidar_perception"))
    default_config = package_share / "config" / "tracking" / "ab3dmot.yaml"

    arguments = [
        DeclareLaunchArgument("enabled", default_value="true"),
        DeclareLaunchArgument(
            "input_topic", default_value="/ad/perception/objects/detected"
        ),
        DeclareLaunchArgument(
            "output_topic", default_value="/experiment/tracked/ab3dmot"
        ),
        DeclareLaunchArgument("target_frame", default_value="odom"),
        DeclareLaunchArgument(
            "ab3dmot_root",
            default_value="",
            description=(
                "Absolute path to the references/ab3dmot submodule checkout; "
                "blank uses the source-tree-relative default."
            ),
        ),
        DeclareLaunchArgument("config_path", default_value=str(default_config)),
        DeclareLaunchArgument(
            "matcher",
            default_value="greedy",
            description=(
                "Association matcher: 'greedy' (default, production-matching "
                "baseline) or 'hungarian' (T-3 opt-in global assignment). "
                "Both use the identical cost matrix and gate for whichever "
                "association_metric is selected."
            ),
        ),
        DeclareLaunchArgument(
            "association_metric",
            default_value="giou_3d",
            description=(
                "T-4: 'giou_3d' (default, production-matching baseline), "
                "'euclidean' (BEV center distance, gated by euclidean_gate_m), "
                "or 'mahalanobis' (BEV innovation distance, gated by "
                "mahalanobis_gate)."
            ),
        ),
        DeclareLaunchArgument("euclidean_gate_m", default_value="3.0"),
        DeclareLaunchArgument("mahalanobis_gate", default_value="11.62"),
        DeclareLaunchArgument(
            "state_estimator",
            default_value="linear_kf",
            description=(
                "T-7A: 'linear_kf' (default, unchanged AB3DMOT reference "
                "constant-velocity filter) or 'ekf' (opt-in planar CTRV "
                "nonlinear model)."
            ),
        ),
        DeclareLaunchArgument(
            "yaw_measurement_mode",
            default_value="detector",
            description=(
                "T-7A.5: 'detector' (default, unchanged -- feed the "
                "detector's own yaw into the estimator) or 'unobserved' "
                "(opt-in -- treat yaw as unmeasured; motion heading is "
                "instead inferred from positional displacement)."
            ),
        ),
        DeclareLaunchArgument(
            "imm_cv_to_cv_probability", default_value="0.95",
            description="T-7B: IMM CV-self-transition probability (state_estimator:=imm only).",
        ),
        DeclareLaunchArgument(
            "imm_ctrv_to_ctrv_probability", default_value="0.95",
            description="T-7B: IMM CTRV-self-transition probability (state_estimator:=imm only).",
        ),
        DeclareLaunchArgument(
            "mahalanobis_max_distance_m",
            default_value="0.0",
            description=(
                "T-5B: optional absolute BEV physical-distance cap (meters) "
                "on Mahalanobis-accepted pairs. <=0 (default) disables the "
                "cap -- pure T-4/T-5A Mahalanobis behavior is unchanged."
            ),
        ),
    ]
    node = Node(
        package="ad_lidar_perception",
        executable="ad_ab3dmot_tracker",
        name="ad_ab3dmot_tracker",
        output="screen",
        parameters=[
            LaunchConfiguration("config_path"),
            {
                "enabled": ParameterValue(LaunchConfiguration("enabled"), value_type=bool),
                "input_topic": LaunchConfiguration("input_topic"),
                "output_topic": LaunchConfiguration("output_topic"),
                "target_frame": LaunchConfiguration("target_frame"),
                "ab3dmot_root": LaunchConfiguration("ab3dmot_root"),
                "matcher": LaunchConfiguration("matcher"),
                "association_metric": LaunchConfiguration("association_metric"),
                "euclidean_gate_m": ParameterValue(
                    LaunchConfiguration("euclidean_gate_m"), value_type=float
                ),
                "mahalanobis_gate": ParameterValue(
                    LaunchConfiguration("mahalanobis_gate"), value_type=float
                ),
                "mahalanobis_max_distance_m": ParameterValue(
                    LaunchConfiguration("mahalanobis_max_distance_m"), value_type=float
                ),
                "state_estimator": LaunchConfiguration("state_estimator"),
                "yaw_measurement_mode": LaunchConfiguration("yaw_measurement_mode"),
                "imm_cv_to_cv_probability": ParameterValue(
                    LaunchConfiguration("imm_cv_to_cv_probability"), value_type=float
                ),
                "imm_ctrv_to_ctrv_probability": ParameterValue(
                    LaunchConfiguration("imm_ctrv_to_ctrv_probability"), value_type=float
                ),
            },
        ],
    )
    return LaunchDescription([*arguments, node])
