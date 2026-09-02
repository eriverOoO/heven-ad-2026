from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    package_share = Path(get_package_share_directory("ad_lidar_perception"))
    default_config = (
        package_share
        / "config"
        / "tracking"
        / "competition_mot_baseline_v1.yaml"
    )

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
            default_value="hungarian",
            description=(
                "Association matcher: 'hungarian' for COMPETITION_MOT_BASELINE_V1 "
                "or the historical research option 'greedy'. "
                "Both use the identical cost matrix and gate for whichever "
                "association_metric is selected."
            ),
        ),
        DeclareLaunchArgument(
            "association_metric",
            default_value="euclidean",
            description=(
                "COMPETITION_MOT_BASELINE_V1 uses 'euclidean' BEV center distance; "
                "historical research options are 'giou_3d' and "
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
            default_value="unobserved",
            description=(
                "COMPETITION_MOT_BASELINE_V1 uses 'unobserved' because adaptive "
                "Euclidean clustering publishes orientation as unavailable; "
                "'detector' remains available for oriented detectors."
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
        DeclareLaunchArgument(
            "kalmannet_checkpoint",
            default_value="",
            description=(
                "T-9B: opt-in, experimental. Absolute path to a frozen "
                "KalmanNet checkpoint (e.g. DENSE-KALMANNET-v2). Required "
                "(must be non-empty) when state_estimator:=kalmannet; never "
                "loaded otherwise. No default path is baked in -- must be "
                "supplied explicitly."
            ),
        ),
        DeclareLaunchArgument(
            "kalmannet_device",
            default_value="cpu",
            description=(
                "T-9B: 'cpu' (default -- the tiny model makes GPU kernel-"
                "launch overhead dominate, per T-9A/T-12's own offline "
                "runtime findings) or 'cuda'. Only read when "
                "state_estimator:=kalmannet."
            ),
        ),
        DeclareLaunchArgument(
            "velocity_audit_enabled", default_value="false"
        ),
        DeclareLaunchArgument(
            "defer_until_tf_ready",
            default_value="true",
            description=(
                "AB3DMOT Exact-Stamp TF Deferred Processing v1: when the "
                "exact-stamp target_frame<-detection transform is not yet "
                "available only because its FUTURE side hasn't arrived "
                "(tf2 'extrapolation into the future'), hold the detection "
                "in a small bounded FIFO queue and process it -- at its "
                "original stamp -- once tf2 can supply the exact "
                "transform, instead of discarding it immediately. Never "
                "extrapolates, blocks, or reorders. 'false' reproduces the "
                "pre-fix unconditional-drop-on-any-TF-failure behavior "
                "exactly."
            ),
        ),
        DeclareLaunchArgument(
            "max_tf_wait_ms",
            default_value="500",
            description=(
                "Bounded wait (ms) a detection may sit in the deferred "
                "queue before being dropped as tf_timeout. Evidence-"
                "derived default -- see "
                "docs/perception/ab3dmot_tf_deferred_processing_v1.md "
                "Phase 2 (measured detection-arrival -> exact-stamp-"
                "transform-ready lag)."
            ),
        ),
        DeclareLaunchArgument(
            "max_pending_detections",
            default_value="8",
            description=(
                "Bounded FIFO depth for deferred detections; oldest "
                "unresolved entry is dropped on overflow (deterministic, "
                "never random)."
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
                "kalmannet_checkpoint": LaunchConfiguration("kalmannet_checkpoint"),
                "kalmannet_device": LaunchConfiguration("kalmannet_device"),
                "velocity_audit_enabled": ParameterValue(
                    LaunchConfiguration("velocity_audit_enabled"),
                    value_type=bool,
                ),
                "defer_until_tf_ready": ParameterValue(
                    LaunchConfiguration("defer_until_tf_ready"), value_type=bool
                ),
                "max_tf_wait_ms": ParameterValue(
                    LaunchConfiguration("max_tf_wait_ms"), value_type=int
                ),
                "max_pending_detections": ParameterValue(
                    LaunchConfiguration("max_pending_detections"), value_type=int
                ),
            },
        ],
    )
    return LaunchDescription([*arguments, node])
