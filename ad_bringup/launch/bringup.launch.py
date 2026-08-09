from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import EnvironmentVariable

from ad_bringup.bringup_stack import build_bringup_stack


def generate_launch_description():
    bringup_share = Path(get_package_share_directory("ad_bringup"))
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "data_dir",
                default_value=EnvironmentVariable(
                    "AD_DATA_DIR", default_value=""
                ),
            ),
            DeclareLaunchArgument("control_enabled", default_value="false"),
            DeclareLaunchArgument(
                "localization_backend", default_value="gnss_imu"
            ),
            DeclareLaunchArgument("path_file", default_value=""),
            DeclareLaunchArgument("path_tracking_backend", default_value=""),
            DeclareLaunchArgument("perception_enabled", default_value=""),
            DeclareLaunchArgument(
                "local_motion_prediction_mode", default_value=""
            ),
            *build_bringup_stack(
                components_file=str(
                    bringup_share / "config" / "components.yaml"
                ),
                status_topic="/ad/vehicle/status",
            ),
        ]
    )
