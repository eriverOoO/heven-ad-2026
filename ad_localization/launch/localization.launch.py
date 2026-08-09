from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessStart
from launch.events import matches_action
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode, Node
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState
from lifecycle_msgs.msg import Transition
import yaml


BACKEND_ODOMETRY_TOPICS = {
    "gnss_imu": "/ad/localization/backends/gnss_imu/odometry",
    "eskf": "/ad/localization/backends/eskf/odometry",
    "imu_quaternion_encoder": (
        "/ad/localization/backends/imu_quaternion_encoder/odometry"
    ),
    "quaternion_wheel_gnss_ekf": (
        "/ad/localization/backends/quaternion_wheel_gnss_ekf/odometry"
    ),
}


def _load_sensor_geometry(
    sensor_config: Path,
    requested_profile: str,
) -> dict[str, object]:
    document = yaml.safe_load(sensor_config.read_text(encoding="utf-8"))
    profile_name = requested_profile or document["active_profile"]
    sensors = document["profiles"][profile_name]["sensors"]
    position = sensors["gps"]["position_m"]
    convention = document["coordinate_convention"]
    static_frames = convention.get("static_frames")
    if static_frames is None:
        base_to_sensor_parent = {axis: 0.0 for axis in ("x", "y", "z")}
    else:
        rear_axle = static_frames["rear_axle"]
        if convention["sensor_parent_frame"] != rear_axle["frame_id"]:
            raise RuntimeError("sensor_parent_frame must match the rear axle frame")
        base_to_sensor_parent = rear_axle["position_m"]
    imu = sensors["imu"]
    rpy = imu["rpy_rad"]
    return {
        "gnss_lever_arm_m": [
            float(base_to_sensor_parent[axis]) + float(position[axis])
            for axis in ("x", "y", "z")
        ],
        "imu_frame": str(imu["frame_id"]),
        "imu_mount_rpy_rad": [
            float(rpy[axis]) for axis in ("roll", "pitch", "yaw")
        ],
    }


def _make_adapter(
    localization_backend, sensor_geometry, imu_topic, eskf_imu_topic
):
    return LifecycleNode(
        package="ad_localization",
        executable="ad_localization_node",
        name="ad_localization",
        namespace="",
        output="screen",
        parameters=[
            LaunchConfiguration("adapter_config"),
            {
                "status_topic": LaunchConfiguration("status_topic"),
                "gnss_lever_arm_m": sensor_geometry["gnss_lever_arm_m"],
                "localization_backend": localization_backend,
                "imu_topic": imu_topic,
                "eskf_imu_topic": eskf_imu_topic,
                "publish_map_to_odom_tf": False,
            },
        ],
    )


def _make_estimator(
    localization_backend, sensor_geometry, imu_topic, eskf_imu_topic
):
    lever_arm = sensor_geometry["gnss_lever_arm_m"]
    output_topic = BACKEND_ODOMETRY_TOPICS[localization_backend]
    if localization_backend == "gnss_imu":
        return Node(
            package="ad_localization",
            executable="gnss_imu_localization_node",
            name="gnss_imu_localization",
            output="screen",
            parameters=[
                LaunchConfiguration("gnss_imu_config"),
                {
                    "imu_topic": imu_topic,
                    "gnss_lever_arm_m": lever_arm,
                    "imu_frame": sensor_geometry["imu_frame"],
                    "imu_mount_rpy_rad": sensor_geometry["imu_mount_rpy_rad"],
                    "output_odometry_topic": output_topic,
                    "publish_tf": False,
                },
            ],
        )

    if localization_backend == "eskf":
        return Node(
            package="kalman_filter_localization",
            executable="ekf_localization_node",
            name="ekf_localization",
            output="screen",
            parameters=[
                LaunchConfiguration("eskf_config"),
                {
                    "imu_topic": eskf_imu_topic,
                    "gnss_lever_arm_x": lever_arm[0],
                    "gnss_lever_arm_y": lever_arm[1],
                    "gnss_lever_arm_z": lever_arm[2],
                    "output_odometry_topic": output_topic,
                    "publish_tf": False,
                },
            ],
        )

    if localization_backend == "imu_quaternion_encoder":
        return Node(
            package="ad_localization",
            executable="imu_quaternion_encoder_node",
            name="imu_quaternion_encoder",
            output="screen",
            parameters=[
                LaunchConfiguration("imu_quaternion_encoder_config"),
                {
                    "mode": LaunchConfiguration(
                        "imu_quaternion_encoder_mode"
                    ),
                    "status_topic": LaunchConfiguration("status_topic"),
                    "imu_topic": imu_topic,
                    "imu_frame": sensor_geometry["imu_frame"],
                    "gnss_lever_arm_m": lever_arm,
                    "imu_mount_rpy_rad": sensor_geometry["imu_mount_rpy_rad"],
                    "output_odometry_topic": output_topic,
                    "publish_tf": False,
                },
            ],
        )

    return Node(
        package="ad_localization",
        executable="quaternion_wheel_gnss_ekf_node",
        name="quaternion_wheel_gnss_ekf",
        output="screen",
        parameters=[
            LaunchConfiguration("quaternion_wheel_gnss_ekf_config"),
            {
                "imu_topic": imu_topic,
                "imu_frame": sensor_geometry["imu_frame"],
                "gnss_lever_arm_m": lever_arm,
                "imu_mount_rpy_rad": sensor_geometry["imu_mount_rpy_rad"],
                "output_odometry_topic": output_topic,
                "publish_tf": False,
            },
        ],
    )


def _make_manager(localization_backend):
    return Node(
        package="ad_localization",
        executable="localization_manager_node",
        name="localization_manager",
        output="screen",
        parameters=[
            LaunchConfiguration("localization_manager_config"),
            {
                "input_odometry_topic": BACKEND_ODOMETRY_TOPICS[
                    localization_backend
                ],
                "canonical_odometry_topic": "/ad/localization/odometry",
                "publish_tf": True,
            },
        ],
    )


def _make_autostart_handlers(adapter):
    autostart = LaunchConfiguration("autostart")
    configure = RegisterEventHandler(
        OnProcessStart(
            target_action=adapter,
            on_start=[
                EmitEvent(
                    event=ChangeState(
                        lifecycle_node_matcher=matches_action(adapter),
                        transition_id=Transition.TRANSITION_CONFIGURE,
                    ),
                    condition=IfCondition(autostart),
                )
            ],
        )
    )
    activate = RegisterEventHandler(
        OnStateTransition(
            target_lifecycle_node=adapter,
            goal_state="inactive",
            entities=[
                EmitEvent(
                    event=ChangeState(
                        lifecycle_node_matcher=matches_action(adapter),
                        transition_id=Transition.TRANSITION_ACTIVATE,
                    ),
                    condition=IfCondition(autostart),
                )
            ],
        )
    )
    return [configure, activate]


def _launch_setup(context):
    sensor_config = Path(LaunchConfiguration("sensor_config").perform(context))
    sensor_profile = LaunchConfiguration("sensor_profile").perform(context)
    localization_backend = LaunchConfiguration("localization_backend").perform(context)
    if localization_backend not in BACKEND_ODOMETRY_TOPICS:
        raise RuntimeError(
            "localization_backend must be exactly 'gnss_imu', 'eskf', "
            "'imu_quaternion_encoder', or 'quaternion_wheel_gnss_ekf'"
        )

    sensor_geometry = _load_sensor_geometry(sensor_config, sensor_profile)
    imu_topic = LaunchConfiguration("imu_topic")
    eskf_imu_topic = LaunchConfiguration("eskf_imu_topic")
    adapter = _make_adapter(
        localization_backend, sensor_geometry, imu_topic, eskf_imu_topic
    )
    estimator = _make_estimator(
        localization_backend,
        sensor_geometry,
        imu_topic,
        eskf_imu_topic,
    )
    manager = _make_manager(localization_backend)
    return [adapter, estimator, manager, *_make_autostart_handlers(adapter)]


def generate_launch_description():
    share = Path(get_package_share_directory("ad_localization"))
    description_share = Path(get_package_share_directory("ad_description"))
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "adapter_config",
                default_value=str(share / "config" / "localization.yaml"),
            ),
            DeclareLaunchArgument(
                "gnss_imu_config",
                default_value=str(share / "config" / "gnss_imu.yaml"),
            ),
            DeclareLaunchArgument(
                "eskf_config",
                default_value=str(share / "config" / "eskf.yaml"),
            ),
            DeclareLaunchArgument(
                "imu_quaternion_encoder_config",
                default_value=str(
                    share / "config" / "imu_quaternion_encoder.yaml"
                ),
            ),
            DeclareLaunchArgument(
                "quaternion_wheel_gnss_ekf_config",
                default_value=str(
                    share / "config" / "quaternion_wheel_gnss_ekf.yaml"
                ),
            ),
            DeclareLaunchArgument(
                "localization_manager_config",
                default_value=str(
                    share / "config" / "localization_manager.yaml"
                ),
            ),
            DeclareLaunchArgument(
                "imu_quaternion_encoder_mode",
                default_value="status_pose",
            ),
            DeclareLaunchArgument(
                "imu_topic",
                default_value="/ad/sensors/imu/data",
                description="Selected normalized IMU input for localization.",
            ),
            DeclareLaunchArgument(
                "eskf_imu_topic",
                default_value="/ad/localization/input/eskf_imu",
            ),
            DeclareLaunchArgument(
                "localization_backend",
                default_value="gnss_imu",
                description=(
                    "Use gnss_imu by default; eskf and "
                    "imu_quaternion_encoder or quaternion_wheel_gnss_ekf "
                    "are opt-in."
                ),
            ),
            DeclareLaunchArgument("status_topic", default_value="/ad/vehicle/status"),
            DeclareLaunchArgument("autostart", default_value="true"),
            DeclareLaunchArgument(
                "sensor_config",
                default_value=str(description_share / "config" / "sensor_mounts.yaml"),
            ),
            DeclareLaunchArgument(
                "sensor_profile",
                default_value="",
                description="Blank selects active_profile from sensor_mounts.yaml.",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
