"""Opt-in one-command training-free perception + camera RViz demo.

Runs the model-free LiDAR perception stack (adaptive Euclidean clustering ->
Euclidean BEV gate + Hungarian association -> Linear KF / AB3DMOT lifecycle ->
analytical CV/CT/IMM prediction -> dynamic + combined occupancy grids) plus the
read-only ``ad_viz`` marker adapters and RViz, and shows the front camera image
in the same RViz session (Level A: 2D image panel, not geometrically projected).

Nothing here needs a learned detector, a learned estimator, a trained-weights
file, a MORAI training dataset, torch, CUDA, or OpenPCDet. Sensor input is
required at runtime and
comes from either live topics (``input_mode:=live``) or an existing rosbag
replay (``input_mode:=replay bag_path:=<abs mcap dir>``) - a rosbag is runtime
replay input, not a training dataset.

This launch is opt-in. It does not change any production launch default and it
is never included by another launch file.

    # live sensors
    ros2 launch ad_lidar_perception training_free_perception_rviz.launch.py

    # rosbag replay, with the front camera panel
    ros2 launch ad_lidar_perception training_free_perception_rviz.launch.py \
        input_mode:=replay bag_path:=/abs/path/to/bag enable_camera:=true
"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import SetParameter


# Training-free backend, locked. Detector: adaptive Euclidean clustering.
# Association: Euclidean BEV gate + Hungarian. Estimator: Linear KF (AB3DMOT
# lifecycle). These are the checked-in defaults of the euclidean detector and
# the ab3dmot_tracker.launch.py path already used by lidar_perception.launch.py.
DEMO_DETECTOR_BACKEND = "euclidean"
DEMO_TRACKER_BACKEND = "ab3dmot"
# lidar_perception.yaml pairs the Euclidean detector with static + dynamic +
# combined occupancy. The ab3dmot override is applied after selection parsing,
# so this composition drives prediction + dynamic OGM + combined OGM through the
# training-free tracker without a new composition file.
DEMO_COMPOSITION = "lidar_perception.yaml"


def _perform(context, name):
    return LaunchConfiguration(name).perform(context)


def _parse_bool(name, value):
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise RuntimeError(f"{name} must be exactly 'true' or 'false'")


def _launch_file(package, name):
    share = Path(get_package_share_directory(package))
    return PythonLaunchDescriptionSource(str(share / "launch" / name))


def _launch_setup(context):
    perception_share = Path(get_package_share_directory("ad_lidar_perception"))

    input_mode = _perform(context, "input_mode").strip().lower()
    if input_mode not in {"live", "replay"}:
        raise RuntimeError("input_mode must be exactly 'live' or 'replay'")
    replay = input_mode == "replay"

    enable_camera = _parse_bool(
        "enable_camera", _perform(context, "enable_camera")
    )
    enable_camera_perception = _parse_bool(
        "enable_camera_perception",
        _perform(context, "enable_camera_perception"),
    )
    enable_dynamic_object_risk = _parse_bool(
        "enable_dynamic_object_risk",
        _perform(context, "enable_dynamic_object_risk"),
    )
    enable_localization = _parse_bool(
        "enable_localization", _perform(context, "enable_localization")
    )
    enable_drivable_mask = _parse_bool(
        "enable_drivable_mask", _perform(context, "enable_drivable_mask")
    )
    start_rviz = _parse_bool("start_rviz", _perform(context, "start_rviz"))

    composition_config = str(
        perception_share / "config" / DEMO_COMPOSITION
    )
    rviz_config = _perform(context, "rviz_config").strip() or str(
        perception_share
        / "rviz"
        / "training_free_perception_camera.rviz"
    )
    # Replay drives /clock from the bag; live sensors run on wall time.
    use_sim_time = "true" if replay else "false"
    risk_flag = "true" if enable_dynamic_object_risk else "false"

    actions = []

    if replay:
        bag_path = _perform(context, "bag_path").strip()
        if not bag_path:
            raise RuntimeError(
                "input_mode:=replay requires bag_path:=<absolute rosbag "
                "directory containing metadata.yaml>"
            )
        # lidar_bag_replay validates the MCAP bag, publishes /clock, starts
        # ad_description (static TF) and the perception stack on sim time.
        actions.append(
            IncludeLaunchDescription(
                _launch_file(
                    "ad_lidar_perception", "lidar_bag_replay.launch.py"
                ),
                launch_arguments={
                    "bag_path": bag_path,
                    "rate": _perform(context, "rate"),
                    "loop": _perform(context, "loop"),
                    "start_paused": _perform(context, "start_paused"),
                    "composition_config": composition_config,
                    "detector_backend": DEMO_DETECTOR_BACKEND,
                    "tracker_backend": DEMO_TRACKER_BACKEND,
                    "dynamic_object_risk": risk_flag,
                    "include_front_camera": (
                        "true" if enable_camera else "false"
                    ),
                    "enable_localization": (
                        "true" if enable_localization else "false"
                    ),
                }.items(),
            )
        )
    else:
        actions.append(
            IncludeLaunchDescription(
                _launch_file("ad_description", "description.launch.py")
            )
        )
        actions.append(
            IncludeLaunchDescription(
                _launch_file(
                    "ad_lidar_perception", "lidar_perception.launch.py"
                ),
                launch_arguments={
                    "composition_config": composition_config,
                    "detector_backend": DEMO_DETECTOR_BACKEND,
                    "tracker_backend": DEMO_TRACKER_BACKEND,
                    "dynamic_object_risk": risk_flag,
                    "platform_profile": "morai",
                    "start_visualization": "false",
                    "start_rviz": "false",
                    "use_sim_time": "false",
                }.items(),
            )
        )

    # Read-only visualization side path. Started here (not via the perception
    # launch) so exactly one visualizer set and one RViz process run, on the
    # correct clock, with the experimental /experiment/tracked/ab3dmot view
    # dropped (the training-free AB3DMOT arm publishes on
    # /ad/perception/objects/tracked, which the primary "A-" visualizer covers).
    actions.append(
        IncludeLaunchDescription(
            _launch_file(
                "ad_lidar_perception", "perception_visualization.launch.py"
            ),
            launch_arguments={
                "start_rviz": "true" if start_rviz else "false",
                "use_sim_time": use_sim_time,
                "rviz_config": rviz_config,
                "enable_experiment_tracker_view": "false",
            }.items(),
        )
    )

    if enable_drivable_mask:
        # Opt-in, read-only drivable-mask producer for the existing
        # road_gate-gated dynamic/static occupancy grids. Reuses the
        # already-existing, already-tested ad_road_corridor_mask_node
        # (ad_planner) unmodified - it rasterizes the committed,
        # checksum-verified competition route corridor
        # (ad_data/map/route_corridor.json) into the exact base_link-frame
        # grid contract the OGM nodes' road_gate already requires. No
        # planner, controller, or CtrlCmd publisher is started; this reads
        # LiDAR/prediction timing + TF only. Scoped so its use_sim_time
        # matches replay/live without leaking into any other node.
        actions.append(
            GroupAction(
                scoped=True,
                actions=[
                    SetParameter(name="use_sim_time", value=replay),
                    IncludeLaunchDescription(
                        _launch_file(
                            "ad_planner", "road_corridor_mask.launch.py"
                        ),
                        launch_arguments={
                            "data_dir": _perform(context, "data_dir"),
                        }.items(),
                    ),
                ],
            )
        )

    if enable_camera_perception:
        # Optional camera perception overlay. Needs ultralytics/torch and the
        # separately-optional ad_camera_perception package; stays fully behind
        # this opt-in flag so the model-free demo never depends on it.
        actions.append(
            GroupAction(
                scoped=True,
                actions=[
                    SetParameter(
                        name="use_sim_time", value=replay
                    ),
                    IncludeLaunchDescription(
                        _launch_file(
                            "ad_camera_perception",
                            "dynamic_obstacle_detection.launch.py",
                        )
                    ),
                ],
            )
        )

    return actions


def generate_launch_description():
    perception_share = Path(get_package_share_directory("ad_lidar_perception"))
    default_rviz = str(
        perception_share / "rviz" / "training_free_perception_camera.rviz"
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "input_mode",
                default_value="live",
                description=(
                    "'live' (sensor topics) or 'replay' (existing rosbag)"
                ),
            ),
            DeclareLaunchArgument(
                "bag_path",
                default_value="",
                description=(
                    "Absolute rosbag directory; required for input_mode:=replay"
                ),
            ),
            DeclareLaunchArgument(
                "rate",
                default_value="0.5",
                description="rosbag playback rate (replay only)",
            ),
            DeclareLaunchArgument(
                "loop",
                default_value="true",
                description="Replay the bag repeatedly (replay only)",
            ),
            DeclareLaunchArgument(
                "start_paused",
                default_value="false",
                description="Start the rosbag paused (replay only)",
            ),
            DeclareLaunchArgument(
                "enable_camera",
                default_value="false",
                description=(
                    "Replay the front compressed-camera topic so the RViz "
                    "image panel is populated (replay only; live camera comes "
                    "from the sensor driver)"
                ),
            ),
            DeclareLaunchArgument(
                "enable_camera_perception",
                default_value="false",
                description=(
                    "Also start the ad_camera_perception YOLO overlay - needs "
                    "ultralytics/torch and is not part of the model-free demo"
                ),
            ),
            DeclareLaunchArgument(
                "enable_dynamic_object_risk",
                default_value="true",
                description=(
                    "Start the observational Dynamic Object Risk node"
                ),
            ),
            DeclareLaunchArgument(
                "enable_localization",
                default_value="false",
                description=(
                    "Replay only: also replay /ad/sensors/gps/fix + "
                    "/ad/vehicle/status and start ad_localization's "
                    "gnss_imu backend for a bag that has raw ego sensors "
                    "but no recorded /ad/localization/odometry"
                ),
            ),
            DeclareLaunchArgument(
                "enable_drivable_mask",
                default_value="false",
                description=(
                    "Also start ad_planner's existing "
                    "ad_road_corridor_mask_node so the already-gated "
                    "dynamic/static occupancy grids (road_gate.enabled: "
                    "true) can publish. Rasterizes the committed "
                    "ad_data/map/route_corridor.json - never a fabricated "
                    "all-drivable mask. No planner/controller/CtrlCmd node "
                    "is started."
                ),
            ),
            DeclareLaunchArgument(
                "data_dir",
                default_value=EnvironmentVariable(
                    "AD_DATA_DIR", default_value=""
                ),
                description=(
                    "Directory containing the committed competition route "
                    "corridor + global path; only read when "
                    "enable_drivable_mask:=true"
                ),
            ),
            DeclareLaunchArgument(
                "start_rviz",
                default_value="true",
                description="Open RViz with the demo config",
            ),
            DeclareLaunchArgument(
                "rviz_config",
                default_value=default_rviz,
                description="RViz config; blank uses the demo default",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
