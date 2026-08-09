import hashlib
import math
import os

import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node


_SUPPORTED_BACKENDS = {"dwa", "frenet_lattice", "mppi_nav2"}
_SUPPORTED_PATH_TRACKING_BACKENDS = {"stanley", "profile_stanley"}
_DWA_FOOTPRINT_GEOMETRY_FIELDS = (
    "center_offset_x_m",
    "half_length_m",
    "half_width_m",
    "clearance_m",
)


def _load_parameter_file(path):
    try:
        with open(path, encoding="utf-8") as stream:
            document = yaml.safe_load(stream)
        parameters = document["ad_planner"]["ros__parameters"]
    except (OSError, TypeError, KeyError, yaml.YAMLError) as error:
        raise RuntimeError(f"invalid planner parameter file: {path}") from error
    if not isinstance(parameters, dict):
        raise RuntimeError(f"invalid planner parameter file: {path}")
    return parameters


def _load_vehicle_parameters():
    vehicle_config = os.path.join(
        get_package_share_directory("ad_description"),
        "config",
        "vehicle_parameters.yaml",
    )
    with open(vehicle_config, encoding="utf-8") as stream:
        vehicle = yaml.safe_load(stream)["vehicle"]
    geometry = vehicle["geometry"]

    return {
        "stanley.control_point_x_m": vehicle["control"][
            "lateral_control_point_x_m"
        ],
        "profile_stanley.control_point_x_m": vehicle["control"][
            "lateral_control_point_x_m"
        ],
        "dwa.wheelbase_m": geometry["wheelbase_m"],
        **{
            f"dwa.footprint.{name}": value
            for name, value in vehicle["collision_footprint"].items()
            if name in _DWA_FOOTPRINT_GEOMETRY_FIELDS
        },
        "perception.front_bumper_x_m": geometry["front_bumper_x_m"],
        "traffic.front_bumper_x_m": geometry["front_bumper_x_m"],
        "local_motion.wheelbase_m": geometry["wheelbase_m"],
        "local_motion.footprint_front_m": geometry["front_bumper_x_m"],
        "local_motion.footprint_rear_m": abs(geometry["rear_bumper_x_m"]),
        "local_motion.footprint_half_width_m": geometry["width_m"] / 2,
        "local_motion.maximum_steering_rad": vehicle["steering"][
            "initial_bicycle_max_road_wheel_angle_rad"
        ],
    }


def _global_path_sha256(data_dir, path_file):
    if not isinstance(path_file, str) or not path_file:
        raise RuntimeError("local motion requires a configured global path")
    path = os.path.join(data_dir, path_file)
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            for chunk in iter(lambda: stream.read(65536), b""):
                digest.update(chunk)
    except OSError as error:
        raise RuntimeError(f"local motion requires global path: {path}") from error
    return digest.hexdigest()


def _positive_float_override(name, value):
    try:
        parsed = float(value)
    except ValueError as error:
        raise RuntimeError(f"{name} must be a positive number") from error
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise RuntimeError(f"{name} must be a positive number")
    return parsed


def _create_planner_node(context):
    config_file = LaunchConfiguration("config_file").perform(context)
    data_dir = LaunchConfiguration("data_dir").perform(context)
    path_override = LaunchConfiguration("path_file", default="").perform(context)
    corridor_override = LaunchConfiguration(
        "route_corridor_file", default=""
    ).perform(context)
    path_tracking_override = LaunchConfiguration(
        "path_tracking_backend", default=""
    ).perform(context)
    target_speed_override = LaunchConfiguration(
        "target_speed_mps", default=""
    ).perform(context).strip()
    perception_enabled_override = LaunchConfiguration(
        "perception_enabled", default=""
    ).perform(context).strip().lower()
    prediction_mode_override = LaunchConfiguration(
        "local_motion_prediction_mode", default=""
    ).perform(context).strip().lower()
    tuning_lease_required = LaunchConfiguration(
        "tuning_lease_required", default=""
    ).perform(context).strip().lower()
    common_parameters = _load_parameter_file(config_file)
    backend = common_parameters.get("local_motion.backend")
    if not isinstance(backend, str) or not backend or backend not in _SUPPORTED_BACKENDS:
        raise RuntimeError(f"unsupported local_motion.backend: {backend!r}")

    backend_file = os.path.join(
        get_package_share_directory("ad_planner"),
        "config",
        "local_planning",
        f"{backend}.yaml",
    )
    _load_parameter_file(backend_file)
    try:
        selected_path = path_override or common_parameters["path_file"]
    except KeyError as error:
        raise RuntimeError(
            "local motion requires a configured global path"
        ) from error
    if not isinstance(selected_path, str) or not selected_path:
        raise RuntimeError(
            "local motion requires a configured global path"
        )
    overrides = {
        "data_dir": data_dir,
        "path_file": selected_path,
    }
    if corridor_override:
        overrides["route_corridor_file"] = corridor_override
    if path_tracking_override:
        if path_tracking_override not in _SUPPORTED_PATH_TRACKING_BACKENDS:
            raise RuntimeError(
                "unsupported path_tracking.backend: "
                f"{path_tracking_override!r}"
            )
        overrides["path_tracking.backend"] = path_tracking_override
    selected_path_tracking = (
        path_tracking_override
        or common_parameters.get("path_tracking.backend")
    )
    for name, value in (("target_speed_mps", target_speed_override),):
        if not value:
            continue
        if (
            not isinstance(selected_path_tracking, str)
            or selected_path_tracking
            not in _SUPPORTED_PATH_TRACKING_BACKENDS
        ):
            raise RuntimeError(
                "speed overrides require a supported path_tracking.backend"
            )
        overrides[f"{selected_path_tracking}.{name}"] = (
            _positive_float_override(name, value)
        )
    if perception_enabled_override:
        if perception_enabled_override not in {"true", "false"}:
            raise RuntimeError(
                "perception_enabled must be empty, true, or false"
            )
        overrides["perception.enabled"] = (
            perception_enabled_override == "true"
        )
    if prediction_mode_override:
        if prediction_mode_override not in {"disabled", "required"}:
            raise RuntimeError(
                "local_motion_prediction_mode must be empty, disabled, "
                "or required"
            )
        overrides["local_motion.prediction.mode"] = (
            prediction_mode_override
        )
    if tuning_lease_required:
        if tuning_lease_required not in {"true", "false"}:
            raise RuntimeError(
                "tuning_lease_required must be empty, true, or false"
            )
        overrides["tuning.lease_required"] = (
            tuning_lease_required == "true"
        )
    overrides["route_corridor.expected_global_path_sha256"] = (
        _global_path_sha256(data_dir, selected_path)
    )

    return Node(
        package="ad_planner",
        executable="ad_planner_node",
        name="ad_planner",
        output="screen",
        parameters=[
            config_file,
            backend_file,
            _load_vehicle_parameters(),
            overrides,
        ],
    )


def _create_road_corridor_mask_node(context):
    config_file = LaunchConfiguration("config_file").perform(context)
    data_dir = LaunchConfiguration("data_dir").perform(context)
    path_override = LaunchConfiguration("path_file", default="").perform(context)
    corridor_override = LaunchConfiguration(
        "route_corridor_file", default=""
    ).perform(context)
    common_parameters = _load_parameter_file(config_file)
    selected_path = path_override or common_parameters.get("path_file")
    if not isinstance(selected_path, str) or not selected_path:
        raise RuntimeError(
            "road corridor mask requires a configured global path"
        )
    selected_corridor = corridor_override or common_parameters.get(
        "route_corridor_file", "map/route_corridor.json"
    )
    if not isinstance(selected_corridor, str) or not selected_corridor:
        raise RuntimeError(
            "road corridor mask requires route_corridor_file"
        )
    package_share = get_package_share_directory("ad_planner")
    return Node(
        package="ad_planner",
        executable="ad_road_corridor_mask_node",
        name="ad_road_corridor_mask",
        output="screen",
        parameters=[
            os.path.join(
                package_share, "config", "road_corridor_mask.yaml"
            ),
            {
                "data_dir": data_dir,
                "route_corridor_file": selected_corridor,
                "route_corridor.expected_global_path_sha256": (
                    _global_path_sha256(data_dir, selected_path)
                ),
            },
        ],
    )


def _create_planner_actions(context):
    planner_node = _create_planner_node(context)
    road_corridor_mask_node = _create_road_corridor_mask_node(context)
    config_file = LaunchConfiguration("config_file").perform(context)
    backend = _load_parameter_file(config_file).get("local_motion.backend")
    if backend != "mppi_nav2":
        return [planner_node, road_corridor_mask_node]

    package_share = get_package_share_directory("ad_planner")
    mppi_runtime = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(package_share, "launch", "mppi_nav2.launch.py")
        ),
        launch_arguments={
            "config_file": os.path.join(
                package_share,
                "config",
                "local_planning",
                "mppi_nav2.yaml",
            ),
        }.items(),
    )
    return [mppi_runtime, planner_node, road_corridor_mask_node]


def generate_launch_description():
    package_share = get_package_share_directory("ad_planner")
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "data_dir",
                default_value=EnvironmentVariable("AD_DATA_DIR", default_value=""),
                description="Directory containing checksum-verified competition data",
            ),
            DeclareLaunchArgument(
                "config_file",
                default_value=os.path.join(package_share, "config", "planner.yaml"),
            ),
            DeclareLaunchArgument("path_file", default_value=""),
            DeclareLaunchArgument("route_corridor_file", default_value=""),
            DeclareLaunchArgument("path_tracking_backend", default_value=""),
            DeclareLaunchArgument("target_speed_mps", default_value=""),
            DeclareLaunchArgument("perception_enabled", default_value=""),
            DeclareLaunchArgument(
                "local_motion_prediction_mode", default_value=""
            ),
            DeclareLaunchArgument(
                "tuning_lease_required", default_value=""
            ),
            OpaqueFunction(function=_create_planner_actions),
        ]
    )
