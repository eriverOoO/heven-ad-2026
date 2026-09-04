"""Launch the Autoware-data-association IMM prediction adapter.

Parameters are read from ``config/tracking/prediction.yaml`` in Python and the
two launch-controlled keys (``yaw_rate_source``,
``runtime_summary_interval_frames``) are overlaid on top *before* launch, so the
node process receives exactly ONE parameter source. This is the same pattern
``ad_planner``'s ``road_corridor_mask.launch.py`` uses, and for the same reason:
passing ``parameters=[config_file, override_dict]`` as two ``--params-file``
arguments for one node lets ``rcl`` resolve the override against the checked-in
YAML by section specificity, not file order. The YAML section is node-name
scoped (``ad_autoware_prediction:``) while ``launch_ros`` always writes an
override dict under the wildcard ``/**:``; when any individual ``-p`` argument
is also present -- ``launch_ros``'s ``SetParameter(use_sim_time=...)`` in
``lidar_bag_replay.launch.py`` emits one -- ``rcl`` then lets the more-specific
YAML section win and the override is silently dropped. Reproduced directly:
``ros2 run ... -p use_sim_time:=True --params-file <node-scoped tracker>
--params-file </**: motion_history>`` yields ``tracker``; drop the ``-p`` and it
yields ``motion_history``. Merging in Python removes the two-file race entirely.
The node's own ``declare_parameter`` code is untouched.
"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import yaml

_NODE_NAME = "ad_autoware_prediction"
_VALID_YAW_RATE_SOURCES = ("tracker", "motion_history")


def _default_config() -> str:
    return str(
        Path(get_package_share_directory("ad_lidar_perception"))
        / "config"
        / "tracking"
        / "prediction.yaml"
    )


def _node_parameters(config_file: str) -> dict:
    with open(config_file, encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    try:
        parameters = document[_NODE_NAME]["ros__parameters"]
    except (KeyError, TypeError) as error:
        raise RuntimeError(
            f"{config_file} must declare {_NODE_NAME}.ros__parameters"
        ) from error
    if not isinstance(parameters, dict):
        raise RuntimeError(
            f"{config_file}: {_NODE_NAME}.ros__parameters must be a mapping"
        )
    return dict(parameters)


def _create_node(context):
    config_file = LaunchConfiguration("config_file").perform(context)
    parameters = _node_parameters(config_file)

    yaw_rate_source = (
        LaunchConfiguration("yaw_rate_source").perform(context).strip()
    )
    if yaw_rate_source not in _VALID_YAW_RATE_SOURCES:
        raise RuntimeError(
            "yaw_rate_source must be 'tracker' or 'motion_history'"
        )
    parameters["yaw_rate_source"] = yaw_rate_source

    runtime_summary_interval_frames = (
        LaunchConfiguration("runtime_summary_interval_frames")
        .perform(context)
        .strip()
    )
    try:
        parameters["runtime_summary_interval_frames"] = int(
            runtime_summary_interval_frames
        )
    except ValueError as error:
        raise RuntimeError(
            "runtime_summary_interval_frames must be an integer"
        ) from error

    return [
        Node(
            package="ad_lidar_perception",
            executable="ad_autoware_prediction_node",
            name=_NODE_NAME,
            output="screen",
            parameters=[parameters],
        )
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value=_default_config(),
                description="Autoware prediction adapter parameter file",
            ),
            DeclareLaunchArgument(
                "runtime_summary_interval_frames",
                default_value="0",
            ),
            DeclareLaunchArgument(
                "yaw_rate_source",
                default_value="tracker",
                description=(
                    "IMM yaw-rate source: 'tracker' (default, reproduces the "
                    "pre-Curve-Aware-Prediction behaviour) or 'motion_history' "
                    "(derive turn rate from recent velocity history). "
                    "Authoritative over the config file's yaw_rate_source key."
                ),
            ),
            OpaqueFunction(function=_create_node),
        ]
    )
