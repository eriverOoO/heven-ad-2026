"""Runtime parameter-propagation contract for ``prediction.launch.py``.

The prediction node's ``yaw_rate_source`` (Curve-Aware Prediction v1) is
selectable from the launch chain
(``study_pipeline_rviz`` -> ``lidar_bag_replay`` -> ``lidar_perception`` ->
``prediction.launch.py``).  A prior integration bug let an explicit
``motion_history`` request be silently dropped: ``prediction.launch.py`` passed
``parameters=[prediction.yaml, {override}]`` as two ``--params-file`` arguments,
and once any individual ``-p`` argument was also present -- ``launch_ros``'s
``SetParameter(use_sim_time=...)`` in ``lidar_bag_replay.launch.py`` emits one --
``rcl`` resolved ``yaw_rate_source`` against the node-name-scoped YAML section
rather than the later wildcard override, keeping the file's ``tracker``.

These tests assert the ACTUAL runtime parameter on a launched node (not just the
launch-description forwarding, which the bug survived), under both the plain
include and the exact ``SetParameter`` + scoped-group condition that triggered
it.  ``config/tracking/prediction.yaml`` is unchanged; the launch file now
merges the override into the config in Python so the node gets one source.
"""

import subprocess
import time
import unittest

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    GroupAction,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import PushRosNamespace, SetParameter
import launch_testing
import launch_testing.actions
import pytest


DOMAIN_ID = 95
_PREDICTION_LAUNCH = str(
    (
        __import__("pathlib").Path(
            get_package_share_directory("ad_lidar_perception")
        )
        / "launch"
        / "prediction.launch.py"
    )
)


def _include(namespace, launch_arguments, *, with_set_parameter):
    inc = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(_PREDICTION_LAUNCH),
        launch_arguments=launch_arguments.items(),
    )
    actions = [PushRosNamespace(namespace)]
    if with_set_parameter:
        # Reproduces lidar_bag_replay.launch.py's own scoped group: an
        # unrelated global parameter that forces launch_ros to emit a leading
        # ``-p`` argument onto every node in the group.
        actions.append(SetParameter(name="use_sim_time", value=True))
    actions.append(inc)
    return GroupAction(scoped=True, actions=actions)


@pytest.mark.launch_test
def generate_test_description():
    return LaunchDescription(
        [
            SetEnvironmentVariable("ROS_DOMAIN_ID", str(DOMAIN_ID)),
            SetEnvironmentVariable("ROS_LOCALHOST_ONLY", "1"),
            # A: plain include, default -> tracker
            _include("plain_default", {}, with_set_parameter=False),
            # B: plain include, explicit override -> motion_history
            _include(
                "plain_override",
                {"yaw_rate_source": "motion_history"},
                with_set_parameter=False,
            ),
            # C: SetParameter + scoped group (the bug trigger), default -> tracker
            _include("setparam_default", {}, with_set_parameter=True),
            # D: SetParameter + scoped group + explicit override -> motion_history
            _include(
                "setparam_override",
                {"yaw_rate_source": "motion_history"},
                with_set_parameter=True,
            ),
            launch_testing.actions.ReadyToTest(),
        ]
    )


def _param(node_fqn, name):
    result = subprocess.run(
        ["ros2", "param", "get", node_fqn, name],
        check=True,
        capture_output=True,
        text=True,
        timeout=15.0,
    )
    return result.stdout.strip().split()[-1]


class TestPredictionYawRatePropagation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Let all four nodes finish construction and register their services.
        deadline = time.time() + 30.0
        needed = {
            "/plain_default/ad_autoware_prediction",
            "/plain_override/ad_autoware_prediction",
            "/setparam_default/ad_autoware_prediction",
            "/setparam_override/ad_autoware_prediction",
        }
        while time.time() < deadline:
            listing = subprocess.run(
                ["ros2", "node", "list"],
                check=False,
                capture_output=True,
                text=True,
                timeout=15.0,
            ).stdout.split()
            if needed.issubset(set(listing)):
                break
            time.sleep(1.0)
        time.sleep(2.0)

    def test_a_plain_include_default_is_tracker(self):
        self.assertEqual(
            _param("/plain_default/ad_autoware_prediction", "yaw_rate_source"),
            "tracker",
        )

    def test_b_plain_include_override_reaches_node(self):
        self.assertEqual(
            _param("/plain_override/ad_autoware_prediction", "yaw_rate_source"),
            "motion_history",
        )

    def test_c_setparameter_scoped_group_default_is_tracker(self):
        self.assertEqual(
            _param(
                "/setparam_default/ad_autoware_prediction", "yaw_rate_source"
            ),
            "tracker",
        )

    def test_d_setparameter_scoped_group_override_reaches_node(self):
        # The exact condition that silently dropped the override before the fix.
        self.assertEqual(
            _param(
                "/setparam_override/ad_autoware_prediction", "yaw_rate_source"
            ),
            "motion_history",
        )

    def test_e_config_file_parameters_survive_the_python_merge(self):
        # The launch file now overlays the override on the YAML in Python; the
        # rest of prediction.yaml (nested dicts, lists) must be unchanged.
        node = "/setparam_override/ad_autoware_prediction"
        self.assertEqual(
            _param(node, "imm.process_variance.coordinated_turn"), "0.35"
        )
        self.assertEqual(_param(node, "motion_history.history_samples"), "4")
        self.assertEqual(_param(node, "expected_frame_id"), "odom")
        self.assertEqual(
            _param(node, "runtime_summary_interval_frames"), "0"
        )


@launch_testing.post_shutdown_test()
class TestPredictionLaunchShutdown(unittest.TestCase):
    def test_exit_codes(self, proc_info):
        launch_testing.asserts.assertExitCodes(proc_info)
