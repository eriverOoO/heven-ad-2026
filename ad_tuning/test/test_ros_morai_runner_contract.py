from types import SimpleNamespace
import threading
import time

import pytest
from rcl_interfaces.msg import ParameterType, ParameterValue

import ad_tuning.ros_morai_runner as runner_module
from ad_tuning.ros_morai_runner import (
    RosMoraiGlobalPathRunner,
    dwa_failure_is_controller_infeasible,
    stale_input_names,
)


def test_stale_input_names_identifies_the_exact_missing_or_old_channels():
    assert stale_input_names(
        now_s=10.0,
        timeout_s=2.0,
        samples={
            "odometry": (object(), 9.0),
            "vehicle_status": (object(), 7.9),
            "target_speed": (None, 9.9),
            "control_command": (object(), 8.0),
        },
    ) == ("vehicle_status", "target_speed")


def test_dwa_dead_end_is_not_mislabeled_as_infrastructure_disconnect():
    for reason in (
        "initial footprint is unsafe",
        "no safe DWA candidate",
    ):
        assert dwa_failure_is_controller_infeasible(
            reason, ("target_speed",)
        )
    assert not dwa_failure_is_controller_infeasible(
        "initial footprint is unsafe",
        ("occupancy_grid", "target_speed"),
    )
    assert not dwa_failure_is_controller_infeasible(
        "occupancy grid and odometry stamps are too far apart",
        ("target_speed",),
    )


class _RecordingLogger:
    def info(self, _message):
        pass


class _RunnerNode:
    def get_logger(self):
        return _RecordingLogger()


class _NeverReadyClient:
    def __init__(self, runner):
        self._runner = runner
        self.calls = 0

    def wait_for_service(self, timeout_sec):
        del timeout_sec
        self.calls += 1
        time.sleep(0.005)
        if self.calls >= 2:
            self._runner._shutdown = True
        return False


def test_wait_until_ready_times_out_instead_of_waiting_forever_for_service():
    runner = object.__new__(RosMoraiGlobalPathRunner)
    runner.startup_timeout_s = 0.001
    runner._shutdown = False
    runner.node = _RunnerNode()
    runner._hold_client = _NeverReadyClient(runner)

    with pytest.raises(
        TimeoutError,
        match="startup timeout.*planner hold service",
    ):
        runner.wait_until_ready()


class _ReadbackClient:
    def __init__(self, values):
        self.values = values
        self.request = None

    def call_async(self, request):
        self.request = request
        return SimpleNamespace(values=self.values)


def _parameter_value(
    parameter_type,
    *,
    bool_value=False,
    integer_value=0,
    double_value=0.0,
    string_value="",
):
    return ParameterValue(
        type=parameter_type,
        bool_value=bool_value,
        integer_value=integer_value,
        double_value=double_value,
        string_value=string_value,
    )


def test_verify_parameters_reads_back_string_bool_integer_and_double_types():
    runner = object.__new__(RosMoraiGlobalPathRunner)
    runner._get_parameter_client = _ReadbackClient(
        [
            _parameter_value(
                ParameterType.PARAMETER_STRING,
                string_value="dwa",
            ),
            _parameter_value(
                ParameterType.PARAMETER_BOOL,
                bool_value=True,
            ),
            _parameter_value(
                ParameterType.PARAMETER_INTEGER,
                integer_value=4,
            ),
            _parameter_value(
                ParameterType.PARAMETER_DOUBLE,
                double_value=1.25,
            ),
        ]
    )
    runner._wait_future = lambda future, _timeout_s: future

    runner.verify_parameters(
        {
            "local_motion.backend": "dwa",
            "perception.enabled": True,
            "example.count": 4,
            "example.gain": 1.25,
        }
    )

    assert runner._get_parameter_client.request.names == [
        "local_motion.backend",
        "perception.enabled",
        "example.count",
        "example.gain",
    ]


def test_verify_parameters_rejects_wrong_ros_type_even_if_default_field_matches():
    runner = object.__new__(RosMoraiGlobalPathRunner)
    runner._get_parameter_client = _ReadbackClient(
        [
            _parameter_value(
                ParameterType.PARAMETER_DOUBLE,
                double_value=0.0,
                string_value="dwa",
            )
        ]
    )
    runner._wait_future = lambda future, _timeout_s: future

    with pytest.raises(
        RuntimeError,
        match="local_motion.backend.*expected string.*got double",
    ):
        runner.verify_parameters({"local_motion.backend": "dwa"})


def test_perception_epoch_wait_uses_post_reset_prediction_and_sim_clock(
    monkeypatch,
):
    runner = object.__new__(RosMoraiGlobalPathRunner)
    runner.require_perception_epoch = True
    runner.reset_timeout_s = 0.5
    runner._shutdown = False
    runner._condition = threading.Condition()
    runner._status_received_s = 11.0
    runner._status_sim_s = 101.2
    runner._prediction_received_s = 11.1
    runner._prediction_sequence = 25
    runner._occupancy_grid_received_s = 11.1
    runner.perception_epoch_minimum_prediction_samples = 5
    runner.perception_epoch_settle_sim_s = 1.05
    runner.stale_timeout_s = 2.0
    runner.require_occupancy_grid = True
    observed = {}

    def ready(**arguments):
        observed.update(arguments)
        return True

    monkeypatch.setattr(
        runner_module, "perception_epoch_is_ready", ready
    )

    assert runner._wait_for_perception_epoch(10.0, 20)
    assert observed["baseline_prediction_sequence"] == 20
    assert observed["prediction_sequence"] == 25
    assert observed["first_post_reset_sim_s"] == pytest.approx(101.2)
    assert observed["current_sim_s"] == pytest.approx(101.2)
