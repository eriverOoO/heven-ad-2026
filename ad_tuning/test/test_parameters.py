import pytest

from ad_tuning.parameters import FloatParameter, IntParameter


class RecordingTrial:
    def __init__(self):
        self.calls = []

    def suggest_float(self, name, low, high, *, step=None, log=False):
        self.calls.append(("float", name, low, high, step, log))
        return 1.25

    def suggest_int(self, name, low, high, *, step=1, log=False):
        self.calls.append(("int", name, low, high, step, log))
        return 7


def test_float_parameter_suggests_plain_range():
    trial = RecordingTrial()

    assert FloatParameter("stanley.lookahead_m", 2.0, 8.0).suggest(trial) == 1.25
    assert trial.calls == [("float", "stanley.lookahead_m", 2.0, 8.0, None, False)]


def test_float_parameter_suggests_log_range():
    trial = RecordingTrial()

    assert FloatParameter("stanley.speed_pid.kp", 0.01, 10.0, log=True).suggest(trial) == 1.25
    assert trial.calls == [("float", "stanley.speed_pid.kp", 0.01, 10.0, None, True)]


def test_float_parameter_suggests_stepped_range():
    trial = RecordingTrial()

    assert FloatParameter("stanley.lookahead_m", 2.0, 8.0, step=0.5).suggest(trial) == 1.25
    assert trial.calls == [("float", "stanley.lookahead_m", 2.0, 8.0, 0.5, False)]


def test_int_parameter_suggests_range():
    trial = RecordingTrial()

    assert IntParameter("stanley.forward_window", 50, 300, step=10).suggest(trial) == 7
    assert trial.calls == [("int", "stanley.forward_window", 50, 300, 10, False)]


@pytest.mark.parametrize(
    "factory",
    [
        lambda: FloatParameter("bad", 2.0, 1.0),
        lambda: FloatParameter("bad", 1.0, 2.0, step=0.0),
        lambda: FloatParameter("bad", 1.0, 2.0, step=0.1, log=True),
        lambda: IntParameter("bad", 2, 1),
        lambda: IntParameter("bad", 1, 2, step=0),
        lambda: IntParameter("bad", 1, 2, step=2, log=True),
        lambda: IntParameter("bad", 1.5, 3),
        lambda: IntParameter("bad", 1, 3.5),
        lambda: IntParameter("bad", 1, 3, step=1.5),
    ],
)
def test_parameter_rejects_invalid_ranges(factory):
    with pytest.raises(ValueError):
        factory()
