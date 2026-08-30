import math
from pathlib import Path

import yaml

from ad_lidar_perception.ab3dmot_config import AB3DMOTConfig
from ad_lidar_perception.ab3dmot_core import AB3DMOTTracker, Detection


PACKAGE = Path(__file__).resolve().parents[1]
TRACKER_CONFIG = (
    PACKAGE / "config" / "tracking" / "competition_mot_baseline_v1.yaml"
)


def detection(x=0.0, y=0.0, yaw=0.0):
    return Detection(x, y, 0.0, yaw, 4.0, 2.0, 1.5, 0, 1.0, 1.0)


def baseline_config():
    document = yaml.safe_load(TRACKER_CONFIG.read_text(encoding="utf-8"))
    parameters = document["/**"]["ros__parameters"]
    fields = AB3DMOTConfig.__dataclass_fields__
    return AB3DMOTConfig(
        **{name: value for name, value in parameters.items() if name in fields}
    )


def run_signature():
    tracker = AB3DMOTTracker(baseline_config())
    frames = [
        [detection(-2.0, -0.3), detection(2.0, 0.3)],
        [detection(-0.8, -0.3), detection(0.8, 0.3)],
        [detection(0.4, -0.3), detection(-0.4, 0.3)],
        [detection(1.6, -0.3), detection(-1.6, 0.3)],
    ]
    signature = []
    for index, detections in enumerate(frames):
        outputs = tracker.step(detections, 10.0 + index * 0.5)
        signature.append(
            tuple(
                (state.track_id, round(state.x, 9), round(state.y, 9))
                for state in outputs
            )
        )
    return tuple(signature)


def test_baseline_config_is_fully_explicit_and_selects_validated_combination():
    config = baseline_config()
    assert config.association_metric == "euclidean"
    assert config.euclidean_gate_m == 3.0
    assert config.matcher == "hungarian"
    assert config.state_estimator == "linear_kf"
    assert config.yaw_measurement_mode == "unobserved"
    assert config.min_hits == 1
    assert config.max_age == 2


def test_zero_one_and_multiple_detections_create_finite_tracks():
    tracker = AB3DMOTTracker(baseline_config())
    assert tracker.step([], 1.0) == []
    one = tracker.step([detection()], 2.0)
    assert len(one) == 1
    multiple = tracker.step(
        [detection(0.2, 0.0), detection(10.0, 1.0)], 3.0
    )
    assert len(multiple) == 2
    assert all(
        math.isfinite(value)
        for state in multiple
        for value in (
            state.x,
            state.y,
            state.z,
            state.yaw,
            state.vx_mps,
            state.vy_mps,
            state.vz_mps,
        )
    )


def test_multiple_tracks_have_private_linear_kf_state():
    tracker = AB3DMOTTracker(baseline_config())
    tracker.step([detection(0.0), detection(10.0)], 1.0)
    first, second = tracker.tracks
    assert first._estimator is not second._estimator
    assert first._estimator._kf is not second._estimator._kf
    first._estimator._kf.x[0, 0] = 123.0
    assert second._estimator._kf.x[0, 0] == 10.0


def test_yaw_unobserved_ignores_placeholder_detector_yaw():
    tracker = AB3DMOTTracker(baseline_config())
    first = tracker.step([detection(yaw=2.8)], 1.0)[0]
    second = tracker.step([detection(0.2, yaw=-2.8)], 2.0)[0]
    assert first.yaw == 0.0
    assert second.yaw == 0.0


def test_larger_dt_propagates_real_seconds_into_velocity_state():
    tracker = AB3DMOTTracker(baseline_config())
    tracker.step([detection(0.0)], 1.0)
    tracker.step([detection(0.2)], 1.1)
    output = tracker.step([detection(4.2)], 3.1)[0]
    assert output.track_id == 1
    assert abs(output.vx_mps - 2.0) < 0.3


def test_crossing_assignment_is_deterministic_for_same_input():
    assert run_signature() == run_signature()


def test_lifecycle_regression_uses_existing_frame_based_max_age():
    tracker = AB3DMOTTracker(baseline_config())
    first_id = tracker.step([detection()], 1.0)[0].track_id
    coast = tracker.step([], 2.0)
    assert [state.track_id for state in coast] == [first_id]
    assert tracker.step([], 3.0) == []
    assert tracker.tracks == ()
    replacement = tracker.step([detection()], 4.0)[0]
    assert replacement.track_id != first_id
