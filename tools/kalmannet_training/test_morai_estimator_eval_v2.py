"""Tests for morai_estimator_eval_v2.py -- MORAI Frozen Estimator
Evaluation Dataset v2 tooling. Uses only synthetic fixture data (never
the real local morai_heven/motion_gt_expansion files), matching the
project's established convention (test_morai_frozen_stream.py) for
tooling that must be verifiable without machine-local assets."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from morai_estimator_eval_v2 import (  # noqa: E402
    FrameMismatchError,
    SequenceBuildError,
    V1_FROZEN_TEST_ACTOR_IDS,
    assert_same_metric_frame,
    assert_valid_transform_chain,
    audit_actor_leakage,
    build_v2_freeze_manifest,
    build_v2_sequences,
    content_hash,
    paired_bootstrap,
    run_level_macro_average,
    save_v2_freeze_manifest,
    sequence_id,
    validate_v2_sequences,
)


# --------------------------------------------------------------------
# leakage audit
# --------------------------------------------------------------------
def test_leakage_audit_marks_train_val_ineligible():
    rows = audit_actor_leakage(
        all_actor_ids=[1, 2, 3, 4], train_ids=(1,), val_ids=(2,), test_ids=(3,), excluded_ids=(4,),
    )
    by_id = {r.actor_id: r for r in rows}
    assert by_id[1].eligible_for_v2 is False and by_id[1].role == "train"
    assert by_id[2].eligible_for_v2 is False and by_id[2].role == "val"
    assert by_id[3].eligible_for_v2 is False and by_id[3].role == "test"
    assert by_id[4].eligible_for_v2 is False and by_id[4].role == "excluded_data_quality"


def test_leakage_audit_unknown_actor_fails_closed():
    rows = audit_actor_leakage(all_actor_ids=[99], train_ids=(), val_ids=(), test_ids=(), excluded_ids=())
    assert rows[0].eligible_for_v2 is False
    assert "fails closed" in rows[0].reason or "ineligible" in rows[0].reason


def test_v1_frozen_test_actor_ids_unchanged():
    assert V1_FROZEN_TEST_ACTOR_IDS == (16, 20, 30)


# --------------------------------------------------------------------
# frame/coordinate assertions
# --------------------------------------------------------------------
def test_assert_same_metric_frame_passes_when_equal():
    assert_same_metric_frame("lidar_link", "lidar_link")  # no raise


def test_assert_same_metric_frame_raises_on_mismatch():
    with pytest.raises(FrameMismatchError):
        assert_same_metric_frame("lidar_link", "map")


def test_assert_valid_transform_chain_passes_on_expected_chain():
    assert_valid_transform_chain(["map", "odom", "base_link", "rear_axle_link", "lidar_link"])


def test_assert_valid_transform_chain_rejects_wrong_chain():
    with pytest.raises(FrameMismatchError):
        assert_valid_transform_chain(["map", "base_link", "lidar_link"])


# --------------------------------------------------------------------
# sequence construction (synthetic fixture GT + detections)
# --------------------------------------------------------------------
def _fixture_gt_and_detections(n=5, actor_id=100, gap_at=None):
    gt_frames = []
    det_frames = []
    for i in range(n):
        x, y = 1.0 * i, 2.0 * i
        gt_frames.append({
            "frame_id": "lidar_link",
            "header_stamp_ns": 1_000_000_000 + i * 100_000_000,
            "objects": [{"actor_id": actor_id, "class": "vehicle", "x": x, "y": y}],
        })
        if gap_at is not None and i == gap_at:
            det_frames.append({"frame_id": "lidar_link", "objects": []})
        else:
            det_frames.append({"frame_id": "lidar_link", "objects": [{"pos": [x + 0.1, y - 0.1, 0.0]}]})
    return gt_frames, det_frames


def test_build_v2_sequences_produces_one_sequence_per_eligible_actor():
    gt, det = _fixture_gt_and_detections(n=5, actor_id=100)
    seqs = build_v2_sequences(gt, det, eligible_actor_ids=[100], run_id="run1", role="test",
                               detector_source_version="v1", gt_source_version="v1")
    assert len(seqs) == 1
    assert seqs[0].actor_id == 100
    assert len(seqs[0].frames) == 5
    assert seqs[0].dt[0] is None
    assert all(d is not None and d > 0 for d in seqs[0].dt[1:])


def test_build_v2_sequences_ignores_ineligible_actors():
    gt, det = _fixture_gt_and_detections(n=5, actor_id=100)
    seqs = build_v2_sequences(gt, det, eligible_actor_ids=[999], run_id="run1", role="test",
                               detector_source_version="v1", gt_source_version="v1")
    assert seqs == []


def test_build_v2_sequences_missing_measurement_is_none_not_zero():
    gt, det = _fixture_gt_and_detections(n=5, actor_id=100, gap_at=2)
    seqs = build_v2_sequences(gt, det, eligible_actor_ids=[100], run_id="run1", role="test",
                               detector_source_version="v1", gt_source_version="v1")
    assert seqs[0].z_meas[2] is None
    assert all(seqs[0].z_meas[i] is not None for i in (0, 1, 3, 4))


def test_build_v2_sequences_rejects_frame_id_mismatch():
    gt, det = _fixture_gt_and_detections(n=3, actor_id=100)
    det[1]["frame_id"] = "map"  # inject a mismatch
    with pytest.raises(FrameMismatchError):
        build_v2_sequences(gt, det, eligible_actor_ids=[100], run_id="run1", role="test",
                            detector_source_version="v1", gt_source_version="v1")


def test_build_v2_sequences_rejects_unequal_length_inputs():
    gt, det = _fixture_gt_and_detections(n=5, actor_id=100)
    with pytest.raises(SequenceBuildError):
        build_v2_sequences(gt, det[:-1], eligible_actor_ids=[100], run_id="run1", role="test",
                            detector_source_version="v1", gt_source_version="v1")


def test_sequence_id_is_deterministic():
    gt, det = _fixture_gt_and_detections(n=5, actor_id=100)
    seqs1 = build_v2_sequences(gt, det, eligible_actor_ids=[100], run_id="run1", role="test",
                                detector_source_version="v1", gt_source_version="v1")
    seqs2 = build_v2_sequences(gt, det, eligible_actor_ids=[100], run_id="run1", role="test",
                                detector_source_version="v1", gt_source_version="v1")
    assert sequence_id(seqs1[0]) == sequence_id(seqs2[0]) == "run1__actor_100"


# --------------------------------------------------------------------
# validator
# --------------------------------------------------------------------
def test_validate_v2_sequences_detects_duplicate_sequence_ids():
    gt, det = _fixture_gt_and_detections(n=5, actor_id=100)
    seqs = build_v2_sequences(gt, det, eligible_actor_ids=[100], run_id="run1", role="test",
                               detector_source_version="v1", gt_source_version="v1")
    issues = validate_v2_sequences(seqs + seqs)
    assert any(i.severity == "error" and "duplicate" in i.message for i in issues)


def test_validate_v2_sequences_clean_sequence_has_no_errors():
    gt, det = _fixture_gt_and_detections(n=20, actor_id=100)
    seqs = build_v2_sequences(gt, det, eligible_actor_ids=[100], run_id="run1", role="test",
                               detector_source_version="v1", gt_source_version="v1")
    issues = validate_v2_sequences(seqs)
    assert not any(i.severity == "error" for i in issues)


def test_validate_v2_sequences_warns_on_short_trajectory():
    gt, det = _fixture_gt_and_detections(n=3, actor_id=100)
    seqs = build_v2_sequences(gt, det, eligible_actor_ids=[100], run_id="run1", role="test",
                               detector_source_version="v1", gt_source_version="v1")
    issues = validate_v2_sequences(seqs)
    assert any(i.severity == "warning" and "short" in i.message for i in issues)


def test_validate_v2_sequences_warns_on_extreme_gap():
    gt, det = _fixture_gt_and_detections(n=25, actor_id=100)
    for i in range(3, 15):
        det[i]["objects"] = []  # 12-frame gap
    seqs = build_v2_sequences(gt, det, eligible_actor_ids=[100], run_id="run1", role="test",
                               detector_source_version="v1", gt_source_version="v1")
    issues = validate_v2_sequences(seqs)
    assert any(i.severity == "warning" and "extreme gap" in i.message for i in issues)


def test_validate_v2_sequences_detects_non_monotonic_timestamp():
    gt, det = _fixture_gt_and_detections(n=5, actor_id=100)
    seqs = build_v2_sequences(gt, det, eligible_actor_ids=[100], run_id="run1", role="test",
                               detector_source_version="v1", gt_source_version="v1")
    seqs[0].timestamps_ns[3] = seqs[0].timestamps_ns[1]  # corrupt
    issues = validate_v2_sequences(seqs)
    assert any(i.severity == "error" and "non-monotonic" in i.message for i in issues)


def test_validate_v2_sequences_detects_non_finite_gt():
    gt, det = _fixture_gt_and_detections(n=5, actor_id=100)
    seqs = build_v2_sequences(gt, det, eligible_actor_ids=[100], run_id="run1", role="test",
                               detector_source_version="v1", gt_source_version="v1")
    seqs[0].x_true[2][0] = float("nan")
    issues = validate_v2_sequences(seqs)
    assert any(i.severity == "error" and "non-finite" in i.message for i in issues)


# --------------------------------------------------------------------
# freeze manifest
# --------------------------------------------------------------------
def test_build_and_save_v2_freeze_manifest_round_trips(tmp_path):
    manifest = build_v2_freeze_manifest(
        source_run_ids=["run1"], actor_ids=[100], sequence_boundaries={"run1__actor_100": {"start": 0, "end": 4}},
        inclusion_rules=["finite GT"], exclusion_rules=["train/val leakage"],
        sequence_file_hashes={"run1__actor_100": content_hash({"x": 1})},
        detector_source_version="v1", gt_source_version="v1",
        alignment_policy="positional frame-index join", frame_convention="lidar_link",
        git_sha="deadbeef",
    )
    path = tmp_path / "manifest.json"
    save_v2_freeze_manifest(manifest, path)
    import json
    loaded = json.loads(path.read_text("utf-8"))
    assert loaded["schema_version"] == "morai_estimator_eval_v2"
    assert loaded["actor_ids"] == [100]


def test_content_hash_is_deterministic_and_order_independent():
    assert content_hash({"a": 1, "b": 2}) == content_hash({"b": 2, "a": 1})
    assert content_hash({"a": 1}) != content_hash({"a": 2})


# --------------------------------------------------------------------
# paired metric aggregation / bootstrap
# --------------------------------------------------------------------
def test_paired_bootstrap_deterministic_given_seed():
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    b = [1.1, 1.9, 3.2, 3.8, 5.1]
    r1 = paired_bootstrap(a, b, seed=0)
    r2 = paired_bootstrap(a, b, seed=0)
    assert r1.observed_diff == r2.observed_diff
    assert r1.ci_low == r2.ci_low and r1.ci_high == r2.ci_high


def test_paired_bootstrap_requires_equal_length():
    with pytest.raises(ValueError):
        paired_bootstrap([1.0, 2.0], [1.0])


def test_paired_bootstrap_stable_when_clearly_separated():
    a = [10.0, 11.0, 10.5, 10.2, 10.8]
    b = [1.0, 1.1, 0.9, 1.05, 0.95]
    result = paired_bootstrap(a, b, seed=0)
    assert result.stable is True
    assert result.ci_low > 0


def test_run_level_macro_average_groups_correctly():
    run_ids = ["r1", "r1", "r2"]
    metrics = [1.0, 3.0, 5.0]
    avg = run_level_macro_average(run_ids, metrics)
    assert avg == {"r1": 2.0, "r2": 5.0}


def test_paired_bootstrap_over_run_level_averages():
    run_ids = ["r1", "r1", "r2", "r2"]
    seq_a = [1.0, 1.2, 5.0, 5.2]
    seq_b = [0.5, 0.6, 4.0, 4.1]
    run_avg_a = run_level_macro_average(run_ids, seq_a)
    run_avg_b = run_level_macro_average(run_ids, seq_b)
    result = paired_bootstrap(list(run_avg_a.values()), list(run_avg_b.values()), unit="run", seed=0)
    assert result.n_units == 2
    assert result.unit == "run"
