"""Integration tests: motion-composition weighted sampling wired into
``batched_trainer.train_one_run_batched`` (section 21: sampler state
resume, no train/val leakage, byte-identical-when-unused)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from batched_trainer import train_one_run_batched  # noqa: E402
from motion_composition import (  # noqa: E402
    BALANCED_MOTION,
    MOVING,
    NEAR_STATIONARY,
    classify_motion,
    compute_segment_motion_stats,
)


def _seq(n, vx, vy, x0, seed, dt=0.1):
    rng = np.random.RandomState(seed)
    frames = list(range(n))
    dts = [None] + [dt] * (n - 1)
    x_true, z_meas = [], []
    x, y = x0, 0.0
    for i in range(n):
        x_true.append([x, y, vx, vy])
        z_meas.append([x + rng.normal(0, 0.01), y + rng.normal(0, 0.01)])
        x += vx * dt
        y += vy * dt
    return {"actor_id": f"a{x0}", "frames": frames, "dt": dts, "x_true": x_true, "z_meas": z_meas,
            "scenario_id": f"scn{x0}", "track_id": f"trk{x0}"}


def _mixed_motion_train_seqs():
    moving = [_seq(n=10, vx=5.0, vy=0.0, x0=float(i), seed=i) for i in range(3)]
    stationary = [_seq(n=10, vx=0.0, vy=0.0, x0=float(i) + 50.0, seed=100 + i) for i in range(9)]
    return moving + stationary  # natural: 3/12 = 25% moving


def _categories_for(seqs):
    return [classify_motion(compute_segment_motion_stats(s)) for s in seqs]


def _common(tmp_path, train_seqs, motion_sampler=None, motion_categories=None):
    val_seqs = [_seq(n=8, vx=2.0, vy=0.0, x0=float(i) + 0.5, seed=200 + i) for i in range(2)]
    return dict(
        train_seqs=train_seqs, val_seqs=val_seqs, seed=1, device="cpu", batch_size=2,
        use_length_bucketing=False, lr=0.01, max_epochs=4, patience=4, hidden_size=4,
        grad_clip=10.0, loss_on_predict_only=True, motion_sampler=motion_sampler,
        motion_categories=motion_categories,
    )


def test_motion_categories_partition_the_mixed_fixture_as_expected():
    seqs = _mixed_motion_train_seqs()
    categories = _categories_for(seqs)
    assert categories.count(MOVING) == 3
    assert categories.count(NEAR_STATIONARY) == 9


def test_motion_sampler_none_matches_omitting_the_kwarg_entirely():
    """motion_sampler defaults to None -- byte-identical to every call
    site that predates this feature (the frozen 10k baseline)."""
    seqs = _mixed_motion_train_seqs()
    common = _common(None, seqs)
    a = train_one_run_batched(**{k: v for k, v in common.items() if k not in ("motion_sampler", "motion_categories")})
    b = train_one_run_batched(**common)
    assert a.best_val == pytest.approx(b.best_val, abs=1e-12)
    assert a.n_epochs_run == b.n_epochs_run
    for ra, rb in zip(a.history, b.history):
        assert ra.train_loss == pytest.approx(rb.train_loss, abs=1e-12)


def test_motion_sampler_requires_categories_when_reweighting_requested():
    seqs = _mixed_motion_train_seqs()
    common = _common(None, seqs, motion_sampler=BALANCED_MOTION, motion_categories=None)
    with pytest.raises(ValueError):
        train_one_run_batched(**common)


def test_motion_sampler_balanced_runs_to_completion_without_crash():
    seqs = _mixed_motion_train_seqs()
    categories = _categories_for(seqs)
    common = _common(None, seqs, motion_sampler=BALANCED_MOTION, motion_categories=categories)
    result = train_one_run_batched(**common)
    assert result.n_epochs_run == 4
    assert not result.catastrophic


def test_motion_sampler_resume_reaches_same_result_as_uninterrupted_run(tmp_path):
    """Section 21 'sampler state resume': the per-epoch draw seed is
    derived purely from (order_seed, epoch), never a persisted RNG
    object -- resuming mid-run must reproduce the exact same draws
    (and therefore the exact same training trajectory) as an
    uninterrupted run, with no new resume-state field needed."""
    seqs = _mixed_motion_train_seqs()
    categories = _categories_for(seqs)
    common = _common(tmp_path, seqs, motion_sampler=BALANCED_MOTION, motion_categories=categories)
    validation_key = {"seed": 1, "lr": 0.01, "batch_size": 2, "hidden_size": 4, "max_epochs": 4}

    uninterrupted = train_one_run_batched(**common)

    resume_path = tmp_path / "resume.pt"
    part1 = train_one_run_batched(
        **{**common, "max_epochs": 2}, resume_checkpoint_path=resume_path, resume_validation_key=validation_key,
    )
    assert part1.n_epochs_run == 2

    resumed = train_one_run_batched(
        **common, resume_checkpoint_path=resume_path, resume_validation_key=validation_key,
    )

    assert resumed.n_epochs_run == uninterrupted.n_epochs_run
    assert resumed.best_epoch == uninterrupted.best_epoch
    assert resumed.best_val == pytest.approx(uninterrupted.best_val, abs=1e-6)
    for a, b in zip(resumed.history, uninterrupted.history):
        assert a.train_loss == pytest.approx(b.train_loss, abs=1e-5)
        assert a.val_loss == pytest.approx(b.val_loss, abs=1e-6)


def test_motion_sampler_never_draws_from_val_seqs():
    """No train/val leakage: the sampler only ever indexes into
    ``train_seqs`` (weights/categories are sized to len(train_seqs),
    val_seqs is a structurally separate argument the sampler code never
    touches)."""
    seqs = _mixed_motion_train_seqs()
    categories = _categories_for(seqs)
    from motion_composition import compute_sample_weights, deterministic_weighted_draw_order

    weights = compute_sample_weights(categories, BALANCED_MOTION)
    draws = deterministic_weighted_draw_order(weights, n_draws=len(seqs), seed=1 * 1_000_003 + 0)
    assert draws.max() < len(seqs)
    assert draws.min() >= 0


# --------------------------------------------------------------------------
# compute_internal_val_motion_report (section 8: INTERNAL VAL ALL/
# NEAR_STATIONARY/MOVING) -- regression test for a real bug this task hit:
# an earlier version built the ALL bucket from raw numpy float32 arithmetic,
# which crashed json.dumps with "Object of type float32 is not JSON
# serializable" only at the very final write step of an ~hours-long
# screening run (the checkpoint itself was already safely saved by then).
# --------------------------------------------------------------------------


def test_internal_val_motion_report_is_json_serializable():
    import json
    from train_kalmannet_motion_composition import compute_internal_val_motion_report

    stationary_seq = _seq(n=4, vx=0.0, vy=0.0, x0=0.0, seed=0)
    moving_seq = _seq(n=4, vx=5.0, vy=0.0, x0=10.0, seed=1)

    def perfect_estimate(seq):
        import numpy as np
        return [np.array(x, dtype=np.float32) for x in seq["x_true"]]  # float32, like real run_knet output

    report = compute_internal_val_motion_report([stationary_seq, moving_seq], perfect_estimate)
    json.dumps(report)  # must not raise
    assert set(report) == {"all", "near_stationary", "moving"}
    assert report["all"]["n"] == 8
    assert report["near_stationary"]["n"] == 4
    assert report["moving"]["n"] == 4


def test_internal_val_motion_report_calls_estimate_fn_exactly_once_per_sequence():
    from train_kalmannet_motion_composition import compute_internal_val_motion_report

    call_count = {"n": 0}
    seqs = [_seq(n=4, vx=5.0, vy=0.0, x0=float(i), seed=i) for i in range(3)]

    def counting_estimate(seq):
        call_count["n"] += 1
        return [list(x) for x in seq["x_true"]]

    compute_internal_val_motion_report(seqs, counting_estimate)
    assert call_count["n"] == 3  # not 6 -- no redundant second pass


def test_internal_val_motion_report_all_bucket_matches_pooled_near_stationary_and_moving():
    import math
    from train_kalmannet_motion_composition import compute_internal_val_motion_report

    stationary_seq = _seq(n=4, vx=0.0, vy=0.0, x0=0.0, seed=0)
    moving_seq = _seq(n=4, vx=5.0, vy=0.0, x0=10.0, seed=1)

    def perfect_estimate(seq):
        return [list(x) for x in seq["x_true"]]

    report = compute_internal_val_motion_report([stationary_seq, moving_seq], perfect_estimate)
    # ALL == exactly the pooled near_stationary + moving samples (n and,
    # for a perfect estimator, RMSE 0 in every bucket).
    assert report["all"]["n"] == report["near_stationary"]["n"] + report["moving"]["n"]
    for bucket in ("all", "near_stationary", "moving"):
        assert math.isclose(report[bucket]["position_rmse_m"], 0.0, abs_tol=1e-9)
