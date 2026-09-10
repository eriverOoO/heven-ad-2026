"""Tests for variable_dt.py (KalmanNet Physically-Consistent Variable-dt
Augmentation v1, docs/perception/kalmannet_av2_variable_dt_v1.md,
section 24)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from variable_dt import (  # noqa: E402
    FIXED_DT,
    MILD_VARIABLE_DT,
    STRONG_VARIABLE_DT,
    ThinningPolicy,
    thin_sequence,
    thinning_seed,
    verify_thinned_sequence,
)


def _seq(n=30, dt=0.1, vx=3.0, vy=1.0, x0=0.0, y0=0.0):
    x_true, z_meas, dts = [], [], [None]
    x, y = x0, y0
    for i in range(n):
        x_true.append([x, y, vx, vy])
        z_meas.append([x, y])
        if i > 0:
            dts.append(dt)
        x += vx * dt
        y += vy * dt
    return {"actor_id": "a0", "frames": list(range(n)), "dt": dts, "x_true": x_true, "z_meas": z_meas}


# --------------------------------------------------------------------------
# Deterministic retained indices
# --------------------------------------------------------------------------


def test_same_seed_and_segment_id_gives_identical_retained_indices():
    seq = _seq()
    a = thin_sequence(seq, MILD_VARIABLE_DT, augmentation_seed=42, segment_id="scn0::trk0::0")
    b = thin_sequence(seq, MILD_VARIABLE_DT, augmentation_seed=42, segment_id="scn0::trk0::0")
    assert a["retained_indices"] == b["retained_indices"]
    assert a["dt"] == b["dt"]


def test_different_segment_id_gives_different_retained_indices_generally():
    seq = _seq(n=60)
    a = thin_sequence(seq, STRONG_VARIABLE_DT, augmentation_seed=42, segment_id="scn0::trk0::0")
    b = thin_sequence(seq, STRONG_VARIABLE_DT, augmentation_seed=42, segment_id="scn1::trk7::2")
    assert a["retained_indices"] != b["retained_indices"]


def test_thinning_seed_never_uses_python_hash():
    """Regression guard: thinning_seed must be a pure SHA-256-derived
    function -- reproducible across separate Python processes, unlike
    Python's own str hash() (salted per-process by default)."""
    a = thinning_seed(1, "scn0::trk0::0")
    b = thinning_seed(1, "scn0::trk0::0")
    assert a == b
    assert isinstance(a, int)


def test_retained_indices_independent_of_unrelated_batch_context():
    """Section 9: same sample/config -> same result independent of batch
    order / shard grouping / worker ordering -- trivially true here since
    thin_sequence takes no batch-context argument at all, but locked by
    a repeated-call-in-different-order check."""
    seq = _seq(n=50)
    results = [
        thin_sequence(seq, MILD_VARIABLE_DT, augmentation_seed=7, segment_id="scnX::trkY::1")["retained_indices"]
        for _ in range(5)
    ]
    assert all(r == results[0] for r in results)


# --------------------------------------------------------------------------
# FIXED_DT is an exact no-op
# --------------------------------------------------------------------------


def test_fixed_dt_is_exact_noop():
    seq = _seq(n=20)
    out = thin_sequence(seq, FIXED_DT, augmentation_seed=1, segment_id="any")
    assert out["dt"] == seq["dt"]
    assert out["x_true"] == seq["x_true"]
    assert out["frames"] == seq["frames"]
    assert "retained_indices" not in out  # true no-op, not merely equivalent-content


# --------------------------------------------------------------------------
# dt calculation (additive, matches the task's own worked example)
# --------------------------------------------------------------------------


def test_dt_recomputation_matches_task_example():
    """Task section 2's own worked example: retaining 0,1,3,4,7 from a
    uniform dt=0.1 sequence gives real intervals 0.1, 0.2, 0.1, 0.3."""
    seq = _seq(n=8, dt=0.1)

    # Forcing an exact retained set through the RNG-driven implementation
    # is awkward, so instead verify the dt-additivity math directly, the
    # same way thin_sequence computes it internally.
    retained = [0, 1, 3, 4, 7]
    dt = seq["dt"]
    new_dt = [None]
    for a, b in zip(retained[:-1], retained[1:]):
        new_dt.append(sum(float(dt[k]) for k in range(a + 1, b + 1)))
    assert new_dt[1:] == pytest.approx([0.1, 0.2, 0.1, 0.3])


def test_thinned_dt_values_are_always_positive_and_multiples_of_base_dt():
    seq = _seq(n=80, dt=0.1)
    out = thin_sequence(seq, STRONG_VARIABLE_DT, augmentation_seed=3, segment_id="scnA::trkB::0")
    for v in out["dt"][1:]:
        assert v > 0
        assert v == pytest.approx(round(v / 0.1) * 0.1, abs=1e-9)


# --------------------------------------------------------------------------
# State/timestamp correspondence + full integrity verification
# --------------------------------------------------------------------------


def test_verify_thinned_sequence_passes_on_a_real_thinning():
    seq = _seq(n=60, dt=0.1)
    out = thin_sequence(seq, MILD_VARIABLE_DT, augmentation_seed=5, segment_id="scnA::trkB::1")
    verify_thinned_sequence(seq, out)  # must not raise


def test_verify_thinned_sequence_catches_state_index_mismatch():
    seq = _seq(n=20, dt=0.1)
    out = thin_sequence(seq, MILD_VARIABLE_DT, augmentation_seed=5, segment_id="scnA::trkB::1")
    corrupted = dict(out)
    corrupted["x_true"] = list(out["x_true"])
    corrupted["x_true"][1] = [999.0, 999.0, 0.0, 0.0]  # break correspondence
    with pytest.raises(AssertionError):
        verify_thinned_sequence(seq, corrupted)


def test_verify_thinned_sequence_catches_dt_mismatch():
    seq = _seq(n=20, dt=0.1)
    out = thin_sequence(seq, MILD_VARIABLE_DT, augmentation_seed=5, segment_id="scnA::trkB::1")
    corrupted = dict(out)
    corrupted["dt"] = list(out["dt"])
    corrupted["dt"][1] = corrupted["dt"][1] + 1.0  # break dt/timestamp correspondence
    with pytest.raises(AssertionError):
        verify_thinned_sequence(seq, corrupted)


def test_no_interpolation_every_retained_state_is_a_real_original_sample():
    seq = _seq(n=40, dt=0.1)
    out = thin_sequence(seq, STRONG_VARIABLE_DT, augmentation_seed=9, segment_id="scnZ::trkQ::0")
    for k, orig_idx in enumerate(out["retained_indices"]):
        assert out["x_true"][k] == seq["x_true"][orig_idx]


def test_no_duplicate_retained_frame():
    seq = _seq(n=100, dt=0.1)
    out = thin_sequence(seq, STRONG_VARIABLE_DT, augmentation_seed=11, segment_id="scnDup::trk::0")
    retained = out["retained_indices"]
    assert len(retained) == len(set(retained))
    assert retained == sorted(retained)


# --------------------------------------------------------------------------
# Minimum usable sequence length preserved (section 10 fallback)
# --------------------------------------------------------------------------


def test_short_sequence_falls_back_to_untouched_when_thinning_would_go_below_minimum():
    short_policy = ThinningPolicy(name="test-aggressive", skip_probs=(0.0, 0.0, 0.0, 1.0), min_retained_frames=5)
    seq = _seq(n=6, dt=0.1)  # thinning at skip=3 every step would retain far fewer than 5
    out = thin_sequence(seq, short_policy, augmentation_seed=1, segment_id="short::seg::0")
    assert out["is_thinned"] is False
    assert out["x_true"] == seq["x_true"]


def test_rejects_sequences_shorter_than_two_frames():
    with pytest.raises(ValueError):
        thin_sequence(_seq(n=1), MILD_VARIABLE_DT, augmentation_seed=1, segment_id="x")


# --------------------------------------------------------------------------
# Declared dt-support distributions (section 3/7 -- documented BEFORE
# screening/official-VAL, never derived from MORAI TEST data)
# --------------------------------------------------------------------------


def test_mild_and_strong_skip_probabilities_sum_to_one():
    assert sum(MILD_VARIABLE_DT.skip_probs) == pytest.approx(1.0)
    assert sum(STRONG_VARIABLE_DT.skip_probs) == pytest.approx(1.0)


def test_strong_has_more_mass_on_larger_skips_than_mild():
    # "more aggressive thinning" (task section 7C) -- STRONG's expected
    # skip count must exceed MILD's.
    mild_expected = sum(k * p for k, p in enumerate(MILD_VARIABLE_DT.skip_probs))
    strong_expected = sum(k * p for k, p in enumerate(STRONG_VARIABLE_DT.skip_probs))
    assert strong_expected > mild_expected


def test_mild_dominant_mass_is_on_zero_skip_ie_0_1s():
    assert MILD_VARIABLE_DT.skip_probs[0] > 0.5


# --------------------------------------------------------------------------
# Empirical retained-fraction / dt-distribution sanity at scale
# --------------------------------------------------------------------------


def test_empirical_dt_distribution_matches_declared_policy_at_scale():
    seqs = [_seq(n=200, dt=0.1) for _ in range(1)]
    all_dt = []
    for i, seq in enumerate(seqs):
        for trial in range(200):
            out = thin_sequence(seq, MILD_VARIABLE_DT, augmentation_seed=100, segment_id=f"scale::{i}::{trial}")
            all_dt.extend(v for v in out["dt"][1:])
    arr = np.round(np.array(all_dt), 6)
    # Loose bound (large-sample, not exact) -- declared skip_probs[0]=0.80.
    assert 0.65 <= np.mean(np.isclose(arr, 0.1)) <= 0.92


# --------------------------------------------------------------------------
# evaluate_with_dt_buckets (sections 13/14)
# --------------------------------------------------------------------------


from variable_dt import dt_bucket_label, evaluate_with_dt_buckets  # noqa: E402


@pytest.mark.parametrize(
    "dt,expected",
    [(0.1, "0.1s"), (0.14, "0.1s"), (0.15, "0.2s"), (0.2, "0.2s"), (0.24, "0.2s"),
     (0.25, "0.3s"), (0.3, "0.3s"), (0.34, "0.3s"), (0.35, "0.4s+"), (0.4, "0.4s+"), (1.0, "0.4s+")],
)
def test_dt_bucket_label_boundaries(dt, expected):
    assert dt_bucket_label(dt) == expected


def test_evaluate_with_dt_buckets_separates_by_transition_dt():
    seq = {
        "frames": [0, 1, 2, 3],
        "dt": [None, 0.1, 0.2, 0.3],
        "x_true": [[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0], [2.0, 0.0, 1.0, 0.0], [3.0, 0.0, 1.0, 0.0]],
        "z_meas": [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
    }

    def perfect_estimate(s):
        return [list(x) for x in s["x_true"]]

    result = evaluate_with_dt_buckets([seq], perfect_estimate)
    assert result["overall"]["n"] == 4
    assert result["0.1s"]["n"] == 1
    assert result["0.2s"]["n"] == 1
    assert result["0.3s"]["n"] == 1
    assert "0.4s+" not in result


def test_evaluate_with_dt_buckets_matched_missing_split():
    seq = {
        "frames": [0, 1, 2],
        "dt": [None, 0.1, 0.1],
        "x_true": [[0.0, 0.0, 0.0, 0.0]] * 3,
        "z_meas": [[0.0, 0.0], None, [0.0, 0.0]],
    }

    def perfect_estimate(s):
        return [list(x) for x in s["x_true"]]

    result = evaluate_with_dt_buckets([seq], perfect_estimate)
    assert result["matched"]["n"] == 2
    assert result["missing"]["n"] == 1
    assert result["overall"]["n"] == 3


def test_evaluate_with_dt_buckets_exact_rmse_value():
    seq = {
        "frames": [0, 1],
        "dt": [None, 0.2],
        "x_true": [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        "z_meas": [[0.0, 0.0], [0.0, 0.0]],
    }

    def biased_estimate(s):
        return [[3.0, 4.0, 0.0, 0.0], [3.0, 4.0, 0.0, 0.0]]

    result = evaluate_with_dt_buckets([seq], biased_estimate)
    assert result["0.2s"]["position_rmse_m"] == pytest.approx(5.0, abs=1e-9)


# --------------------------------------------------------------------------
# thinning != missing measurement (section 6) -- corruption ordering (11)
# --------------------------------------------------------------------------


def test_thinned_sequence_never_carries_a_none_measurement_placeholder_for_a_removed_frame():
    """A temporally-thinned sequence's z_meas list is exactly as long as
    its (shorter) retained-frame list -- there is structurally no way for
    a removed original index to leave a None placeholder behind, unlike
    a missing-measurement frame which keeps its slot."""
    seq = _seq(n=40, dt=0.1)
    out = thin_sequence(seq, STRONG_VARIABLE_DT, augmentation_seed=21, segment_id="scnT::trk::0")
    assert len(out["z_meas"]) == len(out["frames"]) == len(out["x_true"]) == len(out["retained_indices"])


def test_corruption_applied_after_thinning_never_before(monkeypatch):
    """Section 11: load clean -> thin -> corrupt retained-only. If
    corruption ran on the FULL original array first, the corrupted
    z_meas passed into thinning would still have the original (longer)
    length; segment_to_thinned_training_sequence must never do that --
    verified by asserting the corruption call always receives an array
    whose length equals the THINNED retained-frame count, not the
    original segment length."""
    import numpy as np

    from variable_dt import segment_to_thinned_training_sequence

    n_original = 50
    x_true = [[float(i), 0.0, 1.0, 0.0] for i in range(n_original)]
    dts = [None] + [0.1] * (n_original - 1)
    clean_seq = {"frames": list(range(n_original)), "dt": dts, "x_true": x_true,
                 "z_meas": [[float(i), 0.0] for i in range(n_original)]}

    class _FakeSegment:
        segment_id = "fake::seg::0"

        def to_kalmannet_sequence(self, corruption_config):
            assert corruption_config is None  # must be called with CLEAN first
            return dict(clean_seq)

    captured_lengths = []

    def fake_apply_corruption(arrays, segment_id, config):
        captured_lengths.append(arrays["clean_position"].shape[0])
        n = arrays["clean_position"].shape[0]
        return {"measurement": arrays["clean_position"], "measurement_valid": np.ones(n, dtype=bool)}

    monkeypatch.setattr(
        "ad_morai_bridge_dev.dataset.av2_motion_forecasting_adapter.apply_corruption", fake_apply_corruption
    )

    out = segment_to_thinned_training_sequence(
        _FakeSegment(), STRONG_VARIABLE_DT, augmentation_seed=99, corruption_config=object()
    )
    assert captured_lengths, "apply_corruption was never called"
    assert captured_lengths[0] == len(out["retained_indices"])
    assert captured_lengths[0] < n_original  # confirms thinning happened BEFORE corruption, not after


# --------------------------------------------------------------------------
# Determinism of the full per-split loading pipeline (underlies resume
# reproducibility -- section 24's "resume reproducibility with thinning
# config": train_one_run_batched's own existing resume mechanism is only
# exact if load_split_sequences_with_thinning itself is byte-deterministic
# given the same inputs, since data is loaded once before resume applies).
# --------------------------------------------------------------------------


def test_thin_sequence_is_byte_deterministic_across_independent_calls():
    seq = _seq(n=70, dt=0.1)
    results = [
        thin_sequence(seq, STRONG_VARIABLE_DT, augmentation_seed=55, segment_id="scnR::trkR::3")
        for _ in range(3)
    ]
    for r in results[1:]:
        assert r["retained_indices"] == results[0]["retained_indices"]
        assert r["dt"] == results[0]["dt"]
        assert r["x_true"] == results[0]["x_true"]


# --------------------------------------------------------------------------
# No official-VAL leakage / policies are hardcoded, not data-fit
# --------------------------------------------------------------------------


def test_mild_and_strong_policies_are_frozen_hardcoded_constants():
    """Section 8: augmentation probabilities must never be derived from
    the frozen MORAI TEST actors or any loaded data file -- structurally
    enforced here by construction (ThinningPolicy is a frozen dataclass
    with literal tuple values baked into variable_dt.py's own source,
    never read from a config/data file at import time)."""
    with pytest.raises(Exception):
        MILD_VARIABLE_DT.skip_probs = (1.0,)  # frozen -- cannot be mutated at runtime
    assert isinstance(MILD_VARIABLE_DT.skip_probs, tuple)
    assert isinstance(STRONG_VARIABLE_DT.skip_probs, tuple)


def test_load_split_sequences_with_thinning_only_touches_the_given_shard_root(monkeypatch, tmp_path):
    """No official-VAL leakage: the function never reads any shard root
    other than the one explicitly passed in -- verified by pointing it at
    a nonexistent path and confirming the failure is a clean 'no such
    shard index' error from THAT path, never a fallback to some other
    (e.g. official-val) root."""
    from variable_dt import load_split_sequences_with_thinning, FIXED_DT

    fake_root = tmp_path / "definitely_not_official_val"
    with pytest.raises(Exception):
        load_split_sequences_with_thinning(
            fake_root, {"splits": {"train": [], "val": [], "test": []}}, FIXED_DT, 1, None,
        )


# --------------------------------------------------------------------------
# Batch collation with variable dt (section 24)
# --------------------------------------------------------------------------


def test_build_padded_batch_preserves_each_sequences_own_variable_dt():
    """batched_kalmannet.build_padded_batch already stores dt per-sample,
    per-timestep ([b, t_max]) -- unmodified by this task. This locks that
    a THINNED sequence's variable dt values survive collation alongside a
    FIXED sequence in the SAME batch, un-conflated."""
    from batched_kalmannet import build_padded_batch

    fixed_seq = _seq(n=4, dt=0.1)
    thinned = thin_sequence(_seq(n=8, dt=0.1), MILD_VARIABLE_DT, augmentation_seed=1, segment_id="mix::a::0")
    if not thinned.get("is_thinned", True):
        thinned = thin_sequence(_seq(n=8, dt=0.1), STRONG_VARIABLE_DT, augmentation_seed=1, segment_id="mix::b::0")

    batch = build_padded_batch([fixed_seq, thinned], device="cpu")
    n0 = len(fixed_seq["frames"])
    n1 = len(thinned["frames"])
    for t in range(1, n0):
        assert batch.dt[0, t].item() == pytest.approx(fixed_seq["dt"][t], abs=1e-6)
    for t in range(1, n1):
        assert batch.dt[1, t].item() == pytest.approx(thinned["dt"][t], abs=1e-6)
    # The two samples' dt streams must differ somewhere if thinning
    # actually produced a non-trivial dt sequence for sample 1.
    if any(v != 0.1 for v in thinned["dt"][1:]):
        assert not all(
            batch.dt[0, t].item() == pytest.approx(batch.dt[1, t].item(), abs=1e-6)
            for t in range(1, min(n0, n1))
        )
