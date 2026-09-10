"""Tests for dt_curriculum.py (KalmanNet Mixed Fixed/Variable-dt Curriculum
v1, docs/perception/kalmannet_av2_dt_curriculum_v1.md, section 18)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dt_curriculum import (  # noqa: E402
    MIXED_50_MILD_PROBABILITY,
    segment_mix_uniform,
    segment_is_thinned_in_mix,
)
from variable_dt import MILD_VARIABLE_DT  # noqa: E402


# --------------------------------------------------------------------------
# Deterministic fixed-vs-thinned selection
# --------------------------------------------------------------------------

def test_segment_assignment_is_deterministic_for_same_seed_and_id():
    a = [segment_is_thinned_in_mix(7, f"scn__actor_{i}__seg_0", 0.5) for i in range(200)]
    b = [segment_is_thinned_in_mix(7, f"scn__actor_{i}__seg_0", 0.5) for i in range(200)]
    assert a == b


def test_segment_assignment_independent_of_call_context():
    # inserting unrelated draws in between must not shift a segment's own decision
    ref = segment_is_thinned_in_mix(3, "seg-x", 0.5)
    for _ in range(50):
        segment_is_thinned_in_mix(3, "unrelated", 0.5)
    assert segment_is_thinned_in_mix(3, "seg-x", 0.5) == ref


def test_mix_uniform_is_in_unit_interval():
    for i in range(500):
        u = segment_mix_uniform(1, f"seg-{i}")
        assert 0.0 <= u < 1.0


def test_mild_probability_zero_is_all_fixed():
    assert all(not segment_is_thinned_in_mix(1, f"seg-{i}", 0.0) for i in range(300))


def test_mild_probability_one_is_all_thinned():
    assert all(segment_is_thinned_in_mix(1, f"seg-{i}", 1.0) for i in range(300))


def test_mixed_50_realises_roughly_half_thinned_at_scale():
    n = 20000
    thinned = sum(segment_is_thinned_in_mix(42, f"scn_{i}__actor_{i % 97}__seg_{i % 3}", 0.5) for i in range(n))
    frac = thinned / n
    assert abs(frac - 0.5) < 0.02, frac


def test_different_policy_seed_gives_a_different_assignment_generally():
    ids = [f"seg-{i}" for i in range(500)]
    a = [segment_is_thinned_in_mix(1, s, 0.5) for s in ids]
    b = [segment_is_thinned_in_mix(2, s, 0.5) for s in ids]
    disagree = sum(x != y for x, y in zip(a, b))
    # two independent Bernoulli(0.5) assignments disagree ~50% of the time
    assert 150 < disagree < 350, disagree


def test_assignment_seed_never_uses_python_hash():
    import hashlib

    seed, seg_id = 11, "scn__actor_5__seg_2"
    expected = int.from_bytes(
        hashlib.sha256(f"{seed}:mix:{seg_id}".encode("utf-8")).digest()[:8], "big"
    ) / 2**64
    assert segment_mix_uniform(seed, seg_id) == expected


# --------------------------------------------------------------------------
# The mix-assignment RNG is independent of the thinning-internal RNG
# --------------------------------------------------------------------------

def test_mix_infix_makes_assignment_independent_of_thinning_seed():
    from variable_dt import thinning_seed

    # same (seed, segment_id) -> the two SHA-256 derivations must not collide
    seed, seg = 5, "seg-abc"
    assert float(thinning_seed(seed, seg)) != segment_mix_uniform(seed, seg) * (2**31 - 1)


# --------------------------------------------------------------------------
# Resume determinism: the assignment is a pure function of frozen inputs,
# so a resumed run reconstructs the identical fixed/thinned train set.
# --------------------------------------------------------------------------

def test_resume_reconstructs_identical_assignment():
    seg_ids = [f"run__actor_{i}__seg_{i % 4}" for i in range(1000)]
    first = {s: segment_is_thinned_in_mix(20260911, s, MIXED_50_MILD_PROBABILITY) for s in seg_ids}
    # simulate a fresh process / resume: recompute from scratch, shuffled order
    import random

    shuffled = seg_ids[:]
    random.Random(0).shuffle(shuffled)
    second = {s: segment_is_thinned_in_mix(20260911, s, MIXED_50_MILD_PROBABILITY) for s in shuffled}
    assert first == second


# --------------------------------------------------------------------------
# Temporal thinning semantics are unchanged (reused from variable_dt)
# --------------------------------------------------------------------------

def test_thinned_form_uses_the_frozen_mild_policy_unchanged():
    # dt_curriculum must not redefine the MILD distribution
    assert MILD_VARIABLE_DT.skip_probs == (0.80, 0.15, 0.05)
    assert MIXED_50_MILD_PROBABILITY == 0.5


def test_load_split_sequences_mixed_only_touches_the_given_shard_root(monkeypatch, tmp_path):
    import dt_curriculum

    calls = {}

    def fake_load_shard_index(root):
        calls["root"] = root
        return []

    monkeypatch.setattr(
        "ad_morai_bridge_dev.dataset.av2_kalmannet_shard_loader.load_shard_index",
        fake_load_shard_index,
    )
    monkeypatch.setattr(
        "ad_morai_bridge_dev.dataset.av2_kalmannet_shard_loader.iter_segments",
        lambda *a, **k: iter(()),
    )
    manifest = {"splits": {"train": [], "val": [], "test": []}}
    out, counts = dt_curriculum.load_split_sequences_mixed(
        tmp_path, manifest, 0.5, 1, 1, corruption_config=None
    )
    assert calls["root"] == tmp_path
    assert out == {"train": [], "val": [], "test": []}
    assert counts == {"train_fixed": 0, "train_thinned": 0, "train_dropped": 0}


# --------------------------------------------------------------------------
# No official-VAL leakage: the mix policy only consumes a shard root + a
# split manifest handed in by the caller; it never reads AV2's own
# withheld test split and never touches an official-VAL path implicitly.
# --------------------------------------------------------------------------

def test_mix_policy_has_no_hardcoded_official_val_path():
    src = Path(__file__).with_name("dt_curriculum.py").read_text()
    for needle in ("official_val", "av2/val", "scaleup_v2_official", "answer", "withheld"):
        assert needle not in src, needle


# --------------------------------------------------------------------------
# Mixed-policy manifest serialization
# --------------------------------------------------------------------------

def test_mixed_policy_manifest_fields_serialize_to_json():
    import json

    tc = {
        "mix_policy": "MIXED-50",
        "mild_probability": 0.5,
        "policy_seed": 1,
        "augmentation_seed": 1,
        "thinned_policy_skip_probs": list(MILD_VARIABLE_DT.skip_probs),
        "train_fixed_count": 44000,
        "train_thinned_count": 44000,
        "train_dropped_count": 20,
        "realised_mild_fraction": 0.4999,
    }
    round_tripped = json.loads(json.dumps(tc))
    assert round_tripped == tc
    assert round_tripped["thinned_policy_skip_probs"] == [0.80, 0.15, 0.05]


@pytest.mark.parametrize("p", [-0.01, 1.01, 2.0])
def test_train_script_rejects_out_of_range_mild_probability(p):
    import train_kalmannet_dt_curriculum as m

    argv = [
        "--shard-root", "/x", "--split-manifest", "/y", "--seed", "1",
        "--output-checkpoint", "/c.pt", "--output-history", "/h.csv",
        "--mild-probability", str(p),
    ]
    assert m.main(argv) == 2
