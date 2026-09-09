"""Physically-consistent variable-dt augmentation via TEMPORAL THINNING
(docs/perception/kalmannet_av2_variable_dt_v1.md).

**Thinning, never interpolation.** A retained timestep's state/measurement
is the REAL AV2 sample at that original index -- no synthetic state is
ever fabricated. Skipped timesteps are removed from the sequence entirely
(the network never sees them); this is a DIFFERENT mechanism from a
missing-measurement frame (which keeps the timestep, with
``z_meas[t] is None`` and a predict-only transition -- see
``kalmannet_sequences.py`` / ``apply_corruption``'s ``measurement_valid``).
Section 6 of the task spec: these two are never conflated here.

Pure functions only, no torch/ROS dependency. Operates on the plain
``{"x_true","dt","frames","z_meas",...}`` sequence dict shape every other
module in this package already uses -- reads only ``x_true``/``dt``
(dt-additivity is used to recompute post-thinning intervals without
needing raw nanosecond timestamps: dt is elapsed real time, so summing
consecutive original dt values across a thinned gap is exactly the real
elapsed time between the two retained samples).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Deterministic seeding -- same convention already established in this
# project (av2_motion_forecasting_adapter._segment_corruption_seed):
# SHA-256 over stable string identifiers, never Python's hash() (which is
# process-randomized for str by default and therefore NOT reproducible
# across processes/workers/runs).
# ---------------------------------------------------------------------------


def thinning_seed(augmentation_seed: int, segment_id: str) -> int:
    """Same truncation convention as the existing
    ``av2_motion_forecasting_adapter._segment_corruption_seed`` (first 4
    digest bytes, mod 2**31-1) -- ``numpy.random.RandomState`` requires a
    seed in ``[0, 2**32-1]``, and reusing the identical convention keeps
    this and the corruption seed independent-looking despite both being
    SHA-256-derived from a similar-shaped string."""
    digest = hashlib.sha256(f"{augmentation_seed}:thinning:{segment_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % (2**31 - 1)


@dataclass(frozen=True)
class ThinningPolicy:
    """``skip_probs[k]`` = probability of skipping ``k`` original frames
    before the next retained frame (k=0 -> dt stays at the original
    per-step value, e.g. 0.1 s; k=1 -> the next original frame is
    dropped, dt becomes the SUM of 2 original steps, e.g. 0.2 s; etc.).
    Must sum to 1.0. ``FIXED`` (all mass on k=0) is byte-identical to no
    thinning at all -- the existing frozen NATURAL/10k baseline path."""

    name: str
    skip_probs: tuple[float, ...]  # index k = probability of skipping k frames
    min_retained_frames: int = 5  # mirrors TrajectoryConfig.min_gt_samples convention elsewhere


FIXED_DT = ThinningPolicy(name="FIXED-DT", skip_probs=(1.0,))

# Declared BEFORE any screening/official-VAL run (task section 7/8): a
# generic, hand-declared bounded distribution -- NOT fit to the frozen
# MORAI TEST actors or any historical MORAI timing statistic.
MILD_VARIABLE_DT = ThinningPolicy(name="MILD-VARIABLE-DT", skip_probs=(0.80, 0.15, 0.05))
STRONG_VARIABLE_DT = ThinningPolicy(name="STRONG-VARIABLE-DT", skip_probs=(0.55, 0.25, 0.15, 0.05))

POLICY_CHOICES = {"fixed": FIXED_DT, "mild": MILD_VARIABLE_DT, "strong": STRONG_VARIABLE_DT}


def _expected_dt_support(policy: ThinningPolicy, base_dt_s: float = 0.1) -> dict[str, float]:
    """Documentation/reporting helper: the dt VALUE each skip-count
    implies, assuming (as is true for real AV2 data) a locally-uniform
    original step of ``base_dt_s`` -- section 3 of the task doc."""
    return {f"{base_dt_s * (k + 1):.1f}s": p for k, p in enumerate(policy.skip_probs)}


def thin_sequence(
    seq: dict[str, Any],
    policy: ThinningPolicy,
    augmentation_seed: int,
    segment_id: str,
) -> dict[str, Any]:
    """Returns a NEW sequence dict (never mutates ``seq``) whose
    ``frames``/``dt``/``x_true``/``z_meas`` are a thinned subsequence of
    the input's -- every retained state/measurement is the REAL original
    value at its original index (section 3: no interpolation).

    Deterministic given ``(policy, augmentation_seed, segment_id)`` --
    independent of batch order / shard grouping / worker ordering (the
    RNG is freshly seeded from a SHA-256 over stable identifiers,
    ``thinning_seed``, never Python's ``hash()``).

    FIXED_DT policy is a true no-op (returns an equivalent-content copy)
    -- byte-identical to never calling this function, so the existing
    frozen NATURAL/10k baseline path is completely unaffected when this
    policy is selected.
    """
    x_true = seq["x_true"]
    dt = seq["dt"]
    n = len(dt)
    if n < 2:
        raise ValueError(f"sequence too short to thin (n={n})")

    if policy.skip_probs == (1.0,):
        # FIXED-DT: exact no-op, not merely "skip_count always 0" run
        # through the RNG path -- avoids ever perturbing the frozen
        # baseline's own RNG-consumption sequence for any other field.
        return dict(seq)

    rng = np.random.RandomState(thinning_seed(augmentation_seed, segment_id))
    skip_values = np.arange(len(policy.skip_probs))
    probs = np.asarray(policy.skip_probs, dtype=float)
    probs = probs / probs.sum()

    retained = [0]
    i = 0
    while i < n - 1:
        skip = int(rng.choice(skip_values, p=probs))
        nxt = min(i + 1 + skip, n - 1)
        retained.append(nxt)
        i = nxt
        if nxt == n - 1:
            break
    # Deduplicate a possible final-index repeat from the min() clamp above.
    retained = sorted(set(retained))

    if len(retained) < policy.min_retained_frames:
        # Deterministic fallback (section 10: "minimum usable sequence
        # length preserved") -- never silently ship a too-short thinned
        # sequence; fall back to the untouched original for this segment
        # (still logged/countable by the caller via `is_thinned` below).
        out = dict(seq)
        out["is_thinned"] = False
        out["retained_indices"] = list(range(n))
        return out

    new_x_true = [x_true[i] for i in retained]
    new_z_meas_source = seq.get("z_meas")
    new_frames = list(range(len(retained)))

    new_dt: list[Any] = [None]
    for a, b in zip(retained[:-1], retained[1:]):
        # dt-additivity: elapsed real time between two retained samples
        # is the SUM of the original per-step dt values in between --
        # never a fabricated/arbitrary constant (task section 2).
        segment_dt = sum(float(dt[k]) for k in range(a + 1, b + 1))
        new_dt.append(segment_dt)

    out = dict(seq)
    out["frames"] = new_frames
    out["dt"] = new_dt
    out["x_true"] = new_x_true
    if new_z_meas_source is not None:
        out["z_meas"] = [new_z_meas_source[i] for i in retained]
    out["is_thinned"] = True
    out["retained_indices"] = retained
    out["thinning_policy"] = policy.name
    return out


def verify_thinned_sequence(original: dict[str, Any], thinned: dict[str, Any]) -> None:
    """Data-integrity checks (task section 10). Raises ``AssertionError``
    (not silently returns False) on any violation -- integrity bugs here
    would silently corrupt every downstream training/eval number."""
    retained = thinned["retained_indices"]
    dt = thinned["dt"]
    orig_dt = original["dt"]

    assert len(retained) == len(set(retained)), "duplicate retained index"
    assert list(retained) == sorted(retained), "retained indices not strictly increasing"
    assert retained[0] == 0, "thinning must always retain the first frame"

    assert len(dt) == len(retained)
    assert dt[0] is None
    for k in range(1, len(dt)):
        assert dt[k] > 0, f"non-positive dt at position {k}"
        a, b = retained[k - 1], retained[k]
        expected = sum(float(orig_dt[j]) for j in range(a + 1, b + 1))
        assert abs(dt[k] - expected) < 1e-9, f"dt mismatch at position {k}: {dt[k]} != {expected}"

    for k, orig_idx in enumerate(retained):
        assert np.allclose(thinned["x_true"][k], original["x_true"][orig_idx]), (
            f"state at retained position {k} does not match original index {orig_idx} -- "
            "possible interpolation/corruption ordering bug"
        )


def segment_to_thinned_training_sequence(
    segment,
    policy: ThinningPolicy,
    augmentation_seed: int,
    corruption_config,
) -> dict[str, Any] | None:
    """The full per-segment pipeline in the REQUIRED order (task section
    11): load CLEAN sequence -> temporal thinning -> measurement
    corruption applied ONLY to the retained timesteps. A temporally
    removed frame can never later become a "missing measurement" frame
    (measurement_valid/dropout is computed fresh, seeded by
    (corruption_config.seed, segment_id), over the THINNED array length
    only -- it has no way to reference an index that no longer exists).

    Mirrors ``av2_kalmannet_shard_loader.SegmentArrays.to_kalmannet_sequence``'s
    own corruption-application logic exactly (same ``apply_corruption``
    call, same z_meas/None construction) so this is provably the same
    corruption mechanism, just fed a thinned array instead of the full
    one -- no second corruption implementation.
    """
    from ad_morai_bridge_dev.dataset.av2_motion_forecasting_adapter import apply_corruption

    clean_seq = segment.to_kalmannet_sequence(None)
    thinned = thin_sequence(clean_seq, policy, augmentation_seed, segment.segment_id)
    if policy.skip_probs != (1.0,):
        verify_thinned_sequence(clean_seq, thinned)

    x_true = thinned["x_true"]
    length = len(x_true)
    clean_position = np.asarray([[row[0], row[1]] for row in x_true], dtype=float)
    gt_state = np.asarray(x_true, dtype=float)
    corrupted = apply_corruption(
        {"clean_position": clean_position, "gt_state": gt_state}, segment.segment_id, corruption_config
    )
    valid = corrupted["measurement_valid"]
    measurement = corrupted["measurement"]
    z_meas: list[Any] = []
    for k in range(length):
        z_meas.append([float(v) for v in measurement[k]] if bool(valid[k]) else None)

    out = dict(thinned)
    out["z_meas"] = z_meas
    return out


def load_split_sequences_with_thinning(
    shard_output_root,
    split_manifest: dict[str, Any],
    policy: ThinningPolicy,
    augmentation_seed: int,
    corruption_config,
    only_valid_for_kalmannet_gt: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    """Same shape/contract as ``kalmannet_sequences.load_split_sequences``
    (reused verbatim for split-membership resolution, shard iteration,
    and the post-corruption first-measurement truncation), the ONLY
    difference being ``segment_to_thinned_training_sequence`` in place of
    the historical ``_to_training_sequence`` -- so a caller passing
    ``policy=FIXED_DT`` gets byte-identical sequences to the untouched
    historical path (FIXED_DT's ``thin_sequence`` is an exact no-op)."""
    from av2_split import SPLIT_NAMES
    from ad_morai_bridge_dev.dataset.av2_kalmannet_shard_loader import ShardSegmentRef, iter_segments, load_shard_index
    from kalmannet_sequences import _truncate_to_first_measurement

    scenario_to_split: dict[str, str] = {}
    for split_name in SPLIT_NAMES:
        for sid in split_manifest["splits"][split_name]:
            scenario_to_split[sid] = split_name

    refs: list[ShardSegmentRef] = load_shard_index(shard_output_root)
    ref_scenarios = {r.scenario_id for r in refs}
    missing = ref_scenarios - set(scenario_to_split)
    if missing:
        raise ValueError(
            f"{len(missing)} scenario(s) present in the shard dataset are not covered by the "
            f"split manifest (e.g. {sorted(missing)[:3]}) -- split manifest and shard dataset disagree"
        )

    out: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLIT_NAMES}
    for segment in iter_segments(shard_output_root, shard_index=refs):
        if only_valid_for_kalmannet_gt and not segment.valid_for_kalmannet_gt:
            continue
        split_name = scenario_to_split.get(segment.scenario_id)
        if split_name is None:
            continue
        try:
            seq = segment_to_thinned_training_sequence(segment, policy, augmentation_seed, corruption_config)
        except ValueError:
            continue  # too-short segment for even FIXED_DT (n<2) -- pre-existing edge case, counted as dropped
        seq["scenario_id"] = segment.scenario_id
        seq["track_id"] = segment.track_id
        seq["av2_object_type"] = segment.av2_object_type
        seq["coarse_object_type"] = segment.coarse_object_type
        seq = _truncate_to_first_measurement(seq)
        if seq is None:
            continue
        out[split_name].append(seq)
    return out


DT_BUCKET_EDGES = (0.0, 0.15, 0.25, 0.35, float("inf"))
DT_BUCKET_LABELS = ("0.1s", "0.2s", "0.3s", "0.4s+")


def dt_bucket_label(dt_value: float) -> str:
    for i in range(len(DT_BUCKET_EDGES) - 1):
        if DT_BUCKET_EDGES[i] <= dt_value < DT_BUCKET_EDGES[i + 1]:
            return DT_BUCKET_LABELS[i]
    return DT_BUCKET_LABELS[-1]


def evaluate_with_dt_buckets(sequences: list[dict[str, Any]], estimate_fn) -> dict[str, Any]:
    """Task sections 13/14: position/velocity RMSE (+ matched/missing
    split, + N) reported both overall and per dt-bucket (which transition
    dt produced this frame's error -- 0.1/0.2/0.3/0.4+s). ``estimate_fn``
    called exactly once per sequence (same convention as
    ``evaluate_kalmannet_official_val.evaluate_motion_stratified``).
    Frame 0 (dt=None, no transition into it) is excluded from every
    dt-bucket but included in ``overall``/``matched``/``missing``."""
    buckets: dict[str, dict[str, list]] = {}

    def bucket(name):
        if name not in buckets:
            buckets[name] = {"pos_sq_err": [], "vel_sq_err": [], "n": 0}
        return buckets[name]

    for seq in sequences:
        est = estimate_fn(seq)
        dt = seq["dt"]
        z_meas = seq.get("z_meas")
        for i in range(len(seq["frames"])):
            x_est = np.asarray(est[i], dtype=np.float64)
            x_gt = np.asarray(seq["x_true"][i], dtype=np.float64)
            if not np.all(np.isfinite(x_est)) or np.any(np.abs(x_est) > 1e6):
                continue
            pos_err = float(np.sum((x_est[:2] - x_gt[:2]) ** 2))
            vel_err = float(np.sum((x_est[2:] - x_gt[2:]) ** 2))
            names = ["overall"]
            if z_meas is not None:
                names.append("matched" if z_meas[i] is not None else "missing")
            if i > 0 and dt[i] is not None:
                names.append(dt_bucket_label(float(dt[i])))
            for name in names:
                b = bucket(name)
                b["pos_sq_err"].append(pos_err)
                b["vel_sq_err"].append(vel_err)
                b["n"] += 1

    result = {}
    for name, b in buckets.items():
        if b["n"] == 0:
            continue
        result[name] = {
            "n": b["n"],
            "position_rmse_m": float(np.sqrt(np.mean(b["pos_sq_err"]))),
            "velocity_rmse_mps": float(np.sqrt(np.mean(b["vel_sq_err"]))),
        }
    return result
