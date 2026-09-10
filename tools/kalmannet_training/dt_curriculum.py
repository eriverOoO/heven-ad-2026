"""Mixed fixed / physically-consistent-thinned temporal training
(docs/perception/kalmannet_av2_dt_curriculum_v1.md).

ONE minimal mixed strategy: **MIXED-50**. For each training segment, a
deterministic per-segment Bernoulli draw decides whether that segment is
used as its original FIXED sequence or as its (already deterministic)
MILD temporal thinning. Half the segments are thinned, half are not --
one static assignment, decided once from stable identifiers, so it is
resume-safe by construction and needs no per-epoch RNG state.

**Reuses ``variable_dt.py`` entirely** -- ``segment_to_thinned_training_sequence``
is called with either ``FIXED_DT`` (exact no-op) or the frozen
``MILD_VARIABLE_DT`` policy, unchanged. No second thinning implementation,
no change to thinning probabilities / dt calculation / retained-index
semantics.

**No physical duplicate dataset.** A given segment appears exactly once
in the training list, in exactly one form (fixed OR thinned) -- never
both.

Why static per-segment and not a per-epoch schedule: the batched trainer
(``batched_trainer.train_one_run_batched``) consumes a single
precomputed ``train_seqs`` list and resamples indices with replacement
each epoch -- it has no hook to re-thin sequence *content* per epoch. A
true epoch-ramped curriculum would require re-thinning every epoch (or
carrying both forms of every sequence in memory), neither of which is the
"minimal" change this task asks for. If MIXED-50 does not smooth the
tradeoff, a per-epoch curriculum would be a separate, larger task.
"""

from __future__ import annotations

import hashlib
from typing import Any

from variable_dt import (
    FIXED_DT,
    MILD_VARIABLE_DT,
    segment_to_thinned_training_sequence,
)

MIXED_50_MILD_PROBABILITY = 0.5


def segment_mix_uniform(policy_seed: int, segment_id: str) -> float:
    """A deterministic uniform draw in ``[0, 1)`` per segment. Same
    SHA-256-over-stable-identifiers convention as
    ``variable_dt.thinning_seed`` / the corruption seed, with a distinct
    ``:mix:`` infix so this draw is independent of the thinning-internal
    RNG and of the corruption RNG despite all three being SHA-256 of a
    similarly-shaped string. Uses the full 8-byte digest prefix (53-bit
    mantissa range) rather than the 4-byte ``RandomState``-seed
    truncation, since here we want a fine-grained fraction, not a
    ``numpy`` seed."""
    digest = hashlib.sha256(f"{policy_seed}:mix:{segment_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def segment_is_thinned_in_mix(policy_seed: int, segment_id: str, mild_probability: float) -> bool:
    """``True`` -> this segment is used as its MILD temporal thinning;
    ``False`` -> used as its original FIXED sequence. Deterministic,
    independent of batch order / shard grouping / worker ordering / epoch.
    ``mild_probability=0.0`` reproduces the NATURAL FIXED-DT path exactly;
    ``1.0`` reproduces the pure MILD-variable-dt path exactly."""
    return segment_mix_uniform(policy_seed, segment_id) < mild_probability


def load_split_sequences_mixed(
    shard_output_root,
    split_manifest: dict[str, Any],
    mild_probability: float,
    policy_seed: int,
    augmentation_seed: int,
    corruption_config,
    only_valid_for_kalmannet_gt: bool = True,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    """Same shape/contract as
    ``variable_dt.load_split_sequences_with_thinning`` (shard iteration,
    split-membership resolution, post-corruption first-measurement
    truncation all reused verbatim), the ONLY difference being the
    per-segment ``FIXED_DT`` vs ``MILD_VARIABLE_DT`` choice from
    ``segment_is_thinned_in_mix``.

    ``mild_probability`` and ``augmentation_seed`` are frozen policy
    parameters (MIXED-50 -> ``mild_probability=0.5``); ``policy_seed``
    salts the fixed/thinned assignment (kept separate from
    ``augmentation_seed`` so the *which-segments-are-thinned* decision and
    the *how-each-thinned-segment-is-thinned* decision have independent
    RNGs).

    Returns ``(splits, counts)`` where ``counts`` records how many TRAIN
    segments landed in each form -- needed for the freeze manifest and to
    confirm the realised mix ratio matches ``mild_probability``.
    """
    from av2_split import SPLIT_NAMES
    from ad_morai_bridge_dev.dataset.av2_kalmannet_shard_loader import (
        ShardSegmentRef,
        iter_segments,
        load_shard_index,
    )
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
    counts = {"train_fixed": 0, "train_thinned": 0, "train_dropped": 0}
    for segment in iter_segments(shard_output_root, shard_index=refs):
        if only_valid_for_kalmannet_gt and not segment.valid_for_kalmannet_gt:
            continue
        split_name = scenario_to_split.get(segment.scenario_id)
        if split_name is None:
            continue

        thinned = segment_is_thinned_in_mix(policy_seed, segment.segment_id, mild_probability)
        policy = MILD_VARIABLE_DT if thinned else FIXED_DT
        try:
            seq = segment_to_thinned_training_sequence(
                segment, policy, augmentation_seed, corruption_config
            )
        except ValueError:
            if split_name == "train":
                counts["train_dropped"] += 1
            continue  # too-short segment (n<2) -- pre-existing edge case
        seq["scenario_id"] = segment.scenario_id
        seq["track_id"] = segment.track_id
        seq["av2_object_type"] = segment.av2_object_type
        seq["coarse_object_type"] = segment.coarse_object_type
        seq["mix_form"] = "thinned" if thinned else "fixed"
        seq = _truncate_to_first_measurement(seq)
        if seq is None:
            if split_name == "train":
                counts["train_dropped"] += 1
            continue
        if split_name == "train":
            counts["train_thinned" if thinned else "train_fixed"] += 1
        out[split_name].append(seq)
    return out, counts
