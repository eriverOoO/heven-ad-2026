"""Glue between the AV2 sharded dataset loader and the KalmanNet training
loop. **Reuses the existing shard loader as the single data source --
does not implement a second AV2 dataset loader.**

Each training-ready sequence is exactly the
``{"actor_id","frames","dt","x_true","z_meas"}`` dict
``av2_kalmannet_shard_loader.SegmentArrays.to_kalmannet_sequence()``
already produces, with a few extra metadata keys appended
(``scenario_id``, ``track_id``, ``av2_object_type``, ``coarse_object_type``)
for stratified reporting -- the historical training loop never reads
those extra keys, so this stays a strict superset of the historical
sequence contract, not a redesign of it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ad_morai_bridge_dev.dataset.av2_kalmannet_shard_loader import (
    ShardSegmentRef,
    iter_segments,
    load_shard_index,
)
from ad_morai_bridge_dev.dataset.av2_motion_forecasting_adapter import CorruptionConfig

from av2_split import SPLIT_NAMES  # noqa: E402  (sibling module, tools/kalmannet_training on sys.path)


def _truncate_to_first_measurement(seq: dict[str, Any]) -> dict[str, Any] | None:
    """Historical precondition (``state_estimator_gt_comparison/
    train_kalmannet.py``: ``assert first_meas is not None, "sequence not
    truncated to first measurement"``) -- a sequence must start at its
    own first valid measurement. Under the CLEAN condition this is a
    no-op (AV2 measurements are clean by construction, frame 0 is always
    valid). Under GENERIC-ROBUST, dropout can hit frame 0, so this
    function truncates the leading run of ``None`` measurements the same
    way the historical dataset construction did upstream. Returns
    ``None`` (caller drops the sequence, counted) if every frame is
    invalid."""
    z_meas = seq["z_meas"]
    start = 0
    while start < len(z_meas) and z_meas[start] is None:
        start += 1
    if start >= len(z_meas) - 1:  # need at least 2 remaining samples to be useful
        return None
    if start == 0:
        return seq
    truncated = dict(seq)
    truncated["frames"] = list(range(len(seq["frames"]) - start))
    truncated["dt"] = [None] + list(seq["dt"][start + 1 :])
    truncated["x_true"] = seq["x_true"][start:]
    truncated["z_meas"] = seq["z_meas"][start:]
    return truncated


def _to_training_sequence(segment, corruption_config: CorruptionConfig | None) -> dict[str, Any] | None:
    seq = segment.to_kalmannet_sequence(corruption_config)
    seq["scenario_id"] = segment.scenario_id
    seq["track_id"] = segment.track_id
    seq["av2_object_type"] = segment.av2_object_type
    seq["coarse_object_type"] = segment.coarse_object_type
    return _truncate_to_first_measurement(seq)


def load_split_sequences(
    shard_output_root: Path,
    split_manifest: dict[str, Any],
    corruption_config: CorruptionConfig | None = None,
    only_valid_for_kalmannet_gt: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    """Returns ``{"train": [...], "val": [...], "test": [...]}``, each a
    list of training-ready sequence dicts. ``corruption_config`` is
    applied uniformly to every returned sequence (callers building two
    conditions -- CLEAN vs. GENERIC-ROBUST -- call this twice, once per
    config, never mixing conditions within one call)."""

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
            continue  # scenario present in shard but not in this split manifest's scenario set
        seq = _to_training_sequence(segment, corruption_config)
        if seq is None:
            continue  # every frame invalid after truncation to first measurement -- dropped, not fabricated
        out[split_name].append(seq)
    return out


def class_distribution(sequences: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    av2_counts: dict[str, int] = {}
    coarse_counts: dict[str, int] = {}
    for seq in sequences:
        av2_counts[seq["av2_object_type"]] = av2_counts.get(seq["av2_object_type"], 0) + 1
        coarse_counts[seq["coarse_object_type"]] = coarse_counts.get(seq["coarse_object_type"], 0) + 1
    return {"av2_object_type": dict(sorted(av2_counts.items())), "coarse_object_type": dict(sorted(coarse_counts.items()))}
