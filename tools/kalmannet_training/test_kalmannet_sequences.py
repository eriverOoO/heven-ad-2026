"""Unit tests for the shard-loader -> training-sequence glue
(``kalmannet_sequences.py``). Fixture-based (real ``Av2KalmanNetExporter``
export against monkeypatched scenario loading), no real AV2 download
required.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT / "ad_morai_bridge_dev"))

from av2_split import SplitConfig, build_split_manifest  # noqa: E402
from kalmannet_sequences import class_distribution, load_split_sequences  # noqa: E402
from ad_morai_bridge_dev.dataset.av2_motion_forecasting_adapter import (  # noqa: E402
    Av2KalmanNetExporter,
    Av2ObjectStateRecord,
    Av2ScenarioRecord,
    Av2TrackRecord,
    CoordinateConfig,
    CorruptionConfig,
    SegmentConfig,
    ShardConfig,
)


def _straight_track(track_id, n=30, x0=0.0, y0=0.0, vx=5.0, vy=0.0, dt=0.1, object_type="VEHICLE"):
    states = [
        Av2ObjectStateRecord(True, i, x0 + vx * dt * i, y0 + vy * dt * i, 0.0, vx, vy)
        for i in range(n)
    ]
    return Av2TrackRecord(track_id=track_id, object_type=object_type, category="SCORED_TRACK", states=tuple(states))


def _scenario(sid, tracks, n_timestamps=110):
    ts = tuple(int(i * 1e8) for i in range(n_timestamps))
    return Av2ScenarioRecord(scenario_id=sid, city_name="austin", focal_track_id=tracks[0].track_id,
                              timestamps_ns=ts, tracks=tuple(tracks))


def _build_export(tmp_path, scenario_ids, monkeypatch, tracks_per_scenario=2, n=30, object_types=None):
    import ad_morai_bridge_dev.dataset.av2_motion_forecasting_adapter as adapter_mod

    object_types = object_types or ["VEHICLE"]
    scenarios = {
        sid: _scenario(sid, [
            _straight_track(f"{sid}-t{i}", n=n, x0=float(i) * 200.0, object_type=object_types[i % len(object_types)])
            for i in range(tracks_per_scenario)
        ])
        for sid in scenario_ids
    }

    def fake_loader(path: Path):
        return scenarios[path.parent.name]

    monkeypatch.setattr(adapter_mod, "load_av2_scenario_parquet", fake_loader)

    raw_root = tmp_path / "raw"
    for sid in scenario_ids:
        d = raw_root / "train" / sid
        d.mkdir(parents=True)
        (d / f"scenario_{sid}.parquet").write_bytes(b"fixture")

    output_root = tmp_path / "out"
    exporter = Av2KalmanNetExporter(
        input_root=raw_root, output_root=output_root, split="train", scenario_ids=list(scenario_ids),
        segment_config=SegmentConfig(), coordinate_config=CoordinateConfig(), corruption_config=CorruptionConfig(),
        adapter_commit="test", scenario_manifest_provenance={"dataset_source": "argoverse2_motion_forecasting"},
        shard_config=ShardConfig(scenarios_per_shard=3),
    )
    exporter.export()
    return output_root


def test_load_split_sequences_respects_split_manifest(tmp_path, monkeypatch):
    scenario_ids = [f"s{i:02d}" for i in range(9)]
    shard_root = _build_export(tmp_path, scenario_ids, monkeypatch, tracks_per_scenario=2, n=20)
    split_manifest = build_split_manifest(
        scenario_ids, SplitConfig(train_frac=0.6, val_frac=0.2, test_frac=0.2, seed=1), {}
    )
    sequences = load_split_sequences(shard_root, split_manifest, corruption_config=None)
    total = sum(len(v) for v in sequences.values())
    assert total == 18  # 9 scenarios * 2 tracks each
    assert set(sequences) == {"train", "val", "test"}
    # every sequence's scenario_id must belong to the split it was bucketed into
    for split_name, seqs in sequences.items():
        for seq in seqs:
            assert seq["scenario_id"] in split_manifest["splits"][split_name]


def test_no_scenario_appears_in_two_split_buckets(tmp_path, monkeypatch):
    scenario_ids = [f"s{i:02d}" for i in range(12)]
    shard_root = _build_export(tmp_path, scenario_ids, monkeypatch)
    split_manifest = build_split_manifest(scenario_ids, SplitConfig(seed=2), {})
    sequences = load_split_sequences(shard_root, split_manifest, corruption_config=None)
    scenario_to_splits: dict[str, set] = {}
    for split_name, seqs in sequences.items():
        for seq in seqs:
            scenario_to_splits.setdefault(seq["scenario_id"], set()).add(split_name)
    for sid, splits in scenario_to_splits.items():
        assert len(splits) == 1, f"scenario {sid} appears in {splits}"


def test_missing_z_gives_none_after_dropout_corruption(tmp_path, monkeypatch):
    scenario_ids = [f"s{i:02d}" for i in range(6)]
    shard_root = _build_export(tmp_path, scenario_ids, monkeypatch, tracks_per_scenario=1, n=40)
    split_manifest = build_split_manifest(scenario_ids, SplitConfig(seed=3), {})
    config = CorruptionConfig(dropout_prob=0.3, seed=11)
    sequences = load_split_sequences(shard_root, split_manifest, corruption_config=config)
    any_none = False
    for seqs in sequences.values():
        for seq in seqs:
            if any(v is None for v in seq["z_meas"]):
                any_none = True
            assert seq["z_meas"][0] is not None  # truncated to first measurement
    assert any_none


def test_variable_dt_preserved_through_sequence_loading(tmp_path, monkeypatch):
    scenario_ids = ["s00"]
    shard_root = _build_export(tmp_path, scenario_ids, monkeypatch, tracks_per_scenario=1, n=15)
    split_manifest = build_split_manifest(["s00", "s01", "s02"], SplitConfig(seed=4), {})
    # only s00 actually exists in the shard; load_split_sequences must
    # tolerate a split manifest naming scenarios outside this particular
    # shard export (real workflow: one split manifest, many shard runs).
    # s01/s02 aren't in the shard -> fine, they just contribute no sequences.
    sequences = load_split_sequences(shard_root, split_manifest, corruption_config=None)
    all_seqs = [s for v in sequences.values() for s in v]
    assert len(all_seqs) == 1
    seq = all_seqs[0]
    assert seq["dt"][0] is None
    assert all(abs(d - 0.1) < 1e-9 for d in seq["dt"][1:])


def test_class_distribution_reports_av2_and_coarse_types(tmp_path, monkeypatch):
    scenario_ids = [f"s{i:02d}" for i in range(4)]
    shard_root = _build_export(
        tmp_path, scenario_ids, monkeypatch, tracks_per_scenario=2, n=10,
        object_types=["VEHICLE", "PEDESTRIAN"],
    )
    split_manifest = build_split_manifest(scenario_ids, SplitConfig(seed=5), {})
    sequences = load_split_sequences(shard_root, split_manifest, corruption_config=None)
    all_seqs = [s for v in sequences.values() for s in v]
    dist = class_distribution(all_seqs)
    assert dist["av2_object_type"]["VEHICLE"] + dist["av2_object_type"]["PEDESTRIAN"] == len(all_seqs)
    assert dist["coarse_object_type"]["VEHICLE"] > 0
    assert dist["coarse_object_type"]["PEDESTRIAN"] > 0
