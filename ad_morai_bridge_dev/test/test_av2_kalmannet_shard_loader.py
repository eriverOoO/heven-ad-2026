"""Unit tests for ``av2_kalmannet_shard_loader.py`` -- segment recovery
(random access + streaming) and the load-time corruption transform.

Fixture-based (real ``Av2KalmanNetExporter`` runs against monkeypatched
scenario loading), no real AV2 download or ``av2`` pip install required.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ad_morai_bridge_dev.dataset.av2_kalmannet_shard_loader import (
    get_segment,
    iter_segments,
    load_shard_index,
)
from ad_morai_bridge_dev.dataset.av2_motion_forecasting_adapter import (
    Av2KalmanNetExporter,
    Av2ObjectStateRecord,
    Av2ScenarioRecord,
    Av2TrackRecord,
    CoordinateConfig,
    CorruptionConfig,
    SegmentConfig,
    ShardConfig,
)


def _straight_track(track_id, n=30, x0=0.0, y0=0.0, vx=5.0, vy=0.0, dt=0.1):
    states = [
        Av2ObjectStateRecord(True, i, x0 + vx * dt * i, y0 + vy * dt * i, 0.0, vx, vy)
        for i in range(n)
    ]
    return Av2TrackRecord(track_id=track_id, object_type="VEHICLE", category="SCORED_TRACK", states=tuple(states))


def _scenario(sid, tracks, n_timestamps=110):
    ts = tuple(int(i * 1e8) for i in range(n_timestamps))
    return Av2ScenarioRecord(scenario_id=sid, city_name="austin", focal_track_id=tracks[0].track_id,
                              timestamps_ns=ts, tracks=tuple(tracks))


def _build_export(tmp_path, scenario_ids, monkeypatch, tracks_per_scenario=2, n=30, scenarios_per_shard=3):
    """Uses pytest's ``monkeypatch`` (not a plain module-attribute
    assignment) so the ``load_av2_scenario_parquet`` override is
    automatically reverted at the end of the test -- a raw assignment
    would leak across the whole test session."""
    import ad_morai_bridge_dev.dataset.av2_motion_forecasting_adapter as adapter_mod

    scenarios = {
        sid: _scenario(sid, [_straight_track(f"{sid}-t{i}", n=n, x0=float(i) * 200.0) for i in range(tracks_per_scenario)])
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
        shard_config=ShardConfig(scenarios_per_shard=scenarios_per_shard),
    )
    exporter.export()
    return output_root


# ==========================================================================
# C. loader round trip
# ==========================================================================
def test_C_get_segment_round_trips_clean_arrays(tmp_path, monkeypatch):
    output_root = _build_export(tmp_path, [f"s{i:02d}" for i in range(4)], monkeypatch, scenarios_per_shard=2)
    refs = load_shard_index(output_root)
    ref = refs[0]
    segment = get_segment(output_root, ref.segment_id)
    assert segment.segment_id == ref.segment_id
    assert segment.gt_state.shape == (segment.sample_count, 4)
    assert segment.clean_measurement.shape == (segment.sample_count, 2)
    assert np.array_equal(segment.gt_state[:, :2], segment.clean_measurement)  # no corruption yet -> identical
    assert np.all(segment.source_measurement_valid)


def test_get_segment_raises_for_unknown_id(tmp_path, monkeypatch):
    output_root = _build_export(tmp_path, ["s00"], monkeypatch)
    with pytest.raises(KeyError):
        get_segment(output_root, "does-not-exist")


def test_iter_segments_visits_every_segment_exactly_once(tmp_path, monkeypatch):
    scenario_ids = [f"s{i:02d}" for i in range(7)]
    output_root = _build_export(tmp_path, scenario_ids, monkeypatch, tracks_per_scenario=3, scenarios_per_shard=3)
    refs = load_shard_index(output_root)
    seen = [s.segment_id for s in iter_segments(output_root)]
    assert sorted(seen) == sorted(r.segment_id for r in refs)
    assert len(seen) == len(set(seen))


# ==========================================================================
# D. metadata round trip
# ==========================================================================
def test_D_metadata_round_trips_through_shard(tmp_path, monkeypatch):
    output_root = _build_export(tmp_path, ["s00"], monkeypatch, tracks_per_scenario=1, n=12)
    refs = load_shard_index(output_root)
    ref = refs[0]
    segment = get_segment(output_root, ref.segment_id)
    assert segment.scenario_id == ref.scenario_id == "s00"
    assert segment.track_id == ref.track_id
    assert segment.av2_object_type == ref.av2_object_type == "VEHICLE"
    assert segment.coarse_object_type == ref.coarse_object_type == "VEHICLE"
    assert segment.track_category == ref.track_category == "SCORED_TRACK"
    assert segment.city_name == ref.city_name == "austin"
    assert segment.origin_x == pytest.approx(ref.origin_x)
    assert segment.origin_y == pytest.approx(ref.origin_y)
    assert segment.valid_for_kalmannet_gt == ref.valid_for_kalmannet_gt


# ==========================================================================
# L. invalid measurement -> z_meas = None (via to_kalmannet_sequence)
# ==========================================================================
def test_L_to_kalmannet_sequence_no_corruption_is_fully_valid(tmp_path, monkeypatch):
    output_root = _build_export(tmp_path, ["s00"], monkeypatch, tracks_per_scenario=1, n=10)
    segment = get_segment(output_root, load_shard_index(output_root)[0].segment_id)
    seq = segment.to_kalmannet_sequence()  # default: no corruption config
    assert seq["dt"][0] is None
    assert all(v is not None for v in seq["z_meas"])
    assert seq["frames"] == list(range(10))


def test_L_to_kalmannet_sequence_dropout_produces_none_frames(tmp_path, monkeypatch):
    output_root = _build_export(tmp_path, ["s00"], monkeypatch, tracks_per_scenario=1, n=20)
    segment = get_segment(output_root, load_shard_index(output_root)[0].segment_id)
    seq = segment.to_kalmannet_sequence(CorruptionConfig(dropout_prob=1.0, seed=1))
    assert all(v is None for v in seq["z_meas"])
    # never [0, 0], never a repeated previous value -- there is simply no
    # measurement value at all for an invalid frame.
    assert not any(v == [0.0, 0.0] for v in seq["z_meas"] if v is not None)


# ==========================================================================
# I. corruption per-sample determinism
# ==========================================================================
def test_I_corruption_deterministic_given_same_segment_and_config(tmp_path, monkeypatch):
    output_root = _build_export(tmp_path, ["s00"], monkeypatch, tracks_per_scenario=1, n=25)
    segment = get_segment(output_root, load_shard_index(output_root)[0].segment_id)
    config = CorruptionConfig(gaussian_noise_std_m=0.3, dropout_prob=0.2, seed=99)
    seq_1 = segment.to_kalmannet_sequence(config)
    seq_2 = segment.to_kalmannet_sequence(config)
    assert seq_1["z_meas"] == seq_2["z_meas"]


# ==========================================================================
# J. corruption unchanged by dataset reorder
# ==========================================================================
def test_J_corruption_unchanged_by_shard_or_iteration_order(tmp_path, monkeypatch):
    scenario_ids = [f"s{i:02d}" for i in range(5)]
    output_root_a = _build_export(tmp_path / "a", scenario_ids, monkeypatch, tracks_per_scenario=1, n=15, scenarios_per_shard=5)
    output_root_b = _build_export(tmp_path / "b", list(reversed(scenario_ids)), monkeypatch, tracks_per_scenario=1, n=15,
                                   scenarios_per_shard=2)
    config = CorruptionConfig(gaussian_noise_std_m=0.25, dropout_prob=0.15, seed=42)

    for sid in scenario_ids:
        segment_id = f"{sid}__track_{sid}-t0__segment_000"
        seg_a = get_segment(output_root_a, segment_id)
        seg_b = get_segment(output_root_b, segment_id)
        seq_a = seg_a.to_kalmannet_sequence(config)
        seq_b = seg_b.to_kalmannet_sequence(config)
        assert seq_a["z_meas"] == seq_b["z_meas"], (
            f"segment {segment_id} corruption differs depending on which shard/order it was exported/loaded from"
        )


# ==========================================================================
# K. different sample gets different (independent) RNG stream
# ==========================================================================
def test_K_different_segments_get_independent_corruption_streams(tmp_path, monkeypatch):
    output_root = _build_export(tmp_path, ["s00"], monkeypatch, tracks_per_scenario=5, n=40, scenarios_per_shard=1)
    refs = load_shard_index(output_root)
    assert len(refs) == 5
    config = CorruptionConfig(gaussian_noise_std_m=0.4, dropout_prob=0.25, seed=7)

    sequences = []
    for ref in refs:
        segment = get_segment(output_root, ref.segment_id)
        sequences.append(segment.to_kalmannet_sequence(config)["z_meas"])

    # every pair of distinct segments must differ somewhere (independent
    # streams) -- if two happened to be byte-identical it would mean the
    # per-segment seed derivation collided or ignored segment identity.
    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            assert sequences[i] != sequences[j], f"segments {i} and {j} share an identical corruption stream"


def test_corruption_stream_changes_with_global_seed_only_argument(tmp_path, monkeypatch):
    output_root = _build_export(tmp_path, ["s00"], monkeypatch, tracks_per_scenario=1, n=20)
    segment = get_segment(output_root, load_shard_index(output_root)[0].segment_id)
    seq_seed_1 = segment.to_kalmannet_sequence(CorruptionConfig(gaussian_noise_std_m=0.5, seed=1))
    seq_seed_2 = segment.to_kalmannet_sequence(CorruptionConfig(gaussian_noise_std_m=0.5, seed=2))
    assert seq_seed_1["z_meas"] != seq_seed_2["z_meas"]
