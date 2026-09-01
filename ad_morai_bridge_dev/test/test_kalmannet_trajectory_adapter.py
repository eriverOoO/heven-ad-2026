"""Unit + integration tests for the KalmanNet MORAI Trajectory Adapter v1.

Source datasets are built through the ACTUAL MORAI Tracking Dataset Factory
writer, so the adapter runs against the real ``morai_tracking_dataset_v1``
schema. Nothing here trains a model, injects synthetic noise as canonical
data, or asserts any KalmanNet-vs-KF outcome.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import test_dataset_factory as fx
from ad_morai_bridge_dev.dataset.capture import CaptureLimits, CaptureSession
from ad_morai_bridge_dev.dataset.kalmannet_trajectory_adapter import (
    ADAPTER_SCHEMA_VERSION,
    KNET_MEAS_DIM,
    KNET_STATE_DIM,
    NPZ_ARRAY_NAMES,
    AdapterError,
    KalmanNetTrajectoryExporter,
    TrajectoryConfig,
    build_measurement_arrays,
    build_trajectories,
    read_source_runs,
)
from ad_morai_bridge_dev.dataset.kalmannet_trajectory_adapter_validate import (
    validate_export,
)
from ad_morai_bridge_dev.dataset.kalmannet_trajectory_loader import load_trajectory
from ad_morai_bridge_dev.dataset.schema import DatasetPaths, SkewConfig
from ad_morai_bridge_dev.dataset.writer import RunWriter, write_dataset_documents

REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------
# factory-backed source builder with per-frame actor control
# --------------------------------------------------------------------------
def _actor(uid, x, y, *, heading=0.0, vx=None, vy=0.0):
    a = fx.make_actor(uid, 1, x, y, heading)
    if vx is not None:
        a.velocity = SimpleNamespace(x=float(vx), y=float(vy), z=0.0)
    return a


def _write_run(
    paths,
    scenario_id,
    seed,
    *,
    frames_actors,          # list[(stamp_ns, [actor, ...])]
    run_index=0,
):
    run_id = f"{scenario_id}__seed_{seed}__run_{run_index:03d}"
    writer = RunWriter(
        paths,
        scenario_id,
        run_id,
        {
            "seed": {
                "scenario_id": scenario_id,
                "requested_seed": seed,
                "simulator_determinism_guaranteed": False,
            }
        },
    )
    session = CaptureSession(
        writer,
        fx.static_extrinsics(),
        SkewConfig(),
        CaptureLimits(frame_count=len(frames_actors), settle_ns=0),
        "s.json",
        "abc",
    )
    for stamp_ns, actors in frames_actors:
        session.on_ego(fx.make_ego(stamp_ns))
        session.on_actor_gt(fx.make_actor_array(stamp_ns, actors))
        session.on_tf(fx.make_tf(stamp_ns, [("odom", "base_link", (0.0, 0.0, 0.0))]))
        session.on_lidar(
            fx.make_pointcloud(
                stamp_ns,
                [(1.0, 0.0, 0.0, 0.1, 0.0, 1), (25.0, 3.0, -0.5, 0.7, 0.01, 5)],
            )
        )
        if session.complete:
            break
    return session.finalize("limit_reached")


def _uniform(actors_at, *, n, dt_ns=100_000_000, start=1_000):
    return [(start + i * dt_ns, actors_at(i)) for i in range(n)]


def _dataset(root, run_specs, dataset_id="morai_tracking_v1"):
    """run_specs: list[(scenario_id, seed, frames_actors)]"""
    paths = DatasetPaths(root)
    manifests = []
    for idx, (scenario_id, seed, frames_actors) in enumerate(run_specs):
        run_index = sum(
            1 for s, sd, _ in run_specs[:idx] if (s, sd) == (scenario_id, seed)
        )
        manifests.append(
            _write_run(
                paths,
                scenario_id,
                seed,
                frames_actors=frames_actors,
                run_index=run_index,
            )
        )
    write_dataset_documents(
        paths, dataset_id, {"repository_commit": "srcabc"}, manifests
    )
    return paths


def _export(tmp_path, run_specs, *, config=None, split_plan=None, name="knet_out"):
    src = tmp_path / f"src_{name}"
    _dataset(src, run_specs)
    out = tmp_path / name
    exporter = KalmanNetTrajectoryExporter(
        src, out, config or TrajectoryConfig(), "adapterabc"
    )
    result = exporter.export(split_plan)
    return out, result


# --------------------------------------------------------------------------
# contract parity
# --------------------------------------------------------------------------
def test_state_measurement_dims_mirror_kalmannet_core():
    text = (
        REPO_ROOT
        / "ad_lidar_perception"
        / "ad_lidar_perception"
        / "kalmannet_core.py"
    ).read_text(encoding="utf-8")
    assert "STATE_DIM = 4" in text
    assert "MEAS_DIM = 2" in text
    assert KNET_STATE_DIM == 4
    assert KNET_MEAS_DIM == 2


def test_npz_array_names_match_builder(tmp_path):
    out, _ = _export(
        tmp_path,
        [("lead_constant", 0, _uniform(lambda i: [_actor(11, 10.0 + i, 0.0)], n=6))],
    )
    tid = (out / "splits" / "train.txt").read_text().split()[0]
    with np.load(out / "trajectories" / f"{tid}.npz", allow_pickle=False) as data:
        assert set(data.files) == set(NPZ_ARRAY_NAMES)


# --------------------------------------------------------------------------
# basic extraction
# --------------------------------------------------------------------------
def test_single_actor_single_trajectory(tmp_path):
    frames = _uniform(lambda i: [_actor(11, 10.0 + 2.0 * i, 0.0)], n=6)
    runs, stats = read_source_runs(_dataset(tmp_path / "s", [("lead_constant", 0, frames)]).root)
    trajectories = build_trajectories(runs, TrajectoryConfig())
    assert len(trajectories) == 1
    t = trajectories[0]
    assert t.actor_id == 11
    assert t.segment_index == 0
    assert len(t.observations) == 6
    arr = t.npz_arrays()
    assert arr["gt_state"].shape == (6, 4)
    assert np.isnan(arr["dt_s"][0])
    assert np.allclose(arr["dt_s"][1:], 0.1)
    assert stats.boxes_tracking_valid == 6


def test_multi_actor_three_trajectories(tmp_path):
    frames = _uniform(
        lambda i: [
            _actor(11, 10.0 + i, 0.0),
            _actor(22, -5.0 - i, 3.0),
            _actor(33, i, -8.0),
        ],
        n=5,
    )
    runs, _ = read_source_runs(_dataset(tmp_path / "s", [("cut_in", 0, frames)]).root)
    trajectories = build_trajectories(runs, TrajectoryConfig())
    assert sorted(t.actor_id for t in trajectories) == [11, 22, 33]
    assert all(len(t.observations) == 5 for t in trajectories)


def test_same_actor_id_in_two_runs_stays_independent(tmp_path):
    fa = _uniform(lambda i: [_actor(11, 10.0 + i, 0.0)], n=5)
    fb = _uniform(lambda i: [_actor(11, 99.0 + i, 0.0)], n=5)
    runs, _ = read_source_runs(
        _dataset(
            tmp_path / "s",
            [("lead_constant", 0, fa), ("lead_constant", 1, fb)],
        ).root
    )
    trajectories = build_trajectories(runs, TrajectoryConfig())
    assert len(trajectories) == 2
    assert {t.run_id for t in trajectories} == {
        "lead_constant__seed_0__run_000",
        "lead_constant__seed_1__run_000",
    }
    # No trajectory spans runs: each id encodes exactly one run.
    for t in trajectories:
        assert t.trajectory_id.startswith(t.run_id + "__actor_11__segment_")


# --------------------------------------------------------------------------
# native velocity (deliberate mismatch so the test has teeth)
# --------------------------------------------------------------------------
def test_native_velocity_used_not_finite_difference(tmp_path):
    # position advances ~10 m/s, native velocity says 7.0 m/s.
    frames = _uniform(
        lambda i: [_actor(11, 10.0 + 1.0 * i, 0.0, vx=7.0)], n=6
    )
    runs, _ = read_source_runs(_dataset(tmp_path / "s", [("lead_constant", 0, frames)]).root)
    t = build_trajectories(runs, TrajectoryConfig())[0]
    arr = t.npz_arrays()
    assert np.allclose(arr["gt_state"][:, 2], 7.0)
    assert np.allclose(arr["gt_velocity_map"][:, 0], 7.0)
    # the diagnostic finite-diff array records ~10 m/s and is never the state
    mid = arr["finite_diff_velocity"][1:-1, 0]
    assert np.all(np.abs(mid - 10.0) < 1e-6)


# --------------------------------------------------------------------------
# segmentation
# --------------------------------------------------------------------------
def test_time_gap_splits_trajectory(tmp_path):
    def actors_at(i):
        return [] if 4 <= i <= 8 else [_actor(11, 10.0 + i, 0.0)]

    frames = _uniform(actors_at, n=13)
    runs, _ = read_source_runs(_dataset(tmp_path / "s", [("cut_in", 0, frames)]).root)
    trajectories = build_trajectories(
        runs, TrajectoryConfig(max_gt_gap_s=0.3)
    )
    assert len(trajectories) == 2
    assert [len(t.observations) for t in trajectories] == [4, 4]
    assert [t.segment_index for t in trajectories] == [0, 1]


def test_short_gap_does_not_split(tmp_path):
    def actors_at(i):
        return [] if i == 3 else [_actor(11, 10.0 + i, 0.0)]

    frames = _uniform(actors_at, n=8)
    runs, _ = read_source_runs(_dataset(tmp_path / "s", [("cut_in", 0, frames)]).root)
    # 0.2 s gap < default 1.0 s -> one trajectory
    trajectories = build_trajectories(runs, TrajectoryConfig())
    assert len(trajectories) == 1
    assert len(trajectories[0].observations) == 7


def test_teleport_splits_trajectory(tmp_path):
    def actors_at(i):
        return [_actor(11, (1000.0 if i >= 5 else 10.0) + i, 0.0)]

    frames = _uniform(actors_at, n=10)
    runs, _ = read_source_runs(_dataset(tmp_path / "s", [("cut_in", 0, frames)]).root)
    trajectories = build_trajectories(runs, TrajectoryConfig())
    assert len(trajectories) == 2
    assert [len(t.observations) for t in trajectories] == [5, 5]


def test_variable_dt_preserved(tmp_path):
    stamps = [0, 100_000_000, 250_000_000, 400_000_000, 700_000_000]
    frames = [(s + 1_000, [_actor(11, 10.0 + k, 0.0)]) for k, s in enumerate(stamps)]
    runs, _ = read_source_runs(_dataset(tmp_path / "s", [("lead_constant", 0, frames)]).root)
    t = build_trajectories(runs, TrajectoryConfig())[0]
    dt = t.npz_arrays()["dt_s"]
    assert np.isnan(dt[0])
    assert np.allclose(dt[1:], [0.1, 0.15, 0.15, 0.3])


# --------------------------------------------------------------------------
# eligibility accounting
# --------------------------------------------------------------------------
def test_non_tracking_valid_frames_rejected_and_counted(tmp_path):
    paths = DatasetPaths(tmp_path / "s")
    frames = _uniform(lambda i: [_actor(11, 10.0 + i, 0.0)], n=4)
    manifest = _write_run(paths, "lead_constant", 0, frames_actors=frames)
    write_dataset_documents(paths, "morai_tracking_v1", {"repository_commit": "x"}, [manifest])

    # corrupt one frames.jsonl row to be non-tracking-valid
    run_dir = paths.runs_dir / "lead_constant" / "lead_constant__seed_0__run_000"
    lines = (run_dir / "frames.jsonl").read_text().splitlines()
    row = json.loads(lines[1])
    row["valid_for_tracking_gt"] = False
    lines[1] = json.dumps(row, sort_keys=True)
    (run_dir / "frames.jsonl").write_text("\n".join(lines) + "\n")

    runs, stats = read_source_runs(tmp_path / "s")
    t = build_trajectories(runs, TrajectoryConfig())[0]
    assert len(t.observations) == 3
    assert stats.boxes_rejected_frame_not_tracking_valid == 1


# --------------------------------------------------------------------------
# export end to end
# --------------------------------------------------------------------------
def test_export_layout_and_validator(tmp_path):
    specs = [
        ("lead_constant", 0, _uniform(lambda i: [_actor(11, 10.0 + i, 0.0)], n=6)),
        ("lead_constant", 1, _uniform(lambda i: [_actor(11, 5.0 + i, 1.0)], n=6)),
        ("cut_in", 0, _uniform(lambda i: [_actor(7, -3.0 - i, 2.0)], n=6)),
        ("dense_multi_object", 0, _uniform(lambda i: [_actor(9, i, -4.0)], n=6)),
    ]
    out, result = _export(tmp_path, specs)
    assert (out / ".kalmannet_morai_trajectory_adapter").is_file()
    for name in (
        "export_manifest.json",
        "metadata.json",
        "trajectory_index.jsonl",
        "split_manifest.json",
        "splits/train.txt",
    ):
        assert (out / name).exists()
    manifest = json.loads((out / "export_manifest.json").read_text())
    assert manifest["adapter_schema_version"] == ADAPTER_SCHEMA_VERSION
    assert manifest["measurements_present"] is False
    assert manifest["training_ready"] is False
    assert manifest["trajectory_count"] == 4
    report = validate_export(out)
    assert report.ok, report.errors
    assert report.checked_trajectories == 4


def test_zero_trajectory_source_raises(tmp_path):
    # every frame has no actor -> nothing tracking-valid
    frames = _uniform(lambda i: [], n=5)
    src = tmp_path / "s"
    _dataset(src, [("lead_constant", 0, frames)])
    exporter = KalmanNetTrajectoryExporter(
        src, tmp_path / "out", TrajectoryConfig(), "abc"
    )
    with pytest.raises(AdapterError, match="no tracking-valid trajectories"):
        exporter.export(None)


def test_deterministic_repeat_export(tmp_path):
    specs = [
        ("lead_constant", 0, _uniform(lambda i: [_actor(11, 10.0 + i, 0.0)], n=6)),
        ("lead_constant", 1, _uniform(lambda i: [_actor(11, 5.0 + i, 1.0)], n=6)),
        ("cut_in", 0, _uniform(lambda i: [_actor(7, -3.0 - i, 2.0)], n=6)),
    ]
    src = tmp_path / "src"
    _dataset(src, specs)
    res_a = KalmanNetTrajectoryExporter(
        src, tmp_path / "a", TrajectoryConfig(), "adapterabc"
    ).export(None)
    res_b = KalmanNetTrajectoryExporter(
        src, tmp_path / "b", TrajectoryConfig(), "adapterabc"
    ).export(None)
    assert res_a.content_fingerprint == res_b.content_fingerprint
    for tid_file in sorted((tmp_path / "a" / "trajectories").glob("*.npz")):
        assert (
            tmp_path / "b" / "trajectories" / tid_file.name
        ).read_bytes() == tid_file.read_bytes()


def test_min_gt_samples_flag(tmp_path):
    specs = [
        ("lead_constant", 0, _uniform(lambda i: [_actor(11, 10.0 + i, 0.0)], n=6)),
        ("lead_constant", 1, _uniform(lambda i: [_actor(11, 5.0 + i, 1.0)], n=3)),
        ("cut_in", 0, _uniform(lambda i: [_actor(7, -3.0 - i, 2.0)], n=6)),
    ]
    out, _ = _export(tmp_path, specs, config=TrajectoryConfig(min_gt_samples=5))
    rows = {
        json.loads(l)["trajectory_id"]: json.loads(l)
        for l in (out / "trajectory_index.jsonl").read_text().splitlines()
    }
    short = [r for r in rows.values() if r["sample_count"] == 3][0]
    assert short["valid_for_kalmannet_gt"] is False
    long = [r for r in rows.values() if r["sample_count"] == 6][0]
    assert long["valid_for_kalmannet_gt"] is True
    assert all(r["kalmannet_training_ready"] is False for r in rows.values())


# --------------------------------------------------------------------------
# splits + leakage
# --------------------------------------------------------------------------
def test_explicit_split_plan_partitions_by_group(tmp_path):
    specs = [
        ("lead_constant", 0, _uniform(lambda i: [_actor(11, 10.0 + i, 0.0)], n=6)),
        ("cut_in", 0, _uniform(lambda i: [_actor(7, -3.0 - i, 2.0)], n=6)),
        ("dense_multi_object", 0, _uniform(lambda i: [_actor(9, i, -4.0)], n=6)),
    ]
    plan = {
        "splits": {
            "train": [{"scenario": "lead_constant", "seeds": [0]}],
            "val": [{"scenario": "cut_in", "seeds": [0]}],
            "test": [{"scenario": "dense_multi_object", "seeds": [0]}],
        }
    }
    out, _ = _export(tmp_path, specs, split_plan=plan)
    assert len((out / "splits" / "train.txt").read_text().split()) == 1
    assert len((out / "splits" / "val.txt").read_text().split()) == 1
    assert len((out / "splits" / "test.txt").read_text().split()) == 1
    report = validate_export(out)
    assert report.ok, report.errors


def test_multi_actor_run_never_frame_split_across_splits(tmp_path):
    # one run, two actors -> both must land in the same split
    frames = _uniform(
        lambda i: [_actor(11, 10.0 + i, 0.0), _actor(22, -5.0 - i, 3.0)], n=6
    )
    specs = [
        ("lead_constant", 0, frames),
        ("cut_in", 0, _uniform(lambda i: [_actor(7, i, 1.0)], n=6)),
        ("dense_multi_object", 0, _uniform(lambda i: [_actor(9, i, -4.0)], n=6)),
    ]
    plan = {
        "splits": {
            "train": [{"scenario": "lead_constant", "seeds": [0]}],
            "val": [{"scenario": "cut_in", "seeds": [0]}],
            "test": [{"scenario": "dense_multi_object", "seeds": [0]}],
        }
    }
    out, _ = _export(tmp_path, specs, split_plan=plan)
    train_ids = (out / "splits" / "train.txt").read_text().split()
    assert len(train_ids) == 2
    assert all("lead_constant__seed_0" in tid for tid in train_ids)
    assert validate_export(out).ok


def test_validator_catches_injected_leakage(tmp_path):
    specs = [
        ("lead_constant", 0, _uniform(lambda i: [_actor(11, 10.0 + i, 0.0)], n=6)),
        ("cut_in", 0, _uniform(lambda i: [_actor(7, i, 1.0)], n=6)),
        ("dense_multi_object", 0, _uniform(lambda i: [_actor(9, i, -4.0)], n=6)),
    ]
    plan = {
        "splits": {
            "train": [{"scenario": "lead_constant", "seeds": [0]}],
            "val": [{"scenario": "cut_in", "seeds": [0]}],
            "test": [{"scenario": "dense_multi_object", "seeds": [0]}],
        }
    }
    out, _ = _export(tmp_path, specs, split_plan=plan)
    val_id = (out / "splits" / "val.txt").read_text().split()[0]
    train_file = out / "splits" / "train.txt"
    train_file.write_text(train_file.read_text() + val_id + "\n")
    report = validate_export(out)
    assert not report.ok
    assert any("leakage" in e or "multiple splits" in e for e in report.errors)


def test_fewer_than_three_groups_all_train_with_warning(tmp_path):
    specs = [
        ("lead_constant", 0, _uniform(lambda i: [_actor(11, 10.0 + i, 0.0)], n=6)),
        ("cut_in", 0, _uniform(lambda i: [_actor(7, i, 1.0)], n=6)),
    ]
    out, result = _export(tmp_path, specs)
    assert (out / "splits" / "val.txt").read_text().strip() == ""
    assert (out / "splits" / "test.txt").read_text().strip() == ""
    assert result.split_manifest["warnings"]
    assert validate_export(out).ok


def test_split_plan_absent_group_is_hard_error(tmp_path):
    specs = [
        ("lead_constant", 0, _uniform(lambda i: [_actor(11, 10.0 + i, 0.0)], n=6)),
        ("cut_in", 0, _uniform(lambda i: [_actor(7, i, 1.0)], n=6)),
        ("dense_multi_object", 0, _uniform(lambda i: [_actor(9, i, -4.0)], n=6)),
    ]
    plan = {
        "splits": {
            "train": [{"scenario": "lead_constant", "seeds": [0]}],
            "val": [{"scenario": "cut_in", "seeds": [0]}],
            "test": [{"scenario": "roundabout", "seeds": [4]}],
        }
    }
    src = tmp_path / "s"
    _dataset(src, specs)
    exporter = KalmanNetTrajectoryExporter(src, tmp_path / "o", TrajectoryConfig(), "abc")
    with pytest.raises(AdapterError, match="absent groups"):
        exporter.export(plan)


def test_split_plan_double_listed_group_is_hard_error(tmp_path):
    specs = [
        ("lead_constant", 0, _uniform(lambda i: [_actor(11, 10.0 + i, 0.0)], n=6)),
        ("cut_in", 0, _uniform(lambda i: [_actor(7, i, 1.0)], n=6)),
        ("dense_multi_object", 0, _uniform(lambda i: [_actor(9, i, -4.0)], n=6)),
    ]
    plan = {
        "splits": {
            "train": [{"scenario": "lead_constant", "seeds": [0]}],
            "val": [{"scenario": "lead_constant", "seeds": [0]}],
        }
    }
    src = tmp_path / "s"
    _dataset(src, specs)
    exporter = KalmanNetTrajectoryExporter(src, tmp_path / "o", TrajectoryConfig(), "abc")
    with pytest.raises(AdapterError):
        exporter.export(plan)


def test_auto_split_shares_partition_with_centerpoint_adapter(tmp_path):
    from ad_morai_bridge_dev.dataset.centerpoint_adapter import (
        auto_split as cp_auto_split,
    )
    from ad_morai_bridge_dev.dataset.kalmannet_trajectory_adapter import auto_split

    groups = [
        ("lead_constant", "seed_0"),
        ("cut_in", "seed_0"),
        ("cut_in", "seed_1"),
        ("dense_multi_object", "seed_0"),
    ]
    assert auto_split(groups).assignments == cp_auto_split(groups).assignments


# --------------------------------------------------------------------------
# measurement-attachment contract (interface only in v1)
# --------------------------------------------------------------------------
def test_v1_export_has_empty_measurement_slots(tmp_path):
    out, _ = _export(
        tmp_path,
        [("lead_constant", 0, _uniform(lambda i: [_actor(11, 10.0 + i, 0.0)], n=6))],
    )
    tid = (out / "splits" / "train.txt").read_text().split()[0]
    with np.load(out / "trajectories" / f"{tid}.npz", allow_pickle=False) as data:
        assert not np.any(data["measurement_valid"])
        assert np.all(np.isnan(data["measurement"]))
        assert np.all(np.isnan(data["measurement_state"]))


def test_build_measurement_arrays_mask_semantics(tmp_path):
    frames = _uniform(lambda i: [_actor(11, 10.0 + i, 0.0)], n=3)
    runs, _ = read_source_runs(_dataset(tmp_path / "s", [("lead_constant", 0, frames)]).root)
    t = build_trajectories(runs, TrajectoryConfig())[0]
    idx0 = t.observations[0].frame_index
    idx2 = t.observations[2].frame_index
    overlay = build_measurement_arrays(
        t,
        {
            idx0: {"state": [10.05, 0.02], "score": 0.9, "class": "vehicle",
                   "match_distance_m": 0.06},
            idx2: {"state": [12.1, -0.03], "score": 0.7, "class": "vehicle",
                   "match_distance_m": 0.11},
        },
        assignment_method="nearest_center_bev",
        source="euclidean_v1",
    )
    assert overlay["measurement_valid"].tolist() == [True, False, True]
    assert np.isnan(overlay["measurement"][1]).all()
    assert np.allclose(overlay["measurement"][0], [10.05, 0.02])
    assert overlay["measurement_source"][0] == "euclidean_v1"
    assert overlay["measurement_source"][1] == ""
    # no forward-fill: masked-out frame stays nan
    assert np.isnan(overlay["measurement_state"][1]).all()


def test_build_measurement_arrays_rejects_unknown_frame(tmp_path):
    frames = _uniform(lambda i: [_actor(11, 10.0 + i, 0.0)], n=3)
    runs, _ = read_source_runs(_dataset(tmp_path / "s", [("lead_constant", 0, frames)]).root)
    t = build_trajectories(runs, TrajectoryConfig())[0]
    with pytest.raises(AdapterError):
        build_measurement_arrays(
            t, {9999: {"state": [0.0, 0.0]}}, assignment_method="x", source="y"
        )


# --------------------------------------------------------------------------
# loader / historical tensor-layout parity
# --------------------------------------------------------------------------
def test_loader_to_kalmannet_sequence_layout(tmp_path):
    out, _ = _export(
        tmp_path,
        [("lead_constant", 0, _uniform(lambda i: [_actor(11, 10.0 + 2.0 * i, 0.0, vx=20.0)], n=7))],
    )
    tid = (out / "splits" / "train.txt").read_text().split()[0]
    traj = load_trajectory(out / "trajectories" / f"{tid}.npz")
    seq = traj.to_kalmannet_sequence()
    assert seq["dt"][0] is None
    assert all(isinstance(v, float) for v in seq["dt"][1:])
    assert len(seq["x_true"]) == traj.length
    assert all(len(row) == KNET_STATE_DIM for row in seq["x_true"])
    # v1: no measurements -> every z_meas entry is None
    assert all(z is None for z in seq["z_meas"])
    # time-major: rows are timesteps
    assert np.asarray(seq["x_true"]).shape == (7, 4)


def test_source_dataset_is_never_modified(tmp_path):
    src = tmp_path / "s"
    _dataset(src, [("lead_constant", 0, _uniform(lambda i: [_actor(11, 10.0 + i, 0.0)], n=6))])
    before = {p: p.read_bytes() for p in sorted(src.rglob("*")) if p.is_file()}
    out = tmp_path / "o"
    KalmanNetTrajectoryExporter(src, out, TrajectoryConfig(), "abc").export(None)
    after = {p: p.read_bytes() for p in sorted(src.rglob("*")) if p.is_file()}
    assert before == after
