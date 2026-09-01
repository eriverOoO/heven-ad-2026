"""Unit + integration tests for KalmanNet Detector Measurement Attachment v1.

Source datasets are built through the real MORAI Tracking Dataset Factory and
the real KalmanNet trajectory adapter. Detector-prediction records are
synthesised **for unit testing only** (``fixture_detector_records``); no test
asserts detector performance, KNet behaviour, or any KNet-vs-KF outcome.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path


import numpy as np
import pytest

import test_dataset_factory as fx
from ad_morai_bridge_dev.dataset.capture import CaptureLimits, CaptureSession
from ad_morai_bridge_dev.dataset.schema import DatasetPaths, SkewConfig
from ad_morai_bridge_dev.dataset.writer import RunWriter, write_dataset_documents
from ad_morai_bridge_dev.dataset.kalmannet_trajectory_adapter import (
    KalmanNetTrajectoryExporter,
    TrajectoryConfig,
)
from ad_morai_bridge_dev.dataset.kalmannet_supervised_association import (
    Detection,
    GtActor,
    associate_frame,
    hungarian_min_cost,
)
from ad_morai_bridge_dev.dataset.kalmannet_measurement_attachment import (
    MEASUREMENT_SCHEMA_VERSION,
    AdapterError,
    AssociationConfig,
    KalmanNetMeasurementExporter,
    _lidar_to_map_from_tf,
    centerpoint_records_from_heven_offline,
    euclidean_records_from_ros_comparison,
)
from ad_morai_bridge_dev.dataset.kalmannet_measurement_attachment_validate import (
    validate_export,
)


# --------------------------------------------------------------------------
# fixtures: canonical dataset + trajectory export + synthetic detector jsonl
# --------------------------------------------------------------------------
def _actor(uid, x, y, *, heading=0.0):
    return fx.make_actor(uid, 1, x, y, heading)


def _write_run(paths, scenario_id, seed, frames_actors, *, run_index=0, skip_tf_frames=()):
    run_id = f"{scenario_id}__seed_{seed}__run_{run_index:03d}"
    writer = RunWriter(
        paths, scenario_id, run_id,
        {"seed": {"scenario_id": scenario_id, "requested_seed": seed,
                  "simulator_determinism_guaranteed": False}},
    )
    session = CaptureSession(
        writer, fx.static_extrinsics(), SkewConfig(),
        CaptureLimits(frame_count=len(frames_actors), settle_ns=0), "s.json", "abc",
    )
    for i, (stamp_ns, actors) in enumerate(frames_actors):
        session.on_ego(fx.make_ego(stamp_ns))
        session.on_actor_gt(fx.make_actor_array(stamp_ns, actors))
        if i not in skip_tf_frames:
            session.on_tf(fx.make_tf(stamp_ns, [("odom", "base_link", (0.0, 0.0, 0.0))]))
        session.on_lidar(fx.make_pointcloud(stamp_ns, [(1.0, 0.0, 0.0, 0.1, 0.0, 1)]))
        if session.complete:
            break
    return session.finalize("limit_reached")


def _uniform(actors_at, *, n, dt_ns=100_000_000, start=1_000):
    return [(start + i * dt_ns, actors_at(i)) for i in range(n)]


def _canonical(root, run_specs):
    """run_specs: list[(scenario, seed, frames_actors[, skip_tf_frames])]"""
    paths = DatasetPaths(root)
    manifests = []
    for spec in run_specs:
        scenario, seed, frames_actors = spec[0], spec[1], spec[2]
        skip_tf = spec[3] if len(spec) > 3 else ()
        manifests.append(
            _write_run(paths, scenario, seed, frames_actors, skip_tf_frames=skip_tf)
        )
    write_dataset_documents(paths, "morai_tracking_v1", {"repository_commit": "srcabc"}, manifests)
    return paths


def _trajectory_export(tmp_path, run_specs, *, config=None):
    src = tmp_path / "canon"
    _canonical(src, run_specs)
    traj = tmp_path / "traj"
    KalmanNetTrajectoryExporter(src, traj, config or TrajectoryConfig(), "trajabc").export(None)
    return src, traj


def _synth_centerpoint_jsonl(
    canonical_root: Path,
    path: Path,
    *,
    offset=(0.3, -0.1),
    drop_frames=(),
    omit_frames=(),
    extra_fp=None,
):
    """One heven.offline_detection.v1 record per canonical frame: a box near
    each tracking-valid GT actor's lidar_frame centre."""
    lines = []
    for frames_jsonl in sorted((canonical_root / "runs").rglob("frames.jsonl")):
        run_dir = frames_jsonl.parent
        for line in frames_jsonl.read_text().splitlines():
            row = json.loads(line)
            fi = row["frame_index"]
            if fi in omit_frames:
                continue
            dets = []
            if fi not in drop_frames:
                gt = json.loads((run_dir / "gt" / f"{fi:06d}.json").read_text())
                for box in gt["boxes"]:
                    lf = box.get("lidar_frame")
                    if lf and box.get("valid_for_tracking_gt"):
                        c = lf["center"]
                        dets.append({
                            "class_name": "vehicle", "score": 0.9,
                            "box_lidar": [c[0] + offset[0], c[1] + offset[1], c[2],
                                          4.6, 1.9, 1.5, lf["yaw"]],
                        })
                if extra_fp is not None:
                    dets.append({"class_name": "vehicle", "score": 0.5,
                                 "box_lidar": list(extra_fp) + [4.6, 1.9, 1.5, 0.0]})
            lines.append(json.dumps({
                "schema": "heven.offline_detection.v1", "sample_id": row["sample_id"],
                "source_header_stamp_ns": row["lidar_header_stamp_ns"],
                "frame_id": "lidar_link", "inference_time_ms": None, "detections": dets,
            }))
    path.write_text("\n".join(lines) + "\n")
    return path


def _attach(tmp_path, src, traj, det_path, *, source="centerpoint", run_id=None,
            config=None, name="meas", **kw):
    if source == "centerpoint":
        recs = centerpoint_records_from_heven_offline(det_path, detector_config_id="cp_fix")
    else:
        recs = euclidean_records_from_ros_comparison(
            det_path, src, run_id=run_id, detector_config_id="eu_fix")
    out = tmp_path / name
    exp = KalmanNetMeasurementExporter(
        traj, src, out, recs, config or AssociationConfig(), "adaptabc", **kw)
    return out, exp.export()


# --------------------------------------------------------------------------
# Hungarian core
# --------------------------------------------------------------------------
def test_hungarian_matches_brute_force():
    rng = np.random.default_rng(7)

    def brute(cost):
        r, c = cost.shape
        k = min(r, c)
        best = None
        for rows in itertools.permutations(range(r), k):
            for cols in itertools.permutations(range(c), k):
                s = sum(cost[rows[i], cols[i]] for i in range(k))
                if best is None or s < best:
                    best = s
        return best

    for _ in range(200):
        r, c = int(rng.integers(1, 5)), int(rng.integers(1, 5))
        cost = rng.integers(0, 15, size=(r, c)).astype(float)
        pairs = hungarian_min_cost(cost)
        assert len(pairs) == min(r, c)
        assert abs(sum(cost[i, j] for i, j in pairs) - brute(cost)) < 1e-9
        assert pairs == sorted(pairs)


def test_hungarian_deterministic_tie_break():
    cost = np.array([[1.0, 1.0, 5.0], [1.0, 1.0, 5.0]])
    assert hungarian_min_cost(cost) == [(0, 0), (1, 1)]


# --------------------------------------------------------------------------
# frame-local association
# --------------------------------------------------------------------------
def test_positive_match_uses_detection_not_gt():
    gt = [GtActor(11, "vehicle", 10.0, 0.0)]
    det = [Detection(0, "vehicle", 10.4, 0.2, 0.9)]
    assoc = associate_frame(gt, det, max_distance_m=2.5, class_matching=True)
    assert len(assoc.matches) == 1
    m = assoc.matches[0]
    assert m.actor_id == 11 and m.detection_index == 0
    assert abs(m.distance_m - np.hypot(0.4, 0.2)) < 1e-9


def test_physical_gate_blocks_far_detection():
    gt = [GtActor(11, "vehicle", 10.0, 0.0)]
    det = [Detection(0, "vehicle", 30.0, 0.0, 0.9)]
    assoc = associate_frame(gt, det, max_distance_m=2.5, class_matching=True)
    assert assoc.matches == []
    assert assoc.unmatched_actor_ids == [11]
    assert assoc.unmatched_detection_indices == [0]
    assert assoc.gated_out_pairs == 1


def test_class_mismatch_blocks_match():
    gt = [GtActor(11, "vehicle", 10.0, 0.0)]
    det = [Detection(0, "pedestrian", 10.1, 0.0, 0.9)]
    assert associate_frame(gt, det, max_distance_m=2.5, class_matching=True).matches == []


def test_class_agnostic_allows_match_within_gate():
    gt = [GtActor(11, "vehicle", 10.0, 0.0)]
    det = [Detection(0, "unknown", 10.1, 0.0, 0.9)]
    assoc = associate_frame(gt, det, max_distance_m=2.5, class_matching=False)
    assert len(assoc.matches) == 1


def test_one_detection_two_gt_matches_once():
    gt = [GtActor(11, "vehicle", 10.0, 0.0), GtActor(22, "vehicle", 10.5, 0.0)]
    det = [Detection(0, "vehicle", 10.2, 0.0, 0.9)]
    assoc = associate_frame(gt, det, max_distance_m=2.5, class_matching=True)
    assert len(assoc.matches) == 1
    assert len(assoc.unmatched_actor_ids) == 1


def test_two_detections_one_gt_matches_once():
    gt = [GtActor(11, "vehicle", 10.0, 0.0)]
    det = [Detection(0, "vehicle", 10.1, 0.0, 0.9), Detection(1, "vehicle", 9.9, 0.0, 0.8)]
    assoc = associate_frame(gt, det, max_distance_m=2.5, class_matching=True)
    assert len(assoc.matches) == 1
    assert len(assoc.unmatched_detection_indices) == 1


def test_false_positive_detection_is_unmatched():
    gt = [GtActor(11, "vehicle", 10.0, 0.0)]
    det = [Detection(0, "vehicle", 10.1, 0.0, 0.9), Detection(1, "vehicle", 50.0, 50.0, 0.7)]
    assoc = associate_frame(gt, det, max_distance_m=2.5, class_matching=True)
    assert assoc.unmatched_detection_indices == [1]


# --------------------------------------------------------------------------
# transform recomposition
# --------------------------------------------------------------------------
def test_lidar_to_map_reproduces_recorded_lidar_frame(tmp_path):
    # non-trivial ego pose so the transform is not identity
    src = tmp_path / "canon"
    paths = DatasetPaths(src)
    run_id = "lead_constant__seed_0__run_000"
    writer = RunWriter(paths, "lead_constant", run_id,
                       {"seed": {"scenario_id": "lead_constant", "requested_seed": 0}})
    session = CaptureSession(writer, fx.static_extrinsics(), SkewConfig(),
                             CaptureLimits(frame_count=4, settle_ns=0), "s.json", "abc")
    for i in range(4):
        ns = 1_000 + i * 100_000_000
        session.on_ego(fx.make_ego(ns, x=3.0 * i, y=1.0, yaw=0.2))
        session.on_actor_gt(fx.make_actor_array(ns, [fx.make_actor(11, 1, 12.0 + i, 4.0, 0.3)]))
        session.on_tf(fx.make_tf(ns, [("odom", "base_link", (0.05 * i, 0.0, 0.0))]))
        session.on_lidar(fx.make_pointcloud(ns, [(1.0, 0.0, 0.0, 0.1, 0.0, 1)]))
    session.finalize("limit_reached")

    run_dir = paths.runs_dir / "lead_constant" / run_id
    for fi in range(4):
        tf = json.loads((run_dir / "tf" / f"{fi:06d}.json").read_text())
        gt = json.loads((run_dir / "gt" / f"{fi:06d}.json").read_text())
        lidar_to_map = _lidar_to_map_from_tf(tf)
        assert lidar_to_map is not None
        box = gt["boxes"][0]
        recomposed_map = lidar_to_map.apply(tuple(box["lidar_frame"]["center"]))
        assert np.allclose(recomposed_map, box["map_frame"]["center"], atol=1e-6)


def test_no_map_transform_when_tf_absent():
    tf_payload = {"status": "missing"}
    assert _lidar_to_map_from_tf(tf_payload) is None


# --------------------------------------------------------------------------
# end-to-end attachment
# --------------------------------------------------------------------------
def _default_specs():
    return [
        ("lead_constant", 0, _uniform(lambda i: [_actor(11, 10.0 + i, 3.0)], n=6)),
        ("cut_in", 0, _uniform(lambda i: [_actor(7, -3.0 - i, 2.0)], n=6)),
        ("dense_multi_object", 0, _uniform(lambda i: [_actor(9, i, -4.0)], n=6)),
    ]


def test_attach_end_to_end_and_validate(tmp_path):
    src, traj = _trajectory_export(tmp_path, _default_specs())
    det = _synth_centerpoint_jsonl(src, tmp_path / "cp.jsonl", drop_frames=(2,))
    out, result = _attach(tmp_path, src, traj, det)
    assert (out / ".kalmannet_morai_measurement_dataset").is_file()
    manifest = json.loads((out / "export_manifest.json").read_text())
    assert manifest["measurement_schema_version"] == MEASUREMENT_SCHEMA_VERSION
    assert manifest["counts"]["measurements_valid"] == 15  # 18 GT - 3 dropped
    assert manifest["counts"]["zero_detection_frames"] == 3
    assert manifest["counts"]["false_positive_detections"] == 0
    report = validate_export(out, traj)
    assert report.ok, report.errors
    assert report.checked_trajectories == 3


def test_measurement_value_is_detector_center_not_gt(tmp_path):
    src, traj = _trajectory_export(tmp_path, _default_specs())
    det = _synth_centerpoint_jsonl(src, tmp_path / "cp.jsonl", offset=(0.5, -0.25))
    out, _ = _attach(tmp_path, src, traj, det)
    tid = (out / "splits" / "train.txt").read_text().split()[0]
    m = np.load(out / "trajectories" / f"{tid}.npz")
    s = np.load(traj / "trajectories" / f"{tid}.npz")
    valid = m["measurement_valid"]
    # measurement != gt_position, and offset by ~ the injected lidar-frame shift
    d = m["measurement"][valid] - s["gt_position_map"][valid][:, :2]
    assert np.all(np.abs(np.linalg.norm(d, axis=1)) > 0.1)
    assert np.all(np.linalg.norm(d, axis=1) < 1.0)


def test_missed_detection_frame_present_zero_detections(tmp_path):
    src, traj = _trajectory_export(tmp_path, _default_specs())
    det = _synth_centerpoint_jsonl(src, tmp_path / "cp.jsonl", drop_frames=(1, 3))
    out, _ = _attach(tmp_path, src, traj, det)
    rows = {json.loads(l)["trajectory_id"]: json.loads(l)
            for l in (out / "trajectory_index.jsonl").read_text().splitlines()}
    any_row = next(iter(rows.values()))
    assert any_row["miss_reason_counts"]["zero_detections"] == 2
    assert any_row["miss_reason_counts"]["detector_frame_missing"] == 0
    tid = any_row["trajectory_id"]
    m = np.load(out / "trajectories" / f"{tid}.npz")
    assert m["measurement_valid"].tolist() == [True, False, True, False, True, True]
    assert np.isnan(m["measurement"][1]).all()


def test_missing_detector_frame_distinguished_from_zero_detections(tmp_path):
    src, traj = _trajectory_export(tmp_path, _default_specs())
    det = _synth_centerpoint_jsonl(src, tmp_path / "cp.jsonl", omit_frames=(2,))
    out, result = _attach(tmp_path, src, traj, det)
    assert result.manifest["counts"]["detector_frame_missing"] == 3
    assert result.manifest["counts"]["zero_detection_frames"] == 0
    rows = [json.loads(l) for l in (out / "trajectory_index.jsonl").read_text().splitlines()]
    assert all(r["miss_reason_counts"]["detector_frame_missing"] == 1 for r in rows)


def test_no_map_transform_frames_are_masked_and_counted(tmp_path):
    specs = [
        ("lead_constant", 0, _uniform(lambda i: [_actor(11, 10.0 + i, 3.0)], n=6), (1, 2)),
        ("cut_in", 0, _uniform(lambda i: [_actor(7, -3.0 - i, 2.0)], n=6)),
        ("dense_multi_object", 0, _uniform(lambda i: [_actor(9, i, -4.0)], n=6)),
    ]
    src, traj = _trajectory_export(tmp_path, specs)
    det = _synth_centerpoint_jsonl(src, tmp_path / "cp.jsonl")
    out, result = _attach(tmp_path, src, traj, det)
    assert result.manifest["counts"]["no_map_transform_frames"] == 2
    lead_row = [json.loads(l) for l in (out / "trajectory_index.jsonl").read_text().splitlines()
                if "lead_constant" in json.loads(l)["run_id"]][0]
    assert lead_row["miss_reason_counts"]["no_map_transform"] == 2
    tid = lead_row["trajectory_id"]
    m = np.load(out / "trajectories" / f"{tid}.npz")
    assert not m["measurement_valid"][1] and not m["measurement_valid"][2]


def test_false_positive_detection_creates_no_trajectory(tmp_path):
    src, traj = _trajectory_export(tmp_path, _default_specs())
    det = _synth_centerpoint_jsonl(src, tmp_path / "cp.jsonl", extra_fp=(80.0, 80.0, 0.0))
    out, result = _attach(tmp_path, src, traj, det)
    n_traj_before = len(list((traj / "trajectories").glob("*.npz")))
    n_traj_after = len(list((out / "trajectories").glob("*.npz")))
    assert n_traj_after == n_traj_before
    assert result.manifest["counts"]["false_positive_detections"] >= 15


def test_gt_arrays_immutable(tmp_path):
    src, traj = _trajectory_export(tmp_path, _default_specs())
    det = _synth_centerpoint_jsonl(src, tmp_path / "cp.jsonl", drop_frames=(2,))
    out, _ = _attach(tmp_path, src, traj, det)
    from ad_morai_bridge_dev.dataset.kalmannet_measurement_attachment import GT_ARRAY_NAMES
    for npz in sorted((out / "trajectories").glob("*.npz")):
        m = np.load(npz)
        s = np.load(traj / "trajectories" / npz.name)
        for name in GT_ARRAY_NAMES:
            if m[name].dtype.kind in ("U", "S"):
                assert np.array_equal(m[name], s[name]), name
            else:
                assert np.array_equal(m[name], s[name], equal_nan=True), name


def test_natural_missing_run_stats(tmp_path):
    src, traj = _trajectory_export(
        tmp_path,
        [
            ("lead_constant", 0, _uniform(lambda i: [_actor(11, 10.0 + i, 3.0)], n=8)),
            ("cut_in", 0, _uniform(lambda i: [_actor(7, -3.0 - i, 2.0)], n=8)),
            ("dense_multi_object", 0, _uniform(lambda i: [_actor(9, i, -4.0)], n=8)),
        ],
    )
    det = _synth_centerpoint_jsonl(src, tmp_path / "cp.jsonl", drop_frames=(2, 3, 4, 6))
    out, _ = _attach(tmp_path, src, traj, det)
    row = json.loads((out / "trajectory_index.jsonl").read_text().splitlines()[0])
    assert row["measurement_count"] == 4
    assert row["measurement_coverage"] == 0.5
    assert row["longest_missing_run_frames"] == 3
    assert row["longest_missing_run_s"] > 0.0


def test_sample_order_invariance(tmp_path):
    src, traj = _trajectory_export(tmp_path, _default_specs())
    det = _synth_centerpoint_jsonl(src, tmp_path / "cp.jsonl", drop_frames=(2,))
    lines = det.read_text().splitlines()
    shuffled = tmp_path / "cp_shuffled.jsonl"
    shuffled.write_text("\n".join(reversed(lines)) + "\n")
    out_a, res_a = _attach(tmp_path, src, traj, det, name="a")
    out_b, res_b = _attach(tmp_path, src, traj, shuffled, name="b")
    # sample_id is authoritative: the measurement arrays and index rows are
    # identical regardless of record order (only the detector-manifest sha,
    # which hashes the exact input bytes, legitimately differs).
    for npz in sorted((out_a / "trajectories").glob("*.npz")):
        assert (out_b / "trajectories" / npz.name).read_bytes() == npz.read_bytes()
    assert sorted((out_a / "trajectory_index.jsonl").read_text().splitlines()) == sorted(
        (out_b / "trajectory_index.jsonl").read_text().splitlines()
    )


def test_deterministic_repeat_export(tmp_path):
    src, traj = _trajectory_export(tmp_path, _default_specs())
    det = _synth_centerpoint_jsonl(src, tmp_path / "cp.jsonl", drop_frames=(2,))
    out_a, res_a = _attach(tmp_path, src, traj, det, name="a")
    out_b, res_b = _attach(tmp_path, src, traj, det, name="b")
    assert res_a.content_fingerprint == res_b.content_fingerprint
    for npz in sorted((out_a / "trajectories").glob("*.npz")):
        assert (out_b / "trajectories" / npz.name).read_bytes() == npz.read_bytes()


def test_training_ready_criterion(tmp_path):
    src, traj = _trajectory_export(
        tmp_path,
        [
            ("lead_constant", 0, _uniform(lambda i: [_actor(11, 10.0 + i, 3.0)], n=6)),
            ("cut_in", 0, _uniform(lambda i: [_actor(7, -3.0 - i, 2.0)], n=6)),
            ("dense_multi_object", 0, _uniform(lambda i: [_actor(9, i, -4.0)], n=6)),
        ],
    )
    # drop ALL detections for the lead_constant run -> 0 measurements -> not ready
    det = _synth_centerpoint_jsonl(src, tmp_path / "cp.jsonl")
    # rewrite: strip detections whose sample belongs to lead_constant
    lines = []
    for line in det.read_text().splitlines():
        row = json.loads(line)
        if row["sample_id"].startswith("lead_constant__"):
            row["detections"] = []
        lines.append(json.dumps(row))
    det.write_text("\n".join(lines) + "\n")
    out, _ = _attach(tmp_path, src, traj, det)
    rows = {json.loads(l)["trajectory_id"]: json.loads(l)
            for l in (out / "trajectory_index.jsonl").read_text().splitlines()}
    lead = [r for r in rows.values() if r["run_id"].startswith("lead_constant")][0]
    other = [r for r in rows.values() if not r["run_id"].startswith("lead_constant")][0]
    assert lead["measurement_count"] == 0
    assert lead["kalmannet_training_ready"] is False
    assert other["kalmannet_training_ready"] is True


# --------------------------------------------------------------------------
# split preservation + provenance guards
# --------------------------------------------------------------------------
def test_split_preserved_verbatim_and_detector_agnostic(tmp_path):
    plan = {"splits": {
        "train": [{"scenario": "lead_constant", "seeds": [0]}],
        "val": [{"scenario": "cut_in", "seeds": [0]}],
        "test": [{"scenario": "dense_multi_object", "seeds": [0]}],
    }}
    src = tmp_path / "canon"
    _canonical(src, _default_specs())
    traj = tmp_path / "traj"
    KalmanNetTrajectoryExporter(src, traj, TrajectoryConfig(), "t").export(plan)
    det = _synth_centerpoint_jsonl(src, tmp_path / "cp.jsonl")
    out, _ = _attach(tmp_path, src, traj, det, name="cp")
    for split in ("train", "val", "test"):
        assert (out / "splits" / f"{split}.txt").read_text() == (
            traj / "splits" / f"{split}.txt"
        ).read_text()
    report = validate_export(out, traj)
    assert report.ok, report.errors


def test_cross_dataset_pairing_refused(tmp_path):
    src_a, traj_a = _trajectory_export(tmp_path / "a" if False else tmp_path, _default_specs())
    # a second, independent canonical dataset (different manifest sha)
    src_b = tmp_path / "canon_b"
    _canonical(src_b, _default_specs())
    det = _synth_centerpoint_jsonl(src_b, tmp_path / "cp_b.jsonl")
    recs = centerpoint_records_from_heven_offline(det, detector_config_id="x")
    exp = KalmanNetMeasurementExporter(
        traj_a, src_b, tmp_path / "bad", recs, AssociationConfig(), "c")
    with pytest.raises(AdapterError, match="cross-dataset"):
        exp.export()


def test_incomplete_detector_refused(tmp_path):
    src, traj = _trajectory_export(tmp_path, _default_specs())
    det = _synth_centerpoint_jsonl(src, tmp_path / "cp.jsonl")
    recs = centerpoint_records_from_heven_offline(
        det, detector_config_id="x",
        detector_manifest={"expected_frames": 999, "failed_frames": 0})
    exp = KalmanNetMeasurementExporter(
        traj, src, tmp_path / "o", recs, AssociationConfig(), "c")
    with pytest.raises(AdapterError, match="incomplete"):
        exp.export()
    exp2 = KalmanNetMeasurementExporter(
        traj, src, tmp_path / "o2", recs, AssociationConfig(), "c",
        allow_incomplete_detector=True)
    assert exp2.export().manifest["status"] == "complete"


# --------------------------------------------------------------------------
# Euclidean source adapter
# --------------------------------------------------------------------------
def _synth_euclidean_jsonl(canonical_root: Path, path: Path, run_id: str):
    """heven.ros_detection_comparison.v1 records (no sample_id, no yaw, class UNKNOWN)."""
    lines = []
    run_dir = next((canonical_root / "runs").glob(f"*/{run_id}"))
    for line in (run_dir / "frames.jsonl").read_text().splitlines():
        row = json.loads(line)
        fi = row["frame_index"]
        gt = json.loads((run_dir / "gt" / f"{fi:06d}.json").read_text())
        objs = []
        for box in gt["boxes"]:
            lf = box.get("lidar_frame")
            if lf and box.get("valid_for_tracking_gt"):
                c = lf["center"]
                objs.append({"class_name": "unknown", "score": 1.0,
                             "position": [c[0] + 0.2, c[1] + 0.1, c[2]],
                             "dimensions": [4.6, 1.9, 1.5]})
        lines.append(json.dumps({
            "schema": "heven.ros_detection_comparison.v1", "backend": "euclidean",
            "frame_id": "lidar_link", "header_stamp_ns": row["lidar_header_stamp_ns"],
            "received_wall_time_ns": row["lidar_header_stamp_ns"] + 5_000_000,
            "latency_ms": 5.0, "object_count": len(objs), "objects": objs,
        }))
    path.write_text("\n".join(lines) + "\n")
    return path


def test_euclidean_source_adapter_class_agnostic(tmp_path):
    src, traj = _trajectory_export(tmp_path, _default_specs())
    run_id = "lead_constant__seed_0__run_000"
    det = _synth_euclidean_jsonl(src, tmp_path / "eu.jsonl", run_id)
    out, result = _attach(tmp_path, src, traj, det, source="euclidean", run_id=run_id)
    assert result.manifest["detector"]["source"] == "euclidean"
    assert result.manifest["detector"]["class_semantics"] == "class_agnostic"
    assert result.manifest["detector"]["class_matching_effective"] is False
    # only the lead_constant run has aligned euclidean records
    lead = [json.loads(l) for l in (out / "trajectory_index.jsonl").read_text().splitlines()
            if json.loads(l)["run_id"] == run_id][0]
    assert lead["measurement_count"] == 6
    other = [json.loads(l) for l in (out / "trajectory_index.jsonl").read_text().splitlines()
             if json.loads(l)["run_id"] != run_id][0]
    assert other["measurement_count"] == 0
    # no IoU for a yaw-less detector
    m = np.load(out / "trajectories" / f"{lead['trajectory_id']}.npz")
    assert np.all(np.isnan(m["measurement_match_iou"]))
    assert validate_export(out, traj).ok


def test_euclidean_stamp_ambiguity_is_hard_error(tmp_path):
    # two runs whose fixture stamps collide (1000 + i*1e8) -> stamp-only align is ambiguous
    src = tmp_path / "canon"
    _canonical(src, [
        ("lead_constant", 0, _uniform(lambda i: [_actor(11, 10.0 + i, 3.0)], n=4)),
        ("lead_constant", 1, _uniform(lambda i: [_actor(11, 5.0 + i, 1.0)], n=4)),
    ])
    traj = tmp_path / "traj"
    KalmanNetTrajectoryExporter(src, traj, TrajectoryConfig(), "t").export(None)
    det = _synth_euclidean_jsonl(src, tmp_path / "eu.jsonl", "lead_constant__seed_0__run_000")
    # run-scoped index is fine for one run; the guard is on *within-run* dup stamps.
    # Force ambiguity: point at a run and give the canonical run a duplicate stamp row.
    run_dir = src / "runs" / "lead_constant" / "lead_constant__seed_1__run_000"
    lines = (run_dir / "frames.jsonl").read_text().splitlines()
    row = json.loads(lines[1]); row["lidar_header_stamp_ns"] = json.loads(lines[0])["lidar_header_stamp_ns"]
    lines[1] = json.dumps(row)
    (run_dir / "frames.jsonl").write_text("\n".join(lines) + "\n")
    with pytest.raises(AdapterError, match="two frames at stamp"):
        euclidean_records_from_ros_comparison(
            det, src, run_id="lead_constant__seed_1__run_000", detector_config_id="x")
