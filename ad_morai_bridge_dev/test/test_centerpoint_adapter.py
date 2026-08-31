"""Unit + integration tests for the CenterPoint MORAI Data Adapter v1.

The source datasets are built through the ACTUAL MORAI Tracking Dataset
Factory writer (not a hand-rolled directory), so the adapter is exercised
against the real ``morai_tracking_dataset_v1`` schema. The real repo
CenterPoint loader core (``tools/centerpoint_offline/morai_dataset.py``,
torch-free) is instantiated against the adapter output.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

import test_dataset_factory as fx  # helpers: make_ego/make_actor/... static_extrinsics
from ad_morai_bridge_dev.dataset.capture import CaptureLimits, CaptureSession
from ad_morai_bridge_dev.dataset.centerpoint_adapter import (
    TARGET_BOX_FIELDS,
    TARGET_CLASS_NAMES,
    AdapterConfig,
    CenterPointExporter,
    auto_split,
    convert_frame,
    read_source_frames,
)
from ad_morai_bridge_dev.dataset.centerpoint_adapter_validate import validate_export
from ad_morai_bridge_dev.dataset.schema import DatasetPaths, SkewConfig
from ad_morai_bridge_dev.dataset.writer import RunWriter, write_dataset_documents


# --------------------------------------------------------------------------
# build a real factory dataset
# --------------------------------------------------------------------------
def _write_run(paths, scenario_id, seed, *, frames, actors_per_frame, run_index=0):
    run_id = f"{scenario_id}__seed_{seed}__run_{run_index:03d}"
    writer = RunWriter(
        paths, scenario_id, run_id,
        {"seed": {"scenario_id": scenario_id, "requested_seed": seed,
                  "simulator_determinism_guaranteed": False}},
    )
    session = CaptureSession(
        writer, fx.static_extrinsics(), SkewConfig(),
        CaptureLimits(frame_count=frames, settle_ns=0), "s.json", "abc",
    )
    for i in range(frames + 2):
        ns = 1_000 + i * 100_000_000
        session.on_ego(fx.make_ego(ns))
        session.on_actor_gt(fx.make_actor_array(ns, actors_per_frame(i)))
        session.on_tf(fx.make_tf(ns, [("odom", "base_link", (0.0, 0.0, 0.0))]))
        session.on_lidar(
            fx.make_pointcloud(
                ns,
                [(1.0, 0.0, 0.0, 0.1, 0.0, 1), (25.0, 3.0, -0.5, 0.7, 0.01, 5),
                 (60.0, -10.0, 0.2, 0.3, 0.02, 9)],
            )
        )
        if session.complete:
            break
    return session.finalize("limit_reached")


def _vehicle(i):
    return [fx.make_actor(11, 1, 25.0, 3.0, 0.2)]


def make_source_dataset(root: Path, groups, frames=2, dataset_id="morai_tracking_v1"):
    paths = DatasetPaths(root)
    manifests = []
    for scenario_id, seed in groups:
        manifests.append(_write_run(paths, scenario_id, seed, frames=frames,
                                    actors_per_frame=_vehicle))
    write_dataset_documents(paths, dataset_id, {"repository_commit": "srcabc"}, manifests)
    return paths


SIX_GROUPS = [
    ("lead_constant", 0), ("lead_constant", 1), ("lead_constant", 2),
    ("cut_in", 0), ("cut_in", 1), ("dense_multi_object", 0),
]


# --------------------------------------------------------------------------
# repo loader parity
# --------------------------------------------------------------------------
def _load_repo_loader():
    path = Path(__file__).resolve().parents[2] / "tools" / "centerpoint_offline" / "morai_dataset.py"
    spec = importlib.util.spec_from_file_location("_repo_morai_dataset", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("_repo_morai_dataset", module)
    spec.loader.exec_module(module)
    return module


def test_target_contract_matches_repo_centerpoint_loader():
    repo = _load_repo_loader()
    assert TARGET_CLASS_NAMES == repo.CLASS_NAMES
    assert TARGET_BOX_FIELDS == repo.BOX_FIELDS


# --------------------------------------------------------------------------
# source frames + eligibility
# --------------------------------------------------------------------------
def test_reads_only_detection_valid_frames(tmp_path):
    make_source_dataset(tmp_path, [("lead_constant", 0)], frames=2)
    frames, excluded = read_source_frames(tmp_path)
    assert len(frames) == 2
    assert frames[0].scenario_id == "lead_constant"
    assert frames[0].requested_seed == 0
    assert frames[0].group_key == ("lead_constant", "seed_0")
    assert excluded == {}


def test_invalid_detection_frame_is_excluded_and_counted(tmp_path):
    paths = DatasetPaths(tmp_path)
    writer = RunWriter(paths, "lead_constant", "lead_constant__seed_0__run_000",
                       {"seed": {"scenario_id": "lead_constant", "requested_seed": 0}})
    session = CaptureSession(
        writer, fx.static_extrinsics(), SkewConfig(),
        CaptureLimits(frame_count=3, settle_ns=0), "s.json", "abc",
    )
    for i in range(4):
        ns = 1_000 + i * 100_000_000
        session.on_ego(fx.make_ego(ns))
        session.on_actor_gt(fx.make_actor_array(ns, [fx.make_actor(11, 1, 25.0, 3.0, 0.2)]))
        # frame 1 gets no TF -> valid_for_tracking_gt true, detection false
        if i != 1:
            session.on_tf(fx.make_tf(ns, [("odom", "base_link", (0.0, 0.0, 0.0))]))
        session.on_lidar(fx.make_pointcloud(ns, [(1.0, 0.0, 0.0, 0.1, 0.0, 1)]))
        if session.complete:
            break
    m = session.finalize("limit_reached")
    write_dataset_documents(paths, "morai_tracking_v1", {"repository_commit": "x"}, [m])
    frames, excluded = read_source_frames(tmp_path)
    assert len(frames) == 2
    assert sum(excluded.values()) == 1


# --------------------------------------------------------------------------
# box conversion
# --------------------------------------------------------------------------
def _one_box_frame(tmp_path, heading):
    paths = DatasetPaths(tmp_path)
    writer = RunWriter(paths, "lead_constant", "lead_constant__seed_0__run_000",
                       {"seed": {"scenario_id": "lead_constant", "requested_seed": 0}})
    session = CaptureSession(
        writer, fx.static_extrinsics(), SkewConfig(),
        CaptureLimits(frame_count=1, settle_ns=0), "s.json", "abc",
    )
    ns = 1_000
    session.on_ego(fx.make_ego(ns))
    session.on_actor_gt(fx.make_actor_array(ns, [fx.make_actor(11, 1, 25.0, 3.0, heading)]))
    session.on_tf(fx.make_tf(ns, [("odom", "base_link", (0.0, 0.0, 0.0))]))
    session.on_lidar(fx.make_pointcloud(ns, [(25.0, 3.0, -0.5, 0.4, 0.0, 3)]))
    m = session.finalize("limit_reached")
    write_dataset_documents(paths, "morai_tracking_v1", {"repository_commit": "x"}, [m])
    frames, _ = read_source_frames(tmp_path)
    return frames[0]


@pytest.mark.parametrize("heading", [0.0, math.pi / 2, -math.pi / 2, math.pi - 0.01, 3.0])
def test_box_yaw_is_wrapped_and_parity_with_source_lidar_frame(tmp_path, heading):
    frame = _one_box_frame(tmp_path, heading)
    converted = convert_frame(frame, AdapterConfig(class_map={"vehicle": "vehicle"}))
    gt = json.loads(frame.gt_json.read_text())["boxes"][0]["lidar_frame"]
    box = converted.boxes[0]
    # centre / dims are identity copies of the factory's derived lidar_frame box
    assert [box["x"], box["y"], box["z"]] == pytest.approx(gt["center"], abs=1e-9)
    assert box["length"] == pytest.approx(gt["length"])
    assert box["width"] == pytest.approx(gt["width"])
    assert box["height"] == pytest.approx(gt["height"])
    # yaw equals the source yaw modulo 2pi and lies in [-pi, pi]
    assert -math.pi - 1e-6 <= box["yaw"] <= math.pi + 1e-6
    assert math.isclose(
        math.sin(box["yaw"]), math.sin(gt["yaw"]), abs_tol=1e-9
    ) and math.isclose(math.cos(box["yaw"]), math.cos(gt["yaw"]), abs_tol=1e-9)


def test_box_dimension_order_is_length_width_height_not_swapped(tmp_path):
    frame = _one_box_frame(tmp_path, 0.0)
    box = convert_frame(frame, AdapterConfig(class_map={"vehicle": "vehicle"})).boxes[0]
    # fixture actor size is (4.6, 1.9, 1.5) = (length, width, height)
    assert box["length"] == pytest.approx(4.6)
    assert box["width"] == pytest.approx(1.9)
    assert box["height"] == pytest.approx(1.5)
    assert box["length"] > box["width"]  # would fail on an l/w swap


def test_nonzero_z_is_preserved_not_shifted(tmp_path):
    frame = _one_box_frame(tmp_path, 0.0)
    gt = json.loads(frame.gt_json.read_text())["boxes"][0]["lidar_frame"]
    box = convert_frame(frame, AdapterConfig(class_map={"vehicle": "vehicle"})).boxes[0]
    assert box["z"] == pytest.approx(gt["center"][2], abs=1e-9)  # no +-h/2


# --------------------------------------------------------------------------
# class mapping
# --------------------------------------------------------------------------
def test_unmapped_class_is_omitted_from_frame_not_relabelled(tmp_path):
    # a factory-VALID pedestrian box (object_type 0), with the adapter's
    # default vehicle-only map -> the object is dropped, the frame stays.
    paths = DatasetPaths(tmp_path)
    writer = RunWriter(paths, "crossing", "crossing__seed_0__run_000",
                       {"seed": {"scenario_id": "crossing", "requested_seed": 0}})
    session = CaptureSession(
        writer, fx.static_extrinsics(), SkewConfig(),
        CaptureLimits(frame_count=1, settle_ns=0), "s.json", "abc",
    )
    ns = 1_000
    session.on_ego(fx.make_ego(ns))
    session.on_actor_gt(fx.make_actor_array(ns, [
        fx.make_actor(11, 1, 25.0, 3.0, 0.2),
        fx.make_actor(12, 0, 10.0, -2.0, 1.0, size=(0.6, 0.6, 1.7),
                      overhang=0.0, wheelbase=0.0, rear_overhang=0.0),
    ]))
    session.on_tf(fx.make_tf(ns, [("odom", "base_link", (0.0, 0.0, 0.0))]))
    session.on_lidar(fx.make_pointcloud(ns, [(25.0, 3.0, -0.5, 0.4, 0.0, 3)]))
    m = session.finalize("limit_reached")
    write_dataset_documents(paths, "morai_tracking_v1", {"repository_commit": "x"}, [m])
    frames, _ = read_source_frames(tmp_path)
    assert len(frames) == 1  # frame still detection-valid
    converted = convert_frame(frames[0], AdapterConfig(class_map={"vehicle": "vehicle"}))
    assert [b["class_name"] for b in converted.boxes] == ["vehicle"]
    assert converted.omitted_unmapped == {"pedestrian": 1}


# --------------------------------------------------------------------------
# point features
# --------------------------------------------------------------------------
def test_point_export_is_xyzi_only_and_drops_nonfinite(tmp_path):
    paths = DatasetPaths(tmp_path)
    writer = RunWriter(paths, "lead_constant", "lead_constant__seed_0__run_000",
                       {"seed": {"scenario_id": "lead_constant", "requested_seed": 0}})
    session = CaptureSession(
        writer, fx.static_extrinsics(), SkewConfig(),
        CaptureLimits(frame_count=1, settle_ns=0), "s.json", "abc",
    )
    ns = 1_000
    session.on_ego(fx.make_ego(ns))
    session.on_actor_gt(fx.make_actor_array(ns, [fx.make_actor(11, 1, 25.0, 3.0, 0.2)]))
    session.on_tf(fx.make_tf(ns, [("odom", "base_link", (0.0, 0.0, 0.0))]))
    session.on_lidar(fx.make_pointcloud(ns, [
        (1.0, 2.0, 3.0, 0.5, 0.0, 1),
        (float("nan"), 0.0, 0.0, 0.1, 0.0, 2),
        (4.0, 5.0, 6.0, 0.9, 0.0, 3),
    ]))
    m = session.finalize("limit_reached")
    write_dataset_documents(paths, "morai_tracking_v1", {"repository_commit": "x"}, [m])
    frames, _ = read_source_frames(tmp_path)
    converted = convert_frame(frames[0], AdapterConfig(class_map={"vehicle": "vehicle"}))
    assert converted.points.shape == (2, 4)
    assert converted.points.dtype == np.dtype("<f4")
    assert converted.source_point_count == 3
    assert converted.nonfinite_dropped == 1
    assert converted.points[0].tolist() == [1.0, 2.0, 3.0, pytest.approx(0.5)]


# --------------------------------------------------------------------------
# splits
# --------------------------------------------------------------------------
def test_auto_split_no_run_or_group_leakage(tmp_path):
    make_source_dataset(tmp_path, SIX_GROUPS, frames=3)
    out = tmp_path / "cp"
    CenterPointExporter(tmp_path, out, AdapterConfig(class_map={"vehicle": "vehicle"}), "cpabc").export(None)
    labels = {p.stem: json.loads(p.read_text()) for p in (out / "labels").glob("*.json")}
    split_of = {}
    for split in ("train", "val", "test"):
        for sid in (out / "splits" / f"{split}.txt").read_text().split():
            split_of[sid] = split
    by_group = {}
    for sid, split in split_of.items():
        src = labels[sid]["source"]
        by_group.setdefault((src["scenario_id"], src["requested_seed"]), set()).add(split)
    assert all(len(s) == 1 for s in by_group.values())
    assert validate_export(out).ok


def test_explicit_split_plan_is_respected(tmp_path):
    make_source_dataset(tmp_path, SIX_GROUPS, frames=2)
    plan = {"splits": {
        "train": [{"scenario": "lead_constant", "seeds": [0, 1]}, {"scenario": "cut_in", "seeds": [0]}],
        "val": [{"scenario": "lead_constant", "seeds": [2]}],
        "test": [{"scenario": "cut_in", "seeds": [1]}],
    }}
    out = tmp_path / "cp"
    CenterPointExporter(tmp_path, out, AdapterConfig(class_map={"vehicle": "vehicle"}), "x").export(plan)
    sm = json.loads((out / "split_manifest.json").read_text())
    assert sm["frame_counts"]["train"] == 6   # 3 groups x 2 frames
    assert sm["frame_counts"]["val"] == 2
    assert sm["frame_counts"]["test"] == 2
    # dense_multi_object seed 0 was not in the plan -> dropped
    assert json.loads((out / "export_manifest.json").read_text())["counts"]["dropped_by_split_plan"] == 2
    assert validate_export(out).ok


def test_tiny_dataset_does_not_manufacture_val_test(tmp_path):
    make_source_dataset(tmp_path, [("lead_constant", 0)], frames=3)
    out = tmp_path / "cp"
    result = CenterPointExporter(tmp_path, out, AdapterConfig(class_map={"vehicle": "vehicle"}), "x").export(None)
    assert result.manifest["split_frame_counts"] == {"train": 3, "val": 0, "test": 0}
    assert any("too few groups" in w for w in result.split_manifest["warnings"])
    assert (out / "splits" / "val.txt").read_text() == ""


def test_auto_split_is_deterministic():
    a = auto_split([("s", "seed_0"), ("s", "seed_1"), ("s", "seed_2"), ("t", "seed_0")])
    b = auto_split([("t", "seed_0"), ("s", "seed_2"), ("s", "seed_1"), ("s", "seed_0")])
    assert a.assignments == b.assignments


def test_split_leakage_validator_catches_deliberate_overlap(tmp_path):
    make_source_dataset(tmp_path, SIX_GROUPS, frames=2)
    out = tmp_path / "cp"
    CenterPointExporter(tmp_path, out, AdapterConfig(class_map={"vehicle": "vehicle"}), "x").export(None)
    # forge a leak: copy one train id into val.txt
    train_ids = (out / "splits" / "train.txt").read_text().split()
    with (out / "splits" / "val.txt").open("a") as fh:
        fh.write(train_ids[0] + "\n")
    report = validate_export(out)
    assert not report.ok
    assert any("multiple splits" in e or "leakage" in e for e in report.errors)


# --------------------------------------------------------------------------
# negative frame
# --------------------------------------------------------------------------
def test_negative_frame_is_exported_with_empty_gt(tmp_path):
    paths = DatasetPaths(tmp_path)
    writer = RunWriter(paths, "lead_constant", "lead_constant__seed_0__run_000",
                       {"seed": {"scenario_id": "lead_constant", "requested_seed": 0}})
    session = CaptureSession(
        writer, fx.static_extrinsics(), SkewConfig(),
        CaptureLimits(frame_count=1, settle_ns=0), "s.json", "abc",
    )
    ns = 1_000
    session.on_ego(fx.make_ego(ns))
    session.on_actor_gt(fx.make_actor_array(ns, [fx.make_actor(11, 1, 25.0, 3.0, 0.2)]))
    session.on_tf(fx.make_tf(ns, [("odom", "base_link", (0.0, 0.0, 0.0))]))
    session.on_lidar(fx.make_pointcloud(ns, [(1.0, 0.0, 0.0, 0.1, 0.0, 1)]))
    m = session.finalize("limit_reached")
    write_dataset_documents(paths, "morai_tracking_v1", {"repository_commit": "x"}, [m])
    frames, _ = read_source_frames(tmp_path)
    # class map that maps nothing -> the frame has zero target boxes
    converted = convert_frame(frames[0], AdapterConfig(class_map={"pedestrian": "pedestrian"}))
    assert converted.boxes == []
    out = tmp_path / "cp"
    result = CenterPointExporter(
        tmp_path, out, AdapterConfig(class_map={"pedestrian": "pedestrian"}), "x"
    ).export(None)
    assert result.manifest["counts"]["negative_frames"] == 1
    label = json.loads((out / "labels" / f"{frames[0].source_sample_id}.json").read_text())
    assert label["ground_truth"]["boxes"] == []
    assert validate_export(out).ok


# --------------------------------------------------------------------------
# determinism + overwrite + full export
# --------------------------------------------------------------------------
def test_repeat_export_is_bit_identical(tmp_path):
    make_source_dataset(tmp_path, SIX_GROUPS, frames=2)
    r1 = CenterPointExporter(tmp_path, tmp_path / "a", AdapterConfig(class_map={"vehicle": "vehicle"}), "commitX").export(None)
    r2 = CenterPointExporter(tmp_path, tmp_path / "b", AdapterConfig(class_map={"vehicle": "vehicle"}), "commitX").export(None)
    assert r1.content_fingerprint == r2.content_fingerprint
    for name in sorted(p.name for p in (tmp_path / "a" / "points").glob("*.bin")):
        assert (tmp_path / "a" / "points" / name).read_bytes() == (tmp_path / "b" / "points" / name).read_bytes()


def test_completed_export_not_overwritten_silently(tmp_path):
    make_source_dataset(tmp_path, [("lead_constant", 0)], frames=2)
    out = tmp_path / "cp"
    cfg = AdapterConfig(class_map={"vehicle": "vehicle"})
    CenterPointExporter(tmp_path, out, cfg, "x").export(None)
    with pytest.raises(Exception):
        CenterPointExporter(tmp_path, out, cfg, "x").export(None)
    CenterPointExporter(tmp_path, out, cfg, "x", overwrite=True).export(None)


def test_existing_non_marker_directory_is_not_clobbered(tmp_path):
    make_source_dataset(tmp_path, [("lead_constant", 0)], frames=2)
    out = tmp_path / "cp"
    out.mkdir()
    (out / "keepme.txt").write_text("unrelated\n", encoding="utf-8")
    cfg = AdapterConfig(class_map={"vehicle": "vehicle"})
    with pytest.raises(Exception):
        CenterPointExporter(tmp_path, out, cfg, "x").export(None)
    assert (out / "keepme.txt").is_file()
    with pytest.raises(Exception):
        CenterPointExporter(tmp_path, out, cfg, "x", overwrite=True).export(None)
    assert (out / "keepme.txt").is_file()


def test_repo_loader_core_loads_adapter_output(tmp_path):
    make_source_dataset(tmp_path, SIX_GROUPS, frames=3)
    out = tmp_path / "cp"
    CenterPointExporter(tmp_path, out, AdapterConfig(class_map={"vehicle": "vehicle"}), "x").export(None)
    repo = _load_repo_loader()
    core = repo.MoraiHevenDatasetCore(out, split="train")
    assert len(core) > 0
    sample = core[core.first_nonempty_index()]
    assert sample["points"].shape[1] == 4
    assert sample["points"].dtype == np.dtype("<f4") or sample["points"].dtype == np.float32
    assert sample["gt_boxes"].shape[1] == 7
    assert set(sample["gt_names"]) <= set(TARGET_CLASS_NAMES)
    assert sample["coordinate_frame"] == "lidar_link"
    # parity: label box fields == loaded gt_boxes
    label_boxes = np.asarray(
        [[float(b[f]) for f in TARGET_BOX_FIELDS]
         for b in sample["source_label"]["ground_truth"]["boxes"]],
        dtype=np.float32,
    ).reshape(-1, 7)
    assert np.max(np.abs(label_boxes - sample["gt_boxes"])) == pytest.approx(0.0, abs=1e-6)


def test_adapter_rejects_wrong_source_schema(tmp_path):
    make_source_dataset(tmp_path, [("lead_constant", 0)], frames=2)
    manifest = tmp_path / "dataset_manifest.json"
    data = json.loads(manifest.read_text())
    data["schema_version"] = "something_else_v9"
    manifest.write_text(json.dumps(data))
    with pytest.raises(Exception):
        CenterPointExporter(
            tmp_path, tmp_path / "cp", AdapterConfig(class_map={"vehicle": "vehicle"}), "x"
        ).export(None)
