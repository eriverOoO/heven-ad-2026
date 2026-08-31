"""Unit + dry-run tests for the MORAI Tracking Dataset Factory.

No live simulator is required. Synthetic PointCloud2 / ObjectStatusArray /
EgoVehicleStatus / TF messages exercise the same synchronisation, transform,
box and validity code the capture node uses.
"""

from __future__ import annotations

import math
import struct
from types import SimpleNamespace

import numpy as np
import pytest

from ad_morai_bridge_dev.dataset import schema
from ad_morai_bridge_dev.dataset.boxes import ActorGT, canonical_box
from ad_morai_bridge_dev.dataset.capture import CaptureLimits, CaptureSession
from ad_morai_bridge_dev.dataset.frame_builder import FrameAssembler
from ad_morai_bridge_dev.dataset.geometry import RigidTransform, quaternion_from_rpy
from ad_morai_bridge_dev.dataset.pointcloud import (
    PointCloudError,
    decode_pointcloud,
    read_frame_npz,
)
from ad_morai_bridge_dev.dataset.scenario import (
    ScenarioCatalog,
    build_reset_plan,
    default_catalog_path,
    derive_run_parameters,
)
from ad_morai_bridge_dev.dataset.schema import SkewConfig, DatasetPaths
from ad_morai_bridge_dev.dataset.summary import summarize_dataset
from ad_morai_bridge_dev.dataset.sync import TimeBuffer, classify_lidar_stamp
from ad_morai_bridge_dev.dataset.transforms import StaticExtrinsics
from ad_morai_bridge_dev.dataset.validate import validate_dataset
from ad_morai_bridge_dev.dataset.writer import RunWriter, write_dataset_documents


# --------------------------------------------------------------------------
# synthetic messages
# --------------------------------------------------------------------------
def _stamp(ns: int):
    return SimpleNamespace(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)


def _header(ns: int, frame_id: str):
    return SimpleNamespace(stamp=_stamp(ns), frame_id=frame_id)


_PF = {"INT8": 1, "UINT8": 2, "INT16": 3, "UINT16": 4, "INT32": 5, "UINT32": 6,
       "FLOAT32": 7, "FLOAT64": 8}


def make_pointcloud(stamp_ns: int, points, *, frame_id="lidar_link", extra_ring=True):
    """points: list of (x, y, z, intensity, time[, ring])."""

    fields = [
        SimpleNamespace(name="x", datatype=_PF["FLOAT32"], count=1, offset=0),
        SimpleNamespace(name="y", datatype=_PF["FLOAT32"], count=1, offset=4),
        SimpleNamespace(name="z", datatype=_PF["FLOAT32"], count=1, offset=8),
        SimpleNamespace(name="intensity", datatype=_PF["FLOAT32"], count=1, offset=12),
        SimpleNamespace(name="time", datatype=_PF["FLOAT32"], count=1, offset=16),
    ]
    point_step = 20
    fmt = "<5f"
    if extra_ring:
        fields.append(
            SimpleNamespace(name="ring", datatype=_PF["UINT16"], count=1, offset=20)
        )
        point_step = 22
        fmt = "<5fH"
    blob = bytearray()
    for row in points:
        values = list(row[:5])
        if extra_ring:
            values.append(int(row[5]) if len(row) > 5 else 0)
        blob += struct.pack(fmt, *values)
    return SimpleNamespace(
        header=_header(stamp_ns, frame_id),
        is_bigendian=False,
        fields=fields,
        height=1,
        width=len(points),
        point_step=point_step,
        row_step=point_step * len(points),
        data=bytes(blob),
    )


def make_ego(stamp_ns: int, x=0.0, y=0.0, z=0.0, yaw=0.0):
    v = lambda a, b, c: SimpleNamespace(x=a, y=b, z=c)  # noqa: E731
    return SimpleNamespace(
        header=_header(stamp_ns, "map"),
        position=v(x, y, z),
        rpy=v(0.0, 0.0, yaw),
        velocity=v(1.0, 0.0, 0.0),
        angular_velocity=v(0.0, 0.0, 0.05),
        acceleration=v(0.1, 0.0, 0.0),
        signed_velocity=1.0,
    )


def make_actor_array(stamp_ns: int, actors):
    return SimpleNamespace(header=_header(stamp_ns, "map"), objects=list(actors))


def make_actor(uid, otype, x, y, heading, size=(4.6, 1.9, 1.5),
               overhang=0.9, wheelbase=2.8, rear_overhang=0.9):
    v = lambda a, b, c: SimpleNamespace(x=a, y=b, z=c)  # noqa: E731
    return SimpleNamespace(
        unique_id=uid, object_type=otype,
        position=v(x, y, 0.0), heading=heading,
        size=v(*size), overhang=overhang, wheelbase=wheelbase,
        rear_overhang=rear_overhang,
        velocity=v(5.0, 0.0, 0.0), acceleration=v(0.0, 0.0, 0.0), link_id="L1",
    )


def _tf_item(stamp_ns, parent, child, translation=(0.0, 0.0, 0.0)):
    return SimpleNamespace(
        header=_header(stamp_ns, parent),
        child_frame_id=child,
        transform=SimpleNamespace(
            translation=SimpleNamespace(x=translation[0], y=translation[1], z=translation[2]),
            rotation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        ),
    )


def make_tf(stamp_ns, edges):
    return SimpleNamespace(transforms=[_tf_item(stamp_ns, p, c, t) for p, c, t in edges])


def static_extrinsics():
    return StaticExtrinsics.from_tf_static(
        [
            _tf_item(0, "base_link", "rear_axle_link", (-1.4, 0.0, 0.0)),
            _tf_item(0, "rear_axle_link", "lidar_link", (0.0, 0.0, 1.7)),
        ]
    )


# --------------------------------------------------------------------------
# schema
# --------------------------------------------------------------------------
def test_schema_version_is_namespaced():
    assert schema.SCHEMA_VERSION == "morai_tracking_dataset_v1"
    assert schema.RAW_TYPE_TO_CLASS[1] == "vehicle"
    assert schema.CLASS_UNKNOWN == "CLASS_UNKNOWN"


def test_frame_validity_requires_matched_actor_gt():
    record = schema.FrameRecord(
        frame_index=0, sample_id="s", lidar_header_stamp_ns=1,
        lidar_frame_id="lidar_link", lidar_filename="lidar/000000.npz",
        point_count=1, point_fields=["x"], actor_status="skew_exceeded",
    )
    tracking, detection, flags = schema.frame_validity(record, [])
    assert not tracking and not detection
    assert "actor_gt_skew_exceeded" in flags


# --------------------------------------------------------------------------
# pointcloud
# --------------------------------------------------------------------------
def test_pointcloud_roundtrip_preserves_all_fields(tmp_path):
    cloud = make_pointcloud(1, [(1.0, 2.0, 3.0, 0.5, 0.01, 4), (4.0, 5.0, 6.0, 0.9, 0.02, 7)])
    arrays = decode_pointcloud(cloud)
    assert set(arrays) == {"x", "y", "z", "intensity", "time", "ring"}
    assert arrays["ring"].dtype == np.uint16
    assert arrays["ring"].tolist() == [4, 7]
    assert arrays["x"].tolist() == [1.0, 4.0]
    from ad_morai_bridge_dev.dataset.pointcloud import write_frame_npz

    path = tmp_path / "000000.npz"
    write_frame_npz(path, arrays)
    back = read_frame_npz(path)
    for name in arrays:
        assert np.array_equal(back[name], arrays[name])
        assert back[name].dtype == arrays[name].dtype


def test_pointcloud_rejects_bigendian_and_count():
    cloud = make_pointcloud(1, [(0, 0, 0, 0, 0)], extra_ring=False)
    cloud.is_bigendian = True
    with pytest.raises(PointCloudError):
        decode_pointcloud(cloud)
    cloud.is_bigendian = False
    cloud.fields[0].count = 2
    with pytest.raises(PointCloudError):
        decode_pointcloud(cloud)


# --------------------------------------------------------------------------
# boxes + parity with the offline exporter
# --------------------------------------------------------------------------
def _reference_vehicle_box(position, heading, size, overhang, wheelbase, rear_overhang,
                           map_to_lidar):
    """Reference centre-policy formula, verbatim from the offline exporter
    (``tools/morai_dataset_exporter/export_morai_dataset.py::_transform_box``,
    ``rear_axle_ground_to_box_center``)."""

    from ad_morai_bridge_dev.dataset.geometry import normalize_angle

    length, width, height = size
    forward_offset = (wheelbase + overhang - rear_overhang) / 2.0
    center_map = (
        position[0] + forward_offset * math.cos(heading),
        position[1] + forward_offset * math.sin(heading),
        position[2] + height / 2.0,
    )
    center_lidar = map_to_lidar.apply(center_map)
    forward_lidar = map_to_lidar.apply(
        (center_map[0] + math.cos(heading), center_map[1] + math.sin(heading), center_map[2])
    )
    yaw = normalize_angle(
        math.atan2(forward_lidar[1] - center_lidar[1], forward_lidar[0] - center_lidar[0])
    )
    return center_lidar, yaw, forward_offset


def test_box_centre_policy_matches_offline_exporter():
    map_to_lidar = RigidTransform((0.0, 0.0, -1.7), quaternion_from_rpy(0.1, 0.0, 0.2))
    args = ((10.0, -3.0, 0.0), 0.3, (4.6, 1.9, 1.5), 0.9, 2.8, 0.9)
    center, yaw, offset = _reference_vehicle_box(*args, map_to_lidar)
    ours = canonical_box(
        ActorGT(7, 1, args[0], args[1], args[2], (0.0, 0.0, 0.0), args[3], args[4], args[5]),
        map_to_lidar,
    )
    lf = ours["lidar_frame"]
    assert lf["center"] == pytest.approx(list(center), abs=1e-9)
    assert lf["yaw"] == pytest.approx(yaw, abs=1e-9)
    assert ours["forward_center_offset_m"] == pytest.approx(offset)


def test_box_unknown_class_is_flagged_not_dropped():
    box = canonical_box(
        ActorGT(9, 5, (1.0, 2.0, 0.0), 0.0, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), 0.0, 0.0, 0.0),
        None,
    )
    assert box["class_name"] == "CLASS_UNKNOWN"
    assert box["class_mapped"] is False
    assert box["valid_for_detection_gt"] is False


def test_box_vehicle_length_mismatch_flags_actor():
    box = canonical_box(
        ActorGT(3, 1, (0.0, 0.0, 0.0), 0.0, (9.0, 1.9, 1.5), (0.0, 0.0, 0.0), 0.9, 2.8, 0.9),
        RigidTransform((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
    )
    assert "vehicle_length_geometry_mismatch" in box["flags"]
    assert box["valid_for_detection_gt"] is False


# --------------------------------------------------------------------------
# sync / timing
# --------------------------------------------------------------------------
def test_classify_lidar_stamp():
    assert classify_lidar_stamp(0, None) == "nonpositive"
    assert classify_lidar_stamp(10, None) == "first"
    assert classify_lidar_stamp(10, 10) == "duplicate"
    assert classify_lidar_stamp(9, 10) == "backward"
    assert classify_lidar_stamp(11, 10) == "increasing"


@pytest.mark.parametrize(
    "target, skew, expect",
    [
        (100, 0, "matched"),          # exact
        (105, 10, "matched"),         # within
        (110, 10, "matched"),         # exactly threshold
        (111, 10, "skew_exceeded"),   # above
    ],
)
def test_timebuffer_skew_boundary(target, skew, expect):
    buffer = TimeBuffer()
    buffer.add(100, "a")
    match = buffer.nearest(target, skew)
    assert match.status == expect


def test_timebuffer_reports_older_and_newer_and_missing():
    buffer = TimeBuffer()
    assert buffer.nearest(1, 10).status == "missing"
    buffer.add(100, "old")
    buffer.add(130, "new")
    assert buffer.nearest(105, 20).value == "old"   # actor gt older than lidar
    assert buffer.nearest(125, 20).value == "new"   # actor gt newer than lidar


# --------------------------------------------------------------------------
# frame assembler
# --------------------------------------------------------------------------
def _assembler():
    return FrameAssembler("run_x", static_extrinsics(), SkewConfig(), "s.json", "abc")


def test_assemble_full_frame_is_detection_ready():
    asm = _assembler()
    cloud = make_pointcloud(1_000, [(1.0, 0.0, 0.0, 0.1, 0.0, 1)])
    ego = TimeBuffer(); ego.add(1_000, make_ego(1_000))
    actor = TimeBuffer(); actor.add(1_000, make_actor_array(1_000, [make_actor(11, 1, 20.0, 0.0, 0.0)]))
    tf = TimeBuffer(); tf.add(1_000, _tf_item(1_000, "odom", "base_link"))
    frame = asm.assemble(
        0, cloud,
        ego.nearest(1_000, 10), actor.nearest(1_000, 10), tf.nearest(1_000, 10),
    )
    assert frame.record.valid_for_tracking_gt
    assert frame.record.valid_for_detection_gt
    assert frame.record.actor_ids == [11]
    assert "lidar_frame" in frame.gt_payload["boxes"][0]
    assert frame.ego_payload["is_simulator_ground_truth"] is True


def test_assemble_marks_detection_invalid_when_tf_missing():
    asm = _assembler()
    cloud = make_pointcloud(1_000, [(1.0, 0.0, 0.0, 0.1, 0.0, 1)])
    ego = TimeBuffer(); ego.add(1_000, make_ego(1_000))
    actor = TimeBuffer(); actor.add(1_000, make_actor_array(1_000, [make_actor(11, 1, 20.0, 0.0, 0.0)]))
    tf = TimeBuffer()
    frame = asm.assemble(
        0, cloud, ego.nearest(1_000, 10), actor.nearest(1_000, 10), tf.nearest(1_000, 10)
    )
    assert frame.record.valid_for_tracking_gt        # actor gt still fresh
    assert not frame.record.valid_for_detection_gt   # no lidar-frame box
    assert "tf_missing" in frame.record.validation_flags


# --------------------------------------------------------------------------
# writer + capture session
# --------------------------------------------------------------------------
def _capture_run(tmp_path, frames=3, settle_ns=0):
    paths = DatasetPaths(tmp_path)
    writer = RunWriter(paths, "lead_constant", "lead_constant__seed_0__run_000", {"k": "v"})
    session = CaptureSession(
        writer, static_extrinsics(), SkewConfig(),
        CaptureLimits(frame_count=frames, settle_ns=settle_ns),
        "s.json", "abc",
    )
    return paths, writer, session


def test_capture_session_writes_full_run(tmp_path):
    paths, writer, session = _capture_run(tmp_path, frames=3)
    for i in range(6):
        ns = 1_000 + i * 100_000_000
        session.on_ego(make_ego(ns))
        session.on_actor_gt(make_actor_array(ns, [make_actor(11, 1, 20.0, 0.0, 0.0)]))
        session.on_tf(make_tf(ns, [("odom", "base_link", (0.0, 0.0, 0.0))]))
        action = session.on_lidar(make_pointcloud(ns, [(1.0, 0.0, 0.0, 0.1, 0.0, 1)]))
        if action == "captured_final":
            break
    assert session.complete
    manifest = session.finalize("limit_reached")
    assert manifest["status"] == "complete"
    assert manifest["summary"]["captured_lidar_frames"] == 3
    assert manifest["summary"]["unique_actor_ids"] == [11]
    write_dataset_documents(paths, "morai_tracking_v1", {"repository_commit": "x"}, [manifest])
    report = validate_dataset(tmp_path)
    assert report.ok, report.errors


def test_capture_session_settle_frames_are_not_samples(tmp_path):
    _, writer, session = _capture_run(tmp_path, frames=2, settle_ns=500_000_000)
    for i in range(20):
        ns = 1_000 + i * 100_000_000
        session.on_ego(make_ego(ns))
        session.on_actor_gt(make_actor_array(ns, [make_actor(11, 1, 20.0, 0.0, 0.0)]))
        session.on_tf(make_tf(ns, [("odom", "base_link", (0.0, 0.0, 0.0))]))
        session.on_lidar(make_pointcloud(ns, [(1.0, 0.0, 0.0, 0.1, 0.0, 1)]))
        if session.complete:
            break
    assert writer.frame_count == 2
    manifest = session.finalize("limit_reached")
    # first ~5 frames fall in the 0.5 s settle window
    assert manifest["summary"]["captured_lidar_frames"] == 2
    assert "settle_period" in manifest["summary"]["rejected_frames"]


def test_capture_session_rejects_duplicate_and_backward_lidar(tmp_path):
    _, writer, session = _capture_run(tmp_path, frames=5)
    ns = 1_000
    for stamp in (ns, ns, ns - 10):
        session.on_ego(make_ego(stamp))
        session.on_actor_gt(make_actor_array(stamp, [make_actor(11, 1, 20.0, 0.0, 0.0)]))
        session.on_tf(make_tf(stamp, [("odom", "base_link", (0.0, 0.0, 0.0))]))
        session.on_lidar(make_pointcloud(max(stamp, 1), [(1.0, 0.0, 0.0, 0.1, 0.0, 1)]))
    manifest = session.finalize("aborted")
    assert manifest["status"] == "aborted"
    rej = manifest["summary"]["rejected_frames"]
    assert rej.get("lidar_stamp_duplicate", 0) >= 1


def test_interrupted_run_is_aborted_not_complete(tmp_path):
    _, writer, session = _capture_run(tmp_path, frames=10)
    ns = 1_000
    session.on_ego(make_ego(ns))
    session.on_actor_gt(make_actor_array(ns, [make_actor(11, 1, 20.0, 0.0, 0.0)]))
    session.on_tf(make_tf(ns, [("odom", "base_link", (0.0, 0.0, 0.0))]))
    session.on_lidar(make_pointcloud(ns, [(1.0, 0.0, 0.0, 0.1, 0.0, 1)]))
    manifest = session.finalize("keyboard_interrupt")
    assert manifest["status"] == "aborted"
    assert manifest["termination_reason"] == "keyboard_interrupt"


def test_existing_run_collision_is_refused(tmp_path):
    paths = DatasetPaths(tmp_path)
    RunWriter(paths, "s", "r", {})
    with pytest.raises(FileExistsError):
        RunWriter(paths, "s", "r", {})
    RunWriter(paths, "s", "r", {}, overwrite=True)  # allowed


def test_partial_frame_writes_no_index_row_and_is_caught_by_the_validator(
    tmp_path, monkeypatch
):
    paths, writer, _session = _capture_run(tmp_path, frames=2)
    record = schema.FrameRecord(
        frame_index=0, sample_id="s_1", lidar_header_stamp_ns=1,
        lidar_frame_id="lidar_link", lidar_filename="lidar/000000.npz",
        point_count=1, point_fields=["x"],
    )
    import ad_morai_bridge_dev.dataset.writer as writer_mod

    # npz is written first, then the JSON artifacts; fail the gt/ego/tf leg.
    monkeypatch.setattr(
        writer_mod, "_atomic_write_json",
        lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")),
    )
    with pytest.raises(OSError):
        writer.write_frame(
            record, {"x": np.array([1.0], dtype=np.float32)}, {"boxes": []}, {}, {}
        )
    # the guarantee: the index never references the partial frame
    assert (writer.run_dir / "frames.jsonl").read_text() == ""
    assert writer.frame_count == 0
    # and the orphaned npz does not hide - the validator flags it
    monkeypatch.undo()
    writer.finalize("aborted", "partial_frame")
    write_dataset_documents(paths, "d", {"repository_commit": "x"}, [])
    report = validate_dataset(tmp_path)
    assert not report.ok
    assert any("orphaned or missing artifact" in e for e in report.errors)


def test_overwrite_clears_stale_frames_from_a_shorter_run(tmp_path):
    paths = DatasetPaths(tmp_path)
    for frames in (4, 2):
        writer = RunWriter(paths, "s", "r", {}, overwrite=True)
        session = CaptureSession(
            writer, static_extrinsics(), SkewConfig(),
            CaptureLimits(frame_count=frames, settle_ns=0), "s.json", "abc",
        )
        for i in range(frames + 2):
            ns = 1_000 + i * 100_000_000
            session.on_ego(make_ego(ns))
            session.on_actor_gt(make_actor_array(ns, [make_actor(11, 1, 20.0, 0.0, 0.0)]))
            session.on_tf(make_tf(ns, [("odom", "base_link", (0.0, 0.0, 0.0))]))
            session.on_lidar(make_pointcloud(ns, [(1.0, 0.0, 0.0, 0.1, 0.0, 1)]))
            if session.complete:
                break
        manifest = session.finalize("limit_reached")
    write_dataset_documents(paths, "d", {"repository_commit": "x"}, [manifest])
    assert len(list((writer.run_dir / "lidar").glob("*.npz"))) == 2
    assert validate_dataset(tmp_path).ok


# --------------------------------------------------------------------------
# scenario catalog + reset plan
# --------------------------------------------------------------------------
def test_starter_catalog_loads_and_has_six_scenarios():
    catalog = ScenarioCatalog.load(default_catalog_path())
    assert catalog.ids() == sorted(
        ["lead_constant", "lead_brake", "cut_in", "crossing",
         "occlusion_reappear", "dense_multi_object"]
    )
    for scenario in catalog:
        assert scenario.range_regime in ("near", "mid", "far", "mixed")
        assert len(scenario.scenario_sha256) == 64


def test_seed_derivation_is_reproducible_and_seed_sensitive():
    catalog = ScenarioCatalog.load(default_catalog_path())
    scenario = catalog.get("cut_in")
    a = derive_run_parameters(scenario, 0)
    b = derive_run_parameters(scenario, 0)
    c = derive_run_parameters(scenario, 1)
    assert a.actor_jitter == b.actor_jitter
    assert a.actor_jitter != c.actor_jitter
    assert a.simulator_determinism_guaranteed is False


def test_build_reset_plan_applies_jitter_within_bounds():
    catalog = ScenarioCatalog.load(default_catalog_path())
    scenario = catalog.get("dense_multi_object")
    params = derive_run_parameters(scenario, 3)
    base = scenario.base_plan()
    plan = build_reset_plan(scenario, params)
    assert plan.ego == base.ego  # ego never jittered
    for base_actor, new_actor in zip(base.actors, plan.actors):
        shift = math.hypot(
            new_actor.pose.location[0] - base_actor.pose.location[0],
            new_actor.pose.location[1] - base_actor.pose.location[1],
        )
        assert shift <= math.hypot(4.0, 1.0) + 1e-6


# --------------------------------------------------------------------------
# validator catches structural damage
# --------------------------------------------------------------------------
def test_validator_flags_missing_lidar_file(tmp_path):
    paths, _writer, session = _capture_run(tmp_path, frames=2)
    for i in range(3):
        ns = 1_000 + i * 100_000_000
        session.on_ego(make_ego(ns))
        session.on_actor_gt(make_actor_array(ns, [make_actor(11, 1, 20.0, 0.0, 0.0)]))
        session.on_tf(make_tf(ns, [("odom", "base_link", (0.0, 0.0, 0.0))]))
        session.on_lidar(make_pointcloud(ns, [(1.0, 0.0, 0.0, 0.1, 0.0, 1)]))
    manifest = session.finalize("limit_reached")
    write_dataset_documents(paths, "d", {"repository_commit": "x"}, [manifest])
    assert validate_dataset(tmp_path).ok
    next(iter((tmp_path / "runs").rglob("lidar/000000.npz"))).unlink()
    report = validate_dataset(tmp_path)
    assert not report.ok
    assert any("missing" in e for e in report.errors)


def test_summary_reports_counts(tmp_path):
    paths, _writer, session = _capture_run(tmp_path, frames=2)
    for i in range(3):
        ns = 1_000 + i * 100_000_000
        session.on_ego(make_ego(ns))
        session.on_actor_gt(
            make_actor_array(ns, [make_actor(11, 1, 20.0, 0.0, 0.0),
                                  make_actor(12, 1, 40.0, 3.0, 0.1)])
        )
        session.on_tf(make_tf(ns, [("odom", "base_link", (0.0, 0.0, 0.0))]))
        session.on_lidar(make_pointcloud(ns, [(1.0, 0.0, 0.0, 0.1, 0.0, 1)]))
    manifest = session.finalize("limit_reached")
    write_dataset_documents(paths, "d", {"repository_commit": "x"}, [manifest])
    report = summarize_dataset(tmp_path)
    assert report["total_frames"] == 2
    assert sorted(report["unique_gt_actor_ids"]) == [11, 12]
    assert report["class_counts"].get("vehicle") == 4
