import hashlib
import json
import os
from pathlib import Path
import zipfile

import pytest

from scripts import prepare_route_corridor as corridor


FIXTURE = Path(__file__).parent / "fixtures/route_corridor"


def run_fixture():
    return corridor.build_route_corridor(
        FIXTURE, FIXTURE / "global_path.txt", match_tolerance_m=0.05
    )


def test_builds_continuous_primary_link_sequence():
    corridor_document = run_fixture()

    assert corridor_document["schema_version"] == 1
    assert corridor_document["primary_lane_sequence_id"] == "route:0"
    assert corridor_document["lanes"][0]["source_link_ids"] == ["L0", "L1"]


def test_includes_only_explicitly_permitted_adjacent_lane():
    corridor_document = run_fixture()

    ids = {lane["lane_sequence_id"] for lane in corridor_document["lanes"]}
    assert ids == {"route:0", "route:0:left:1"}


def test_keeps_parallel_lanes_as_separate_point_arrays():
    corridor_document = run_fixture()

    assert corridor_document["lanes"][0]["points"] != (
        corridor_document["lanes"][1]["points"]
    )
    assert all(
        "adjacent_lane_sequence_ids" in lane
        for lane in corridor_document["lanes"]
    )


def test_records_all_source_sha256_values():
    corridor_document = run_fixture()

    assert set(corridor_document["source_sha256"]) == {
        "global_info.json",
        "node_set.json",
        "link_set.json",
        "global_path",
    }
    assert corridor_document["source_sha256"]["global_path"] == (
        hashlib.sha256((FIXTURE / "global_path.txt").read_bytes()).hexdigest()
    )


def test_points_keep_source_attributes_and_route_geometry():
    primary = run_fixture()["lanes"][0]

    assert primary["source_link_attributes"] == [
        {"id": "L0", "max_speed": 50.0, "width_end": 3.2, "width_start": 3.1},
        {"id": "L1", "max_speed": 60.0, "width_end": 3.3, "width_start": 3.2},
    ]
    assert [point["route_s_m"] for point in primary["points"]] == [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
    ]
    assert all(point["yaw_rad"] == pytest.approx(0.0) for point in primary["points"])
    assert all(
        point["curvature_inv_m"] == pytest.approx(0.0)
        for point in primary["points"]
    )


def test_rejects_path_points_outside_the_strict_match_tolerance(tmp_path):
    global_path = tmp_path / "outside.txt"
    global_path.write_text("0.06, 0, 0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="does not match an MGeo link point"):
        corridor.build_route_corridor(FIXTURE, global_path, match_tolerance_m=0.05)


def test_writes_sorted_json_with_trailing_newline_atomically(tmp_path):
    output = tmp_path / "map/route_corridor.json"

    corridor.write_route_corridor(run_fixture(), output)

    contents = output.read_bytes()
    assert contents.endswith(b"\n")
    assert json.loads(contents) == run_fixture()
    assert contents == json.dumps(
        run_fixture(), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8") + b"\n"


def _link(identifier, from_node_id, to_node_id, points, **overrides):
    values = {
        "identifier": identifier,
        "from_node_id": from_node_id,
        "to_node_id": to_node_id,
        "points": tuple(points),
        "max_speed": 50.0,
        "width_start": 3.0,
        "width_end": 3.0,
        "can_move_left_lane": False,
        "left_destination_id": None,
        "can_move_right_lane": False,
        "right_destination_id": None,
    }
    values.update(overrides)
    return corridor.Link(**values)


def test_ambiguous_coordinates_choose_the_geometrically_continuous_route():
    links = {
        "A": _link("A", "N0", "N1", ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))),
        "B": _link("B", "N1", "N2", ((1.1, 0.0, 0.0), (2.0, 0.0, 0.0))),
        "C": _link(
            "C", "N1", "N3", ((1.0, 0.0, 0.0), (1.1, 0.0, 0.0), (2.0, 0.0, 0.0))
        ),
    }
    path = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.1, 0.0, 0.0), (2.0, 0.0, 0.0)]

    assert corridor._match_primary_link_runs(
        corridor._point_candidates(links, path, 0.05), links
    ) == ["A", "C"]


def test_permission_gap_splits_otherwise_connected_adjacent_destinations():
    links = {
        "P0": _link(
            "P0", "P0A", "P0B", ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
            can_move_left_lane=True, left_destination_id="L0",
        ),
        "P1": _link("P1", "P0B", "P1B", ((1.0, 0.0, 0.0), (2.0, 0.0, 0.0))),
        "P2": _link(
            "P2", "P1B", "P2B", ((2.0, 0.0, 0.0), (3.0, 0.0, 0.0)),
            can_move_left_lane=True, left_destination_id="L1",
        ),
        "L0": _link("L0", "L0A", "L0B", ((0.0, 3.0, 0.0), (1.0, 3.0, 0.0))),
        "L1": _link("L1", "L0B", "L1B", ((1.0, 3.0, 0.0), (2.0, 3.0, 0.0))),
    }

    assert corridor._split_adjacent_sequences(["P0", "P1", "P2"], links, "left") == [
        ["L0"],
        ["L1"],
    ]


def test_adjacent_lane_uses_primary_global_station_instead_of_zero():
    links = {
        "P0": _link(
            "P0", "P0A", "P0B",
            ((0.0, 0.0, 0.0), (100.0, 0.0, 0.0)),
        ),
        "P1": _link(
            "P1", "P0B", "P1B",
            ((100.0, 0.0, 0.0), (110.0, 0.0, 0.0)),
            can_move_left_lane=True, left_destination_id="L1",
        ),
        "L1": _link(
            "L1", "L1A", "L1B",
            ((100.0, 3.0, 0.0), (110.0, 3.0, 0.0)),
        ),
    }

    aligned = corridor._adjacent_link_station_ranges(
        ["P0", "P1"], links, "left"
    )
    document = corridor._lane_document(
        "route:0:left:1",
        ["L1"],
        links,
        {"right": ["route:0"]},
        aligned,
    )

    assert [point["route_s_m"] for point in document["points"]] == [
        100.0,
        110.0,
    ]


def test_equal_node_ids_do_not_allow_a_primary_endpoint_gap_over_tolerance():
    links = {
        "A": _link("A", "N0", "N1", ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))),
        "B": _link("B", "N1", "N2", ((1.1, 0.0, 0.0), (2.0, 0.0, 0.0))),
    }

    assert not corridor._is_legal_transition(
        corridor.PointCandidate("A", 1, 0.0),
        corridor.PointCandidate("B", 0, 0.0),
        links,
    )


def test_zip_and_directory_inputs_have_identical_source_digests(tmp_path):
    archive = tmp_path / "route_corridor.zip"
    with zipfile.ZipFile(archive, "w") as output:
        for name in ("global_info.json", "node_set.json", "link_set.json"):
            output.write(FIXTURE / name, arcname=f"nested/{name}")

    directory_document = run_fixture()
    zip_document = corridor.build_route_corridor(
        archive, FIXTURE / "global_path.txt", match_tolerance_m=0.05
    )

    assert zip_document["source_sha256"] == directory_document["source_sha256"]


def test_cleanup_failure_does_not_mask_the_original_replace_error(tmp_path, monkeypatch):
    output = tmp_path / "route_corridor.json"

    def fail_replace(_temporary, _output):
        raise RuntimeError("replace failed")

    def fail_unlink(self, *args, **kwargs):
        raise OSError("unlink failed")

    monkeypatch.setattr(corridor.os, "replace", fail_replace)
    monkeypatch.setattr(Path, "unlink", fail_unlink)

    with pytest.raises(RuntimeError, match="replace failed"):
        corridor.write_route_corridor(run_fixture(), output)


def test_fdopen_failure_closes_the_raw_temporary_descriptor(tmp_path, monkeypatch):
    descriptors = []

    def fail_fdopen(descriptor, *_args, **_kwargs):
        descriptors.append(descriptor)
        raise OSError("fdopen failed")

    monkeypatch.setattr(corridor.os, "fdopen", fail_fdopen)

    with pytest.raises(OSError, match="fdopen failed"):
        corridor.write_route_corridor(run_fixture(), tmp_path / "route_corridor.json")

    assert len(descriptors) == 1
    with pytest.raises(OSError):
        os.fstat(descriptors[0])
