"""Golden-fixture equivalence for the online highway-merge reference path.

The committed validation oracle `ad_data/path/test_highway_merge_ego_ramp_path.txt`
(PR #25) was hand-built as: all real `route:0:left:1` points + real `route:0`
points past merge completion. This test reproduces the C++ builder's algorithm
from the same corridor and proves the online output is the SAME real corridor
geometry (x/y) -- no synthetic lane-change curve -- and that the production
route file is untouched.

`ReferencePoint` carries no z, so the builder's Route z is 0 and every
comparison here is x/y only (the Stanley / Profile-Stanley controller is 2D).
"""

import hashlib
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "ad_data"
ROUTE_CORRIDOR = DATA_DIR / "map" / "route_corridor.json"
GOLDEN = DATA_DIR / "path" / "test_highway_merge_ego_ramp_path.txt"
PRODUCTION_PATH = DATA_DIR / "path" / "2026_molit_comp_global_path.txt"
MERGE_GEOMETRY = REPO / "ad_planner" / "config" / "highway_merge.json"

PRODUCTION_PATH_SHA256 = (
    "50658991e607d9339d76e4cd6cb169dfc733ea53b93de2c3e222460bb497cc05"
)
TARGET_CONTINUATION_M = 200.0


def _lanes():
    doc = json.loads(ROUTE_CORRIDOR.read_text())
    return {lane["lane_sequence_id"]: lane for lane in doc["lanes"]}


def _build_reference_xy():
    """Port of build_highway_merge_reference_path (x/y only)."""
    lanes = _lanes()
    source = lanes["route:0:left:1"]["points"]
    target = lanes["route:0"]["points"]
    merge_complete = source[-1]["route_s_m"]
    points = [(p["x_m"], p["y_m"]) for p in source]
    n_source = len(points)
    for p in target:
        if merge_complete + 1e-6 < p["route_s_m"] <= merge_complete + TARGET_CONTINUATION_M + 1e-6:
            points.append((p["x_m"], p["y_m"]))
    return points, n_source, merge_complete


def _golden_xy():
    rows = []
    for raw in GOLDEN.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            parts = line.split()
            rows.append((float(parts[0]), float(parts[1])))
    return rows


def test_online_builder_matches_the_golden_fixture_xy():
    built, n_source, _ = _build_reference_xy()
    golden = _golden_xy()
    assert len(built) == len(golden)
    for (bx, by), (gx, gy) in zip(built, golden):
        assert math.isclose(bx, gx, abs_tol=1e-9)
        assert math.isclose(by, gy, abs_tol=1e-9)


def test_source_section_is_route_0_left_1_verbatim():
    built, n_source, _ = _build_reference_xy()
    source = _lanes()["route:0:left:1"]["points"]
    assert n_source == len(source)
    for (bx, by), p in zip(built[:n_source], source):
        assert (bx, by) == (p["x_m"], p["y_m"])


def test_target_section_is_a_subsequence_of_route_0():
    built, n_source, merge_complete = _build_reference_xy()
    route0 = {(round(p["x_m"], 9), round(p["y_m"], 9)) for p in _lanes()["route:0"]["points"]}
    target_section = built[n_source:]
    assert len(target_section) > 100
    for bx, by in target_section:
        assert (round(bx, 9), round(by, 9)) in route0


def test_geometric_splice_is_merge_complete_not_commit():
    _, _, merge_complete = _build_reference_xy()
    zone = json.loads(MERGE_GEOMETRY.read_text())["merge_zones"][0]
    # splice == merge complete (~1286.15), never the ~1213 m commit station
    assert math.isclose(merge_complete, zone["route_s_merge_complete_m"], abs_tol=2.0)
    assert merge_complete - zone["route_s_zone_entry_m"] > 150.0


def test_splice_join_is_continuous():
    built, n_source, _ = _build_reference_xy()
    a = built[n_source - 1]
    b = built[n_source]
    gap = math.hypot(b[0] - a[0], b[1] - a[1])
    assert gap < 1.0
    before = built[n_source - 2]
    h0 = math.atan2(a[1] - before[1], a[0] - before[0])
    h1 = math.atan2(b[1] - a[1], b[0] - a[0])
    assert abs(math.remainder(h1 - h0, 2 * math.pi)) < 0.05


def test_production_route_is_unchanged():
    assert hashlib.sha256(PRODUCTION_PATH.read_bytes()).hexdigest() == PRODUCTION_PATH_SHA256
    planner_yaml = (REPO / "ad_planner" / "config" / "planner.yaml").read_text()
    assert "enable_highway_merge_reference_path: false" in planner_yaml
    assert '"path/2026_molit_comp_global_path.txt"' in planner_yaml or \
        "2026_molit_comp_global_path.txt" in planner_yaml
