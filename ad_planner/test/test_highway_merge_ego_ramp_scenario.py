"""Pure-data validation of the Highway Merge Ego-on-Ramp Scenario v1 fixture.

CASE B (PR #24) established that the committed competition global path is
`route:0` (the mainline) and the production ego never drives the acceleration
lane `route:0:left:1`. This scenario adds an OPT-IN, source-grounded validation
route so the existing Highway Merge Gap Risk -> Response -> Response Integration
-> Mission Primitive chain can be exercised with the ego physically on the ramp.

These tests touch no ROS graph. They lock, against the real checksum-verified
route corridor:

  * the fixture route is a verbatim concatenation of real corridor geometry
    (source section == route:0:left:1, target section == route:0), no invented
    Cartesian waypoints, deterministically regenerable;
  * the source/target join is geometrically continuous (no teleport);
  * an ego swept along the real acceleration-lane samples keeps a monotonic
    `project_primary_route` station on `route:0` (the projection the mission
    primitive and the gap-risk node already use) -- so no mission-progress
    architecture change is required;
  * the derived mission commit station sits exactly on the acceleration-lane
    taper (|lateral separation| ~ commit_lateral_separation_m), and the
    completion station sits where the ramp has fully merged (~0 separation);
  * the production competition route file is byte-for-byte unchanged.
"""

import hashlib
import importlib.util
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "ad_data"
ROUTE_CORRIDOR = DATA_DIR / "map" / "route_corridor.json"
MERGE_GEOMETRY = REPO / "ad_planner" / "config" / "highway_merge.json"
FIXTURE_PATH = DATA_DIR / "path" / "test_highway_merge_ego_ramp_path.txt"
PRODUCTION_PATH = DATA_DIR / "path" / "2026_molit_comp_global_path.txt"
SCENARIO_JSON = DATA_DIR / "scenarios" / "kcity_highway_ego_onramp_v1.json"

SOURCE_LANE_ID = "route:0:left:1"
TARGET_LANE_ID = "route:0"
CONTINUATION_M = 200.0

# Frozen when this fixture was authored (PR #24 main). The production route is a
# competition input and must never change as a side effect of this scenario.
PRODUCTION_PATH_SHA256 = (
    "50658991e607d9339d76e4cd6cb169dfc733ea53b93de2c3e222460bb497cc05"
)


def _lanes():
    doc = json.loads(ROUTE_CORRIDOR.read_text())
    return {lane["lane_sequence_id"]: lane for lane in doc["lanes"]}, doc


def build_fixture_lines():
    """Deterministically reconstruct the fixture body (no header comments).

    Every coordinate is copied verbatim from route_corridor.json; the function
    never synthesises a point.
    """
    lanes, _ = _lanes()
    source = lanes[SOURCE_LANE_ID]["points"]
    target = lanes[TARGET_LANE_ID]["points"]
    merge_complete_s = source[-1]["route_s_m"]
    lines = [f"{p['x_m']!r} {p['y_m']!r} {p['z_m']!r}" for p in source]
    n_source = len(lines)
    for p in target:
        if merge_complete_s + 1e-6 < p["route_s_m"] <= merge_complete_s + CONTINUATION_M + 1e-6:
            lines.append(f"{p['x_m']!r} {p['y_m']!r} {p['z_m']!r}")
    return lines, n_source, len(lines) - n_source


def _fixture_points():
    rows = []
    for raw in FIXTURE_PATH.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        rows.append((float(parts[0]), float(parts[1]), float(parts[2])))
    return rows


def _project_primary(target_points, ex, ey, eyaw):
    """Faithful port of ad_planner project_primary_route (pi/2 heading gate)."""
    best = None
    for i in range(1, len(target_points)):
        a, b = target_points[i - 1], target_points[i]
        dx, dy = b["x_m"] - a["x_m"], b["y_m"] - a["y_m"]
        length = math.hypot(dx, dy)
        if length <= 0.0:
            continue
        ux, uy = dx / length, dy / length
        rx, ry = ex - a["x_m"], ey - a["y_m"]
        along = rx * ux + ry * uy
        clamped = min(max(along, 0.0), length)
        lateral = -rx * uy + ry * ux
        distance = math.hypot(along - clamped, lateral)
        seg_heading = math.atan2(dy, dx)
        heading_error = abs(math.remainder(seg_heading - eyaw, 2.0 * math.pi))
        if heading_error > math.pi / 2.0 + 1e-12:
            continue
        ratio = clamped / length
        route_s = a["route_s_m"] + ratio * (b["route_s_m"] - a["route_s_m"])
        cand = (distance, route_s, abs(lateral), heading_error)
        if best is None or cand[0] < best[0] or (
            cand[0] == best[0] and cand[3] < best[3]
        ):
            best = cand
    return best


def _frenet_lateral(target_points, ex, ey):
    """Faithful port of project_to_frenet's closest-point + signed offset."""
    closest_d = math.inf
    closest_s = target_points[0]["route_s_m"]
    for i in range(1, len(target_points)):
        a, b = target_points[i - 1], target_points[i]
        sx, sy = b["x_m"] - a["x_m"], b["y_m"] - a["y_m"]
        length = math.hypot(sx, sy)
        ux, uy = sx / length, sy / length
        rx, ry = ex - a["x_m"], ey - a["y_m"]
        ratio = min(max((rx * ux + ry * uy) / length, 0.0), 1.0)
        px, py = a["x_m"] + ratio * sx, a["y_m"] + ratio * sy
        dist = math.hypot(ex - px, ey - py)
        if dist < closest_d:
            closest_d = dist
            closest_s = a["route_s_m"] + ratio * (b["route_s_m"] - a["route_s_m"])
    # interpolate reference yaw at closest_s
    pts = target_points
    if closest_s <= pts[0]["route_s_m"]:
        ryaw, rx, ry = pts[0]["yaw_rad"], pts[0]["x_m"], pts[0]["y_m"]
    elif closest_s >= pts[-1]["route_s_m"]:
        ryaw, rx, ry = pts[-1]["yaw_rad"], pts[-1]["x_m"], pts[-1]["y_m"]
    else:
        for i in range(1, len(pts)):
            if pts[i]["route_s_m"] >= closest_s:
                a, b = pts[i - 1], pts[i]
                t = (closest_s - a["route_s_m"]) / (b["route_s_m"] - a["route_s_m"])
                rx = a["x_m"] + t * (b["x_m"] - a["x_m"])
                ry = a["y_m"] + t * (b["y_m"] - a["y_m"])
                dyaw = math.atan2(
                    math.sin(b["yaw_rad"] - a["yaw_rad"]),
                    math.cos(b["yaw_rad"] - a["yaw_rad"]),
                )
                ryaw = a["yaw_rad"] + t * dyaw
                break
    d = -math.sin(ryaw) * (ex - rx) + math.cos(ryaw) * (ey - ry)
    return closest_s, d


# --------------------------------------------------------------------------- #
# 1-7: fixture structure and provenance
# --------------------------------------------------------------------------- #
def test_source_and_target_lanes_exist_in_the_corridor():
    lanes, doc = _lanes()
    assert SOURCE_LANE_ID in lanes
    assert TARGET_LANE_ID in lanes
    assert doc["primary_lane_sequence_id"] == TARGET_LANE_ID


def test_fixture_is_deterministically_regenerable_from_the_corridor():
    expected, _, _ = build_fixture_lines()
    committed = [
        raw for raw in FIXTURE_PATH.read_text().splitlines()
        if raw.split("#", 1)[0].strip()
    ]
    assert committed == expected
    # regeneration is stable across repeated calls
    assert build_fixture_lines()[0] == expected


def test_fixture_source_section_matches_route_0_left_1_verbatim():
    lanes, _ = _lanes()
    source = lanes[SOURCE_LANE_ID]["points"]
    _, n_source, _ = build_fixture_lines()
    assert n_source == len(source)
    rows = _fixture_points()
    for corridor_pt, row in zip(source, rows[:n_source]):
        assert math.isclose(corridor_pt["x_m"], row[0], abs_tol=1e-9)
        assert math.isclose(corridor_pt["y_m"], row[1], abs_tol=1e-9)


def test_fixture_target_section_matches_route_0_verbatim():
    lanes, _ = _lanes()
    target = {round(p["route_s_m"], 6): p for p in lanes[TARGET_LANE_ID]["points"]}
    _, n_source, n_target = build_fixture_lines()
    rows = _fixture_points()
    assert n_target > 100
    for row in rows[n_source:]:
        match = min(
            target.values(),
            key=lambda p: math.hypot(p["x_m"] - row[0], p["y_m"] - row[1]),
        )
        assert math.hypot(match["x_m"] - row[0], match["y_m"] - row[1]) < 1e-6


def test_ego_start_is_on_the_source_lane_and_before_the_merge_zone():
    lanes, _ = _lanes()
    source = lanes[SOURCE_LANE_ID]["points"]
    zone = json.loads(MERGE_GEOMETRY.read_text())["merge_zones"][0]
    start = _fixture_points()[0]
    assert math.isclose(start[0], source[0]["x_m"], abs_tol=1e-9)
    assert math.isclose(start[1], source[0]["y_m"], abs_tol=1e-9)
    # the acceleration lane begins exactly at the merge zone entry on this map;
    # the ego cannot start further back on the ramp because route:0:left:1 does
    # not extend upstream of the zone entry (a documented CASE B consequence).
    ego_s, _ = _frenet_lateral(lanes[TARGET_LANE_ID]["points"], start[0], start[1])
    assert abs(ego_s - zone["route_s_zone_entry_m"]) < 1.0
    assert ego_s <= zone["route_s_merge_complete_m"]


def test_source_target_join_is_continuous_no_teleport():
    lanes, _ = _lanes()
    source = lanes[SOURCE_LANE_ID]["points"]
    target = lanes[TARGET_LANE_ID]["points"]
    _, n_source, _ = build_fixture_lines()
    rows = _fixture_points()
    join_a = rows[n_source - 1]
    join_b = rows[n_source]
    gap = math.hypot(join_b[0] - join_a[0], join_b[1] - join_a[1])
    assert gap < 1.0  # one nominal point spacing, no jump
    # the accel-lane end coincides with a route:0 point (0 m separation)
    _, sep = _frenet_lateral(target, source[-1]["x_m"], source[-1]["y_m"])
    assert abs(sep) < 0.05
    # every consecutive spacing along the fixture is a plausible sample step
    spacings = [
        math.hypot(rows[k][0] - rows[k - 1][0], rows[k][1] - rows[k - 1][1])
        for k in range(1, len(rows))
    ]
    assert max(spacings) < 1.0
    assert min(spacings) > 0.01


def test_fixture_has_no_large_heading_discontinuity():
    rows = _fixture_points()
    headings = [
        math.atan2(rows[k][1] - rows[k - 1][1], rows[k][0] - rows[k - 1][0])
        for k in range(1, len(rows))
    ]
    for k in range(1, len(headings)):
        turn = abs(math.remainder(headings[k] - headings[k - 1], 2.0 * math.pi))
        assert turn < 0.35  # rad, per ~0.5 m step


# --------------------------------------------------------------------------- #
# 8-11: primary-route projection compatibility (CRITICAL)
# --------------------------------------------------------------------------- #
def test_primary_route_projection_is_monotonic_for_a_ramp_ego():
    lanes, _ = _lanes()
    source = lanes[SOURCE_LANE_ID]["points"]
    target = lanes[TARGET_LANE_ID]["points"]
    previous = None
    max_abs_error = 0.0
    lateral_first = lateral_last = None
    for pt in source:
        proj = _project_primary(target, pt["x_m"], pt["y_m"], pt["yaw_rad"])
        assert proj is not None, "ramp ego has no forward-facing route:0 segment"
        _, route_s, lateral, _ = proj
        if previous is not None:
            assert route_s >= previous - 1e-6, "primary route_s went backwards"
        previous = route_s
        max_abs_error = max(max_abs_error, abs(route_s - pt["route_s_m"]))
        if lateral_first is None:
            lateral_first = lateral
        lateral_last = lateral
    # source-lane station already lives in the route:0 frame, so the projection
    # tracks it almost exactly
    assert max_abs_error < 0.5
    # ramp lateral offset to route:0 tapers from ~3.9 m to ~0 m
    assert 3.5 < lateral_first < 4.3
    assert lateral_last < 0.3


def test_frenet_projection_is_monotonic_for_a_ramp_ego():
    lanes, _ = _lanes()
    source = lanes[SOURCE_LANE_ID]["points"]
    target = lanes[TARGET_LANE_ID]["points"]
    previous = None
    for pt in source:
        s, _ = _frenet_lateral(target, pt["x_m"], pt["y_m"])
        if previous is not None:
            assert s >= previous - 1e-6
        previous = s


def test_commit_station_corresponds_to_the_taper():
    lanes, _ = _lanes()
    source = lanes[SOURCE_LANE_ID]["points"]
    target = lanes[TARGET_LANE_ID]["points"]
    zone = json.loads(MERGE_GEOMETRY.read_text())["merge_zones"][0]
    threshold = 3.5  # commit_lateral_separation_m (config/planner.yaml)
    commit_s = None
    for pt in source:
        _, d = _frenet_lateral(target, pt["x_m"], pt["y_m"])
        if abs(d) < threshold:
            commit_s = pt["route_s_m"]
            commit_sep = abs(d)
            break
    assert commit_s is not None
    assert zone["route_s_zone_entry_m"] < commit_s < zone["route_s_merge_complete_m"]
    # the ego is genuinely inside the taper here: separation just under threshold
    assert threshold - 0.2 < commit_sep < threshold
    # commit leaves a real committed span before merge completion
    assert zone["route_s_merge_complete_m"] - commit_s > 60.0


def test_completion_station_corresponds_to_full_merge():
    lanes, _ = _lanes()
    source = lanes[SOURCE_LANE_ID]["points"]
    target = lanes[TARGET_LANE_ID]["points"]
    _, sep = _frenet_lateral(target, source[-1]["x_m"], source[-1]["y_m"])
    assert abs(sep) < 0.05  # source and target coincide at merge completion


# --------------------------------------------------------------------------- #
# 12-15: production safety, opt-in, scenario manifest
# --------------------------------------------------------------------------- #
def test_production_global_path_is_unchanged():
    digest = hashlib.sha256(PRODUCTION_PATH.read_bytes()).hexdigest()
    assert digest == PRODUCTION_PATH_SHA256
    corridor_digest = json.loads(ROUTE_CORRIDOR.read_text())["source_sha256"][
        "global_path"
    ]
    assert corridor_digest == PRODUCTION_PATH_SHA256


def test_fixture_is_not_referenced_by_production_config():
    planner_yaml = (REPO / "ad_planner" / "config" / "planner.yaml").read_text()
    assert "test_highway_merge_ego_ramp" not in planner_yaml
    assert "2026_molit_comp_global_path.txt" in planner_yaml


def test_scenario_manifest_is_consistent_with_the_geometry():
    scenario = json.loads(SCENARIO_JSON.read_text())
    lanes, _ = _lanes()
    source = lanes[SOURCE_LANE_ID]["points"]
    ego = scenario["egoVehicle"]["initPosition"]
    assert math.isclose(ego["pos"]["x"], source[0]["x_m"], abs_tol=1e-3)
    assert math.isclose(ego["pos"]["y"], source[0]["y_m"], abs_tol=1e-3)
    tangent_deg = math.degrees(source[0]["yaw_rad"])
    assert abs(
        math.remainder(float(ego["rot"]["yaw"]) - tangent_deg, 360.0)
    ) < 1.0
    assert ego["initLink"] == "A2256W000409"  # the route:0:left:1 source link


def test_opt_in_scenario_launch_builds_the_fixture_stack():
    """The scenario launch loads and its OpaqueFunction produces the 3-node
    stack on the fixture route with a generated corridor whose recorded
    global-path digest matches the fixture (exercising the digest-rewrite path),
    without standing up a graph.
    """
    launch_path = REPO / "ad_planner" / "launch" / "highway_merge_ego_ramp_scenario.launch.py"
    spec = importlib.util.spec_from_file_location("ego_ramp_scenario_launch", launch_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    module.generate_launch_description()  # import-time smoke

    captured = []

    def _fake_node(**kwargs):
        captured.append(kwargs)
        return kwargs

    class _Cfg:
        def __init__(self, name, default=None):
            self._name = name

        def perform(self, _context):
            return str(DATA_DIR) if self._name == "data_dir" else ""

    module.Node = _fake_node
    module.LaunchConfiguration = _Cfg
    result = module._prepare(context=object())

    assert len(result) == 3
    assert [c["executable"] for c in captured] == [
        "ad_highway_merge_gap_risk_node",
        "ad_highway_merge_gap_response_node",
        "ad_planner_node",
    ]
    fixture_digest = hashlib.sha256(FIXTURE_PATH.read_bytes()).hexdigest()
    generated = None
    for node in captured:
        for entry in node["parameters"]:
            if isinstance(entry, dict):
                if "route_corridor_file" in entry:
                    generated = entry["route_corridor_file"]
                if "route_corridor.expected_global_path_sha256" in entry:
                    assert entry["route_corridor.expected_global_path_sha256"] == fixture_digest
                if entry.get("path_file"):
                    assert entry["path_file"].endswith("test_highway_merge_ego_ramp_path.txt")
    assert generated and Path(generated).is_file()
    corridor = json.loads(Path(generated).read_text())
    assert corridor["source_sha256"]["global_path"] == fixture_digest
    original = json.loads(ROUTE_CORRIDOR.read_text())
    assert [lane["lane_sequence_id"] for lane in corridor["lanes"]] == [
        lane["lane_sequence_id"] for lane in original["lanes"]
    ]
    # the planner node also gets the merge feature flags on
    planner_flags = {}
    for entry in captured[-1]["parameters"]:
        if isinstance(entry, dict):
            planner_flags.update(entry)
    assert planner_flags["enable_highway_merge_mission"] is True
    assert planner_flags["enable_highway_merge_response_integration"] is True


def test_fixture_route_is_loadable_shaped():
    """Every fixture line is `x y z` parseable and consecutive points are
    distinct -- the shape DataLoader::load_path expects.
    """
    rows = _fixture_points()
    assert len(rows) > 500
    for a, b in zip(rows, rows[1:]):
        assert (a[0], a[1], a[2]) != (b[0], b[1], b[2])
    # first == ramp start, last is on route:0 well past merge complete
    lanes, _ = _lanes()
    assert math.isclose(rows[0][0], lanes[SOURCE_LANE_ID]["points"][0]["x_m"], abs_tol=1e-9)


def test_scenario_actors_do_not_collide_at_t0():
    scenario = json.loads(SCENARIO_JSON.read_text())
    ego = scenario["egoVehicle"]["initPosition"]["pos"]
    positions = [(ego["x"], ego["y"])]
    for vehicle in scenario["vehicleList"]:
        pos = vehicle["initPosition"]["pos"]
        positions.append((pos["x"], pos["y"]))
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            separation = math.hypot(
                positions[i][0] - positions[j][0],
                positions[i][1] - positions[j][1],
            )
            assert separation > 6.0  # longer than one IONIQ 5
    # no test NPC is placed on the ego's own acceleration lane
    for vehicle in scenario["vehicleList"]:
        pos = vehicle["initPosition"]["pos"]
        _, lateral = _frenet_lateral(
            _lanes()[0][TARGET_LANE_ID]["points"], pos["x"], pos["y"]
        )
        assert abs(lateral) < 2.0  # on route:0, not on route:0:left:1 (~3.9 m)
