"""Architecture-lock test for Highway Merge Mission Primitive v1.

CONCLUSION (locked here as a regression): CASE B - the competition global path
does NOT encode a ramp -> mainline merge maneuver. The ego is never on the
acceleration lane.

Evidence, asserted below against the real committed data
(ad_data/path/2026_molit_comp_global_path.txt + ad_data/map/route_corridor.json):

  * the global path lies on the mainline `route:0` centerline (lateral offset
    ~0.0 m) at every station across the merge region (1080-1320 m route_s);
  * the global path is ~3.9 m away from the acceleration lane `route:0:left:1`
    at the merge-zone entry, closing to ~0 only where that lane tapers into
    `route:0` at merge-complete - i.e. the ego drives straight down the
    mainline while `route:0:left:1` merges into it from the side.

So there is no lateral transition in the global path to reuse, and no lateral
maneuver for the ego to perform on the committed route. The mission primitive
therefore tracks mission STATE and exposes a source/target corridor intent for
a future lateral executor; it generates no path.

Also locked: the source-grounded commit boundary derivation (first
source-lane station whose lateral separation from `route:0` falls below
`commit_lateral_separation_m` = 3.5 m) lands inside (zone_entry, merge_complete)
near route_s ~= 1213 m.

If a future route reroutes the ego via `route:0:left:1`, this test fails and
the CASE A / CASE B decision must be revisited.
"""

import json
import math
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
REPO = PACKAGE.parent
COMMIT_LATERAL_SEPARATION_M = 3.5  # config/planner.yaml default


def _load_global_path():
    text = (REPO / "ad_data" / "path" / "2026_molit_comp_global_path.txt").read_text(
        encoding="utf-8"
    )
    points = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.replace(",", " ").split()
        points.append((float(parts[0]), float(parts[1])))
    return points


def _lanes():
    corridor = json.loads(
        (REPO / "ad_data" / "map" / "route_corridor.json").read_text(encoding="utf-8")
    )
    return {lane["lane_sequence_id"]: lane["points"] for lane in corridor["lanes"]}


def _project(polyline, x, y):
    """Nearest point on a route-corridor polyline: (distance, route_s_m)."""
    best = None
    for i in range(1, len(polyline)):
        ax, ay = polyline[i - 1]["x_m"], polyline[i - 1]["y_m"]
        bx, by = polyline[i]["x_m"], polyline[i]["y_m"]
        dx, dy = bx - ax, by - ay
        length_sq = dx * dx + dy * dy
        if length_sq == 0.0:
            continue
        t = max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / length_sq))
        px, py = ax + t * dx, ay + t * dy
        dist = math.hypot(x - px, y - py)
        if best is None or dist < best[0]:
            route_s = polyline[i - 1]["route_s_m"] + t * (
                polyline[i]["route_s_m"] - polyline[i - 1]["route_s_m"]
            )
            best = (dist, route_s)
    return best


def test_global_path_is_the_mainline_not_the_acceleration_lane():
    lanes = _lanes()
    route0 = lanes["route:0"]
    accel = lanes["route:0:left:1"]
    gp = _load_global_path()

    # For each global-path point whose route:0 projection lands in the merge
    # region, record the lateral distance to route:0 and to route:0:left:1.
    samples = {}
    for x, y in gp:
        d0, s0 = _project(route0, x, y)
        if d0 > 6.0 or not (1080.0 <= s0 <= 1320.0):
            continue
        for target_s in (1120, 1160, 1200, 1240, 1286):
            if target_s in samples or abs(s0 - target_s) > 0.6:
                continue
            d_accel, _ = _project(accel, x, y)
            samples[target_s] = (d0, d_accel)

    assert set(samples) == {1120, 1160, 1200, 1240, 1286}, samples

    # CASE B evidence 1: the global path IS route:0 (lateral ~ 0) everywhere in
    # the merge region.
    for target_s, (d_route0, _) in samples.items():
        assert d_route0 < 0.5, f"global path is off route:0 at s={target_s}: {d_route0}"

    # CASE B evidence 2: the global path is ~3.9 m from route:0:left:1 at the
    # zone entry and only converges where that lane tapers into route:0.
    assert samples[1120][1] > 3.0, samples[1120]
    assert samples[1200][1] > 3.0, samples[1200]
    assert samples[1286][1] < 0.5, samples[1286]  # accel lane has merged by here


def test_commit_boundary_is_source_grounded_and_inside_the_zone():
    lanes = _lanes()
    route0 = lanes["route:0"]
    accel = lanes["route:0:left:1"]
    document = json.loads(
        (PACKAGE / "config" / "highway_merge.json").read_text(encoding="utf-8")
    )
    zone = document["merge_zones"][0]
    zone_entry = zone["route_s_zone_entry_m"]
    merge_complete = zone["route_s_merge_complete_m"]

    commit = None
    for point in accel:
        separation, _ = _project(route0, point["x_m"], point["y_m"])
        if separation < COMMIT_LATERAL_SEPARATION_M:
            commit = point["route_s_m"]
            break

    assert commit is not None, "the acceleration lane never taper reaches route:0"
    assert zone_entry < commit < merge_complete
    # Taper onset for kcity_highway_onramp.
    assert 1205.0 < commit < 1225.0, commit
    # A meaningful committed span remains (~9 s at 8 m/s).
    assert merge_complete - commit > 60.0


def test_merge_zone_is_the_only_configured_zone():
    document = json.loads(
        (PACKAGE / "config" / "highway_merge.json").read_text(encoding="utf-8")
    )
    assert len(document["merge_zones"]) == 1
    zone = document["merge_zones"][0]
    assert zone["merge_zone_id"] == "kcity_highway_onramp"
    assert zone["source_lane_sequence_id"] == "route:0:left:1"
    assert zone["target_lane_sequence_id"] == "route:0"
