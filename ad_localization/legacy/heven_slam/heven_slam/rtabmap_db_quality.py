"""Read-only quality checks for an RTAB-Map SQLite database."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import struct
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


LINK_TYPES = {
    0: "neighbor",
    1: "global_closure",
    2: "local_space_closure",
    3: "local_time_closure",
    4: "user_closure",
    5: "virtual_closure",
    6: "neighbor_merged",
    7: "pose_prior",
    8: "landmark",
    9: "gravity",
}
CLOSURE_TYPES = {1, 2, 3, 4, 5}
GRAPH_TYPES = {0, 1, 2, 3, 4, 5, 6}


@dataclass(frozen=True)
class SessionQuality:
    map_id: int
    nodes: int
    duration_s: float
    path_m: float
    end_gap_m: float
    max_step_m: float
    median_step_m: float
    yaw_change_deg: float
    closed_loop_candidate: bool


@dataclass(frozen=True)
class DatabaseQuality:
    database: str
    nodes: int
    scans: int
    gps_measurements: int
    graph_components: int
    closure_links: int
    link_types: dict[str, int]
    sessions: list[SessionQuality]
    max_neighbor_link_m: float
    checks: dict[str, bool]

    @property
    def ready_for_loop_validation(self) -> bool:
        return all(self.checks.values())


def _translation(blob: bytes) -> tuple[float, float, float]:
    if len(blob) != 48:
        raise ValueError(f"expected a 48-byte RTAB-Map transform, got {len(blob)}")
    transform = struct.unpack("<12f", blob)
    return transform[3], transform[7], transform[11]


def _yaw(blob: bytes) -> float:
    transform = struct.unpack("<12f", blob)
    return math.atan2(transform[4], transform[0])


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _component_count(node_ids: Iterable[int], links: Iterable[tuple[int, int]]) -> int:
    parents = {node_id: node_id for node_id in node_ids}

    def find(node_id: int) -> int:
        while parents[node_id] != node_id:
            parents[node_id] = parents[parents[node_id]]
            node_id = parents[node_id]
        return node_id

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left, right in links:
        if left in parents and right in parents:
            union(left, right)
    return len({find(node_id) for node_id in parents})


def analyze_database(
    database: Path,
    *,
    min_closed_path_m: float = 50.0,
    max_closure_gap_m: float = 5.0,
    max_neighbor_step_m: float = 3.0,
) -> DatabaseQuality:
    """Analyze graph connectivity and trajectory closure without changing the DB."""

    database = database.expanduser().resolve()
    if not database.is_file():
        raise FileNotFoundError(database)

    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        node_rows = connection.execute(
            "SELECT id, map_id, stamp, pose FROM Node "
            "WHERE pose IS NOT NULL ORDER BY map_id, id"
        ).fetchall()
        link_rows = connection.execute(
            "SELECT from_id, to_id, type, transform FROM Link"
        ).fetchall()
        scan_count = connection.execute(
            "SELECT COUNT(*) FROM Data WHERE scan IS NOT NULL"
        ).fetchone()[0]
        gps_count = connection.execute(
            "SELECT COUNT(*) FROM Node WHERE gps IS NOT NULL"
        ).fetchone()[0]
    finally:
        connection.close()

    trajectories: dict[int, list[tuple[float, tuple[float, float, float], float]]] = (
        defaultdict(list)
    )
    for _node_id, map_id, stamp, pose in node_rows:
        trajectories[map_id].append((stamp, _translation(pose), _yaw(pose)))

    sessions = []
    for map_id, trajectory in sorted(trajectories.items()):
        steps = [
            math.dist(previous[1], current[1])
            for previous, current in zip(trajectory, trajectory[1:])
        ]
        path_m = sum(steps)
        end_gap_m = math.dist(trajectory[0][1], trajectory[-1][1])
        yaw_delta = trajectory[-1][2] - trajectory[0][2]
        yaw_delta = math.atan2(math.sin(yaw_delta), math.cos(yaw_delta))
        sessions.append(
            SessionQuality(
                map_id=map_id,
                nodes=len(trajectory),
                duration_s=trajectory[-1][0] - trajectory[0][0],
                path_m=path_m,
                end_gap_m=end_gap_m,
                max_step_m=max(steps, default=0.0),
                median_step_m=_median(steps),
                yaw_change_deg=math.degrees(yaw_delta),
                closed_loop_candidate=(
                    path_m >= min_closed_path_m and end_gap_m <= max_closure_gap_m
                ),
            )
        )

    type_counts = Counter(link_type for _, _, link_type, _ in link_rows)
    graph_links = [
        (from_id, to_id)
        for from_id, to_id, link_type, _ in link_rows
        if link_type in GRAPH_TYPES
    ]
    neighbor_distances = [
        math.dist((0.0, 0.0, 0.0), _translation(transform))
        for _, _, link_type, transform in link_rows
        if link_type in {0, 6} and transform is not None
    ]
    closure_links = sum(type_counts[link_type] for link_type in CLOSURE_TYPES)
    node_ids = [node_id for node_id, _, _, _ in node_rows]
    components = _component_count(node_ids, graph_links)
    max_neighbor_link = max(neighbor_distances, default=0.0)

    checks = {
        "lidar_scan_for_every_node": scan_count == len(node_rows),
        "single_connected_pose_graph": components == 1,
        "loop_closure_detected": closure_links > 0,
        "closed_route_recorded": any(s.closed_loop_candidate for s in sessions),
        "neighbor_motion_within_limit": max_neighbor_link <= max_neighbor_step_m,
    }
    named_link_types = {
        LINK_TYPES.get(link_type, f"unknown_{link_type}"): count
        for link_type, count in sorted(type_counts.items())
    }
    return DatabaseQuality(
        database=str(database),
        nodes=len(node_rows),
        scans=scan_count,
        gps_measurements=gps_count,
        graph_components=components,
        closure_links=closure_links,
        link_types=named_link_types,
        sessions=sessions,
        max_neighbor_link_m=max_neighbor_link,
        checks=checks,
    )


def format_report(report: DatabaseQuality) -> str:
    lines = [
        f"Database: {report.database}",
        (
            f"Nodes/scans/GPS: {report.nodes}/{report.scans}/"
            f"{report.gps_measurements}"
        ),
        (
            f"Graph components: {report.graph_components}, "
            f"closure links: {report.closure_links}"
        ),
        "Sessions:",
    ]
    for session in report.sessions:
        lines.append(
            "  map_id={map_id}: nodes={nodes}, path={path_m:.2f} m, "
            "end_gap={end_gap_m:.2f} m, max_step={max_step_m:.2f} m, "
            "closed={closed_loop_candidate}".format(**asdict(session))
        )
    lines.append("Checks:")
    for name, passed in report.checks.items():
        lines.append(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    lines.append(
        "Verdict: "
        + ("READY for loop-quality evaluation" if report.ready_for_loop_validation else "NOT READY; collect a complete returning route")
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--min-closed-path-m", type=float, default=50.0)
    parser.add_argument("--max-closure-gap-m", type=float, default=5.0)
    parser.add_argument("--max-neighbor-step-m", type=float, default=3.0)
    args = parser.parse_args(argv)
    report = analyze_database(
        args.database,
        min_closed_path_m=args.min_closed_path_m,
        max_closure_gap_m=args.max_closure_gap_m,
        max_neighbor_step_m=args.max_neighbor_step_m,
    )
    if args.as_json:
        payload = asdict(report)
        payload["ready_for_loop_validation"] = report.ready_for_loop_validation
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(format_report(report))
    return 0 if report.ready_for_loop_validation else 2


if __name__ == "__main__":
    raise SystemExit(main())
