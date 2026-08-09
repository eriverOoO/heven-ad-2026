"""Extract selected MGeo link centerlines from a MORAI Unity map bundle.

Generated route data belongs outside the repository and must not be committed.
"""

import argparse
import heapq
import json
import math
from pathlib import Path

import UnityPy


COMMISSIONING_LINKS = [
    "A2256W000172",
    "A2256W000361",
    "A2256W000370",
    "A2256W000360",
]

# Directed MGeo links selected from the rulebook checkpoint order. The route
# returns to the starting area so it can be rotated to any point on the loop.
FULL_COURSE_LINKS = [
    "A2256W000172",
    "A2256W000361",
    "A2256W000370",
    "A2256W000360",
    "A2256W000372-A2256W000375",
    "A2256W000376",
    "A2256W000274",
    "A2256W000266",
    "A2256W000015",
    "A2256W000301",
    "A2256W000074",
    "A2256W000302",
    "A2256W000090",
    "A2256W000246",
    "A2256W000265",
    "A2256W000266",
    "A2256W000015-A2256W000286",
    "A2256W000301-A2256W000282",
    "A2256W000300",
    "A2256W000304",
    "A2256W000148",
    "A2256W000146",
    "A2256W000135",
    "A2256W000132",
    "A2256W000165",
    "A2256W000107",
    "A2256W000600",
    "A2256W000138",
    "A2256W000157",
    "A2256W000868",
    "A2256W000168",
    "A2256W000419",
    "A2256W000415",
    "A2256W000404",
    "A2256W000010-A2256W000175",
    "A2256W000410",
    "A2256W000430-A2256W000431",
    "A2256W000435",
    "A2256W000423",
    "A2256W000153",
    "A2256W000451",
    "A2256W000446",
    "A2256W000448",
    "A2256W000126",
    "A2256W000128",
    "A2256W000333",
    "A2256W000154",
    "A2256W000332",
    "A2256W000751",
    "A2256W000748",
    "A2256W000078",
    "A2256W000324",
    "A2256W000597",
    "A2256W000241",
    "A2256W000243",
    "A2256W000231",
    "A2256W000085",
    "A2256W000234",
    "A2256W000086",
]

ROUTE_PROFILES = {
    "commissioning": COMMISSIONING_LINKS,
    "full-course": FULL_COURSE_LINKS,
}


def load_links(bundle_path: Path) -> dict:
    environment = UnityPy.load(str(bundle_path))
    for obj in environment.objects:
        if obj.type.name != "TextAsset":
            continue
        data = obj.read()
        if data.m_Name != "link_set":
            continue
        script = data.m_Script
        if isinstance(script, (bytes, bytearray)):
            script = script.decode("utf-8")
        return {link["idx"]: link for link in json.loads(script)}
    raise RuntimeError("link_set TextAsset not found in bundle")


def stitch_points(links: dict, link_ids: list[str]) -> list[list[float]]:
    points = []
    for link_id in link_ids:
        link_points = [point[:2] for point in links[link_id]["points"]]
        if points and link_points:
            join_gap = math.dist(points[-1], link_points[0])
            if join_gap > 25.0:
                raise RuntimeError(
                    f"route discontinuity before {link_id}: {join_gap:.2f} m"
                )
            if join_gap < 0.01:
                link_points = link_points[1:]
        points.extend(link_points)
    return points


def rotate_closed_route(
    points: list[list[float]], start_x: float, start_y: float
) -> list[list[float]]:
    closure_gap = math.dist(points[-1], points[0])
    if closure_gap > 25.0:
        raise RuntimeError(f"route is not closed: endpoint gap is {closure_gap:.2f} m")
    nearest = min(
        range(len(points)),
        key=lambda index: math.dist(points[index], (start_x, start_y)),
    )
    return points[nearest:] + points[: nearest + 1]


def densify_points(
    points: list[list[float]], max_step_m: float = 0.5
) -> list[list[float]]:
    dense = [points[0]]
    for left, right in zip(points, points[1:]):
        distance = math.dist(left, right)
        segments = max(1, math.ceil(distance / max_step_m))
        for segment in range(1, segments + 1):
            ratio = segment / segments
            dense.append(
                [
                    left[0] + (right[0] - left[0]) * ratio,
                    left[1] + (right[1] - left[1]) * ratio,
                ]
            )
    return dense


def shortest_entry_path(
    links: dict, entry_link: str, route_links: list[str]
) -> tuple[list[str], int]:
    """Return a directed path from entry_link to the nearest full-route link."""
    if entry_link not in links:
        raise RuntimeError(f"entry link {entry_link} does not exist")
    from_node = {}
    for link_id, link in links.items():
        from_node.setdefault(link["from_node_idx"], []).append(link_id)
    distances = {entry_link: 0.0}
    previous = {}
    queue = [(0.0, entry_link)]
    while queue:
        distance, current = heapq.heappop(queue)
        if distance != distances[current]:
            continue
        for target in from_node.get(links[current]["to_node_idx"], []):
            cost = float(links[target].get("link_length", 0.0))
            candidate = distance + cost
            if candidate < distances.get(target, math.inf):
                distances[target] = candidate
                previous[target] = current
                heapq.heappush(queue, (candidate, target))
    reachable = [
        (distances[link_id], index, link_id)
        for index, link_id in enumerate(route_links)
        if link_id in distances
    ]
    if not reachable:
        raise RuntimeError(f"no directed path from {entry_link} to the full route")
    _, join_index, join_link = min(reachable)
    path = [join_link]
    while path[-1] != entry_link:
        path.append(previous[path[-1]])
    path.reverse()
    return path, join_index


def link_ranges_for_points(links: dict, link_ids: list[str]) -> tuple[list, list]:
    """Stitch/densify links while retaining point-index ranges for each link."""
    points = []
    point_links = []
    for link_id in link_ids:
        link_points = [point[:2] for point in links[link_id]["points"]]
        if points and link_points:
            join_gap = math.dist(points[-1], link_points[0])
            if join_gap > 25.0:
                raise RuntimeError(
                    f"route discontinuity before {link_id}: {join_gap:.2f} m"
                )
            if join_gap < 0.01:
                link_points = link_points[1:]
        for point in link_points:
            points.append(point)
            point_links.append(link_id)
    dense = [points[0]]
    dense_links = [point_links[0]]
    for left, right, right_link in zip(points, points[1:], point_links[1:]):
        distance = math.dist(left, right)
        segments = max(1, math.ceil(distance / 0.5))
        for segment in range(1, segments + 1):
            ratio = segment / segments
            dense.append(
                [
                    left[0] + (right[0] - left[0]) * ratio,
                    left[1] + (right[1] - left[1]) * ratio,
                ]
            )
            dense_links.append(right_link)
    ranges = []
    start = 0
    for index in range(1, len(dense_links) + 1):
        if index == len(dense_links) or dense_links[index] != dense_links[start]:
            ranges.append(
                {
                    "link_id": dense_links[start],
                    "start_index": start,
                    "end_index": index - 1,
                }
            )
            start = index
    return dense, ranges


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--profile",
        choices=ROUTE_PROFILES,
        default="commissioning",
    )
    parser.add_argument("--start-x", type=float)
    parser.add_argument("--start-y", type=float)
    parser.add_argument(
        "--entry-link",
        help="prepend a directed path from the vehicle's current MORAI link",
    )
    args = parser.parse_args()
    if (args.start_x is None) != (args.start_y is None):
        parser.error("--start-x and --start-y must be used together")
    links = load_links(args.bundle)
    link_ids = ROUTE_PROFILES[args.profile]
    if args.entry_link:
        if args.profile != "full-course":
            parser.error("--entry-link requires --profile full-course")
        entry_path, join_index = shortest_entry_path(links, args.entry_link, link_ids)
        # Enter the loop at the nearest reachable link, drive one complete
        # circuit, and finish on that join link.
        link_ids = (
            entry_path
            + link_ids[join_index + 1 :]
            + link_ids[: join_index + 1]
        )
        points, link_ranges = link_ranges_for_points(links, link_ids)
    else:
        points = stitch_points(links, link_ids)
        link_ranges = []
    if args.profile == "full-course" and not args.entry_link:
        closure_gap = math.dist(points[-1], points[0])
        if closure_gap > 25.0:
            raise RuntimeError(
                f"full-course route endpoint gap is {closure_gap:.2f} m"
            )
        if closure_gap >= 0.01:
            points.append(points[0])
    if not args.entry_link:
        points = densify_points(points)
    if args.start_x is not None:
        if args.entry_link:
            parser.error("start rotation cannot be combined with --entry-link")
        if args.profile != "full-course":
            parser.error("start rotation requires --profile full-course")
        points = rotate_closed_route(points, args.start_x, args.start_y)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "purpose": args.profile,
                "source_links": link_ids,
                "points": points,
                "link_ranges": link_ranges,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    length = sum(math.dist(left, right) for left, right in zip(points, points[1:]))
    print(
        f"wrote {len(points)} points ({length:.1f} m, "
        f"closure gap {math.dist(points[0], points[-1]):.2f} m) to {args.output}"
    )


if __name__ == "__main__":
    main()
