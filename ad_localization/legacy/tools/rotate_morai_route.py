#!/usr/bin/env python3
"""Rotate an already extracted closed MORAI route to a new start position."""

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int)
    parser.add_argument("--start-x", type=float, required=True)
    parser.add_argument("--start-y", type=float, required=True)
    args = parser.parse_args()

    document = json.loads(args.input.read_text(encoding="utf-8"))
    all_points = document["points"]
    end_index = len(all_points) - 1 if args.end_index is None else args.end_index
    if not 0 <= args.start_index < end_index < len(all_points):
        raise RuntimeError("route slice indices are invalid")
    points = all_points[args.start_index : end_index + 1]
    closure_gap = math.dist(points[0], points[-1])
    if closure_gap > 0.01:
        raise RuntimeError(f"selected route is not closed: {closure_gap:.3f} m")

    point_links = [None] * len(all_points)
    for item in document.get("link_ranges", []):
        for index in range(int(item["start_index"]), int(item["end_index"]) + 1):
            point_links[index] = item["link_id"]
    labels = point_links[args.start_index : end_index + 1]
    if any(label is None for label in labels):
        raise RuntimeError("selected route contains points without link metadata")

    # Exclude the duplicated closure point while choosing and rotating, then
    # append the selected start point so the result is exactly closed.
    nearest = min(
        range(len(points) - 1),
        key=lambda index: math.dist(
            points[index], (args.start_x, args.start_y)
        ),
    )
    rotated_points = points[nearest:-1] + points[: nearest + 1]
    rotated_labels = labels[nearest:-1] + labels[: nearest + 1]

    ranges = []
    range_start = 0
    for index in range(1, len(rotated_labels) + 1):
        if (
            index == len(rotated_labels)
            or rotated_labels[index] != rotated_labels[range_start]
        ):
            ranges.append({
                "link_id": rotated_labels[range_start],
                "start_index": range_start,
                "end_index": index - 1,
            })
            range_start = index

    result = {
        "purpose": "full-course-closed-loop",
        "source": str(args.input),
        "points": rotated_points,
        "link_ranges": ranges,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    length = sum(
        math.dist(left, right)
        for left, right in zip(rotated_points, rotated_points[1:])
    )
    print(
        f"wrote {len(rotated_points)} points ({length:.1f} m, "
        f"closure {math.dist(rotated_points[0], rotated_points[-1]):.3f} m, "
        f"rotation index {nearest}) to {args.output}"
    )


if __name__ == "__main__":
    main()
