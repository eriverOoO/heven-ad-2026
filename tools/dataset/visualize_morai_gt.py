#!/usr/bin/env python3
"""Numerically validate and render MORAI lidar-frame oriented 3D boxes."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import numpy as np
from PIL import Image, ImageDraw


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


BOX_FIELDS = ("x", "y", "z", "length", "width", "height", "yaw")
CLASS_COLORS = {"vehicle": "tab:blue", "pedestrian": "tab:orange", "obstacle": "tab:red"}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sample-count", type=int, default=48)
    parser.add_argument("--very-small-threshold", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_points(dataset: Path, label: dict[str, Any]) -> np.ndarray:
    point_info = label["points"]
    points = np.fromfile(dataset / point_info["path"], dtype="<f4")
    if points.size % 4:
        raise ValueError(f"invalid XYZI byte count for {label['sample_id']}")
    points = points.reshape(-1, 4)
    if len(points) != int(point_info["finite_count"]):
        raise ValueError(f"point count mismatch for {label['sample_id']}")
    return points


def points_inside_box(
    points: np.ndarray,
    box: dict[str, Any],
    *,
    yaw: float | None = None,
    length: float | None = None,
    width: float | None = None,
    center_xy: tuple[float, float] | None = None,
    margin: float = 0.0,
) -> np.ndarray:
    """Return a strict oriented 3D-box membership mask."""
    box_yaw = float(box["yaw"] if yaw is None else yaw)
    box_length = float(box["length"] if length is None else length)
    box_width = float(box["width"] if width is None else width)
    center_x, center_y = center_xy or (float(box["x"]), float(box["y"]))
    dx = points[:, 0] - center_x
    dy = points[:, 1] - center_y
    cosine, sine = math.cos(box_yaw), math.sin(box_yaw)
    local_x = cosine * dx + sine * dy
    local_y = -sine * dx + cosine * dy
    local_z = points[:, 2] - float(box["z"])
    return (
        (np.abs(local_x) <= box_length / 2.0 + margin)
        & (np.abs(local_y) <= box_width / 2.0 + margin)
        & (np.abs(local_z) <= float(box["height"]) / 2.0 + margin)
    )


def _materially_better(candidate: int, nominal: int) -> bool:
    return candidate >= nominal + 5 and candidate >= 1.5 * max(nominal, 1)


def analyze_object(
    sample_id: str,
    box_index: int,
    box: dict[str, Any],
    points: np.ndarray,
    roi: dict[str, list[float]],
    very_small_threshold: int,
) -> dict[str, Any]:
    values = [float(box[field]) for field in BOX_FIELDS]
    finite = all(math.isfinite(value) for value in values)
    valid_dimensions = finite and min(values[3:6]) > 0.0
    center_in_roi = finite and all(
        float(roi[axis][0]) <= float(box[axis]) <= float(roi[axis][1])
        for axis in ("x", "y", "z")
    )
    if not finite or not valid_dimensions:
        nominal = expanded = swapped = negated = rear_origin = 0
    else:
        nominal = int(points_inside_box(points, box).sum())
        expanded = int(points_inside_box(points, box, margin=0.25).sum())
        swapped = int(
            points_inside_box(
                points,
                box,
                length=float(box["width"]),
                width=float(box["length"]),
            ).sum()
        )
        negated = int(points_inside_box(points, box, yaw=-float(box["yaw"])).sum())
        offset = float(box.get("forward_center_offset_m", 0.0))
        rear_center = (
            float(box["x"]) - offset * math.cos(float(box["yaw"])),
            float(box["y"]) - offset * math.sin(float(box["yaw"])),
        )
        rear_origin = int(points_inside_box(points, box, center_xy=rear_center).sum())
    distance = math.hypot(float(box["x"]), float(box["y"])) if finite else math.nan
    exported_visibility = box.get("num_lidar_points_inside_box")
    visibility_matches = (
        exported_visibility is None or int(exported_visibility) == nominal
    )
    return {
        "sample_id": sample_id,
        "box_index": box_index,
        "actor_id": box.get("actor_id"),
        "class_name": box.get("class_name"),
        "x": box.get("x"),
        "y": box.get("y"),
        "z": box.get("z"),
        "length": box.get("length"),
        "width": box.get("width"),
        "height": box.get("height"),
        "yaw": box.get("yaw"),
        "distance_m": distance,
        "inside_points": nominal,
        "exported_num_lidar_points_inside_box": exported_visibility,
        "exported_visibility_matches": visibility_matches,
        "inside_points_margin_0_25m": expanded,
        "inside_points_swapped_lw": swapped,
        "inside_points_negated_yaw": negated,
        "inside_points_actor_origin_xy": rear_origin,
        "zero_points": nominal == 0,
        "very_small_points": 0 < nominal <= very_small_threshold,
        "margin_materially_better": _materially_better(expanded, nominal),
        "swap_materially_better": _materially_better(swapped, nominal),
        "negated_yaw_materially_better": _materially_better(negated, nominal),
        "actor_origin_materially_better": _materially_better(rear_origin, nominal),
        "finite": finite,
        "valid_dimensions": valid_dimensions,
        "center_in_roi": center_in_roi,
    }


def _box_corners(
    box: dict[str, Any], *, center_xy: tuple[float, float] | None = None
) -> np.ndarray:
    half_length = float(box["length"]) / 2.0
    half_width = float(box["width"]) / 2.0
    local = np.array(
        [
            [half_length, half_width],
            [half_length, -half_width],
            [-half_length, -half_width],
            [-half_length, half_width],
        ]
    )
    cosine, sine = math.cos(float(box["yaw"])), math.sin(float(box["yaw"]))
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    center = center_xy or (float(box["x"]), float(box["y"]))
    return local @ rotation.T + np.array(center)


def _draw_bev(
    axis: Any,
    points: np.ndarray,
    boxes: list[dict[str, Any]],
    object_rows: list[dict[str, Any]],
    limits: tuple[tuple[float, float], tuple[float, float]],
    title: str,
    show_rear_axle_hypothesis: bool = False,
) -> None:
    axis.scatter(points[:, 0], points[:, 1], s=0.25, c=points[:, 2], cmap="Greys", alpha=0.5)
    for box, row in zip(boxes, object_rows):
        color = CLASS_COLORS.get(str(box["class_name"]), "magenta")
        corners = _box_corners(box)
        closed = np.vstack([corners, corners[0]])
        axis.plot(closed[:, 0], closed[:, 1], color=color, linewidth=1.5)
        center = (float(box["x"]), float(box["y"]))
        axis.scatter(*center, s=14, color=color, marker="x")
        arrow_length = min(3.0, max(1.0, float(box["length"]) / 2.0))
        axis.arrow(
            center[0],
            center[1],
            arrow_length * math.cos(float(box["yaw"])),
            arrow_length * math.sin(float(box["yaw"])),
            width=0.04,
            head_width=0.35,
            color=color,
            length_includes_head=True,
        )
        axis.text(
            center[0],
            center[1],
            f"{box['class_name']}:{row['inside_points']}",
            fontsize=6,
            color=color,
        )
        offset = float(box.get("forward_center_offset_m", 0.0))
        if show_rear_axle_hypothesis and box["class_name"] == "vehicle" and offset:
            rear_center = (
                center[0] - offset * math.cos(float(box["yaw"])),
                center[1] - offset * math.sin(float(box["yaw"])),
            )
            rear_corners = _box_corners(box, center_xy=rear_center)
            rear_closed = np.vstack([rear_corners, rear_corners[0]])
            axis.plot(
                rear_closed[:, 0],
                rear_closed[:, 1],
                color="tab:purple",
                linewidth=1.0,
                linestyle="--",
                alpha=0.8,
            )
            axis.scatter(*rear_center, s=16, color="tab:purple", marker="+")
    axis.scatter([0.0], [0.0], marker="*", s=60, color="lime", edgecolor="black", label="LiDAR")
    axis.set_xlim(*limits[0])
    axis.set_ylim(*limits[1])
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x forward [m]")
    axis.set_ylabel("y left [m]")
    axis.set_title(title)
    axis.grid(alpha=0.2)


def render_frame(
    path: Path,
    label: dict[str, Any],
    points: np.ndarray,
    object_rows: list[dict[str, Any]],
    categories: Iterable[str],
    roi: dict[str, list[float]],
) -> None:
    boxes = label["ground_truth"]["boxes"]
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    _draw_bev(
        axes[0, 0],
        points,
        boxes,
        object_rows,
        ((float(roi["x"][0]), float(roi["x"][1])), (float(roi["y"][0]), float(roi["y"][1]))),
        "Full configured ROI",
    )
    if boxes:
        corners = np.vstack([_box_corners(box) for box in boxes])
        x_limits = (max(float(roi["x"][0]), float(corners[:, 0].min()) - 8.0), min(float(roi["x"][1]), float(corners[:, 0].max()) + 8.0))
        y_limits = (max(float(roi["y"][0]), float(corners[:, 1].min()) - 8.0), min(float(roi["y"][1]), float(corners[:, 1].max()) + 8.0))
    else:
        x_limits, y_limits = (-4.0, 40.0), (-20.0, 20.0)
    _draw_bev(
        axes[0, 1],
        points,
        boxes,
        object_rows,
        (x_limits, y_limits),
        "Object-focused BEV (purple dashed: actor/rear-axle origin hypothesis)",
        show_rear_axle_hypothesis=True,
    )

    axes[1, 0].scatter(points[:, 0], points[:, 2], s=0.25, color="0.35", alpha=0.5)
    for box, row in zip(boxes, object_rows):
        color = CLASS_COLORS.get(str(box["class_name"]), "magenta")
        corners = _box_corners(box)
        x_min, x_max = float(corners[:, 0].min()), float(corners[:, 0].max())
        z_min = float(box["z"]) - float(box["height"]) / 2.0
        rectangle = plt.Rectangle(
            (x_min, z_min),
            x_max - x_min,
            float(box["height"]),
            fill=False,
            color=color,
            linewidth=1.5,
        )
        axes[1, 0].add_patch(rectangle)
        axes[1, 0].text(x_min, z_min, str(row["inside_points"]), fontsize=6, color=color)
    axes[1, 0].set_xlim(float(roi["x"][0]), float(roi["x"][1]))
    axes[1, 0].set_ylim(float(roi["z"][0]) - 1.0, float(roi["z"][1]) + 1.0)
    axes[1, 0].set_xlabel("x forward [m]")
    axes[1, 0].set_ylabel("z up [m]")
    axes[1, 0].set_title("Side projection")
    axes[1, 0].grid(alpha=0.2)

    axes[1, 1].axis("off")
    lines = [
        label["sample_id"],
        f"categories: {', '.join(sorted(categories))}",
        f"points: {len(points)}  boxes: {len(boxes)}",
        "class actor  range  yaw  L/W/H  in  +.25  swap  -yaw  rear",
    ]
    for box, row in zip(boxes, object_rows):
        lines.append(
            f"{box['class_name'][:4]:4s} {int(box['actor_id']):5d} "
            f"{row['distance_m']:5.1f} {float(box['yaw']):+5.2f} "
            f"{float(box['length']):.1f}/{float(box['width']):.1f}/{float(box['height']):.1f} "
            f"{row['inside_points']:4d} {row['inside_points_margin_0_25m']:5d} "
            f"{row['inside_points_swapped_lw']:5d} {row['inside_points_negated_yaw']:5d} "
            f"{row['inside_points_actor_origin_xy']:5d}"
        )
    axes[1, 1].text(0.0, 1.0, "\n".join(lines), va="top", family="monospace", fontsize=8)
    figure.suptitle("MORAI GT validation: center + heading + oriented box", fontsize=12)
    figure.savefig(path, dpi=130)
    plt.close(figure)


def frame_categories(boxes: list[dict[str, Any]], rows: list[dict[str, Any]]) -> set[str]:
    result = set()
    if any(row["distance_m"] < 20.0 for row in rows):
        result.add("near")
    if any(row["distance_m"] >= 40.0 for row in rows):
        result.add("far")
    if any(float(box["y"]) >= 3.0 for box in boxes):
        result.add("left")
    if any(float(box["y"]) <= -3.0 for box in boxes):
        result.add("right")
    if any(
        box["class_name"] == "vehicle" and abs(math.sin(float(box["yaw"]))) >= 0.5
        for box in boxes
    ):
        result.add("rotated_vehicle")
    if len(boxes) >= 3:
        result.add("multi_object")
    if any(row["zero_points"] for row in rows):
        result.add("zero_points")
    if any(row["very_small_points"] for row in rows):
        result.add("very_small_points")
    if any(row["margin_materially_better"] for row in rows):
        result.add("margin_sensitive")
    if any(box["class_name"] == "pedestrian" and row["inside_points"] > 0 for box, row in zip(boxes, rows)):
        result.add("visible_pedestrian")
    if any(box["class_name"] == "obstacle" and row["inside_points"] > 0 for box, row in zip(boxes, rows)):
        result.add("visible_obstacle")
    if any(
        row["swap_materially_better"]
        or row["negated_yaw_materially_better"]
        or row["actor_origin_materially_better"]
        for row in rows
    ):
        result.add("alternative_better")
    if not boxes:
        result.add("empty_gt")
    return result


def _evenly_spaced(values: list[str], count: int) -> list[str]:
    if len(values) <= count:
        return values
    indices = np.linspace(0, len(values) - 1, count, dtype=int)
    return [values[int(index)] for index in indices]


def select_frames(
    ordered_ids: list[str], categories: dict[str, set[str]], count: int
) -> tuple[list[str], dict[str, list[str]]]:
    priority = [
        "visible_pedestrian",
        "visible_obstacle",
        "margin_sensitive",
        "zero_points",
        "very_small_points",
        "alternative_better",
        "near",
        "far",
        "left",
        "right",
        "rotated_vehicle",
        "multi_object",
    ]
    per_category = max(4, math.ceil(count / max(len(priority), 1)))
    selected: list[str] = []
    category_selection: dict[str, list[str]] = {}
    for category in priority:
        candidates = [sample_id for sample_id in ordered_ids if category in categories[sample_id]]
        chosen = _evenly_spaced(candidates, per_category)
        category_selection[category] = chosen
        for sample_id in chosen:
            if sample_id not in selected:
                selected.append(sample_id)
    for sample_id in _evenly_spaced(ordered_ids, count):
        if len(selected) >= count:
            break
        if sample_id not in selected:
            selected.append(sample_id)
    return selected[:count], category_selection


def create_contact_sheets(render_paths: list[Path], output: Path) -> list[Path]:
    sheets = []
    for sheet_index, start in enumerate(range(0, len(render_paths), 12), start=1):
        batch = render_paths[start : start + 12]
        thumbnails = []
        for path in batch:
            image = Image.open(path).convert("RGB")
            image.thumbnail((600, 390))
            thumbnails.append((path.stem, image.copy()))
        row_count = math.ceil(len(thumbnails) / 2)
        canvas = Image.new("RGB", (1200, row_count * 430), "white")
        drawing = ImageDraw.Draw(canvas)
        for index, (name, image) in enumerate(thumbnails):
            x = (index % 2) * 600
            y = (index // 2) * 430
            canvas.paste(image, (x, y + 20))
            drawing.text((x + 5, y + 3), name, fill="black")
        path = output / f"contact_sheet_{sheet_index:02d}.jpg"
        canvas.save(path, quality=88)
        sheets.append(path)
    return sheets


def _prepare_output(output: Path, overwrite: bool) -> None:
    marker = output / ".morai_gt_validation"
    if output.exists():
        if not overwrite:
            raise RuntimeError(f"output exists: {output}; pass --overwrite")
        if not marker.is_file():
            raise RuntimeError(f"refusing to replace unrecognized directory: {output}")
        shutil.rmtree(output)
    (output / "renders").mkdir(parents=True)
    marker.write_text("HEVEN MORAI GT visual validation\n", encoding="utf-8")


def _percentiles(values: list[int]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "p10": None, "p50": None, "p90": None, "p95": None, "max": None}
    return {
        "min": float(min(values)),
        "p10": float(np.percentile(values, 10)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": float(max(values)),
    }


def main() -> int:
    arguments = parse_arguments()
    dataset = arguments.dataset.resolve()
    output = arguments.output.resolve()
    _prepare_output(output, arguments.overwrite)
    metadata = json.loads((dataset / "metadata.json").read_text())
    roi = metadata["evaluation_roi_lidar_m"]
    label_paths = sorted((dataset / "labels").glob("*.json"))
    object_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    labels: dict[str, dict[str, Any]] = {}
    categories: dict[str, set[str]] = {}
    map_to_odom_values = []
    by_class: dict[str, list[int]] = defaultdict(list)
    by_distance: dict[str, list[int]] = defaultdict(list)
    for label_path in label_paths:
        label = json.loads(label_path.read_text())
        sample_id = label["sample_id"]
        labels[sample_id] = label
        points = load_points(dataset, label)
        boxes = label["ground_truth"]["boxes"]
        rows = [
            analyze_object(sample_id, index, box, points, roi, arguments.very_small_threshold)
            for index, box in enumerate(boxes)
        ]
        object_rows.extend(rows)
        for row in rows:
            by_class[str(row["class_name"])].append(int(row["inside_points"]))
            distance = float(row["distance_m"])
            band = "0-20m" if distance < 20.0 else "20-40m" if distance < 40.0 else "40-60m" if distance < 60.0 else "60m+"
            by_distance[band].append(int(row["inside_points"]))
        frame_category = frame_categories(boxes, rows)
        categories[sample_id] = frame_category
        transform = label["transform_alignment"]["map_to_odom"]
        quaternion = transform["quaternion_xyzw"]
        yaw = math.atan2(
            2.0 * (quaternion[3] * quaternion[2] + quaternion[0] * quaternion[1]),
            1.0 - 2.0 * (quaternion[1] ** 2 + quaternion[2] ** 2),
        )
        map_to_odom_values.append((*transform["translation"], yaw))
        frame_rows.append(
            {
                "sample_id": sample_id,
                "point_count": len(points),
                "object_count": len(boxes),
                "zero_point_objects": sum(row["zero_points"] for row in rows),
                "very_small_point_objects": sum(row["very_small_points"] for row in rows),
                "categories": ";".join(sorted(frame_category)),
            }
        )

    ordered_ids = [row["sample_id"] for row in frame_rows]
    selected, category_selection = select_frames(ordered_ids, categories, arguments.sample_count)
    rows_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in object_rows:
        rows_by_sample[row["sample_id"]].append(row)
    render_paths = []
    for sample_id in selected:
        label = labels[sample_id]
        points = load_points(dataset, label)
        render_path = output / "renders" / f"{sample_id}.png"
        render_frame(render_path, label, points, rows_by_sample[sample_id], categories[sample_id], roi)
        render_paths.append(render_path)
    sheets = create_contact_sheets(render_paths, output)

    with (output / "object_stats.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(object_rows[0]) if object_rows else [])
        if object_rows:
            writer.writeheader()
            writer.writerows(object_rows)
    with (output / "frame_stats.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(frame_rows[0]) if frame_rows else [])
        if frame_rows:
            writer.writeheader()
            writer.writerows(frame_rows)
    (output / "selected_samples.txt").write_text(
        "".join(f"{sample_id}\n" for sample_id in selected), encoding="utf-8"
    )

    zero_rows = [row for row in object_rows if row["zero_points"]]
    small_rows = [row for row in object_rows if row["very_small_points"]]
    suspicious_rows = [
        row
        for row in object_rows
        if row["margin_materially_better"]
        or row["swap_materially_better"]
        or row["negated_yaw_materially_better"]
        or row["actor_origin_materially_better"]
        or not row["finite"]
        or not row["valid_dimensions"]
        or not row["center_in_roi"]
        or not row["exported_visibility_matches"]
    ]
    review_rows = []
    for row in object_rows:
        reasons = []
        for field, reason in (
            ("zero_points", "zero_points"),
            ("very_small_points", "inside_points_1_to_threshold"),
            ("margin_materially_better", "margin_0_25m_materially_better"),
            ("swap_materially_better", "swapped_lw_materially_better"),
            ("negated_yaw_materially_better", "negated_yaw_materially_better"),
            ("actor_origin_materially_better", "actor_origin_materially_better"),
        ):
            if row[field]:
                reasons.append(reason)
        if not row["finite"]:
            reasons.append("nonfinite_box")
        if not row["valid_dimensions"]:
            reasons.append("invalid_dimensions")
        if not row["center_in_roi"]:
            reasons.append("center_outside_roi")
        if not row["exported_visibility_matches"]:
            reasons.append("exported_visibility_mismatch")
        if reasons:
            review_rows.append({**row, "review_reasons": ";".join(reasons)})
    with (output / "human_review_samples.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(review_rows[0]) if review_rows else [])
        if review_rows:
            writer.writeheader()
            writer.writerows(review_rows)
    transforms = np.asarray(map_to_odom_values, dtype=float)
    transform_steps = np.diff(transforms, axis=0) if len(transforms) > 1 else np.empty((0, 4))
    summary = {
        "frames_analyzed": len(frame_rows),
        "objects_analyzed": len(object_rows),
        "frames_rendered": len(selected),
        "rendered_samples": selected,
        "render_category_selection": category_selection,
        "zero_point_objects": len(zero_rows),
        "very_small_point_objects_1_to_threshold": len(small_rows),
        "very_small_threshold": arguments.very_small_threshold,
        "objects_requiring_human_review": len(review_rows),
        "hypothesis_or_geometry_review_objects": len(suspicious_rows),
        "human_review_samples": sorted({row["sample_id"] for row in review_rows}),
        "invalid": {
            "nonfinite_boxes": sum(not row["finite"] for row in object_rows),
            "invalid_dimensions": sum(not row["valid_dimensions"] for row in object_rows),
            "centers_outside_roi": sum(not row["center_in_roi"] for row in object_rows),
            "exported_visibility_mismatches": sum(
                not row["exported_visibility_matches"] for row in object_rows
            ),
        },
        "alternative_hypotheses": {
            "margin_0_25m_materially_better": sum(row["margin_materially_better"] for row in object_rows),
            "swapped_lw_materially_better": sum(row["swap_materially_better"] for row in object_rows),
            "negated_yaw_materially_better": sum(row["negated_yaw_materially_better"] for row in object_rows),
            "actor_origin_materially_better": sum(row["actor_origin_materially_better"] for row in object_rows),
        },
        "inside_points": {
            "overall": _percentiles([int(row["inside_points"]) for row in object_rows]),
            "by_class": {key: _percentiles(values) for key, values in sorted(by_class.items())},
            "by_distance": {key: _percentiles(values) for key, values in sorted(by_distance.items())},
        },
        "yaw_rad": {
            "min": min((float(row["yaw"]) for row in object_rows), default=None),
            "max": max((float(row["yaw"]) for row in object_rows), default=None),
        },
        "map_to_odom": {
            "translation_min": transforms[:, :3].min(axis=0).tolist() if len(transforms) else [],
            "translation_max": transforms[:, :3].max(axis=0).tolist() if len(transforms) else [],
            "yaw_min": float(transforms[:, 3].min()) if len(transforms) else None,
            "yaw_max": float(transforms[:, 3].max()) if len(transforms) else None,
            "maximum_adjacent_translation_step_m": float(np.linalg.norm(transform_steps[:, :3], axis=1).max()) if len(transform_steps) else 0.0,
            "maximum_adjacent_yaw_step_rad": float(np.abs(transform_steps[:, 3]).max()) if len(transform_steps) else 0.0,
        },
        "contact_sheets": [str(path) for path in sheets],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
