#!/usr/bin/env python3
"""Summarize paired AV2 native/VLP16-like CenterPoint stage captures."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path

import numpy as np

from av2_vlp16_adapter import AdapterConfig
from evaluate_av2_xyzirc_pair import (
    DEFAULT_POINT_CLOUD_RANGE,
    detections_from_ros_dict,
    gated_hungarian_matches,
    roi_point_count,
)
from publish_av2_xyzirc import load_npz


STAGE_FILES = {
    "post_score": "post_score.yaml",
    "post_circle_nms": "post_circle_nms.yaml",
    "pre_iou": "pre_iou.yaml",
    "post_iou": "post_iou.yaml",
    "final": "final_stage.yaml",
}
DISTANCE_BINS = ((0.0, 20.0), (20.0, 40.0), (40.0, 60.0), (60.0, 80.0))


def load_ros_yaml(path: Path) -> dict:
    import yaml

    messages = [row for row in yaml.safe_load_all(path.read_text()) if row]
    if len(messages) != 1:
        raise ValueError(f"expected one complete ROS message in {path}, got {len(messages)}")
    return messages[0]


def describe(values: list[float]) -> dict:
    if not values:
        result = {
            key: None for key in ("mean", "median", "p10", "p25", "p75", "p90", "max")
        }
        return {"count": 0, **result}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p10": float(np.percentile(array, 10)),
        "p25": float(np.percentile(array, 25)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "max": float(array.max()),
    }


def wrapped_angle_difference(a: float, b: float) -> float:
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))


def box_axis_angle_difference(a: float, b: float) -> float:
    """Return yaw-axis difference when front/back direction is ignored."""
    difference = wrapped_angle_difference(a, b)
    return min(difference, abs(math.pi - difference))


def validate_stage_counts(counts: dict[str, int]) -> None:
    if counts["post_circle_nms"] != counts["pre_iou"]:
        raise ValueError(f"circle/pre-IoU count mismatch: {counts}")
    if not (
        counts["post_score"] >= counts["post_circle_nms"]
        and counts["pre_iou"] >= counts["post_iou"]
        and counts["post_iou"] == counts["final"]
    ):
        raise ValueError(f"invalid stage count flow: {counts}")


def stage_rows(run_dir: Path) -> dict[str, list[dict]]:
    stages = {
        stage: detections_from_ros_dict(load_ros_yaml(run_dir / filename), label=None)
        for stage, filename in STAGE_FILES.items()
    }
    counts = {stage: len(rows) for stage, rows in stages.items()}
    try:
        validate_stage_counts(counts)
    except ValueError as error:
        raise ValueError(f"{error} in {run_dir}") from error
    return stages


def gt_for_timestamp(columns: dict[str, list], timestamp_ns: int) -> list[dict]:
    x_min, y_min, z_min, x_max, y_max, z_max = DEFAULT_POINT_CLOUD_RANGE
    result = []
    for index, value in enumerate(columns["timestamp_ns"]):
        if int(value) != timestamp_ns or columns["category"][index] != "REGULAR_VEHICLE":
            continue
        x = float(columns["tx_m"][index])
        y = float(columns["ty_m"][index])
        z = float(columns["tz_m"][index])
        if not (x_min <= x < x_max and y_min <= y < y_max and z_min <= z < z_max):
            continue
        result.append(
            {
                "track_uuid": columns["track_uuid"][index],
                "x": x,
                "y": y,
                "z": z,
                "length": float(columns["length_m"][index]),
                "width": float(columns["width_m"][index]),
                "height": float(columns["height_m"][index]),
                "quaternion": tuple(
                    float(columns[name][index]) for name in ("qw", "qx", "qy", "qz")
                ),
            }
        )
    return result


def points_in_box(cloud: np.ndarray, gt: dict) -> int:
    qw, qx, qy, qz = gt["quaternion"]
    rotation = np.asarray(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )
    xyz = np.column_stack((cloud["x"], cloud["y"], cloud["z"]))
    local = (xyz - np.asarray((gt["x"], gt["y"], gt["z"]))) @ rotation
    half = np.asarray((gt["length"], gt["width"], gt["height"])) / 2.0
    return int(np.count_nonzero(np.all(np.abs(local) <= half, axis=1)))


def paired_gt_stability(gt: list[dict], native: list[dict], adapted: list[dict], gate_m: float) -> list[dict]:
    native_by_gt = {gi: native[di] for gi, di, _ in gated_hungarian_matches(gt, native, gate_m)}
    adapted_by_gt = {gi: adapted[di] for gi, di, _ in gated_hungarian_matches(gt, adapted, gate_m)}
    rows = []
    for gi in sorted(native_by_gt.keys() & adapted_by_gt.keys()):
        left, right = native_by_gt[gi], adapted_by_gt[gi]
        rows.append(
            {
                "track_uuid": gt[gi]["track_uuid"],
                "native_center_error_m": math.hypot(
                    left["x"] - gt[gi]["x"], left["y"] - gt[gi]["y"]
                ),
                "adapted_center_error_m": math.hypot(
                    right["x"] - gt[gi]["x"], right["y"] - gt[gi]["y"]
                ),
                "center_shift_m": math.hypot(left["x"] - right["x"], left["y"] - right["y"]),
                "yaw_shift_rad": wrapped_angle_difference(left["yaw"], right["yaw"]),
                "yaw_axis_shift_rad": box_axis_angle_difference(left["yaw"], right["yaw"]),
                "size_shift_m": float(
                    np.linalg.norm(
                        np.asarray((left["length"], left["width"], left["height"]))
                        - np.asarray((right["length"], right["width"], right["height"]))
                    )
                ),
                "score_shift": right["score"] - left["score"],
                "class_changed": left["label"] != right["label"],
            }
        )
    return rows


def outputs_equivalent(
    left: list[dict],
    right: list[dict],
    *,
    center_tolerance_m: float = 0.005,
    scalar_tolerance: float = 0.005,
) -> bool:
    """Compare two final outputs without relying on object array order."""
    if len(left) != len(right):
        return False
    matches = gated_hungarian_matches(left, right, center_tolerance_m)
    if len(matches) != len(left):
        return False
    for left_index, right_index, _ in matches:
        a, b = left[left_index], right[right_index]
        if a["label"] != b["label"]:
            return False
        for field in ("score", "z", "length", "width", "height"):
            if abs(a[field] - b[field]) > scalar_tolerance:
                return False
        if wrapped_angle_difference(a["yaw"], b["yaw"]) > scalar_tolerance:
            return False
    return True


def assign_beam_candidates(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, config: AdapterConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return elevation, nearest logical beam, and pre-azimuth keep mask."""
    ranges = np.sqrt(x * x + y * y + z * z)
    elevation = np.degrees(np.arctan2(z, np.hypot(x, y)))
    targets = np.asarray(config.target_elevations_deg)
    nearest = np.abs(elevation[:, None] - targets[None, :]).argmin(axis=1)
    keep = (
        (ranges >= config.min_range_m)
        & (ranges <= config.max_range_m)
        & (np.abs(elevation - targets[nearest]) <= config.vertical_tolerance_deg)
    )
    return elevation, nearest, keep


def summarize_mode(frames: list[dict]) -> dict:
    total_gt = sum(row["gt"] for row in frames)
    total_matches = sum(row["matched"] for row in frames)
    return {
        "frames": len(frames),
        "gt": total_gt,
        "final_detections_per_frame": describe([row["final_total"] for row in frames]),
        "vehicle_detections_per_frame": describe([row["vehicle_detections"] for row in frames]),
        "micro_recall": total_matches / total_gt if total_gt else None,
        "per_frame_recall": describe([row["recall"] for row in frames if row["recall"] is not None]),
        "false_positives_per_frame": sum(row["false_positives"] for row in frames) / len(frames),
        "matched_center_error_m": describe(
            [value for row in frames for value in row["match_errors"]]
        ),
        "matched_score": describe([value for row in frames for value in row["matched_scores"]]),
    }


def beam_audit(lidar_paths: list[Path], derived_root: Path) -> dict:
    import pyarrow.feather as feather
    from av2.utils.io import read_ego_SE3_sensor

    config = AdapterConfig()
    targets = np.asarray(config.target_elevations_deg)
    candidates = np.zeros(16, dtype=np.int64)
    survivors = np.zeros(16, dtype=np.int64)
    occupied_frames = np.zeros(16, dtype=np.int64)
    accepted_elevations: list[list[float]] = [[] for _ in range(16)]
    native_ring_elevations: dict[int, list[np.ndarray]] = defaultdict(list)
    sensor_local_ring_elevations: dict[int, list[np.ndarray]] = defaultdict(list)
    sensor_local_candidates = np.zeros(16, dtype=np.int64)
    log_dir = lidar_paths[0].parent.parent.parent
    sensor_poses = read_ego_SE3_sensor(log_dir)
    for path in lidar_paths:
        table = feather.read_table(path, columns=["x", "y", "z", "laser_number"])
        x = table["x"].to_numpy().astype(np.float32)
        y = table["y"].to_numpy().astype(np.float32)
        z = table["z"].to_numpy().astype(np.float32)
        ring = table["laser_number"].to_numpy().astype(np.int64)
        elevation, nearest, keep = assign_beam_candidates(x, y, z, config)
        for beam in range(16):
            values = elevation[keep & (nearest == beam)]
            candidates[beam] += len(values)
            accepted_elevations[beam].extend(values.tolist())
        for laser_number in np.unique(ring):
            native_ring_elevations[int(laser_number)].append(elevation[ring == laser_number])
        # AV2 publishes these points in the ego frame. This diagnostic split is
        # validated below by the narrow, matching local-angle bands n and n+32;
        # it is never applied to the coordinates used for inference.
        for mask, sensor_name in ((ring < 32, "up_lidar"), (ring >= 32, "down_lidar")):
            sensor_xyz = sensor_poses[sensor_name].inverse().transform_point_cloud(
                np.column_stack((x[mask], y[mask], z[mask]))
            )
            sensor_elevation = np.degrees(
                np.arctan2(sensor_xyz[:, 2], np.hypot(sensor_xyz[:, 0], sensor_xyz[:, 1]))
            )
            source_rings = ring[mask]
            local_nearest = np.abs(sensor_elevation[:, None] - targets[None, :]).argmin(axis=1)
            local_keep = (
                np.abs(sensor_elevation - targets[local_nearest])
                <= config.vertical_tolerance_deg
            )
            sensor_local_candidates += np.bincount(
                local_nearest[local_keep], minlength=len(targets)
            )
            for laser_number in np.unique(source_rings):
                sensor_local_ring_elevations[int(laser_number)].append(
                    sensor_elevation[source_rings == laser_number]
                )
        with np.load(derived_root / "vlp16_like" / f"{path.stem}.npz", allow_pickle=False) as data:
            channel = data["channel"]
        for beam in range(16):
            count = int(np.count_nonzero(channel == beam))
            survivors[beam] += count
            occupied_frames[beam] += count > 0

    beams = []
    for beam, target in enumerate(targets):
        values = accepted_elevations[beam]
        beams.append(
            {
                "channel": beam,
                "nominal_elevation_deg": float(target),
                "accepted_observed_min_deg": min(values) if values else None,
                "accepted_observed_max_deg": max(values) if values else None,
                "candidate_points_before_azimuth": int(candidates[beam]),
                "surviving_points_after_azimuth": int(survivors[beam]),
                "occupied_frames": int(occupied_frames[beam]),
            }
        )
    native_rings = []
    for ring, chunks in sorted(native_ring_elevations.items()):
        values = np.concatenate(chunks)
        local_values = np.concatenate(sensor_local_ring_elevations[ring])
        native_rings.append(
            {
                "laser_number": ring,
                "points": len(values),
                "inferred_sensor": "up_lidar" if ring < 32 else "down_lidar",
                "ego_origin_median_elevation_deg": float(np.median(values)),
                "ego_origin_p05_elevation_deg": float(np.percentile(values, 5)),
                "ego_origin_p95_elevation_deg": float(np.percentile(values, 95)),
                "sensor_local_median_elevation_deg": float(np.median(local_values)),
                "sensor_local_p05_elevation_deg": float(np.percentile(local_values, 5)),
                "sensor_local_p95_elevation_deg": float(np.percentile(local_values, 95)),
            }
        )
    ring_pair_differences = [
        abs(
            native_rings[ring]["sensor_local_median_elevation_deg"]
            - native_rings[ring + 32]["sensor_local_median_elevation_deg"]
        )
        for ring in range(32)
    ]
    return {
        "logical_beams": beams,
        "native_laser_numbers": native_rings,
        "sensor_local_target_candidates_before_azimuth": sensor_local_candidates.tolist(),
        "paired_ring_local_median_difference_deg": describe(ring_pair_differences),
        "diagnosis": (
            "Current adapter gates elevation about the AV2 ego origin. Sensor-local inverse "
            "extrinsics recover narrow matching ring bands for laser n and n+32, so ego-origin "
            "elevation is not a physical beam angle. Near-empty logical beams are present before "
            "azimuth reduction; strict tolerance and source/target ring mismatch also contribute."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--lidar-dir", type=Path, required=True)
    parser.add_argument("--derived-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--score-csv", type=Path)
    parser.add_argument("--gate-m", type=float, default=3.0)
    args = parser.parse_args()

    import pyarrow.feather as feather

    columns = feather.read_table(args.annotations).to_pydict()
    lidar_paths = sorted(args.lidar_dir.glob("*.feather"))
    modes = {"native": [], "vlp16_like": []}
    paired_rows = []
    stage_scores: dict[str, dict[str, list[float]]] = {
        mode: {stage: [] for stage in STAGE_FILES} for mode in modes
    }
    stage_counts: dict[str, dict[str, list[int]]] = {
        mode: {stage: [] for stage in STAGE_FILES} for mode in modes
    }
    distance_rows = defaultdict(lambda: {"gt": 0, "native_matches": 0, "adapted_matches": 0,
                                         "native_points": [], "adapted_points": [],
                                         "native_scores": [], "adapted_scores": []})

    for lidar_path in lidar_paths:
        timestamp_ns = int(lidar_path.stem)
        gt = gt_for_timestamp(columns, timestamp_ns)
        clouds = {}
        final_all = {}
        match_maps = {}
        for mode in modes:
            stages = stage_rows(args.run_root / str(timestamp_ns) / mode)
            for stage, rows in stages.items():
                stage_counts[mode][stage].append(len(rows))
                stage_scores[mode][stage].extend(row["score"] for row in rows)
            final_all[mode] = stages["final"]
            vehicle = [row for row in stages["final"] if row["label"] == 1]
            matches = gated_hungarian_matches(gt, vehicle, args.gate_m)
            match_maps[mode] = {gi: vehicle[di] for gi, di, _ in matches}
            clouds[mode], _ = load_npz(args.derived_root / mode / f"{timestamp_ns}.npz")
            modes[mode].append(
                {
                    "timestamp_ns": timestamp_ns,
                    "gt": len(gt),
                    "final_total": len(stages["final"]),
                    "vehicle_detections": len(vehicle),
                    "matched": len(matches),
                    "recall": len(matches) / len(gt) if gt else None,
                    "false_positives": len(vehicle) - len(matches),
                    "match_errors": [distance for _, _, distance in matches],
                    "matched_scores": [vehicle[di]["score"] for _, di, _ in matches],
                    "input_points": len(clouds[mode]),
                    "roi_points": roi_point_count(clouds[mode], DEFAULT_POINT_CLOUD_RANGE),
                    "stage_counts": {stage: len(rows) for stage, rows in stages.items()},
                }
            )

        paired_rows.extend(paired_gt_stability(gt, final_all["native"], final_all["vlp16_like"], args.gate_m))
        for gi, actor in enumerate(gt):
            distance = math.hypot(actor["x"], actor["y"])
            bin_name = next(
                (f"{int(low)}-{int(high)}m" for low, high in DISTANCE_BINS if low <= distance < high),
                None,
            )
            if bin_name is None:
                continue
            row = distance_rows[bin_name]
            row["gt"] += 1
            row["native_points"].append(points_in_box(clouds["native"], actor))
            row["adapted_points"].append(points_in_box(clouds["vlp16_like"], actor))
            if gi in match_maps["native"]:
                row["native_matches"] += 1
                row["native_scores"].append(match_maps["native"][gi]["score"])
            if gi in match_maps["vlp16_like"]:
                row["adapted_matches"] += 1
                row["adapted_scores"].append(match_maps["vlp16_like"][gi]["score"])

    stage_summary = {}
    for mode in modes:
        stage_summary[mode] = {
            stage: {
                "count_per_frame": describe(stage_counts[mode][stage]),
                "score": describe(stage_scores[mode][stage]),
            }
            for stage in STAGE_FILES
        }
        s1 = sum(stage_counts[mode]["post_score"])
        s2 = sum(stage_counts[mode]["post_circle_nms"])
        s3 = sum(stage_counts[mode]["pre_iou"])
        s4 = sum(stage_counts[mode]["post_iou"])
        stage_summary[mode]["circle_reduction"] = 1.0 - s2 / s1 if s1 else None
        stage_summary[mode]["iou_reduction"] = 1.0 - s4 / s3 if s3 else None

    result = {
        "frame_count": len(lidar_paths),
        "modes": {mode: summarize_mode(rows) for mode, rows in modes.items()},
        "per_frame": modes,
        "stage_summary": stage_summary,
        "gt_conditioned_stability": {
            "objects_detected_in_both": len(paired_rows),
            "native_center_error_m": describe(
                [row["native_center_error_m"] for row in paired_rows]
            ),
            "adapted_center_error_m": describe(
                [row["adapted_center_error_m"] for row in paired_rows]
            ),
            "center_shift_m": describe([row["center_shift_m"] for row in paired_rows]),
            "yaw_shift_rad": describe([row["yaw_shift_rad"] for row in paired_rows]),
            "yaw_axis_shift_rad": describe(
                [row["yaw_axis_shift_rad"] for row in paired_rows]
            ),
            "front_back_yaw_flips": sum(
                row["yaw_shift_rad"] > math.pi / 2.0 for row in paired_rows
            ),
            "size_shift_m": describe([row["size_shift_m"] for row in paired_rows]),
            "score_shift": describe([row["score_shift"] for row in paired_rows]),
            "class_changes": sum(row["class_changed"] for row in paired_rows),
        },
        "distance_bins": {
            name: {
                "gt": row["gt"],
                "native_recall": row["native_matches"] / row["gt"] if row["gt"] else None,
                "adapted_recall": row["adapted_matches"] / row["gt"] if row["gt"] else None,
                "native_points_per_object": describe(row["native_points"]),
                "adapted_points_per_object": describe(row["adapted_points"]),
                "native_matched_score": describe(row["native_scores"]),
                "adapted_matched_score": describe(row["adapted_scores"]),
            }
            for name, row in distance_rows.items()
        },
        "beam_audit": beam_audit(lidar_paths, args.derived_root),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.score_csv:
        args.score_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.score_csv.open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=("mode", "stage", "candidate_index", "score")
            )
            writer.writeheader()
            for mode in modes:
                for stage in STAGE_FILES:
                    for candidate_index, score in enumerate(stage_scores[mode][stage]):
                        writer.writerow(
                            {
                                "mode": mode,
                                "stage": stage,
                                "candidate_index": candidate_index,
                                "score": score,
                            }
                        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
