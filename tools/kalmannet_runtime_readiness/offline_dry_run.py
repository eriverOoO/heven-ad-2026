"""KalmanNet Production Candidate Freeze + Runtime Integration Readiness v1.

Offline integration smoke test (section 14): drives the REAL, unmodified
`ab3dmot_core.AB3DMOTTracker` -- the same class the ROS node
(`ab3dmot_tracker_node.py`) constructs -- with `state_estimator=linear_kf`
and then `state_estimator=kalmannet` (the frozen production-candidate
checkpoint), on the identical, real, previously-recorded detection
stream, with no ROS runtime involved.

This is NOT a scientific benchmark and produces NO accuracy claim -- it
only verifies: startup, checkpoint loading, tracks created, outputs
produced, finite positions/velocities, no exception, and (for the
KalmanNet arm) distinct per-track hidden-state object identities.

Usage:
    python3 offline_dry_run.py <detections.jsonl> <kalmannet_checkpoint.pt> [--max-frames N]

Detection stream format: one JSON object per line, matching the existing
`heven.ros_detection_comparison.v1` convention already used by
`~/heven_presentation_assets/end_to_end_detector_tracker/euclidean_detections.jsonl`
(fields: `stamp_ns`, `objects: [{x,y,z,qw, length,width,height,
class_label,score,existence_probability}, ...]`).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ad_lidar_perception"))

from ad_lidar_perception.ab3dmot_config import AB3DMOTConfig
from ad_lidar_perception.ab3dmot_core import AB3DMOTTracker, Detection


def _yaw_from_quaternion(qx: float, qy: float, qz: float, qw: float) -> float:
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def load_frames(path: Path, max_frames: int | None = None) -> list[tuple[float, list[Detection]]]:
    """Returns `[(timestamp_seconds, [Detection, ...]), ...]`, timestamps
    derived from each row's own real `stamp_ns` (never a synthetic 0.1s
    step) -- the actual, previously-recorded detector cadence."""
    frames: list[tuple[float, list[Detection]]] = []
    with path.open() as fh:
        for i, line in enumerate(fh):
            if max_frames is not None and i >= max_frames:
                break
            row = json.loads(line)
            t = row["stamp_ns"] / 1e9
            detections = [
                Detection(
                    x=obj["x"], y=obj["y"], z=obj["z"],
                    yaw=_yaw_from_quaternion(obj["qx"], obj["qy"], obj["qz"], obj["qw"]),
                    length=obj["length"], width=obj["width"], height=obj["height"],
                    label=int(obj.get("class_label", 0)),
                    label_probability=float(obj.get("score", 1.0)),
                    existence_probability=float(obj.get("existence_probability", 1.0)),
                )
                for obj in row["objects"]
            ]
            frames.append((t, detections))
    return frames


def run_dry_run(
    frames: list[tuple[float, list[Detection]]],
    *,
    state_estimator: str,
    kalmannet_checkpoint: str = "",
    kalmannet_device: str = "cpu",
) -> dict:
    config = AB3DMOTConfig(
        association_metric="euclidean",
        matcher="hungarian",
        euclidean_gate_m=3.0,
        yaw_measurement_mode="unobserved",
        state_estimator=state_estimator,
        kalmannet_checkpoint=kalmannet_checkpoint,
        kalmannet_device=kalmannet_device,
    )
    tracker = AB3DMOTTracker(config)

    n_frames = 0
    n_output_states = 0
    n_nonfinite = 0
    unique_track_ids: set[int] = set()
    last_t: float | None = None
    n_duplicate_or_backward_skipped = 0

    for t, detections in frames:
        if last_t is not None and t <= last_t:
            n_duplicate_or_backward_skipped += 1
            continue
        out = tracker.step(detections, t)
        last_t = t
        n_frames += 1
        for state in out:
            n_output_states += 1
            unique_track_ids.add(state.track_id)
            values = (state.x, state.y, state.z, state.vx_mps, state.vy_mps, state.vz_mps)
            if not all(math.isfinite(v) for v in values):
                n_nonfinite += 1

    hidden_state_object_ids: set[int] = set()
    if state_estimator == "kalmannet":
        for track in tracker._tracks:
            hidden_state_object_ids.add(id(track._estimator._hidden_state))

    return {
        "state_estimator": state_estimator,
        "kalmannet_provenance": tracker.kalmannet_provenance,
        "n_frames_processed": n_frames,
        "n_frames_skipped_duplicate_or_backward": n_duplicate_or_backward_skipped,
        "n_output_states": n_output_states,
        "n_nonfinite": n_nonfinite,
        "n_unique_track_ids": len(unique_track_ids),
        "n_live_tracks_at_end": len(tracker._tracks),
        "n_distinct_hidden_state_objects_at_end": len(hidden_state_object_ids),
        "exception": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("detections_jsonl", type=Path)
    parser.add_argument("kalmannet_checkpoint", type=Path)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    frames = load_frames(args.detections_jsonl, args.max_frames)
    print(f"loaded {len(frames)} frames from {args.detections_jsonl}")

    results = {}
    for label, kwargs in [
        ("A_linear_kf", {"state_estimator": "linear_kf"}),
        ("B_kalmannet", {
            "state_estimator": "kalmannet",
            "kalmannet_checkpoint": str(args.kalmannet_checkpoint),
            "kalmannet_device": "cpu",
        }),
    ]:
        try:
            result = run_dry_run(frames, **kwargs)
        except Exception as exc:  # noqa: BLE001 -- report, do not hide
            result = {"state_estimator": kwargs["state_estimator"], "exception": repr(exc)}
        results[label] = result
        print(f"--- {label} ---")
        for k, v in result.items():
            print(f"  {k}: {v}")

    if args.output:
        args.output.write_text(json.dumps(results, indent=2, default=str))
        print(f"wrote {args.output}")

    ok = all(r.get("exception") is None and r.get("n_nonfinite", 1) == 0 for r in results.values())
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
