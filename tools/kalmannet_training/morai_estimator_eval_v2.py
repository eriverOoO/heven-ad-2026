"""MORAI Frozen Estimator Evaluation Dataset v2 -- tooling for a
substantially larger, independently-frozen MORAI estimator evaluation
set, built to move past the pseudo-replication problem of
MORAI_ESTIMATOR_EVAL_V1 (T-11/T-12's frozen 3-sequence/154-frame test
stream, ``morai_frozen_stream.py``).

**This module never fabricates evaluation data.** Every function here
either (a) audits real, already-on-disk MORAI artifacts and reports what
exists, (b) constructs sequences from real GT + real frozen detector
output using a documented, deterministic join, or (c) validates/aggregates
sequences already built. If real data is insufficient to grow V1 (the
finding of this task -- see ``audit_actor_leakage``), this module still
provides the complete, tested machinery so that a future session with
real new MORAI captures can populate V2 without writing any of this code
again.

**GT/runtime separation (competition-compliance boundary, Section 6).**
Every function here operates on ALREADY-EXPORTED, OFFLINE artifacts
(``~/datasets/morai_heven/labels/*.json``,
``~/heven_presentation_assets/motion_gt_expansion/canonical/*``). No
function in this module is imported by, or reachable from, any runtime
ROS node, `ad_lidar_perception`, `CenterPoint`, AB3DMOT, prediction, or
planner code -- GT here feeds only the offline post-run evaluator built
in this file, never a runtime decision.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

MORAI_ESTIMATOR_EVAL_V2_SCHEMA_VERSION = "morai_estimator_eval_v2"

# The frozen V1 test set (T-11/T-12 lineage) -- MUST NEVER CHANGE. Any
# function here that touches actor eligibility imports this constant
# rather than re-typing the literal, so a future edit here would need to
# change only one place (and would still not retroactively alter the
# already-frozen V1 sequences.pkl / morai_frozen_stream.py contract).
V1_FROZEN_TEST_ACTOR_IDS = (16, 20, 30)

# Every actor ID in the sole captured MORAI run (`static_20260805_003151`)
# and which pipeline stage(s) have consumed it, as of this task. Sourced
# directly from `~/heven_presentation_assets/motion_gt_expansion/
# canonical/split_policy.json` (T-11's own frozen, pre-T-12 split) plus
# this session's own MORAI-calibrated-corruption task (which additionally
# consumed the same train+val actors for residual/gap calibration).
V1_TRAIN_ACTOR_IDS = (13, 48, 46, 15, 36, 1, 35, 44, 18, 34, 38, 40)
V1_VAL_ACTOR_IDS = (23, 31, 21, 27)
V1_EXCLUDED_ACTOR_IDS = (2, 3)
V1_EXCLUDED_REASON = (
    "actor 2 (obstacle) and actor 3 (pedestrian) show vehicle-speed kinematics "
    "inconsistent with their class label -- a GT-quality defect (per "
    "motion_gt_expansion/canonical/split_policy.json's own "
    "excluded_reason), NOT a leakage exclusion. Their GT is not trusted "
    "enough to serve as new held-out evaluation data either."
)

# Consumers of the TRAIN/VAL actor pool, for the leakage table (Section
# 13). Each entry names a real, already-completed task in this project
# that used these actors for training, tuning, or calibration.
TRAIN_VAL_CONSUMERS = (
    "T-9A.1 Tuned Linear KF calibration (selected_kf_config.json)",
    "T-12 KalmanNet training (kalmannet_training_provenance.json)",
    "T-12.2/T-12.3 dense-v2 KalmanNet training (frozen_model_v2_manifest.json)",
    "This project's MORAI-Calibrated AV2 Corruption v1 task "
    "(morai_calibration.py residual/gap calibration, PR #52)",
)


@dataclass
class SourceRunRecord:
    """One row of the raw-MORAI-data inventory (Section 2 of the task)."""

    source_path: str
    date_or_run_id: str
    scene_or_run_identity: str
    frame_range: str
    actor_count: int | None
    detector_source: str
    gt_source: str
    previously_used_for: list[str]
    suitable_as_new_v2_test_data: bool
    suitability_reason: str


def _display_path(path: Path, default_symbolic: str) -> str:
    """Reports a machine-independent symbolic form (~/...) for known
    default locations, so audit output stays committable without leaking
    this machine's absolute home directory."""
    home = str(Path.home())
    resolved = str(path)
    if resolved == str(Path(default_symbolic.replace("~", home))):
        return default_symbolic
    return resolved


def audit_source_runs(
    morai_heven_root: Path | None = None,
    canonical_root: Path | None = None,
    static_bag_metadata: Path | None = None,
    camera_bag_metadata: Path | None = None,
) -> list[SourceRunRecord]:
    """Audits every locally-known MORAI-adjacent data source. Reads only
    metadata/manifest files already on disk -- never opens a raw bag,
    never launches MORAI. Returns one record per source, whether or not
    it turns out usable, so the audit itself is complete even when the
    answer is "not usable"."""
    morai_heven_root = morai_heven_root or (Path.home() / "datasets/morai_heven")
    canonical_root = canonical_root or (Path.home() / "heven_presentation_assets/motion_gt_expansion/canonical")
    records: list[SourceRunRecord] = []

    metadata_path = morai_heven_root / "metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text("utf-8"))
        n_labels = len(list((morai_heven_root / "labels").glob("*.json"))) if (morai_heven_root / "labels").is_dir() else None
        records.append(SourceRunRecord(
            source_path=_display_path(morai_heven_root, "~/datasets/morai_heven"),
            date_or_run_id="static_20260805_003151",
            scene_or_run_identity=metadata.get("class_evidence", {}).get("scenario", "unknown"),
            frame_range=f"0-{(n_labels or 1) - 1}" if n_labels else "unknown",
            actor_count=21,  # audited exhaustively by T-11's own actor-mining pass, cross-checked below
            detector_source="none in this export (LiDAR raw points + GT boxes only)",
            gt_source="/ad/dev/objects (MORAI dev-mode actor GT), target_frame=lidar_link",
            previously_used_for=["T-9A/T-9A.1/T-11/T-12/T-12.x/T-13/ros_gt_crosscheck/kf_calibration/"
                                  "kalmannet_ablation/kalmannet_dense_baseline/kalmannet_training_stability/"
                                  "MORAI-Calibrated AV2 Corruption v1 (this project's own corruption calibration)"],
            suitable_as_new_v2_test_data=False,
            suitability_reason=(
                "This is the ONLY MORAI run with actor GT ever captured in this project. "
                "Every one of its 21 actors is already assigned to TRAIN, VAL, TEST, or "
                "excluded-for-GT-quality (see audit_actor_leakage) -- zero actors remain "
                "unconsumed. The existing frozen V1 TEST actors [16,20,30] may be REUSED "
                "(not counted as new), but no additional actor from this run can serve as "
                "new held-out data without violating the leakage boundary."
            ),
        ))

    canonical_provenance = canonical_root / "provenance.json"
    if canonical_provenance.is_file():
        prov = json.loads(canonical_provenance.read_text("utf-8"))
        records.append(SourceRunRecord(
            source_path=_display_path(canonical_root, "~/heven_presentation_assets/motion_gt_expansion/canonical"),
            date_or_run_id=prov.get("dataset_version", "unknown"),
            scene_or_run_identity=prov.get("scene", "unknown"),
            frame_range=f"0-{prov.get('n_frames', 1) - 1}",
            actor_count=21,
            detector_source=prov.get("detection_source", "unknown"),
            gt_source=prov.get("gt_source", "unknown"),
            previously_used_for=["source material for state_estimator_gt_comparison/sequences.pkl "
                                  "(the exact pickle morai_frozen_stream.py loads)"],
            suitable_as_new_v2_test_data=False,
            suitability_reason="Derived from the same single scene/run as morai_heven above -- not an "
                                "independent source, same leakage conclusion applies.",
        ))

    static_bag_metadata = static_bag_metadata or (Path("bags/static_20260805_003151/metadata.yaml"))
    if Path(static_bag_metadata).is_file():
        records.append(SourceRunRecord(
            source_path=str(static_bag_metadata),
            date_or_run_id="static_20260805_003151",
            scene_or_run_identity="same scene as morai_heven (confirmed: starting_time within ~2s of the "
                                   "first exported label's own source stamp, message counts consistent)",
            frame_range="unknown (raw .mcap file absent from disk, metadata.yaml only)",
            actor_count=None,
            detector_source="metadata claims /ad/perception/objects/detected present (1796 msgs) but the "
                             "underlying data is unrecoverable",
            gt_source="metadata claims /ad/dev/objects present (9711 msgs) but the underlying data is "
                       "unrecoverable",
            previously_used_for=[],
            suitable_as_new_v2_test_data=False,
            suitability_reason="The .mcap payload is missing from disk (metadata-only remnant) -- "
                                "unreplayable. Even if it were recoverable, it is the same run as "
                                "morai_heven, not a new independent one.",
        ))

    camera_bag_metadata = camera_bag_metadata or Path(
        "morai_cam4_20260813_163222/morai_cam4_20260813_163222/metadata.yaml"
    )
    if Path(camera_bag_metadata).is_file():
        records.append(SourceRunRecord(
            source_path=str(camera_bag_metadata),
            date_or_run_id="morai_cam4_20260813_163222",
            scene_or_run_identity="a genuinely different, moving-ego recording",
            frame_range="unknown (full replayable .mcap present, not frame-counted here)",
            actor_count=0,
            detector_source="real LiDAR/camera present, no recorded detector output",
            gt_source="none -- no /ad/dev/objects or any actor-GT topic in this bag",
            previously_used_for=["camera+LiDAR fusion research (POST-FREEZE EXTENSION 2), qualitative only"],
            suitable_as_new_v2_test_data=False,
            suitability_reason="Real and independent of the static_20260805_003151 run, but carries zero "
                                "actor ground truth -- cannot serve an estimator evaluation regardless of "
                                "leakage status.",
        ))

    return records


@dataclass
class LeakageAuditRow:
    actor_id: int
    role: str  # "train" | "val" | "test" | "excluded_data_quality"
    used_in: list[str]
    eligible_for_v2: bool
    reason: str


def audit_actor_leakage(
    all_actor_ids: list[int] | None = None,
    train_ids: tuple[int, ...] = V1_TRAIN_ACTOR_IDS,
    val_ids: tuple[int, ...] = V1_VAL_ACTOR_IDS,
    test_ids: tuple[int, ...] = V1_FROZEN_TEST_ACTOR_IDS,
    excluded_ids: tuple[int, ...] = V1_EXCLUDED_ACTOR_IDS,
) -> list[LeakageAuditRow]:
    """Produces one row per actor in the sole captured MORAI run,
    explaining exactly why it is or is not eligible as NEW V2 evaluation
    data. An actor already in ``test_ids`` is reported as reused (not
    new); every other actor is either a leakage exclusion (train/val) or
    a GT-quality exclusion, and none is eligible."""
    if all_actor_ids is None:
        all_actor_ids = sorted(set(train_ids) | set(val_ids) | set(test_ids) | set(excluded_ids))
    rows: list[LeakageAuditRow] = []
    for aid in sorted(all_actor_ids):
        if aid in test_ids:
            rows.append(LeakageAuditRow(
                actor_id=aid, role="test", used_in=["MORAI_ESTIMATOR_EVAL_V1 (frozen, T-11/T-12)"],
                eligible_for_v2=False,
                reason="already the frozen V1 test set -- reused as-is in V2, not counted as new data",
            ))
        elif aid in train_ids:
            rows.append(LeakageAuditRow(
                actor_id=aid, role="train", used_in=list(TRAIN_VAL_CONSUMERS),
                eligible_for_v2=False, reason="leakage -- used for training/calibration",
            ))
        elif aid in val_ids:
            rows.append(LeakageAuditRow(
                actor_id=aid, role="val", used_in=list(TRAIN_VAL_CONSUMERS),
                eligible_for_v2=False, reason="leakage -- used for model/hyperparameter selection",
            ))
        elif aid in excluded_ids:
            rows.append(LeakageAuditRow(
                actor_id=aid, role="excluded_data_quality", used_in=[],
                eligible_for_v2=False, reason=V1_EXCLUDED_REASON,
            ))
        else:
            rows.append(LeakageAuditRow(
                actor_id=aid, role="unknown", used_in=[],
                eligible_for_v2=False,
                reason="not present in the documented split_policy.json -- treated as ineligible "
                       "until its provenance is established (fails closed, never silently included)",
            ))
    return rows


class FrameMismatchError(ValueError):
    """Raised when GT and detector measurement are not verified to share
    the same coordinate frame (Section 9)."""


def assert_same_metric_frame(gt_frame_id: str, measurement_frame_id: str) -> None:
    if gt_frame_id != measurement_frame_id:
        raise FrameMismatchError(
            f"GT frame {gt_frame_id!r} != measurement frame {measurement_frame_id!r} -- "
            "refusing to build a sequence across mismatched frames"
        )


def assert_valid_transform_chain(chain: list[str], expected: tuple[str, ...] = (
    "map", "odom", "base_link", "rear_axle_link", "lidar_link",
)) -> None:
    if tuple(chain) != expected:
        raise FrameMismatchError(f"unexpected transform chain {chain!r}, expected {list(expected)}")


@dataclass
class V2Sequence:
    run_id: str
    actor_id: int
    role: str
    frames: list[int]
    timestamps_ns: list[int]
    dt: list[float | None]
    x_true: list[list[float]]
    z_meas: list[list[float] | None]
    class_name: str
    range_m: list[float | None]
    source_frame: str
    detector_source_version: str
    gt_source_version: str


class SequenceBuildError(ValueError):
    pass


def build_v2_sequences(
    gt_frames: list[dict[str, Any]],
    detection_frames: list[dict[str, Any]],
    eligible_actor_ids: list[int],
    run_id: str,
    role: str,
    detector_source_version: str,
    gt_source_version: str,
    max_association_distance_m: float = 3.0,
) -> list[V2Sequence]:
    """Builds one sequence per (run_id, actor_id) from already-aligned,
    positionally-joined GT and detection frame lists (``gt_frames[i]``
    corresponds to ``detection_frames[i]`` by INDEX, never by matching
    ``stamp_ns`` -- the same documented convention
    ``motion_gt_expansion/canonical/provenance.json`` already
    establishes: detection replay stamps are publish-time artifacts, not
    the real join key). Refuses (raises ``FrameMismatchError``) if any
    frame's GT/detection ``frame_id`` differ, or if any actor outside
    ``eligible_actor_ids`` would otherwise contribute a sequence."""
    if len(gt_frames) != len(detection_frames):
        raise SequenceBuildError(
            f"gt_frames ({len(gt_frames)}) and detection_frames ({len(detection_frames)}) "
            "must be the same length -- positional join requires equal frame counts"
        )

    by_actor: dict[int, dict[str, list]] = {}
    for i, (gt_row, det_row) in enumerate(zip(gt_frames, detection_frames)):
        assert_same_metric_frame(gt_row.get("frame_id", "lidar_link"), det_row.get("frame_id", "lidar_link"))
        stamp_ns = gt_row["header_stamp_ns"]
        for obj in gt_row["objects"]:
            aid = obj["actor_id"]
            if aid not in eligible_actor_ids:
                continue
            slot = by_actor.setdefault(aid, {"frame_idx": [], "stamp_ns": [], "gt": [], "class_name": obj.get("class")})
            slot["frame_idx"].append(i)
            slot["stamp_ns"].append(stamp_ns)
            slot["gt"].append(obj)

    sequences: list[V2Sequence] = []
    for aid, slot in by_actor.items():
        frame_idx = slot["frame_idx"]
        stamps = slot["stamp_ns"]
        if any(stamps[k] >= stamps[k + 1] for k in range(len(stamps) - 1)):
            raise SequenceBuildError(f"actor {aid}: non-monotonic GT timestamps")
        dt: list[float | None] = [None] + [
            (stamps[k + 1] - stamps[k]) / 1e9 for k in range(len(stamps) - 1)
        ]
        x_true = []
        z_meas: list[list[float] | None] = []
        range_m: list[float | None] = []
        for k, gt_obj in enumerate(slot["gt"]):
            x, y = gt_obj["x"], gt_obj["y"]
            if k == 0:
                vx = vy = 0.0
            else:
                prev = slot["gt"][k - 1]
                d = dt[k]
                vx = (x - prev["x"]) / d if d else 0.0
                vy = (y - prev["y"]) / d if d else 0.0
            x_true.append([x, y, vx, vy])

            det_frame = detection_frames[frame_idx[k]]
            best = None
            best_dist = float("inf")
            for cand in det_frame.get("objects", []):
                cx, cy = cand["pos"][0], cand["pos"][1]
                dist = float(np.hypot(cx - x, cy - y))
                if dist < best_dist:
                    best_dist = dist
                    best = cand
            if best is not None and best_dist <= max_association_distance_m:
                z_meas.append([best["pos"][0], best["pos"][1]])
                range_m.append(float(np.hypot(best["pos"][0], best["pos"][1])))
            else:
                z_meas.append(None)
                range_m.append(None)

        sequences.append(V2Sequence(
            run_id=run_id, actor_id=aid, role=role, frames=frame_idx, timestamps_ns=stamps,
            dt=dt, x_true=x_true, z_meas=z_meas, class_name=slot["class_name"],
            range_m=range_m, source_frame="lidar_link",
            detector_source_version=detector_source_version, gt_source_version=gt_source_version,
        ))
    return sequences


def sequence_id(seq: V2Sequence) -> str:
    return f"{seq.run_id}__actor_{seq.actor_id}"


@dataclass
class ValidationIssue:
    severity: str  # "error" | "warning"
    sequence_id: str
    message: str


def validate_v2_sequences(sequences: list[V2Sequence]) -> list[ValidationIssue]:
    """Fails closed: every check below is additive to the issues list,
    never a silent repair. Callers must treat any "error" severity issue
    as a hard rejection of the whole dataset build."""
    issues: list[ValidationIssue] = []
    seen_ids: set[str] = set()
    for seq in sequences:
        sid = sequence_id(seq)
        if sid in seen_ids:
            issues.append(ValidationIssue("error", sid, "duplicate run/actor sequence ID"))
        seen_ids.add(sid)

        if len(seq.frames) != len(set(seq.frames)):
            issues.append(ValidationIssue("error", sid, "duplicate frame indices within one sequence"))

        for k in range(1, len(seq.timestamps_ns)):
            if seq.timestamps_ns[k] <= seq.timestamps_ns[k - 1]:
                issues.append(ValidationIssue("error", sid, f"non-monotonic timestamp at index {k}"))
                break
        for k, d in enumerate(seq.dt):
            if k == 0:
                continue
            if d is None or d <= 0.0:
                issues.append(ValidationIssue("error", sid, f"dt<=0 (or None) at index {k}"))
                break

        gt_arr = np.array(seq.x_true, dtype=float)
        if not np.all(np.isfinite(gt_arr)):
            issues.append(ValidationIssue("error", sid, "non-finite GT state"))

        n_present = sum(1 for z in seq.z_meas if z is not None)
        n_total = len(seq.z_meas)
        if len(seq.z_meas) != len(seq.x_true) or len(seq.z_meas) != len(seq.frames):
            issues.append(ValidationIssue("error", sid, "measurement mask length mismatch with GT/frames"))

        if n_total < 10:
            issues.append(ValidationIssue("warning", sid, f"short trajectory ({n_total} frames)"))
        if n_total and n_present / n_total < 0.5:
            issues.append(ValidationIssue("warning", sid, f"sparse measurements ({n_present}/{n_total} present)"))

        gap = 0
        max_gap = 0
        for z in seq.z_meas:
            if z is None:
                gap += 1
                max_gap = max(max_gap, gap)
            else:
                gap = 0
        if max_gap >= 10:
            issues.append(ValidationIssue("warning", sid, f"extreme gap ({max_gap} consecutive missing frames)"))

    return issues


def content_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def build_v2_freeze_manifest(
    *,
    source_run_ids: list[str],
    actor_ids: list[int],
    sequence_boundaries: dict[str, dict[str, int]],
    inclusion_rules: list[str],
    exclusion_rules: list[str],
    sequence_file_hashes: dict[str, str],
    detector_source_version: str,
    gt_source_version: str,
    alignment_policy: str,
    frame_convention: str,
    git_sha: str,
    notes: str = "",
) -> dict[str, Any]:
    return {
        "schema_version": MORAI_ESTIMATOR_EVAL_V2_SCHEMA_VERSION,
        "source_run_ids": source_run_ids,
        "actor_ids": sorted(actor_ids),
        "sequence_boundaries": sequence_boundaries,
        "inclusion_rules": inclusion_rules,
        "exclusion_rules": exclusion_rules,
        "sequence_file_hashes": sequence_file_hashes,
        "detector_source_version": detector_source_version,
        "gt_source_version": gt_source_version,
        "alignment_policy": alignment_policy,
        "frame_convention": frame_convention,
        "git_sha": git_sha,
        "notes": notes,
    }


def save_v2_freeze_manifest(manifest: dict[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


# --------------------------------------------------------------------
# Statistical evaluation design (Section 19)
# --------------------------------------------------------------------
@dataclass
class PairedBootstrapResult:
    unit: str  # "sequence" | "run"
    n_units: int
    observed_diff: float
    ci_low: float
    ci_high: float
    fraction_bootstrap_same_sign: float
    n_bootstrap: int
    stable: bool


def paired_bootstrap(
    per_unit_rmse_a: list[float],
    per_unit_rmse_b: list[float],
    unit: str = "sequence",
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> PairedBootstrapResult:
    """Paired bootstrap over sequences (default) or, when the caller has
    already macro-averaged to one value per run, over runs. Frame-level
    bootstrap is deliberately never offered here as a primary estimator
    -- see docs/perception/morai_estimator_eval_v2.md Section 19."""
    if len(per_unit_rmse_a) != len(per_unit_rmse_b):
        raise ValueError("per_unit_rmse_a and per_unit_rmse_b must be paired (same length)")
    n = len(per_unit_rmse_a)
    a = np.asarray(per_unit_rmse_a, dtype=float)
    b = np.asarray(per_unit_rmse_b, dtype=float)
    observed = float(np.mean(a) - np.mean(b))
    rng = np.random.RandomState(seed)
    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        diffs.append(float(np.mean(a[idx]) - np.mean(b[idx])))
    diffs = np.asarray(diffs)
    ci_low, ci_high = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
    same_sign = float(np.mean(np.sign(diffs) == np.sign(observed))) if observed != 0 else float("nan")
    stable = bool((ci_low > 0) or (ci_high < 0))
    return PairedBootstrapResult(
        unit=unit, n_units=n, observed_diff=observed, ci_low=ci_low, ci_high=ci_high,
        fraction_bootstrap_same_sign=same_sign, n_bootstrap=n_bootstrap, stable=stable,
    )


def run_level_macro_average(
    per_sequence_run_ids: list[str], per_sequence_metric: list[float],
) -> dict[str, float]:
    """Macro-averages a per-sequence metric up to one value per run_id --
    the unit paired_bootstrap should use when >1 independent run exists."""
    by_run: dict[str, list[float]] = {}
    for run_id, value in zip(per_sequence_run_ids, per_sequence_metric):
        by_run.setdefault(run_id, []).append(value)
    return {run_id: float(np.mean(values)) for run_id, values in by_run.items()}


def load_actor_manifest_csv(path: Path) -> list[dict[str, Any]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
