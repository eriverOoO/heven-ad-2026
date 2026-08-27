"""POST-FREEZE EXTENSION 2, Stage 3 semantic-assisted association.

RESEARCH ONLY, opt-in, and disabled by default.  Camera semantics originate
only from the validated Stage-1/2 camera-to-projected-LiDAR IoU path.  They
never enter the Kalman state, never change the geometric gate, and never hard
reject a candidate.  Missing or insufficient evidence contributes exactly
zero, reducing the solver matrix to the original geometric matrix.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from .ab3dmot_core import (
    AB3DMOTTracker,
    Detection,
    _greedy_matching,
    _hungarian_matching,
)

__all__ = [
    "SemanticEvidence",
    "SemanticDetection",
    "SemanticMemory",
    "SemanticMemorySnapshot",
    "SemanticAssociationConfig",
    "SemanticAB3DMOTTracker",
    "SEMANTIC_CLASS_NAMES",
    "DEFAULT_COMPATIBILITY_PENALTIES",
    "normalize_semantic_class",
    "semantic_reliability",
    "compatibility_penalty",
    "semantic_penalty_matrix",
]


SEMANTIC_CLASS_NAMES = (
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "person",
)

_CLASS_ALIASES = {
    "pedestrian": "person",
    "motorbike": "motorcycle",
    "vehicle": "vehicle",
    "unknown": "unknown",
}


def _compatibility_table() -> dict[str, dict[str, float]]:
    """Explicit symmetric mismatch penalties; 0 compatible, 1 incompatible."""
    classes = SEMANTIC_CLASS_NAMES + ("vehicle", "unknown")
    table = {
        first: {second: (0.0 if first == second else 1.0) for second in classes}
        for first in classes
    }
    # A broad LiDAR-side vehicle label is compatible with all road vehicles.
    for road_vehicle in ("car", "truck", "bus"):
        table["vehicle"][road_vehicle] = 0.0
        table[road_vehicle]["vehicle"] = 0.0
    # COCO often separates visually similar road-vehicle subclasses weakly;
    # retain a small, bounded distinction rather than treating it like a
    # vehicle-vs-person contradiction.
    for first in ("car", "truck", "bus"):
        for second in ("car", "truck", "bus"):
            if first != second:
                table[first][second] = 0.25
    # Two-wheel classes are related but not identical.
    table["motorcycle"]["bicycle"] = 0.5
    table["bicycle"]["motorcycle"] = 0.5
    # Unknown is always neutral; it cannot constitute contradictory evidence.
    for semantic_class in classes:
        table["unknown"][semantic_class] = 0.0
        table[semantic_class]["unknown"] = 0.0
    return table


DEFAULT_COMPATIBILITY_PENALTIES = _compatibility_table()


def normalize_semantic_class(value: str) -> str:
    normalized = str(value).strip().lower()
    normalized = _CLASS_ALIASES.get(normalized, normalized)
    if normalized not in DEFAULT_COMPATIBILITY_PENALTIES:
        raise ValueError(f"unsupported semantic class {value!r}")
    return normalized


@dataclass(frozen=True)
class SemanticEvidence:
    """One validated camera-to-LiDAR semantic observation."""

    semantic_class: str
    camera_confidence: float
    camera_lidar_iou: float
    semantic_valid: bool
    source_timestamp_seconds: float

    def __post_init__(self) -> None:
        normalized = normalize_semantic_class(self.semantic_class)
        object.__setattr__(self, "semantic_class", normalized)
        if self.semantic_valid and normalized in ("unknown", "vehicle"):
            raise ValueError("valid camera evidence requires a concrete camera class")
        for name in ("camera_confidence", "camera_lidar_iou"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and within [0, 1]")
        if not math.isfinite(self.source_timestamp_seconds):
            raise ValueError("source_timestamp_seconds must be finite")


@dataclass(frozen=True)
class SemanticDetection(Detection):
    """An AB3DMOT detection with optional side-channel camera evidence."""

    semantic_evidence: SemanticEvidence | None = None
    source_detection_index: int = -1


def semantic_reliability(evidence: SemanticEvidence | None) -> float:
    """Transparent bounded reliability: camera confidence times fusion IoU."""
    if evidence is None or not evidence.semantic_valid:
        return 0.0
    return min(1.0, max(0.0, evidence.camera_confidence * evidence.camera_lidar_iou))


def compatibility_penalty(
    track_class: str,
    detection_class: str,
    table: Mapping[str, Mapping[str, float]] = DEFAULT_COMPATIBILITY_PENALTIES,
) -> float:
    track_class = normalize_semantic_class(track_class)
    detection_class = normalize_semantic_class(detection_class)
    try:
        value = float(table[track_class][detection_class])
    except KeyError as exc:
        raise ValueError("compatibility table is missing a declared class pair") from exc
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("compatibility penalties must be finite and within [0, 1]")
    return value


@dataclass(frozen=True)
class SemanticMemorySnapshot:
    modal_class: str | None
    modal_confidence: float
    total_weight: float
    evidence_updates: int
    sufficient: bool


@dataclass
class SemanticMemory:
    """Track-private, bounded, exponentially decayed weighted class counts."""

    class_weights: dict[str, float] = field(default_factory=dict)
    evidence_updates: int = 0
    latest_evidence: SemanticEvidence | None = None

    def advance_frame(self, decay: float) -> None:
        if not math.isfinite(decay) or not 0.0 <= decay <= 1.0:
            raise ValueError("semantic memory decay must be within [0, 1]")
        self.class_weights = {
            semantic_class: weight * decay
            for semantic_class, weight in self.class_weights.items()
            if weight * decay > 1.0e-12
        }

    def update(self, evidence: SemanticEvidence | None, *, confidence_weighted: bool) -> None:
        if evidence is None or not evidence.semantic_valid:
            return
        weight = semantic_reliability(evidence) if confidence_weighted else 1.0
        if weight <= 0.0:
            return
        self.class_weights[evidence.semantic_class] = (
            self.class_weights.get(evidence.semantic_class, 0.0) + weight
        )
        self.evidence_updates += 1
        self.latest_evidence = evidence

    def clear_current_frame_evidence(self) -> None:
        self.latest_evidence = None

    def snapshot(
        self,
        *,
        memory_enabled: bool,
        confidence_weighted: bool,
        min_updates: int,
        min_total_weight: float,
    ) -> SemanticMemorySnapshot:
        if not memory_enabled:
            evidence = self.latest_evidence
            if evidence is None or not evidence.semantic_valid:
                return SemanticMemorySnapshot(None, 0.0, 0.0, 0, False)
            confidence = semantic_reliability(evidence) if confidence_weighted else 1.0
            return SemanticMemorySnapshot(
                evidence.semantic_class, confidence, confidence, 1, confidence > 0.0
            )

        total = float(sum(self.class_weights.values()))
        if not self.class_weights or total <= 0.0:
            return SemanticMemorySnapshot(None, 0.0, total, self.evidence_updates, False)
        modal_class = min(
            self.class_weights,
            key=lambda name: (-self.class_weights[name], name),
        )
        modal_confidence = self.class_weights[modal_class] / total
        sufficient = self.evidence_updates >= min_updates and total >= min_total_weight
        return SemanticMemorySnapshot(
            modal_class, modal_confidence, total, self.evidence_updates, sufficient
        )


@dataclass(frozen=True)
class SemanticAssociationConfig:
    """Research configuration; disabled and neutral by default."""

    semantic_association_enabled: bool = False
    shadow_mode: bool = False
    collect_counterfactual_diagnostics: bool = False
    lambda_sem: float = 0.0
    confidence_weighted: bool = True
    semantic_memory_enabled: bool = True
    memory_decay: float = 0.95
    min_track_evidence_updates: int = 2
    min_track_total_weight: float = 0.20
    compatibility_penalties: Mapping[str, Mapping[str, float]] = field(
        default_factory=_compatibility_table
    )

    def __post_init__(self) -> None:
        if not math.isfinite(self.lambda_sem) or self.lambda_sem < 0.0:
            raise ValueError("lambda_sem must be finite and non-negative")
        if not math.isfinite(self.memory_decay) or not 0.0 <= self.memory_decay <= 1.0:
            raise ValueError("memory_decay must be within [0, 1]")
        if self.min_track_evidence_updates < 1:
            raise ValueError("min_track_evidence_updates must be >= 1")
        if (
            not math.isfinite(self.min_track_total_weight)
            or self.min_track_total_weight < 0.0
        ):
            raise ValueError("min_track_total_weight must be finite and non-negative")
        # Validate every declared pair eagerly; no hidden magic fallback.
        for first in DEFAULT_COMPATIBILITY_PENALTIES:
            for second in DEFAULT_COMPATIBILITY_PENALTIES:
                compatibility_penalty(first, second, self.compatibility_penalties)


def semantic_penalty_matrix(
    detections: Sequence[Detection],
    track_snapshots: Sequence[SemanticMemorySnapshot],
    *,
    confidence_weighted: bool,
    compatibility_penalties: Mapping[str, Mapping[str, float]],
) -> np.ndarray:
    """Dimensionless [0,1] soft mismatch penalty, never a hard gate."""
    matrix = np.zeros((len(detections), len(track_snapshots)), dtype=float)
    for det_index, detection in enumerate(detections):
        evidence = getattr(detection, "semantic_evidence", None)
        if evidence is None or not evidence.semantic_valid:
            continue
        detection_weight = semantic_reliability(evidence) if confidence_weighted else 1.0
        if detection_weight <= 0.0:
            continue
        for track_index, snapshot in enumerate(track_snapshots):
            if not snapshot.sufficient or snapshot.modal_class is None:
                continue
            mismatch = compatibility_penalty(
                snapshot.modal_class,
                evidence.semantic_class,
                compatibility_penalties,
            )
            track_weight = snapshot.modal_confidence if confidence_weighted else 1.0
            matrix[det_index, track_index] = min(
                1.0, max(0.0, mismatch * track_weight * detection_weight)
            )
    return matrix


class SemanticAB3DMOTTracker(AB3DMOTTracker):
    """AB3DMOT with an opt-in soft semantic solver-cost augmentation.

    The superclass still owns prediction, geometric metric construction,
    matcher invocation, geometric gating, KF update, birth, and lifecycle.
    This subclass owns only side-channel semantic memories and the bounded
    auxiliary cost supplied through the no-op protected seam.
    """

    def __init__(self, config, semantic_config: SemanticAssociationConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self.semantic_config = semantic_config
        self.semantic_memories: dict[int, SemanticMemory] = {}
        self.semantic_diagnostics: list[dict] = []
        self._last_selected_pairs: list[tuple[int, int]] = []
        self._last_semantic_detail: dict | None = None
        self.last_semantic_memory_update_ms = 0.0

    def _matcher(self, cost_matrix: np.ndarray) -> np.ndarray:
        if self.config.matcher == "hungarian":
            return _hungarian_matching(cost_matrix)
        return _greedy_matching(cost_matrix)

    def _valid_mask(self, metric_matrix: np.ndarray) -> np.ndarray:
        if self.config.association_metric == "giou_3d":
            return metric_matrix >= self.config.giou_gate
        if self.config.association_metric == "euclidean":
            return metric_matrix <= self.config.euclidean_gate_m
        valid = metric_matrix <= self.config.mahalanobis_gate
        if self.config.mahalanobis_max_distance_m > 0.0:
            # The established core computes this optional physical cap after
            # the protected seam. Stage 3 deliberately uses Euclidean-3m, so
            # reject this unsupported diagnostic combination rather than
            # silently approximating its second gate here.
            raise ValueError("Stage-3 semantic diagnostics do not support Mahalanobis hybrid cap")
        return valid

    def _native_cost_scale(self) -> float:
        # lambda_sem is dimensionless relative to the active geometric gate.
        # For Stage 3's established Euclidean-3m baseline, lambda=0.1 means a
        # maximum 0.3 m-equivalent additive solver penalty.
        if self.config.association_metric == "euclidean":
            return self.config.euclidean_gate_m
        if self.config.association_metric == "mahalanobis":
            return self.config.mahalanobis_gate
        return 1.0

    def _accept_raw(self, raw: np.ndarray, valid_mask: np.ndarray) -> list[tuple[int, int]]:
        return [
            (int(det_index), int(track_index))
            for det_index, track_index in raw
            if valid_mask[int(det_index), int(track_index)]
        ]

    def _association_solver_cost_matrix(
        self,
        detections: Sequence[Detection],
        metric_matrix: np.ndarray,
        cost_matrix: np.ndarray,
    ) -> np.ndarray:
        cfg = self.semantic_config
        should_compute = cfg.shadow_mode or cfg.semantic_association_enabled
        if not should_compute or cfg.lambda_sem == 0.0:
            self._last_semantic_detail = {
                "semantic_cost_ms": 0.0,
                "shadow_solver_ms": 0.0,
                "penalty_matrix": np.zeros_like(cost_matrix),
                "augmented_cost_matrix": cost_matrix,
                "baseline_pairs": None,
                "semantic_pairs": None,
                "valid_mask": self._valid_mask(metric_matrix),
            }
            return cost_matrix

        started = time.perf_counter()
        snapshots = [
            self.semantic_memories.get(track.track_id, SemanticMemory()).snapshot(
                memory_enabled=cfg.semantic_memory_enabled,
                confidence_weighted=cfg.confidence_weighted,
                min_updates=cfg.min_track_evidence_updates,
                min_total_weight=cfg.min_track_total_weight,
            )
            for track in self._tracks
        ]
        unit_penalty = semantic_penalty_matrix(
            detections,
            snapshots,
            confidence_weighted=cfg.confidence_weighted,
            compatibility_penalties=cfg.compatibility_penalties,
        )
        penalty_matrix = unit_penalty * cfg.lambda_sem * self._native_cost_scale()
        augmented = cost_matrix + penalty_matrix
        semantic_cost_ms = (time.perf_counter() - started) * 1000.0

        valid_mask = self._valid_mask(metric_matrix)
        baseline_pairs = None
        semantic_pairs = None
        shadow_solver_ms = 0.0
        if cfg.shadow_mode or cfg.collect_counterfactual_diagnostics:
            shadow_started = time.perf_counter()
            baseline_raw = self._matcher(cost_matrix)
            semantic_raw = self._matcher(augmented)
            shadow_solver_ms = (time.perf_counter() - shadow_started) * 1000.0
            baseline_pairs = self._accept_raw(baseline_raw, valid_mask)
            semantic_pairs = self._accept_raw(semantic_raw, valid_mask)
        self._last_semantic_detail = {
            "semantic_cost_ms": semantic_cost_ms,
            "shadow_solver_ms": shadow_solver_ms,
            "penalty_matrix": penalty_matrix,
            "augmented_cost_matrix": augmented,
            "baseline_pairs": baseline_pairs,
            "semantic_pairs": semantic_pairs,
            "valid_mask": valid_mask,
            "track_snapshots": snapshots,
        }
        if cfg.semantic_association_enabled and not cfg.shadow_mode:
            return augmented
        return cost_matrix

    def _associate(self, detections: Sequence[Detection]) -> tuple[np.ndarray, np.ndarray]:
        matched_dets, matched_tracks = super()._associate(detections)
        self._last_selected_pairs = [
            (int(det_index), int(track_index))
            for det_index, track_index in zip(matched_dets, matched_tracks)
        ]
        if not self.collect_diagnostics:
            self._last_semantic_detail = None
            return matched_dets, matched_tracks
        detail = self._last_semantic_detail
        if detail is None:
            detail = {
                "semantic_cost_ms": 0.0,
                "shadow_solver_ms": 0.0,
                "penalty_matrix": np.zeros((len(detections), len(self._tracks))),
                "augmented_cost_matrix": None,
                "baseline_pairs": list(self._last_selected_pairs),
                "semantic_pairs": list(self._last_selected_pairs),
                "valid_mask": np.zeros((len(detections), len(self._tracks)), dtype=bool),
            }
        baseline_pairs = detail["baseline_pairs"]
        semantic_pairs = detail["semantic_pairs"]
        if baseline_pairs is None:
            baseline_pairs = list(self._last_selected_pairs)
        if semantic_pairs is None:
            semantic_pairs = list(self._last_selected_pairs)
        valid_mask = detail["valid_mask"]
        penalty_matrix = detail["penalty_matrix"]
        track_ids = [track.track_id for track in self._tracks]
        detection_source_indices = [
            int(getattr(detection, "source_detection_index", index))
            for index, detection in enumerate(detections)
        ]
        track_snapshots = detail.get("track_snapshots")
        if track_snapshots is None:
            track_snapshots = [
                self.semantic_memories.get(track_id, SemanticMemory()).snapshot(
                    memory_enabled=self.semantic_config.semantic_memory_enabled,
                    confidence_weighted=self.semantic_config.confidence_weighted,
                    min_updates=self.semantic_config.min_track_evidence_updates,
                    min_total_weight=self.semantic_config.min_track_total_weight,
                )
                for track_id in track_ids
            ]
        self.semantic_diagnostics.append(
            {
                "frame_index": self._frame_index,
                "n_detections": len(detections),
                "n_tracks": len(self._tracks),
                "candidate_pairs": int(np.sum(valid_mask)),
                "pairs_with_semantic_evidence": int(
                    np.sum(
                        valid_mask
                        & np.asarray(
                            [
                                [
                                    bool(
                                        getattr(det, "semantic_evidence", None)
                                        and getattr(det, "semantic_evidence").semantic_valid
                                        and self.semantic_memories.get(track.track_id, SemanticMemory())
                                        .snapshot(
                                            memory_enabled=self.semantic_config.semantic_memory_enabled,
                                            confidence_weighted=self.semantic_config.confidence_weighted,
                                            min_updates=self.semantic_config.min_track_evidence_updates,
                                            min_total_weight=self.semantic_config.min_track_total_weight,
                                        ).sufficient
                                    )
                                    for track in self._tracks
                                ]
                                for det in detections
                            ],
                            dtype=bool,
                        )
                    )
                ) if valid_mask.size else 0,
                "pairs_affected": int(np.sum((penalty_matrix > 0.0) & valid_mask)),
                "baseline_pairs": baseline_pairs,
                "semantic_pairs": semantic_pairs,
                "selected_pairs": list(self._last_selected_pairs),
                "assignment_changes": len(set(baseline_pairs) ^ set(semantic_pairs)),
                "track_ids": track_ids,
                "detection_source_indices": detection_source_indices,
                "track_semantic_snapshots": [
                    {
                        "modal_class": snapshot.modal_class,
                        "modal_confidence": snapshot.modal_confidence,
                        "total_weight": snapshot.total_weight,
                        "evidence_updates": snapshot.evidence_updates,
                        "sufficient": snapshot.sufficient,
                    }
                    for snapshot in track_snapshots
                ],
                "detection_semantics": [
                    {
                        "semantic_valid": bool(
                            getattr(detection, "semantic_evidence", None)
                            and getattr(detection, "semantic_evidence").semantic_valid
                        ),
                        "semantic_class": (
                            getattr(detection, "semantic_evidence").semantic_class
                            if getattr(detection, "semantic_evidence", None) else None
                        ),
                        "reliability": semantic_reliability(
                            getattr(detection, "semantic_evidence", None)
                        ),
                        "camera_confidence": (
                            getattr(detection, "semantic_evidence").camera_confidence
                            if getattr(detection, "semantic_evidence", None) else None
                        ),
                        "camera_lidar_iou": (
                            getattr(detection, "semantic_evidence").camera_lidar_iou
                            if getattr(detection, "semantic_evidence", None) else None
                        ),
                    }
                    for detection in detections
                ],
                "semantic_cost_ms": float(detail["semantic_cost_ms"]),
                "shadow_solver_ms": float(detail["shadow_solver_ms"]),
                "penalty_matrix": penalty_matrix.tolist(),
                "augmented_cost_matrix": (
                    detail["augmented_cost_matrix"].tolist()
                    if detail["augmented_cost_matrix"] is not None else None
                ),
            }
        )
        self._last_semantic_detail = None
        return matched_dets, matched_tracks

    def step(self, detections: Sequence[Detection], timestamp_seconds: float):
        pre_track_ids = [track.track_id for track in self._tracks]
        if self.semantic_config.semantic_memory_enabled:
            for track_id in pre_track_ids:
                self.semantic_memories.setdefault(track_id, SemanticMemory()).advance_frame(
                    self.semantic_config.memory_decay
                )

        outputs = super().step(detections, timestamp_seconds)

        update_started = time.perf_counter()
        matched_det_indices = {det_index for det_index, _ in self._last_selected_pairs}
        matched_track_indices = {track_index for _, track_index in self._last_selected_pairs}
        for det_index, track_index in self._last_selected_pairs:
            track_id = pre_track_ids[track_index]
            memory = self.semantic_memories.setdefault(track_id, SemanticMemory())
            if not self.semantic_config.semantic_memory_enabled:
                memory.clear_current_frame_evidence()
            memory.update(
                getattr(detections[det_index], "semantic_evidence", None),
                confidence_weighted=self.semantic_config.confidence_weighted,
            )

        if not self.semantic_config.semantic_memory_enabled:
            for track_index, track_id in enumerate(pre_track_ids):
                if track_index not in matched_track_indices:
                    self.semantic_memories.setdefault(
                        track_id, SemanticMemory()
                    ).clear_current_frame_evidence()

        unmatched_det_indices = [
            det_index for det_index in range(len(detections))
            if det_index not in matched_det_indices
        ]
        pre_track_id_set = set(pre_track_ids)
        new_tracks = [track for track in self._tracks if track.track_id not in pre_track_id_set]
        if len(new_tracks) != len(unmatched_det_indices):
            raise RuntimeError("semantic memory birth mapping diverged from AB3DMOT lifecycle")
        for det_index, track in zip(unmatched_det_indices, new_tracks):
            memory = self.semantic_memories.setdefault(track.track_id, SemanticMemory())
            memory.update(
                getattr(detections[det_index], "semantic_evidence", None),
                confidence_weighted=self.semantic_config.confidence_weighted,
            )

        live_ids = {track.track_id for track in self._tracks}
        self.semantic_memories = {
            track_id: memory
            for track_id, memory in self.semantic_memories.items()
            if track_id in live_ids
        }
        self.last_semantic_memory_update_ms = (time.perf_counter() - update_started) * 1000.0
        if self.semantic_diagnostics:
            self.semantic_diagnostics[-1]["semantic_memory_update_ms"] = (
                self.last_semantic_memory_update_ms
            )
        return outputs
