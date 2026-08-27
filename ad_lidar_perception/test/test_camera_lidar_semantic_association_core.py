"""Focused Stage-3 semantic-assisted AB3DMOT association tests."""
import math
import unittest

import numpy as np

from ad_lidar_perception.ab3dmot_config import AB3DMOTConfig
from ad_lidar_perception.ab3dmot_core import AB3DMOTTracker
from ad_lidar_perception.camera_lidar_semantic_association_core import (
    SemanticAB3DMOTTracker,
    SemanticAssociationConfig,
    SemanticDetection,
    SemanticEvidence,
    SemanticMemory,
    SemanticMemorySnapshot,
    compatibility_penalty,
    semantic_penalty_matrix,
    semantic_reliability,
)


def evidence(label="car", confidence=0.8, iou=0.5, valid=True, timestamp=1.0):
    return SemanticEvidence(label, confidence, iou, valid, timestamp)


def detection(x=0.0, semantic=None, source_index=-1):
    return SemanticDetection(
        x=x,
        y=0.0,
        z=0.0,
        yaw=0.0,
        length=4.0,
        width=2.0,
        height=1.5,
        semantic_evidence=semantic,
        source_detection_index=source_index,
    )


def tracker_config():
    return AB3DMOTConfig(
        association_metric="euclidean",
        matcher="hungarian",
        euclidean_gate_m=3.0,
        state_estimator="linear_kf",
        yaw_measurement_mode="unobserved",
    )


class EvidenceAndCompatibilityTest(unittest.TestCase):
    def test_reliability_is_confidence_times_iou_and_bounded(self):
        self.assertAlmostEqual(semantic_reliability(evidence(confidence=0.8, iou=0.25)), 0.2)
        self.assertEqual(semantic_reliability(None), 0.0)
        self.assertEqual(semantic_reliability(evidence(valid=False)), 0.0)
        with self.assertRaises(ValueError):
            evidence(confidence=1.01)
        with self.assertRaises(ValueError):
            evidence(iou=-0.01)

    def test_compatibility_matrix_is_explicit_and_alias_aware(self):
        self.assertEqual(compatibility_penalty("car", "car"), 0.0)
        self.assertEqual(compatibility_penalty("vehicle", "bus"), 0.0)
        self.assertEqual(compatibility_penalty("pedestrian", "person"), 0.0)
        self.assertEqual(compatibility_penalty("car", "truck"), 0.25)
        self.assertEqual(compatibility_penalty("motorcycle", "bicycle"), 0.5)
        self.assertEqual(compatibility_penalty("person", "car"), 1.0)
        self.assertEqual(compatibility_penalty("unknown", "person"), 0.0)

    def test_valid_unknown_camera_evidence_is_rejected(self):
        with self.assertRaises(ValueError):
            evidence(label="unknown", valid=True)


class SemanticMemoryTest(unittest.TestCase):
    def test_memory_initializes_empty_and_requires_repeated_evidence(self):
        memory = SemanticMemory()
        empty = memory.snapshot(
            memory_enabled=True, confidence_weighted=True,
            min_updates=2, min_total_weight=0.2,
        )
        self.assertFalse(empty.sufficient)
        memory.update(evidence(), confidence_weighted=True)
        once = memory.snapshot(
            memory_enabled=True, confidence_weighted=True,
            min_updates=2, min_total_weight=0.2,
        )
        self.assertFalse(once.sufficient)
        memory.update(evidence(), confidence_weighted=True)
        twice = memory.snapshot(
            memory_enabled=True, confidence_weighted=True,
            min_updates=2, min_total_weight=0.2,
        )
        self.assertTrue(twice.sufficient)
        self.assertEqual(twice.modal_class, "car")

    def test_missing_evidence_is_not_a_contradiction(self):
        memory = SemanticMemory()
        memory.update(evidence(), confidence_weighted=True)
        before = dict(memory.class_weights)
        memory.update(None, confidence_weighted=True)
        memory.update(evidence(valid=False), confidence_weighted=True)
        self.assertEqual(memory.class_weights, before)
        self.assertEqual(memory.evidence_updates, 1)

    def test_decay_bounds_memory_without_changing_modal_class(self):
        memory = SemanticMemory()
        memory.update(evidence(), confidence_weighted=True)
        before = sum(memory.class_weights.values())
        memory.advance_frame(0.95)
        self.assertAlmostEqual(sum(memory.class_weights.values()), before * 0.95)
        snapshot = memory.snapshot(
            memory_enabled=True, confidence_weighted=True,
            min_updates=1, min_total_weight=0.0,
        )
        self.assertEqual(snapshot.modal_class, "car")

    def test_single_class_flip_cannot_replace_stable_history(self):
        memory = SemanticMemory()
        for _ in range(5):
            memory.update(evidence("car", 0.9, 0.5), confidence_weighted=True)
        memory.update(evidence("truck", 0.3, 0.11), confidence_weighted=True)
        snapshot = memory.snapshot(
            memory_enabled=True, confidence_weighted=True,
            min_updates=2, min_total_weight=0.2,
        )
        self.assertEqual(snapshot.modal_class, "car")
        self.assertGreater(snapshot.modal_confidence, 0.95)


class SemanticPenaltyTest(unittest.TestCase):
    def test_missing_track_or_detection_evidence_is_exactly_neutral(self):
        ready = SemanticMemorySnapshot("car", 1.0, 1.0, 2, True)
        insufficient = SemanticMemorySnapshot("car", 1.0, 1.0, 1, False)
        matrix = semantic_penalty_matrix(
            [detection(semantic=None), detection(semantic=evidence("person"))],
            [ready, insufficient],
            confidence_weighted=True,
            compatibility_penalties=SemanticAssociationConfig().compatibility_penalties,
        )
        self.assertTrue(np.array_equal(matrix[0], np.zeros(2)))
        self.assertEqual(matrix[1, 1], 0.0)

    def test_same_class_is_zero_and_incompatible_class_is_bounded(self):
        ready = SemanticMemorySnapshot("car", 0.8, 1.0, 3, True)
        matrix = semantic_penalty_matrix(
            [detection(semantic=evidence("car")), detection(semantic=evidence("person"))],
            [ready],
            confidence_weighted=True,
            compatibility_penalties=SemanticAssociationConfig().compatibility_penalties,
        )
        self.assertEqual(matrix[0, 0], 0.0)
        self.assertGreater(matrix[1, 0], 0.0)
        self.assertLessEqual(matrix[1, 0], 1.0)

    def test_low_confidence_and_low_iou_limit_penalty(self):
        ready = SemanticMemorySnapshot("car", 1.0, 1.0, 3, True)
        matrix = semantic_penalty_matrix(
            [detection(semantic=evidence("person", 0.26, 0.11))],
            [ready],
            confidence_weighted=True,
            compatibility_penalties=SemanticAssociationConfig().compatibility_penalties,
        )
        self.assertAlmostEqual(matrix[0, 0], 0.0286)


class TrackerIntegrationTest(unittest.TestCase):
    def _run(self, semantic_config, frames):
        tracker = SemanticAB3DMOTTracker(
            tracker_config(), semantic_config, collect_diagnostics=True
        )
        outputs = []
        for index, frame in enumerate(frames):
            outputs.append(tracker.step(frame, 1.0 + index * 0.1))
        signature = [
            [(state.track_id, state.x, state.hits, state.time_since_update) for state in frame]
            for frame in outputs
        ]
        return tracker, signature

    def test_semantic_disabled_matches_original_tracker_numerically(self):
        frames = [
            [detection(0.0, evidence("car"))],
            [detection(0.1, evidence("car"))],
            [detection(0.2, evidence("person"))],
            [detection(0.3, None)],
        ]
        baseline = AB3DMOTTracker(tracker_config())
        baseline_signature = []
        for index, frame in enumerate(frames):
            out = baseline.step(frame, 1.0 + index * 0.1)
            baseline_signature.append(
                [(state.track_id, state.x, state.hits, state.time_since_update) for state in out]
            )
        _, semantic_signature = self._run(SemanticAssociationConfig(), frames)
        self.assertEqual(semantic_signature, baseline_signature)

    def test_lambda_zero_is_exact_baseline_even_when_enabled(self):
        frames = [[detection(i * 0.1, evidence("car"))] for i in range(5)]
        disabled, sig_disabled = self._run(SemanticAssociationConfig(), frames)
        enabled, sig_enabled = self._run(
            SemanticAssociationConfig(semantic_association_enabled=True, lambda_sem=0.0), frames
        )
        self.assertEqual(sig_enabled, sig_disabled)
        for diag in enabled.semantic_diagnostics:
            self.assertEqual(diag["pairs_affected"], 0)

    def test_shadow_mode_does_not_drive_track_update(self):
        frames = [
            [detection(0.0, evidence("car")), detection(2.0, evidence("person"))],
            [detection(0.1, evidence("car")), detection(2.1, evidence("person"))],
            [detection(1.0, evidence("person")), detection(1.1, evidence("car"))],
        ]
        _, baseline = self._run(SemanticAssociationConfig(), frames)
        shadow_tracker, shadow = self._run(
            SemanticAssociationConfig(shadow_mode=True, lambda_sem=1.0), frames
        )
        self.assertEqual(shadow, baseline)
        self.assertTrue(any(d["pairs_affected"] >= 0 for d in shadow_tracker.semantic_diagnostics))

    def test_track_memories_are_private(self):
        tracker, _ = self._run(
            SemanticAssociationConfig(),
            [[detection(0.0, evidence("car")), detection(10.0, evidence("person"))]],
        )
        self.assertEqual(len(tracker.semantic_memories), 2)
        memories = list(tracker.semantic_memories.values())
        self.assertIsNot(memories[0], memories[1])
        self.assertNotEqual(memories[0].latest_evidence.semantic_class,
                            memories[1].latest_evidence.semantic_class)

    def test_no_hard_rejection_and_one_to_one_assignment_preserved(self):
        frames = [
            [detection(0.0, evidence("car")), detection(5.0, evidence("person"))],
            [detection(0.1, evidence("car")), detection(5.1, evidence("person"))],
            [detection(0.2, evidence("person")), detection(5.2, evidence("car"))],
        ]
        tracker, _ = self._run(
            SemanticAssociationConfig(
                semantic_association_enabled=True,
                lambda_sem=1.0,
                min_track_evidence_updates=2,
            ),
            frames,
        )
        diag = tracker.semantic_diagnostics[-1]
        self.assertTrue(np.all(np.isfinite(np.asarray(diag["augmented_cost_matrix"]))))
        selected = diag["selected_pairs"]
        self.assertEqual(len({pair[0] for pair in selected}), len(selected))
        self.assertEqual(len({pair[1] for pair in selected}), len(selected))

    def test_assignment_is_deterministic(self):
        frames = [
            [detection(0.0, evidence("car")), detection(3.0, evidence("person"))],
            [detection(0.2, evidence("car")), detection(3.2, evidence("person"))],
            [detection(1.4, evidence("person")), detection(1.6, evidence("car"))],
        ]
        cfg = SemanticAssociationConfig(
            semantic_association_enabled=True, lambda_sem=0.5,
            min_track_evidence_updates=2,
        )
        first, signature_first = self._run(cfg, frames)
        second, signature_second = self._run(cfg, frames)
        self.assertEqual(signature_first, signature_second)
        diagnostic_signature_first = [
            (entry["baseline_pairs"], entry["semantic_pairs"], entry["selected_pairs"],
             entry["candidate_pairs"], entry["pairs_affected"])
            for entry in first.semantic_diagnostics
        ]
        diagnostic_signature_second = [
            (entry["baseline_pairs"], entry["semantic_pairs"], entry["selected_pairs"],
             entry["candidate_pairs"], entry["pairs_affected"])
            for entry in second.semantic_diagnostics
        ]
        self.assertEqual(diagnostic_signature_first, diagnostic_signature_second)


if __name__ == "__main__":
    unittest.main()
