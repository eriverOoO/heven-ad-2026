"""Runtime Readiness v1: unit tests for offline_dry_run.py and
latency_benchmark.py, using a tiny synthetic checkpoint fixture (never
the large real production candidate) and a small synthetic detection
stream (never requiring the real recorded JSONL to be present)."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ad_lidar_perception"))

from ad_lidar_perception.kalmannet_core import KalmanNetGRU, set_seed
from offline_dry_run import load_frames, run_dry_run
from latency_benchmark import benchmark


def make_tiny_checkpoint(path: Path, hidden_size: int = 4, seed: int = 0) -> None:
    set_seed(seed)
    net = KalmanNetGRU(hidden_size=hidden_size)
    torch.save(net.state_dict(), path)


def write_synthetic_jsonl(path: Path, n_frames: int = 20) -> None:
    with path.open("w") as fh:
        for i in range(n_frames):
            row = {
                "stamp_ns": 1_000_000_000 + i * 100_000_000,
                "objects": [
                    {
                        "x": float(i) * 0.3, "y": 0.0, "z": 1.0,
                        "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0,
                        "length": 4.0, "width": 2.0, "height": 1.5,
                        "class_label": 0, "score": 1.0, "existence_probability": 1.0,
                    }
                ],
            }
            fh.write(json.dumps(row) + "\n")


class OfflineDryRunTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.ckpt_path = Path(self.tmpdir.name) / "tiny.pt"
        make_tiny_checkpoint(self.ckpt_path)
        self.jsonl_path = Path(self.tmpdir.name) / "detections.jsonl"
        write_synthetic_jsonl(self.jsonl_path)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_load_frames_derives_real_timestamps(self):
        frames = load_frames(self.jsonl_path)
        self.assertEqual(len(frames), 20)
        t0, dets0 = frames[0]
        self.assertAlmostEqual(t0, 1.0)
        self.assertEqual(len(dets0), 1)

    def test_linear_kf_dry_run_produces_finite_output_no_exception(self):
        frames = load_frames(self.jsonl_path)
        result = run_dry_run(frames, state_estimator="linear_kf")
        self.assertIsNone(result["exception"])
        self.assertEqual(result["n_nonfinite"], 0)
        self.assertGreater(result["n_output_states"], 0)

    def test_kalmannet_dry_run_produces_finite_output_no_exception(self):
        frames = load_frames(self.jsonl_path)
        result = run_dry_run(
            frames, state_estimator="kalmannet", kalmannet_checkpoint=str(self.ckpt_path),
        )
        self.assertIsNone(result["exception"])
        self.assertEqual(result["n_nonfinite"], 0)
        self.assertGreater(result["n_output_states"], 0)
        self.assertIsNotNone(result["kalmannet_provenance"])
        # exactly one hidden-state object for the one persistent track.
        self.assertEqual(result["n_distinct_hidden_state_objects_at_end"], 1)

    def test_max_frames_truncates(self):
        frames = load_frames(self.jsonl_path, max_frames=5)
        self.assertEqual(len(frames), 5)


class LatencyBenchmarkTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.ckpt_path = Path(self.tmpdir.name) / "tiny.pt"
        make_tiny_checkpoint(self.ckpt_path)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_linear_kf_benchmark_runs_and_reports_finite_stats(self):
        result = benchmark("linear_kf", n_tracks=3, iterations=5)
        self.assertGreater(result["mean_ms"], 0.0)
        self.assertGreaterEqual(result["p95_ms"], result["median_ms"])
        self.assertEqual(result["n_live_tracks_at_end"], 3)

    def test_kalmannet_benchmark_runs_and_reports_finite_stats(self):
        result = benchmark(
            "kalmannet", n_tracks=3, iterations=5, kalmannet_checkpoint=str(self.ckpt_path),
        )
        self.assertGreater(result["mean_ms"], 0.0)
        self.assertEqual(result["n_live_tracks_at_end"], 3)


if __name__ == "__main__":
    unittest.main()
