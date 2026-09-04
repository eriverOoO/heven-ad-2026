"""Unit tests for ``evaluate_kalmannet.py``: KF/KNet common sample
stream, metric bucketing, and the "KF params are transferred-without-
retuning constants, never fit against test data" invariant.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

_AD_LIDAR_PKG_DIR = Path(__file__).resolve().parents[2] / "ad_lidar_perception"
sys.path.insert(0, str(_AD_LIDAR_PKG_DIR))

import evaluate_kalmannet as ek  # noqa: E402
from ad_lidar_perception.kalmannet_core import KalmanNetGRU  # noqa: E402


def _cv_sequence(n=15, dt=0.1, x0=0.0, y0=0.0, vx=3.0, vy=1.0, missing_indices=()):
    frames = list(range(n))
    dts = [None] + [dt] * (n - 1)
    x_true, z_meas = [], []
    for i in range(n):
        x, y = x0 + vx * dt * i, y0 + vy * dt * i
        x_true.append([x, y, vx, vy])
        z_meas.append(None if i in missing_indices else [x, y])
    return {"actor_id": f"synthetic-{x0}", "frames": frames, "dt": dts, "x_true": x_true,
            "z_meas": z_meas, "coarse_object_type": "VEHICLE"}


def test_kf_baseline_tracks_a_clean_constant_velocity_sequence_well():
    seq = _cv_sequence(n=30)
    out = ek.run_kf(seq)
    assert len(out) == 30
    final_err = abs(out[-1][0] - seq["x_true"][-1][0])
    assert final_err < 5.0  # loose bound -- just confirms the KF is actually tracking, not diverging


def test_knet_and_kf_see_the_identical_sequence_stream():
    """Both estimators must be evaluated against the SAME z_meas per
    frame -- confirmed by checking evaluate_estimator produces identical
    bucket sample counts (n) for both, which can only happen if both
    walked the identical set of sequences/frames."""
    torch.manual_seed(0)
    net = KalmanNetGRU(hidden_size=8)
    net.eval()
    sequences = [_cv_sequence(n=20, x0=float(i), missing_indices={5, 6} if i % 2 else set()) for i in range(4)]

    knet_report = ek.evaluate_estimator(sequences, lambda s: ek.run_knet(net, s, "cpu"))
    kf_report = ek.evaluate_estimator(sequences, ek.run_kf)

    assert set(knet_report["buckets"]) == set(kf_report["buckets"])
    for name in knet_report["buckets"]:
        assert knet_report["buckets"][name]["n"] == kf_report["buckets"][name]["n"], (
            f"bucket {name!r}: knet saw {knet_report['buckets'][name]['n']} samples, "
            f"kf saw {kf_report['buckets'][name]['n']} -- not the same sample stream"
        )


def test_matched_and_missing_buckets_partition_overall():
    sequences = [_cv_sequence(n=20, missing_indices={3, 4, 5})]
    report = ek.evaluate_estimator(sequences, ek.run_kf)
    n_matched = report["buckets"]["matched"]["n"]
    n_missing = report["buckets"]["missing"]["n"]
    n_overall = report["buckets"]["overall"]["n"]
    assert n_matched + n_missing == n_overall


def test_gap_length_bucketing_classifies_correctly():
    assert ek._bucket_for_gap(0) == "matched"
    assert ek._bucket_for_gap(1) == "gap_1_2"
    assert ek._bucket_for_gap(2) == "gap_1_2"
    assert ek._bucket_for_gap(3) == "gap_3_5"
    assert ek._bucket_for_gap(5) == "gap_3_5"
    assert ek._bucket_for_gap(6) == "gap_6_plus"
    assert ek._bucket_for_gap(20) == "gap_6_plus"


def test_class_stratified_buckets_report_n_alongside_metric():
    sequences = [_cv_sequence(n=15)]
    report = ek.evaluate_estimator(sequences, ek.run_kf)
    assert "class::VEHICLE" in report["buckets"]
    assert report["buckets"]["class::VEHICLE"]["n"] > 0
    assert "position_rmse_m" in report["buckets"]["class::VEHICLE"]


def test_divergence_and_nonfinite_are_counted_not_silently_dropped():
    def exploding_estimator(seq):
        return [[1e9, 1e9, 0.0, 0.0] for _ in seq["frames"]]

    sequences = [_cv_sequence(n=10)]
    report = ek.evaluate_estimator(sequences, exploding_estimator)
    assert report["divergence_count"] == 10
    assert "overall" not in report["buckets"]  # every sample diverged, none counted toward RMSE


def test_tuned_kf_params_are_literal_constants_never_fit_on_test_data():
    """The KF baseline's sigma_a/r_std/p0_scale must be literal module
    constants (transferred from the historical HEVEN tuning, never a
    function call that could read/fit against the AV2 test split)."""
    source = inspect.getsource(ek)
    assert "TUNED_HEVEN_KF_SIGMA_A = 5.0" in source
    assert "TUNED_HEVEN_KF_R_STD = 0.2254682076528578" in source
    assert "TUNED_HEVEN_KF_P0_SCALE = 10.0" in source
    # the default run_kf() signature must reference these constants directly,
    # not a fit_kf(...) / calibrate(...) style call.
    default_sigma_a = inspect.signature(ek.run_kf).parameters["sigma_a"].default
    assert default_sigma_a == ek.TUNED_HEVEN_KF_SIGMA_A
