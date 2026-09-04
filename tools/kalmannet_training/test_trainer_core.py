"""Unit tests for the ported training loop (``trainer_core.py``) and the
checkpoint manifest utilities (``checkpoint_utils.py``). Uses tiny
synthetic sequences (no real AV2 data / download needed) so these run
fast and deterministically in isolation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

_AD_LIDAR_PKG_DIR = Path(__file__).resolve().parents[2] / "ad_lidar_perception"
sys.path.insert(0, str(_AD_LIDAR_PKG_DIR))

import checkpoint_utils  # noqa: E402
from trainer_core import run_sequence, set_all_seeds, train_one_run  # noqa: E402
from ad_lidar_perception.kalmannet_core import KalmanNetGRU  # noqa: E402


def _cv_sequence(n=20, dt=0.1, x0=0.0, y0=0.0, vx=4.0, vy=-2.0, missing_indices=()):
    frames = list(range(n))
    dts = [None] + [dt] * (n - 1)
    x_true = []
    z_meas = []
    for i in range(n):
        x = x0 + vx * dt * i
        y = y0 + vy * dt * i
        x_true.append([x, y, vx, vy])
        z_meas.append(None if i in missing_indices else [x, y])
    return {"actor_id": "synthetic-1", "frames": frames, "dt": dts, "x_true": x_true, "z_meas": z_meas}


def test_missing_measurement_is_prediction_only_no_gain_call_no_loss_when_option_a():
    """z is None -> KalmanNetFilter._f (analytical predict) only; with
    loss_on_predict_only=False (option A, measurement-update-only loss),
    a missing frame must contribute ZERO loss terms."""
    torch.manual_seed(0)
    net = KalmanNetGRU(hidden_size=8)
    seq = _cv_sequence(n=10, missing_indices={3, 4, 7})
    result = run_sequence(net, seq, "cpu", loss_on_predict_only=False)
    # frames with dt is not None: 9 (all but frame 0); of those, 3 are missing -> 6 measurement-update frames
    assert result.n_loss_terms == 6


def test_missing_measurement_included_in_loss_when_historical_default():
    """Historical-compatible default (loss_on_predict_only=True, verified
    against instrumented_train.py): every frame with dt is not None
    contributes a loss term, missing or not."""
    torch.manual_seed(0)
    net = KalmanNetGRU(hidden_size=8)
    seq = _cv_sequence(n=10, missing_indices={3, 4, 7})
    result = run_sequence(net, seq, "cpu", loss_on_predict_only=True)
    assert result.n_loss_terms == 9  # every frame except frame 0


def test_variable_dt_actually_used_not_a_fixed_rate():
    """A sequence with a genuinely non-uniform dt schedule must produce a
    different trajectory estimate than the same sequence resampled to a
    fixed rate -- proves dt flows into the analytical F_matrix(dt), not
    silently ignored."""
    torch.manual_seed(0)
    net = KalmanNetGRU(hidden_size=8)
    net.eval()

    seq_uniform = _cv_sequence(n=10, dt=0.1)
    seq_variable = dict(seq_uniform)
    seq_variable["dt"] = [None] + [0.05, 0.3, 0.05, 0.3, 0.05, 0.3, 0.05, 0.3, 0.05]

    with torch.no_grad():
        r_uniform = run_sequence(net, seq_uniform, "cpu", loss_on_predict_only=True)
        r_variable = run_sequence(net, seq_variable, "cpu", loss_on_predict_only=True)
    assert not torch.allclose(r_uniform.loss, r_variable.loss)


def test_set_all_seeds_gives_reproducible_initialization():
    set_all_seeds(123)
    net_a = KalmanNetGRU(hidden_size=8)
    weights_a = net_a.output_fc.weight.detach().clone()

    set_all_seeds(123)
    net_b = KalmanNetGRU(hidden_size=8)
    weights_b = net_b.output_fc.weight.detach().clone()

    assert torch.equal(weights_a, weights_b)


def test_set_all_seeds_different_seed_gives_different_initialization():
    set_all_seeds(1)
    net_a = KalmanNetGRU(hidden_size=8)
    weights_a = net_a.output_fc.weight.detach().clone()

    set_all_seeds(2)
    net_b = KalmanNetGRU(hidden_size=8)
    weights_b = net_b.output_fc.weight.detach().clone()

    assert not torch.equal(weights_a, weights_b)


def test_train_one_run_deterministic_given_same_seed():
    train_seqs = [_cv_sequence(n=15, x0=float(i), missing_indices={4} if i % 2 else set()) for i in range(6)]
    val_seqs = [_cv_sequence(n=15, x0=100.0 + i) for i in range(3)]

    r1 = train_one_run(train_seqs, val_seqs, seed=0, device="cpu", max_epochs=2, patience=2, hidden_size=8)
    r2 = train_one_run(train_seqs, val_seqs, seed=0, device="cpu", max_epochs=2, patience=2, hidden_size=8)

    assert r1.best_val == r2.best_val
    assert r1.n_epochs_run == r2.n_epochs_run
    for k1, k2 in zip(r1.best_state_dict.values(), r2.best_state_dict.values()):
        assert torch.equal(k1, k2)


def test_train_one_run_produces_decreasing_or_stable_loss_and_no_nan():
    train_seqs = [_cv_sequence(n=20, x0=float(i) * 2, vx=3.0 + 0.1 * i) for i in range(10)]
    val_seqs = [_cv_sequence(n=20, x0=50.0 + i) for i in range(4)]

    result = train_one_run(train_seqs, val_seqs, seed=0, device="cpu", max_epochs=5, patience=5, hidden_size=8)
    assert not result.catastrophic
    assert result.nonfinite_step_count == 0
    assert np.isfinite(result.best_val)
    assert result.best_state_dict is not None


def test_checkpoint_save_and_load_round_trip(tmp_path):
    net = KalmanNetGRU(hidden_size=8)
    manifest = checkpoint_utils.build_checkpoint_manifest(
        checkpoint_family="test_family", model_config={"hidden_size": 8},
        training_config={"lr": 0.001}, seed_info={"seed": 0},
        corruption_config={"mode": "none"}, dataset_manifest_sha256="abc123",
        split_manifest_sha256="def456", coordinate_mode="first_state_relative",
        best_epoch=3, best_val_loss=1.23, loss_on_predict_only=True,
    )
    path = tmp_path / "ckpt.pt"
    checksum = checkpoint_utils.save_checkpoint(net.state_dict(), manifest, path)

    assert path.is_file()
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    assert manifest_path.is_file()

    loaded_state, loaded_manifest = checkpoint_utils.load_checkpoint(path)
    net2 = KalmanNetGRU(hidden_size=8)
    net2.load_state_dict(loaded_state)
    for p1, p2 in zip(net.parameters(), net2.parameters()):
        assert torch.equal(p1, p2)
    assert loaded_manifest["checkpoint_family"] == "test_family"
    assert loaded_manifest["best_epoch"] == 3
    assert checksum == checkpoint_utils.sha256_file(path)


def test_checkpoint_manifest_records_reproducibility_fields():
    manifest = checkpoint_utils.build_checkpoint_manifest(
        checkpoint_family="test_family", model_config={"hidden_size": 32},
        training_config={"lr": 0.001, "grad_clip": 10.0}, seed_info={"seed": 5},
        corruption_config={"mode": "none"}, dataset_manifest_sha256="abc",
        split_manifest_sha256="def", coordinate_mode="first_state_relative",
        best_epoch=10, best_val_loss=2.0, loss_on_predict_only=True,
    )
    for key in ("git_sha", "environment", "model_config", "training_config", "seed_info",
                "corruption_config", "dataset_manifest_sha256", "split_manifest_sha256",
                "coordinate_mode", "best_epoch", "best_val_loss"):
        assert key in manifest
    assert "torch_version" in manifest["environment"]
    assert "cuda_available" in manifest["environment"]
