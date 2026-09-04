"""Versioned checkpoint + reproducibility-manifest helpers for the AV2
KalmanNet Stage-0 trainer. A distinct checkpoint family from the
historical ``dense_kalmannet_v2.pt`` -- never overwrites it.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import torch

CHECKPOINT_FAMILY = "av2_stage0_kalmannet_v1"
CHECKPOINT_SCHEMA_VERSION = "av2_stage0_kalmannet_checkpoint_v1"


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repository_root(),
            check=True, text=True, capture_output=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def environment_info() -> dict[str, Any]:
    info = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    return info


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_json(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def build_checkpoint_manifest(
    *,
    checkpoint_family: str,
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    seed_info: dict[str, Any],
    corruption_config: dict[str, Any],
    dataset_manifest_sha256: str,
    split_manifest_sha256: str,
    coordinate_mode: str,
    best_epoch: int | None,
    best_val_loss: float,
    loss_on_predict_only: bool,
) -> dict[str, Any]:
    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_family": checkpoint_family,
        "architecture": "KalmanNetGRU (ad_lidar_perception/ad_lidar_perception/kalmannet_core.py, unmodified)",
        "state_fields": ["x", "y", "vx", "vy"],
        "measurement_fields": ["x", "y"],
        "model_config": model_config,
        "training_config": training_config,
        "loss_on_predict_only": loss_on_predict_only,
        "seed_info": seed_info,
        "corruption_config": corruption_config,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "split_manifest_sha256": split_manifest_sha256,
        "coordinate_mode": coordinate_mode,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "git_sha": git_sha(),
        "environment": environment_info(),
    }


def save_checkpoint(state_dict: dict, manifest: dict[str, Any], path: Path) -> str:
    """Saves ``{"state_dict": ..., **manifest}`` -- the same wrapped-dict
    shape ``ab3dmot_core.load_kalmannet_network`` already knows how to
    read (``state_dict = raw["state_dict"] if ... else raw``), so this
    checkpoint family stays structurally loadable by the existing runtime
    loader in principle, even though this task does not wire it in (task
    instruction: no runtime integration in this task)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state_dict": state_dict, **manifest}
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    checksum = sha256_file(path)
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    manifest_with_hash = {**manifest, "checkpoint_sha256": checksum, "checkpoint_path": str(path)}
    manifest_path.write_text(json.dumps(manifest_with_hash, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return checksum


def load_checkpoint(path: Path) -> tuple[dict, dict[str, Any]]:
    raw = torch.load(Path(path), map_location="cpu")
    if isinstance(raw, dict) and "state_dict" in raw:
        state_dict = raw["state_dict"]
        manifest = {k: v for k, v in raw.items() if k != "state_dict"}
    else:
        state_dict = raw
        manifest = {}
    return state_dict, manifest
