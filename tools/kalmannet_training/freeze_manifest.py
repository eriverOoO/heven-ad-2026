"""Freeze-manifest helpers for the batched-optimizer calibration task.

A "freeze" is the explicit, written record of the ONE selected training
configuration, produced from TRAIN/VALIDATION evidence only -- it exists
so that Stage-0 TEST is evaluated exactly ONCE, against a configuration
already committed to disk BEFORE that evaluation happens, never
re-selected afterward. ``require_freeze_manifest_exists`` is a small,
practical guard a TEST-evaluation script can call first, so "evaluate
TEST before freezing" is a hard, catchable error rather than a silent
process discipline.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


class FreezeManifestMissingError(RuntimeError):
    """Raised when TEST evaluation is attempted before a freeze manifest
    has been written -- see ``require_freeze_manifest_exists``."""


@dataclass
class FreezeManifest:
    schema_version: str
    batch_size: int
    learning_rate: float
    max_epochs: int
    patience: int
    gradient_clip: float
    loss_on_predict_only: bool
    bucket_policy: dict[str, Any]
    shuffle_seed_policy: dict[str, Any]
    corruption_policy: dict[str, Any]
    model_architecture: dict[str, Any]
    selection_evidence: dict[str, Any]
    three_seed_validation_summary: dict[str, Any]
    frozen_at_git_sha: str
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


FREEZE_MANIFEST_SCHEMA_VERSION = "kalmannet_batch_calibration_freeze_v1"


def build_freeze_manifest(
    *,
    batch_size: int,
    learning_rate: float,
    max_epochs: int,
    patience: int,
    gradient_clip: float,
    loss_on_predict_only: bool,
    bucket_policy: dict[str, Any],
    shuffle_seed_policy: dict[str, Any],
    corruption_policy: dict[str, Any],
    model_architecture: dict[str, Any],
    selection_evidence: dict[str, Any],
    three_seed_validation_summary: dict[str, Any],
    git_sha: str,
    notes: str = "",
) -> FreezeManifest:
    return FreezeManifest(
        schema_version=FREEZE_MANIFEST_SCHEMA_VERSION,
        batch_size=batch_size, learning_rate=learning_rate, max_epochs=max_epochs,
        patience=patience, gradient_clip=gradient_clip, loss_on_predict_only=loss_on_predict_only,
        bucket_policy=bucket_policy, shuffle_seed_policy=shuffle_seed_policy,
        corruption_policy=corruption_policy, model_architecture=model_architecture,
        selection_evidence=selection_evidence,
        three_seed_validation_summary=three_seed_validation_summary,
        frozen_at_git_sha=git_sha, notes=notes,
    )


def save_freeze_manifest(manifest: FreezeManifest, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_freeze_manifest(path: Path) -> FreezeManifest:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return FreezeManifest(**data)


def require_freeze_manifest_exists(path: Path) -> FreezeManifest:
    """Call this at the top of any Stage-0 TEST evaluation script. Raises
    ``FreezeManifestMissingError`` if no freeze manifest exists yet --
    TEST must never be evaluated before the configuration is frozen."""
    path = Path(path)
    if not path.exists():
        raise FreezeManifestMissingError(
            f"no freeze manifest at {path} -- write one (build_freeze_manifest + "
            "save_freeze_manifest) BEFORE evaluating Stage-0 TEST"
        )
    return load_freeze_manifest(path)
