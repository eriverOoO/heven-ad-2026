"""Per-seed artifact path generation for the AV2 10k GENERIC-ROBUST
multi-seed experiment (docs/perception/kalmannet_av2_10k_multiseed_v1.md).

A single pure function, no I/O -- used both to generate the actual
commands for seeds 1/2 and as the thing this task's own "per-seed
artifact separation" tests exercise. Every path embeds the seed number,
so two different seeds can never collide by construction (verified by
test, not just assumed); the naming convention matches the REAL files
already on disk for seed 0 (``av2_scaleup_v2_10k_generic_robust_bs64_
lr004_seed0.pt`` etc., from PR #55/PR #56/PR #57), so seed 0's existing
artifacts are never at risk of being overwritten by this module's own
seed 1/2 paths.
"""

from __future__ import annotations

from pathlib import Path

CHECKPOINT_NAME_TEMPLATE = "av2_scaleup_v2_10k_generic_robust_bs64_lr004_seed{seed}"


def seed_artifact_paths(root: Path, seed: int) -> dict[str, Path]:
    """Returns every per-seed artifact path this experiment produces,
    all rooted at ``root`` (e.g. ``~/datasets/av2/checkpoints``) and all
    embedding ``seed`` in their filename."""
    root = Path(root)
    stem = CHECKPOINT_NAME_TEMPLATE.format(seed=seed)
    return {
        "checkpoint": root / f"{stem}.pt",
        "resume_checkpoint": root / f"{stem}.pt.resume.pt",
        "checkpoint_manifest": root / f"{stem}.pt.manifest.json",
        "history_csv": root / f"{stem}_history.csv",
        "freeze_manifest": root / f"{stem}_FREEZE.json",
        "official_val_report": root / f"av2_10k_generic_seed{seed}_official_val_report.json",
        "train_log": root.parent / "logs" / f"{stem}_train.log",
        "train_stdout_log": root.parent / "logs" / f"{stem}_stdout.log",
    }
