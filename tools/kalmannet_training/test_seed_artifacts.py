"""Tests for seed_artifacts.py -- per-seed artifact path generation,
uniqueness across seeds, and non-collision with seed 0's real existing
files (docs/perception/kalmannet_av2_10k_multiseed_v1.md sections 5/23)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from seed_artifacts import seed_artifact_paths  # noqa: E402


def test_seed_artifact_paths_all_embed_the_seed_number():
    root = Path("/tmp/fake_root")
    for seed in (0, 1, 2):
        paths = seed_artifact_paths(root, seed)
        for key, path in paths.items():
            assert f"seed{seed}" in path.name, f"{key} path {path} does not embed seed{seed}"


def test_seed_artifact_paths_are_unique_across_seeds():
    root = Path("/tmp/fake_root")
    p0 = seed_artifact_paths(root, 0)
    p1 = seed_artifact_paths(root, 1)
    p2 = seed_artifact_paths(root, 2)
    for key in p0:
        assert p0[key] != p1[key] != p2[key] != p0[key]


def test_seed0_paths_match_the_real_pre_existing_naming_convention():
    """Section 4/10: never overwrite seed 0's already-frozen artifacts --
    verified by matching the EXACT real filenames already on disk from
    PR #55/PR #56/PR #57 (av2_scaleup_v2_10k_generic_robust_bs64_lr004_
    seed0.pt etc.), not a guessed/different convention."""
    root = Path("/home/user/datasets/av2/checkpoints")
    paths = seed_artifact_paths(root, 0)
    assert paths["checkpoint"].name == "av2_scaleup_v2_10k_generic_robust_bs64_lr004_seed0.pt"
    assert paths["freeze_manifest"].name == "av2_scaleup_v2_10k_generic_robust_bs64_lr004_seed0_FREEZE.json"
    assert paths["history_csv"].name == "av2_scaleup_v2_10k_generic_robust_bs64_lr004_seed0_history.csv"
    assert paths["resume_checkpoint"].name == "av2_scaleup_v2_10k_generic_robust_bs64_lr004_seed0.pt.resume.pt"


def test_all_seeds_share_the_same_root_directory():
    """Artifacts are separated by FILENAME (seed-embedded), not by
    directory -- confirms this is a real collision-avoidance property,
    not merely "they happen to be in different folders.\""""
    root = Path("/tmp/fake_root")
    p0 = seed_artifact_paths(root, 0)
    p1 = seed_artifact_paths(root, 1)
    for key in p0:
        assert p0[key].parent == p1[key].parent


def test_no_two_of_the_three_seeds_produce_any_shared_path():
    root = Path("/tmp/fake_root")
    all_paths = set()
    for seed in (0, 1, 2):
        paths = seed_artifact_paths(root, seed)
        for path in paths.values():
            assert path not in all_paths, f"seed {seed} path {path} collides with an earlier seed"
            all_paths.add(path)
