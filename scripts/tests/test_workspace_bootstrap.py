from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP = ROOT / "scripts" / "bootstrap_workspace.sh"
README = ROOT / "README.md"
README_BOOTSTRAP_MARKER = "<!-- heven-ad-workspace-bootstrap -->"


def run_bootstrap(*arguments: str, script: Path = BOOTSTRAP):
    assert script.is_file(), "workspace bootstrap script is missing"
    return subprocess.run(
        [str(script), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )


def test_check_mode_resolves_the_assembled_workspace_without_mutating_it():
    status_before = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    result = run_bootstrap("--check")

    status_after = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert result.returncode == 0, result.stderr
    assert f"repository: {ROOT}" in result.stdout
    assert f"workspace: {ROOT.parents[1]}" in result.stdout
    assert "ROS environment: humble" in result.stdout
    assert result.stdout.rstrip().endswith("preflight OK")
    assert status_after == status_before


def test_unknown_argument_is_rejected_without_starting_bootstrap():
    result = run_bootstrap("--unknown")

    assert result.returncode == 2
    assert "usage:" in result.stderr
    assert "vcs import" not in result.stdout


def test_immutable_workspace_paths_are_exported_without_reassignment():
    script = BOOTSTRAP.read_text(encoding="utf-8")

    assert 'export REPOSITORY_ROOT WORKSPACE_SOURCE_DIRECTORY' in script
    assert 'REPOSITORY_ROOT="$REPOSITORY_ROOT" \\\n' not in script
    assert 'WORKSPACE_SOURCE_DIRECTORY="$WORKSPACE_SOURCE_DIRECTORY" \\\n' not in script


def test_rosdep_install_uses_options_supported_by_humble():
    script = BOOTSTRAP.read_text(encoding="utf-8")

    assert "rosdep install --from-paths" in script
    assert "--yes" not in script
    assert "--ignore-src" in script
    assert "--rosdistro humble" in script


def test_bootstrap_covers_lfs_sources_patches_python_build_and_test():
    script = BOOTSTRAP.read_text(encoding="utf-8")

    required_fragments = (
        "lfs pull",
        "lfs fsck",
        "submodule update --init --recursive",
        "vcs import",
        "dependencies.repos",
        "apply_dependency_patches.sh",
        "verify_ad_data.py",
        "setup_dev_env.sh",
        "colcon list",
        "--packages-up-to",
        "autoware_multi_object_tracker",
        "colcon build",
        "colcon test",
        "colcon test-result --verbose",
    )
    missing = [fragment for fragment in required_fragments if fragment not in script]

    assert missing == []


def test_repository_tests_import_python_packages_from_source_before_build():
    runner = (ROOT / "scripts" / "test_python.sh").read_text(encoding="utf-8")

    assert "PYTHON_PACKAGE_ROOTS" in runner
    assert "PYTHONPATH" in runner
    assert "package.xml" in runner


def test_bootstrap_preserves_an_existing_lfs_pre_push_hook():
    script = BOOTSTRAP.read_text(encoding="utf-8")

    assert "rev-parse --git-path hooks/pre-push" in script
    assert '[[ "$LFS_PRE_PUSH_HOOK" != /* ]]' in script
    assert 'LFS_PRE_PUSH_HOOK="$REPOSITORY_ROOT/$LFS_PRE_PUSH_HOOK"' in script
    assert "grep -Fq -- 'git lfs pre-push'" in script
    assert "lfs install --local" in script


def test_repository_outside_workspace_src_is_rejected(tmp_path):
    assert BOOTSTRAP.is_file(), "workspace bootstrap script is missing"
    wrong_repo = tmp_path / "heven_ad_2026"
    wrong_script = wrong_repo / "scripts" / BOOTSTRAP.name
    wrong_script.parent.mkdir(parents=True)
    shutil.copy2(BOOTSTRAP, wrong_script)

    result = run_bootstrap("--check", script=wrong_script)

    assert result.returncode == 1
    assert "<workspace>/src/heven_ad_2026" in result.stderr
    assert "preflight OK" not in result.stdout


def test_readme_one_shot_bootstrap_block_is_valid_bash(tmp_path):
    readme = README.read_text(encoding="utf-8")
    assert README_BOOTSTRAP_MARKER in readme, "README one-shot block is missing"
    marker_index = readme.index(README_BOOTSTRAP_MARKER)
    fence_start = readme.index("```bash\n", marker_index) + len("```bash\n")
    fence_end = readme.index("\n```", fence_start)
    bootstrap_block = readme[fence_start:fence_end]
    extracted = tmp_path / "readme_bootstrap.sh"
    extracted.write_text(bootstrap_block, encoding="utf-8")

    result = subprocess.run(
        ["bash", "-n", str(extracted)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_readme_private_clone_reuses_gh_authentication_without_relogin():
    readme = README.read_text(encoding="utf-8")

    assert "gh auth status -h github.com" in readme
    assert "gh auth login -h github.com -w" in readme
    assert "gh auth setup-git" in readme
    assert readme.index("gh auth status -h github.com") < readme.index(
        "gh auth login -h github.com -w"
    )
    assert readme.count("git ls-remote") >= 2
    assert "이후 HTTPS clone·fetch·pull에서는 GitHub 비밀번호를 다시 묻지 않는다" in readme


def test_readme_installs_and_uses_the_mcap_rosbag_storage_plugin():
    readme = README.read_text(encoding="utf-8")

    assert "ros-humble-rosbag2-storage-mcap" in readme
    assert "ros2 bag record -s mcap" in readme
    assert '$HOME/heven_ad_2026_ws/bags' in readme
