from pathlib import Path
import re
import stat
import subprocess

import yaml


ROOT = Path(__file__).resolve().parents[2]


def test_gitignore_preserves_runtime_artifacts_and_uv_environment():
    ignored_paths = (
        ".venv/bin/python",
        "ad_data/experiments/run.mcap",
        "ad_data/local_archive/heven_temp_2026_ws/metadata.json",
        "third_party/fast_lio/PCD/scans.pcd",
        "models/detector.pt",
        "runtime.db",
        "route_corridor.json",
    )
    for relative_path in ignored_paths:
        result = subprocess.run(
            ["git", "check-ignore", "--no-index", "--quiet", relative_path],
            cwd=ROOT,
            check=False,
        )
        assert result.returncode == 0, f"expected ignored path: {relative_path}"

    descriptor = "ad_morai_bridge_dev/data/morai_api.desc"
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "--quiet", descriptor],
        cwd=ROOT,
        check=False,
    )
    assert result.returncode == 1, f"generated descriptor must be trackable: {descriptor}"

    curated_assets = (
        "ad_data/map/global_info.json",
        "ad_data/path/2026_molit_comp_global_path.txt",
        "ad_data/tuning/best_parameters/profile_stanley.yaml",
    )
    for relative_path in curated_assets:
        result = subprocess.run(
            ["git", "check-ignore", "--no-index", "--quiet", relative_path],
            cwd=ROOT,
            check=False,
        )
        assert result.returncode == 1, (
            f"curated repository asset must remain trackable: {relative_path}"
        )


def test_vcs_manifest_pins_required_external_repositories():
    repositories = yaml.safe_load(
        (ROOT / "dependencies.repos").read_text(encoding="utf-8")
    )["repositories"]

    assert set(repositories) == {
        "patchwork-plusplus",
        "MORAI-DriveExample_GRPC",
        "kalman-filter-localization-ros2",
        "autoware_universe",
        "managed_transform_buffer",
        "muSSP",
    }
    for name, repository in repositories.items():
        assert repository["type"] == "git"
        assert repository["url"].startswith("https://github.com/")
        assert len(repository["version"]) == 40, name


def test_eskf_overlay_is_versioned_and_fail_closed():
    patch_directory = ROOT / "patches" / "kalman-filter-localization-ros2"
    patches = sorted(patch_directory.glob("*.patch"))
    script = ROOT / "scripts" / "apply_dependency_patches.sh"
    script_text = script.read_text(encoding="utf-8")

    assert [patch.name for patch in patches] == [
        "0001-large-imu-gap-recovery.patch",
        "0002-stationary-accel-bias-initialization.patch",
        "0003-wheel-confirmed-zupt.patch",
        "0004-gate-preinitialization-output.patch",
    ]
    assert script.stat().st_mode & stat.S_IXUSR
    assert "git -C \"$DEPENDENCY_DIRECTORY\" apply --check" in script_text
    assert "refusing partially applied" in script_text
    assert "refusing to patch a dirty" in script_text
    assert (patch_directory / "LICENSE.upstream").is_file()
    assert (patch_directory / "README.md").is_file()


def test_uv_lock_covers_repository_development_and_tuning_tools():
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    lock = (ROOT / "uv.lock").read_text(encoding="utf-8")
    setup = (ROOT / "scripts" / "setup_dev_env.sh").read_text(encoding="utf-8")

    assert 'requires-python = "==3.10.*"' in project
    for dependency in ("grpcio", "matplotlib", "numpy", "optuna", "pytest"):
        assert f'"{dependency}' in project
        assert f'name = "{dependency}"' in lock
    assert "uv sync --locked" in setup
    assert "--no-python-downloads" in setup
    assert "include-system-site-packages = false" in setup


def test_python_runner_defaults_to_repository_tooling_tests():
    runner = (ROOT / "scripts" / "test_python.sh").read_text(encoding="utf-8")

    assert 'if [[ "$#" -eq 0 ]]' in runner
    assert "set -- scripts/tests" in runner


def test_obsolete_unity_route_tools_are_quarantined_under_localization_legacy():
    assert not (ROOT / "tools").exists()
    legacy_tools = ROOT / "ad_localization" / "legacy" / "tools"
    assert {
        path.name for path in legacy_tools.iterdir() if path.is_file()
    } == {
        "generate_morai_route.py",
        "requirements.txt",
        "rotate_morai_route.py",
    }


def test_planner_declares_the_nav2_packages_required_by_its_mppi_launch():
    package = (ROOT / "ad_planner" / "package.xml").read_text(encoding="utf-8")

    for dependency in (
        "nav2_controller",
        "nav2_costmap_2d",
        "nav2_lifecycle_manager",
        "nav2_mppi_controller",
        "nav2_msgs",
    ):
        assert f"<exec_depend>{dependency}</exec_depend>" in package


def test_ci_prepares_and_verifies_the_complete_heven_workspace():
    workflow = yaml.safe_load(
        (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    )
    job = workflow["jobs"]["build"]
    steps = job["steps"]
    assert job["timeout-minutes"] >= 90
    assert job["needs"] == "prepare-image"
    assert job["container"]["image"] == "${{ needs.prepare-image.outputs.image }}"
    assert job["container"]["credentials"] == {
        "username": "${{ github.actor }}",
        "password": "${{ secrets.GITHUB_TOKEN }}",
    }
    assert job["permissions"]["contents"] == "read"
    assert job["permissions"]["packages"] == "read"

    checkout = next(step for step in steps if step.get("name") == "Checkout")
    assert re.fullmatch(r"actions/checkout@[0-9a-f]{40}", checkout["uses"])
    assert checkout["with"]["path"] == "ws/src/heven_ad_2026"
    assert checkout["with"]["lfs"] is True

    assert not any(step.get("name") == "Install ROS 2 Humble" for step in steps)
    assert not any(step.get("name") == "Install package dependencies" for step in steps)

    commands = " ".join(
        "\n".join(step.get("run", "") for step in steps).split()
    )
    required_commands = (
        "git lfs pull",
        "vcs import src",
        "--shallow",
        "dependencies.repos",
        "scripts/apply_dependency_patches.sh",
        "colcon list --packages-up-to",
        "--paths-only",
        "autoware_multi_object_tracker",
        "scripts/verify_ad_data.py",
        "scripts/test_python.sh",
        "colcon list --base-paths src/heven_ad_2026",
        "colcon build",
        "--executor sequential",
        "-DPython3_EXECUTABLE=",
        "--packages-up-to",
        "source src/heven_ad_2026/.venv/bin/activate",
        "VENV_SITE_PACKAGES=",
        "export PYTHONPATH=\"$VENV_SITE_PACKAGES",
        "export CTEST_PARALLEL_LEVEL=1",
        "colcon test",
        "--packages-select",
        "--metas src/heven_ad_2026/colcon.meta",
        "kalman_filter_localization_core",
        "colcon test-result --verbose",
    )
    missing = [command for command in required_commands if command not in commands]
    assert missing == []
    assert "apt-get" not in commands
    assert "rosdep install" not in commands

    setup_uv = next(step for step in steps if step.get("name") == "Install uv")
    assert setup_uv["with"]["working-directory"] == "ws/src/heven_ad_2026"
    assert setup_uv["with"]["cache-dependency-glob"] == "uv.lock"


def test_ci_reuses_compiler_outputs_through_a_bounded_ccache():
    workflow = yaml.safe_load(
        (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    )
    steps = workflow["jobs"]["build"]["steps"]

    restore = next(
        step for step in steps if step.get("name") == "Restore compiler cache"
    )
    save = next(step for step in steps if step.get("name") == "Save compiler cache")
    assert re.fullmatch(r"actions/cache/restore@[0-9a-f]{40}", restore["uses"])
    assert re.fullmatch(r"actions/cache/save@[0-9a-f]{40}", save["uses"])
    assert restore["id"] == "ccache-restore"
    assert restore["with"]["path"] == "~/.cache/ccache"
    assert "github.sha" in restore["with"]["key"]
    assert "ccache-v1-${{ runner.os }}-humble-" in restore["with"][
        "restore-keys"
    ]
    assert save["with"]["path"] == restore["with"]["path"]
    assert save["with"]["key"] == restore["with"]["key"]
    assert "always()" in save["if"]
    assert "steps.ccache-restore.outputs.cache-hit != 'true'" in save["if"]

    step_names = [step.get("name") for step in steps]
    assert step_names.index("Build complete HEVEN stack") < step_names.index(
        "Save compiler cache"
    )
    assert step_names.index("Save compiler cache") < step_names.index(
        "Test complete HEVEN stack"
    )

    commands = "\n".join(step.get("run", "") for step in steps)
    assert "ccache --max-size=8G" in commands
    assert "ccache --zero-stats" in commands
    assert "-DCMAKE_C_COMPILER_LAUNCHER=ccache" in commands
    assert "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache" in commands
    assert "ccache --show-stats" in commands


def test_ci_builds_or_reuses_a_content_addressed_ghcr_dependency_image():
    workflow = yaml.safe_load(
        (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    )
    prepare = workflow["jobs"]["prepare-image"]
    steps = prepare["steps"]

    assert prepare["permissions"] == {"contents": "read", "packages": "write"}
    assert prepare["outputs"]["image"] == "${{ steps.final-image.outputs.image }}"

    checkout = next(step for step in steps if step.get("name") == "Checkout")
    assert checkout["with"]["ref"] == (
        "${{ github.event.pull_request.head.sha || github.sha }}"
    )

    commands = "\n".join(step.get("run", "") for step in steps)
    assert "scripts/ci_image_fingerprint.py" in commands
    assert "ghcr.io/${GITHUB_REPOSITORY,,}-ci" in commands
    assert "docker buildx imagetools inspect" in commands
    assert "@sha256:" in commands

    for step_name in ("Set up Docker Buildx", "Log in to GHCR", "Build and push dependency image"):
        step = next(step for step in steps if step.get("name") == step_name)
        assert re.fullmatch(r"[^@]+@[0-9a-f]{40}", step["uses"])

    login = next(step for step in steps if step.get("name") == "Log in to GHCR")
    assert login["with"] == {
        "registry": "ghcr.io",
        "username": "${{ github.actor }}",
        "password": "${{ secrets.GITHUB_TOKEN }}",
    }
    build = next(
        step for step in steps if step.get("name") == "Build and push dependency image"
    )
    assert build["with"]["context"] == "."
    assert build["with"]["file"] == "docker/ci/Dockerfile"
    assert build["with"]["push"] is True
    assert build["with"]["platforms"] == "linux/amd64"


def test_dependency_image_is_source_free_and_installs_the_build_closure():
    dockerfile = (ROOT / "docker" / "ci" / "Dockerfile").read_text(
        encoding="utf-8"
    )
    normalized = " ".join(dockerfile.split())

    assert re.search(
        r"^FROM ros:humble-ros-base-jammy@sha256:[0-9a-f]{64}$",
        dockerfile,
        re.MULTILINE,
    )
    assert "--mount=type=bind,target=/tmp/ws/src/heven_ad_2026,readonly" in dockerfile
    assert "dependencies.repos" in dockerfile
    assert "vcs import" in dockerfile
    assert "scripts/apply_dependency_patches.sh" in dockerfile
    assert "set +u; source /opt/ros/humble/setup.bash; set -u" in normalized
    assert "colcon list --packages-up-to" in normalized
    assert "autoware_multi_object_tracker" in dockerfile
    assert "rosdep install --from-paths" in normalized
    assert "-r -y" not in dockerfile
    assert "libunwind-dev" in dockerfile
    assert "ccache" in dockerfile
    assert "git-lfs" in dockerfile
    assert "python3-vcstool" in dockerfile
    assert "COPY ." not in dockerfile
    assert "colcon build" not in dockerfile
