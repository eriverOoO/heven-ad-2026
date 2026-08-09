#!/usr/bin/env bash

set -euo pipefail

usage() {
  echo "usage: $0 [--check]" >&2
}

source_external_setup() {
  local setup_file="$1"
  set +u
  # shellcheck disable=SC1090
  source "$setup_file"
  set -u
}

MODE="bootstrap"
if [[ "$#" -eq 1 && "$1" == "--check" ]]; then
  MODE="check"
elif [[ "$#" -ne 0 ]]; then
  usage
  exit 2
fi

readonly SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly REPOSITORY_ROOT="$(cd -- "$SCRIPT_DIRECTORY/.." && pwd -P)"

if [[ "$(basename -- "$REPOSITORY_ROOT")" != "heven_ad_2026" ||
  "$(basename -- "$(dirname -- "$REPOSITORY_ROOT")")" != "src" ]]
then
  echo "repository must be placed at <workspace>/src/heven_ad_2026" >&2
  exit 1
fi

readonly WORKSPACE_SOURCE_DIRECTORY="$(cd -- "$REPOSITORY_ROOT/.." && pwd -P)"
readonly WORKSPACE_ROOT="$(cd -- "$WORKSPACE_SOURCE_DIRECTORY/.." && pwd -P)"
readonly ROS_SETUP_FILE="/opt/ros/humble/setup.bash"
export REPOSITORY_ROOT WORKSPACE_SOURCE_DIRECTORY

if [[ ! -r /etc/os-release ]]; then
  echo "cannot identify the operating system: /etc/os-release is missing" >&2
  exit 1
fi
# shellcheck disable=SC1091
source /etc/os-release
if [[ "${ID:-}" != "ubuntu" || "${VERSION_ID:-}" != "22.04" ]]; then
  echo "Ubuntu 22.04 is required; found ${PRETTY_NAME:-unknown system}" >&2
  exit 1
fi
if [[ ! -r "$ROS_SETUP_FILE" ]]; then
  echo "ROS 2 Humble is required: $ROS_SETUP_FILE is missing" >&2
  exit 1
fi

for required_command in git git-lfs python3 rosdep vcs colcon uv; do
  if ! command -v "$required_command" >/dev/null 2>&1; then
    echo "required command is missing: $required_command" >&2
    exit 1
  fi
done
for required_file in \
  dependencies.repos \
  colcon.meta \
  pyproject.toml \
  uv.lock \
  scripts/apply_dependency_patches.sh \
  scripts/setup_dev_env.sh \
  scripts/test_python.sh \
  scripts/verify_ad_data.py
do
  if [[ ! -f "$REPOSITORY_ROOT/$required_file" ]]; then
    echo "repository file is missing: $required_file" >&2
    exit 1
  fi
done

echo "repository: $REPOSITORY_ROOT"
echo "workspace: $WORKSPACE_ROOT"
source_external_setup "$ROS_SETUP_FILE"
if [[ "${ROS_DISTRO:-}" != "humble" ]]; then
  echo "ROS 2 Humble setup did not select the humble distribution" >&2
  exit 1
fi
echo "ROS environment: $ROS_DISTRO"
if [[ "$MODE" == "check" ]]; then
  echo "preflight OK"
  exit 0
fi

git -C "$REPOSITORY_ROOT" submodule update --init --recursive
LFS_PRE_PUSH_HOOK="$(
  git -C "$REPOSITORY_ROOT" rev-parse --git-path hooks/pre-push
)"
if [[ "$LFS_PRE_PUSH_HOOK" != /* ]]; then
  LFS_PRE_PUSH_HOOK="$REPOSITORY_ROOT/$LFS_PRE_PUSH_HOOK"
fi
readonly LFS_PRE_PUSH_HOOK
if [[ -f "$LFS_PRE_PUSH_HOOK" ]] && \
  grep -Fq -- 'git lfs pre-push' "$LFS_PRE_PUSH_HOOK"
then
  echo "existing Git LFS pre-push hook is valid"
else
  git -C "$REPOSITORY_ROOT" lfs install --local
fi
git -C "$REPOSITORY_ROOT" lfs pull
git -C "$REPOSITORY_ROOT" lfs fsck
python3 "$REPOSITORY_ROOT/scripts/verify_ad_data.py" \
  --root "$REPOSITORY_ROOT"

if [[ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]]; then
  sudo rosdep init
fi
rosdep update --rosdistro humble

vcs import "$WORKSPACE_SOURCE_DIRECTORY" \
  --input "$REPOSITORY_ROOT/dependencies.repos" \
  --skip-existing

python3 - <<'PY'
import os
from pathlib import Path
import subprocess

import yaml


repository_root = Path(os.environ["REPOSITORY_ROOT"])
source_directory = Path(os.environ["WORKSPACE_SOURCE_DIRECTORY"])
manifest = yaml.safe_load(
    (repository_root / "dependencies.repos").read_text(encoding="utf-8")
)["repositories"]

errors = []
for name, expected in manifest.items():
    checkout = source_directory / name
    if not (checkout / ".git").exists():
        errors.append(f"missing dependency checkout: {checkout}")
        continue
    actual_head = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    actual_origin = subprocess.check_output(
        ["git", "-C", str(checkout), "remote", "get-url", "origin"],
        text=True,
    ).strip()
    if actual_head != expected["version"]:
        errors.append(
            f"{name} HEAD is {actual_head}; expected {expected['version']}"
        )
    if actual_origin != expected["url"]:
        errors.append(
            f"{name} origin is {actual_origin}; expected {expected['url']}"
        )

if errors:
    raise SystemExit("\n".join(errors))
print(f"verified {len(manifest)} pinned dependency checkouts")
PY

"$REPOSITORY_ROOT/scripts/apply_dependency_patches.sh" \
  "$WORKSPACE_SOURCE_DIRECTORY"
"$REPOSITORY_ROOT/scripts/setup_dev_env.sh"
"$REPOSITORY_ROOT/scripts/test_python.sh"

cd -- "$WORKSPACE_ROOT"
mapfile -t FIRST_PARTY_PACKAGES < <(
  colcon list --base-paths "$REPOSITORY_ROOT" --names-only | sort -u
)
if [[ "${#FIRST_PARTY_PACKAGES[@]}" -eq 0 ]]; then
  echo "no first-party ROS packages were discovered" >&2
  exit 1
fi
readonly -a REQUIRED_RUNTIME_PACKAGES=(
  autoware_multi_object_tracker
)
readonly -a BUILD_ROOT_PACKAGES=(
  "${FIRST_PARTY_PACKAGES[@]}"
  "${REQUIRED_RUNTIME_PACKAGES[@]}"
)

mapfile -t AVAILABLE_PACKAGES < <(
  colcon list --base-paths "$WORKSPACE_SOURCE_DIRECTORY" --names-only
)
for required_package in "${BUILD_ROOT_PACKAGES[@]}"; do
  if ! printf '%s\n' "${AVAILABLE_PACKAGES[@]}" \
    | grep -Fxq -- "$required_package"
  then
    echo "required source package is missing: $required_package" >&2
    exit 1
  fi
done

mapfile -t BUILD_PACKAGE_PATHS < <(
  colcon list \
    --base-paths "$WORKSPACE_SOURCE_DIRECTORY" \
    --packages-up-to "${BUILD_ROOT_PACKAGES[@]}" \
    --paths-only \
    | sort -u
)
if [[ "${#BUILD_PACKAGE_PATHS[@]}" -eq 0 ]]; then
  echo "HEVEN build dependency closure is empty" >&2
  exit 1
fi

rosdep install --from-paths "${BUILD_PACKAGE_PATHS[@]}" \
  --ignore-src \
  --rosdistro humble \
  -r -y

readonly PROJECT_PYTHON="$REPOSITORY_ROOT/.venv/bin/python3"
export AD_DATA_DIR="$REPOSITORY_ROOT/ad_data"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-2}"
export MAKEFLAGS="${MAKEFLAGS:--j2 -l2}"

colcon build \
  --executor sequential \
  --symlink-install \
  --packages-up-to "${BUILD_ROOT_PACKAGES[@]}" \
  --cmake-args \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DPython3_EXECUTABLE="$PROJECT_PYTHON" \
    -DPYTHON_EXECUTABLE="$PROJECT_PYTHON"

source_external_setup "$WORKSPACE_ROOT/install/setup.bash"
source_external_setup "$REPOSITORY_ROOT/.venv/bin/activate"
export CTEST_PARALLEL_LEVEL="${CTEST_PARALLEL_LEVEL:-2}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

colcon test \
  --executor sequential \
  --packages-select "${FIRST_PARTY_PACKAGES[@]}" \
  --metas "$REPOSITORY_ROOT/colcon.meta"
colcon test \
  --executor sequential \
  --packages-select \
    kalman_filter_localization_core \
    kalman_filter_localization \
  --ctest-args -LE linter
colcon test-result --verbose

echo
echo "HEVEN AD workspace bootstrap complete: $WORKSPACE_ROOT"
echo "For a new terminal:"
echo "  source /opt/ros/humble/setup.bash"
echo "  source $WORKSPACE_ROOT/install/setup.bash"
echo "  export AD_DATA_DIR=$REPOSITORY_ROOT/ad_data"
