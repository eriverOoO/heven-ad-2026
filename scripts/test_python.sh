#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$script_dir/.." && pwd)"

"$script_dir/setup_dev_env.sh"
cd "$repo_dir"

export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
mapfile -t PYTHON_PACKAGE_ROOTS < <(
  for package_manifest in "$repo_dir"/*/package.xml; do
    package_root="$(dirname -- "$package_manifest")"
    python_package="$package_root/$(basename -- "$package_root")"
    if [[ -d "$python_package" ]]; then
      printf '%s\n' "$package_root"
    fi
  done | sort -u
)
if [[ "${#PYTHON_PACKAGE_ROOTS[@]}" -gt 0 ]]; then
  printf -v python_source_path '%s:' "${PYTHON_PACKAGE_ROOTS[@]}"
  export PYTHONPATH="${python_source_path%:}${PYTHONPATH:+:$PYTHONPATH}"
fi

# ROS package tests require a colcon installation and sourced workspace.
# With no explicit path, only run repository-local tooling tests.
if [[ "$#" -eq 0 ]]; then
  set -- scripts/tests
fi

exec uv run --locked python -m pytest "$@"
