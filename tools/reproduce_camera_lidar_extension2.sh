#!/usr/bin/env bash
# GT-FREE DESCRIPTIVE REPRODUCTION. No downloads; no production config writes.
set -euo pipefail

script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd "${script_directory}/.." && pwd)"
python_command="${HEVEN_EXT2_PYTHON:-python3}"

export PYTHONPATH="${repository_root}/ad_lidar_perception${PYTHONPATH:+:${PYTHONPATH}}"
exec "${python_command}" "${script_directory}/reproduce_camera_lidar_extension2.py" "$@"
