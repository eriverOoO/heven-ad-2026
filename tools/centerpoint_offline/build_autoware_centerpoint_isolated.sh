#!/usr/bin/env bash
# Build only the Autoware dependency closure of lidar_centerpoint in this worktree.
# It must never write the shared heven_ros_ws build/install/log directories.
set -euo pipefail

WORKTREE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
AUTOWARE_SRC=${AUTOWARE_SRC:-/home/didgang1203/projects/autoware_tracker_ws/src}
RUNTIME_ROOT=${AUTOWARE_RUNTIME_ROOT:-"$WORKTREE/.autoware_runtime"}

for binary in colcon nvcc; do
  command -v "$binary" >/dev/null || { echo "missing required command: $binary" >&2; exit 2; }
done
for library in nvinfer nvonnxparser; do
  ldconfig -p | grep -q "lib${library}\\.so" || {
    echo "missing TensorRT library: lib${library}.so" >&2
    echo "Install the pinned TensorRT runtime before invoking this build." >&2
    exit 2
  }
done
[[ -d "$AUTOWARE_SRC/autoware_universe" ]] || { echo "AUTOWARE_SRC is not an Autoware source root: $AUTOWARE_SRC" >&2; exit 2; }

source /opt/ros/humble/setup.bash
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}
export CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL:-2}

mkdir -p "$RUNTIME_ROOT"/{build,install,log,cache}
colcon build --symlink-install \
  --base-paths "$AUTOWARE_SRC" \
  --packages-up-to autoware_lidar_centerpoint \
  --build-base "$RUNTIME_ROOT/build" \
  --install-base "$RUNTIME_ROOT/install" \
  --log-base "$RUNTIME_ROOT/log" \
  --cmake-args -DCMAKE_BUILD_TYPE=Release

source "$RUNTIME_ROOT/install/setup.bash"
ros2 pkg prefix autoware_lidar_centerpoint
ros2 pkg executables autoware_lidar_centerpoint
