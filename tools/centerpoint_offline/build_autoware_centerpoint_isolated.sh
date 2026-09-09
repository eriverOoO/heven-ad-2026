#!/usr/bin/env bash
# Build only the Autoware dependency closure of lidar_centerpoint in this worktree.
# It must never write the shared heven_ros_ws build/install/log directories.
set -euo pipefail

WORKTREE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
AUTOWARE_SRC=${AUTOWARE_SRC:-/home/didgang1203/projects/autoware_tracker_ws/src}
RUNTIME_ROOT=${AUTOWARE_RUNTIME_ROOT:-"$WORKTREE/.autoware_runtime"}
TENSORRT_ROOT=${TENSORRT_ROOT:-}

for binary in colcon nvcc; do
  command -v "$binary" >/dev/null || { echo "missing required command: $binary" >&2; exit 2; }
done
if [[ -n "$TENSORRT_ROOT" ]]; then
  bash "$WORKTREE/tools/centerpoint_offline/check_local_tensorrt.sh"
  export LD_LIBRARY_PATH="$TENSORRT_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  TRT_CMAKE_ARGS=("-DCMAKE_LIBRARY_PATH=$TENSORRT_ROOT/lib" "-DCMAKE_INCLUDE_PATH=$TENSORRT_ROOT/include")
else
  for library in nvinfer nvonnxparser; do
    ldconfig -p | grep -q "lib${library}\\.so" || {
      echo "missing TensorRT library: lib${library}.so" >&2
      echo "Set TENSORRT_ROOT to a verified local TensorRT 10.8 root, or install no global packages from this script." >&2
      exit 2
    }
  done
  TRT_CMAKE_ARGS=()
fi
[[ -d "$AUTOWARE_SRC/autoware_universe" ]] || { echo "AUTOWARE_SRC is not an Autoware source root: $AUTOWARE_SRC" >&2; exit 2; }

# Some ROS Humble setup scripts reference optional variables unset in a clean
# shell. Keep strict mode for this helper, but source that generated script
# with nounset temporarily disabled.
set +u
source /opt/ros/humble/setup.bash
set -u
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}
export CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL:-2}

mkdir -p "$RUNTIME_ROOT"/{build,install,log,cache}
colcon --log-base "$RUNTIME_ROOT/log" build --symlink-install \
  --base-paths "$AUTOWARE_SRC" \
  --packages-up-to autoware_lidar_centerpoint \
  --build-base "$RUNTIME_ROOT/build" \
  --install-base "$RUNTIME_ROOT/install" \
  --cmake-args -DCMAKE_BUILD_TYPE=Release "${TRT_CMAKE_ARGS[@]}"

source "$RUNTIME_ROOT/install/setup.bash"
ros2 pkg prefix autoware_lidar_centerpoint
ros2 pkg executables autoware_lidar_centerpoint
