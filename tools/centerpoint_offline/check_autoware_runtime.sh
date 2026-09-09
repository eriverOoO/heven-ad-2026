#!/usr/bin/env bash
# Read-only readiness report for the isolated Autoware CenterPoint audit.
set -u

audit_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
selection="$audit_root/ad_lidar_perception/config/experiments/autoware_centerpoint_audit_v1.yaml"
data_root="${AD_DATA_DIR:-/home/didgang1203}"
trt_root="${TENSORRT_ROOT:-/home/didgang1203/opt/tensorrt/10.8.0.43-cuda11.8}"

printf 'ROS_DISTRO=%s\n' "${ROS_DISTRO:-unset}"
printf 'ROS_DOMAIN_ID=%s\n' "${ROS_DOMAIN_ID:-unset}"
printf 'AD_DATA_DIR=%s\n' "$data_root"
printf 'CUDA compiler: '
nvcc --version 2>/dev/null | tail -1 || printf 'unavailable\n'
printf 'TensorRT root: %s\n' "$trt_root"
printf 'TensorRT libraries: '
if [[ -r "$trt_root/lib/libnvinfer.so" && -r "$trt_root/lib/libnvonnxparser.so" ]]; then
  printf 'local nvinfer + nvonnxparser available\n'
else
  printf 'unavailable\n'
fi
printf 'TensorRT CLI: '
if [[ -x "$trt_root/bin/trtexec" ]]; then
  printf '%s\n' "$trt_root/bin/trtexec"
else
  command -v trtexec || printf 'unavailable\n'
fi
printf 'CenterPoint executable: '
ros2 pkg prefix autoware_lidar_centerpoint 2>/dev/null || printf 'unavailable\n'
printf 'Model root: %s\n' "$data_root/models/autoware/lidar_centerpoint"
find "$data_root/models/autoware/lidar_centerpoint" -maxdepth 1 -type f \
  \( -name 'pts_*centerpoint*.onnx' -o -name '*centerpoint*.engine' \) \
  -printf '%f %s bytes\n' 2>/dev/null || true
printf 'Pinned HEVEN artifact verification:\n'
if [[ ${CHECK_HEVEN_INTEGRATION:-0} == 1 ]]; then
  PYTHONPATH="$audit_root/ad_lidar_perception${PYTHONPATH:+:$PYTHONPATH}" \
    python3 "$audit_root/scripts/check_autoware_perception.py" \
    --selection "$selection" \
    --lock "$audit_root/ad_lidar_perception/config/autoware_perception.lock.yaml" \
    --data-root "$data_root"
else
  printf 'skipped (set CHECK_HEVEN_INTEGRATION=1 in a sourced HEVEN overlay)\n'
fi
