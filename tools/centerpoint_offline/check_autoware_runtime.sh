#!/usr/bin/env bash
# Read-only readiness report for the isolated Autoware CenterPoint audit.
set -u

audit_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
selection="$audit_root/ad_lidar_perception/config/experiments/autoware_centerpoint_audit_v1.yaml"
data_root="${AD_DATA_DIR:-/home/didgang1203}"

printf 'ROS_DISTRO=%s\n' "${ROS_DISTRO:-unset}"
printf 'ROS_DOMAIN_ID=%s\n' "${ROS_DOMAIN_ID:-unset}"
printf 'AD_DATA_DIR=%s\n' "$data_root"
printf 'CUDA compiler: '
nvcc --version 2>/dev/null | tail -1 || printf 'unavailable\n'
printf 'TensorRT libraries:\n'
ldconfig -p 2>/dev/null | grep -E 'libnvinfer\.so|libnvonnxparser\.so' || printf 'unavailable\n'
printf 'TensorRT CLI: '
command -v trtexec || printf 'unavailable\n'
printf 'CenterPoint executable: '
ros2 pkg prefix autoware_lidar_centerpoint 2>/dev/null || printf 'unavailable\n'
printf 'Model root: %s\n' "$data_root/models/autoware/lidar_centerpoint"
find "$data_root/models/autoware/lidar_centerpoint" -maxdepth 1 -type f \
  \( -name 'pts_*centerpoint*.onnx' -o -name '*centerpoint*.engine' \) \
  -printf '%f %s bytes\n' 2>/dev/null || true
printf 'Pinned HEVEN artifact verification:\n'
PYTHONPATH="$audit_root/ad_lidar_perception${PYTHONPATH:+:$PYTHONPATH}" \
  python3 "$audit_root/scripts/check_autoware_perception.py" \
  --selection "$selection" \
  --lock "$audit_root/ad_lidar_perception/config/autoware_perception.lock.yaml" \
  --data-root "$data_root" || true
