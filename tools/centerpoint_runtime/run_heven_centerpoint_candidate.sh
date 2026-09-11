#!/usr/bin/env bash
# Launch the opt-in pinned-Autoware CenterPoint candidate; never changes defaults.
set -euo pipefail

runtime_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
trt_root="${TENSORRT_ROOT:-/home/didgang1203/opt/tensorrt/10.8.0.43-cuda11.8}"
model_data_root="${AD_DATA_DIR:-/home/didgang1203}"
ros_overlay="${HEVEN_ROS_OVERLAY:-}"
autoware_overlay="${HEVEN_AUTOWARE_OVERLAY:-}"

fail() { printf 'CENTERPOINT_CANDIDATE_FAIL: %s\n' "$*" >&2; exit 2; }
[[ -r /opt/ros/humble/setup.bash ]] || fail "ROS Humble setup missing: /opt/ros/humble/setup.bash"
[[ -n "$ros_overlay" ]] || fail "set HEVEN_ROS_OVERLAY to a persistent HEVEN install prefix (not /tmp)"
[[ -r "$ros_overlay/setup.bash" ]] || fail "HEVEN_ROS_OVERLAY/setup.bash missing"
[[ -n "$autoware_overlay" ]] || fail "set HEVEN_AUTOWARE_OVERLAY to a persistent Autoware install prefix (not /tmp)"
[[ -r "$autoware_overlay/setup.bash" ]] || fail "HEVEN_AUTOWARE_OVERLAY/setup.bash missing"
[[ -r "$trt_root/lib/libnvinfer.so" ]] || fail "TensorRT nvinfer missing under $trt_root"
[[ -r "$trt_root/lib/libnvonnxparser.so" ]] || fail "TensorRT ONNX parser missing under $trt_root"
[[ "${AD_AUTOWARE_MODEL_LICENSE_REVIEWED:-}" == 1 ]] || fail "AD_AUTOWARE_MODEL_LICENSE_REVIEWED=1 is required"

model_root="$model_data_root/models/autoware/lidar_centerpoint"
for artifact in \
  pts_voxel_encoder_centerpoint.onnx \
  pts_backbone_neck_head_centerpoint.onnx \
  centerpoint_ml_package.param.yaml \
  detection_class_remapper.param.yaml \
  pts_voxel_encoder_centerpoint.engine \
  pts_backbone_neck_head_centerpoint.engine; do
  [[ -r "$model_root/$artifact" ]] || fail "required model artifact missing: $model_root/$artifact"
done

source /opt/ros/humble/setup.bash
source "$autoware_overlay/setup.bash"
source "$ros_overlay/setup.bash"
export TENSORRT_ROOT="$trt_root"
export AD_DATA_DIR="$model_data_root"
export LD_LIBRARY_PATH="$trt_root/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$trt_root/bin:$PATH"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-77}"

command -v ros2 >/dev/null || fail "ros2 unavailable after sourcing overlays"
ros2 pkg prefix autoware_lidar_centerpoint >/dev/null 2>&1 || fail "autoware_lidar_centerpoint package unavailable"
ros2 pkg prefix autoware_multi_object_tracker >/dev/null 2>&1 || fail "autoware_multi_object_tracker package unavailable"

selection="$runtime_root/ad_lidar_perception/config/experiments/autoware_centerpoint_competition_candidate_v1.yaml"
if [[ -n "${HEVEN_CENTERPOINT_BAG_PATH:-}" ]]; then
  exec ros2 launch ad_lidar_perception lidar_bag_replay.launch.py \
    bag_path:="$HEVEN_CENTERPOINT_BAG_PATH" \
    detector_backend:=autoware_centerpoint \
    autoware_selection_config:="$selection" \
    composition_config:="$selection" \
    "$@"
fi

exec ros2 launch ad_lidar_perception lidar_perception.launch.py \
  detector_backend:=autoware_centerpoint \
  autoware_selection_config:="$selection" \
  composition_config:="$selection" \
  point_layout_adapter_enabled:=true \
  "$@"
