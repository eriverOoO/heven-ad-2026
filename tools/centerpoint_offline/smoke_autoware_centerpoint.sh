#!/usr/bin/env bash
# Run a bounded, isolated CenterPoint runtime smoke test with synthetic XYZIRC.
# This verifies transport, engine loading and postprocessing only; it is not an
# accuracy or MORAI evaluation.
set -euo pipefail

WORKTREE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RUNTIME_ROOT=${AUTOWARE_RUNTIME_ROOT:-"$WORKTREE/.autoware_runtime"}
MODEL_DATA_ROOT=${AUTOWARE_MODEL_DATA_ROOT:-/home/didgang1203/models/autoware}
MODEL_NAME=${AUTOWARE_CENTERPOINT_MODEL_NAME:-centerpoint}
MODEL_ROOT="$MODEL_DATA_ROOT/lidar_centerpoint"
TENSORRT_ROOT=${TENSORRT_ROOT:?Set TENSORRT_ROOT to the local TensorRT 10.8 root.}
SMOKE_ROOT=${AUTOWARE_CENTERPOINT_SMOKE_ROOT:-/home/didgang1203/datasets/centerpoint/autoware_runtime_smoke_v1}
ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-77}

for path in \
  "$RUNTIME_ROOT/install/setup.bash" \
  "$MODEL_ROOT/pts_voxel_encoder_${MODEL_NAME}.engine" \
  "$MODEL_ROOT/pts_backbone_neck_head_${MODEL_NAME}.engine" \
  "$TENSORRT_ROOT/lib/libnvinfer.so" \
  "$TENSORRT_ROOT/lib/libnvonnxparser.so"; do
  [[ -e "$path" ]] || { echo "missing required runtime artifact: $path" >&2; exit 2; }
done

mkdir -p "$SMOKE_ROOT/logs"
set +u
source /opt/ros/humble/setup.bash
source "$RUNTIME_ROOT/install/setup.bash"
set -u
export ROS_DOMAIN_ID
export LD_LIBRARY_PATH="$TENSORRT_ROOT/lib:/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

tf_pid=''
node_pid=''
echo_pid=''
cleanup() {
  for pid in "$echo_pid" "$node_pid" "$tf_pid"; do
    [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
  done
  for pid in "$echo_pid" "$node_pid" "$tf_pid"; do
    [[ -n "$pid" ]] && wait "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT

ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 map lidar_link \
  >"$SMOKE_ROOT/logs/static_tf.log" 2>&1 &
tf_pid=$!
timeout 40s ros2 launch autoware_lidar_centerpoint lidar_centerpoint.launch.xml \
  input/pointcloud:=/ad/perception/lidar/points_xyzirc \
  output/objects:=/ad/perception/objects/detected \
  data_path:="$MODEL_DATA_ROOT" node_name:=lidar_centerpoint \
  model_name:="$MODEL_NAME" build_only:=false \
  >"$SMOKE_ROOT/logs/node.log" 2>&1 &
node_pid=$!

sleep 8
timeout 20s ros2 topic echo --once /ad/perception/objects/detected \
  >"$SMOKE_ROOT/logs/detected_objects_once.yaml" 2>&1 &
echo_pid=$!
python3 "$WORKTREE/tools/centerpoint_offline/synthetic_xyzirc.py" \
  --topic /ad/perception/lidar/points_xyzirc --frame-id lidar_link --count 15 \
  >"$SMOKE_ROOT/logs/publisher.log" 2>&1
sleep 3

if [[ ! -s "$SMOKE_ROOT/logs/detected_objects_once.yaml" ]]; then
  echo "no DetectedObjects message received; inspect $SMOKE_ROOT/logs/node.log" >&2
  exit 1
fi
grep -q '^objects:' "$SMOKE_ROOT/logs/detected_objects_once.yaml" || {
  echo "DetectedObjects output was malformed; inspect $SMOKE_ROOT/logs/detected_objects_once.yaml" >&2
  exit 1
}
echo "Autoware CenterPoint synthetic smoke: PASS"
echo "Logs: $SMOKE_ROOT/logs"
