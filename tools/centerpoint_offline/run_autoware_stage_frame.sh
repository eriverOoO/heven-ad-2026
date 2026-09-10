#!/usr/bin/env bash
# Run one canonical XYZIRC NPZ through the isolated CenterPoint stage overlay.
set -euo pipefail

usage() {
  echo "usage: $0 --npz PATH --output-dir DIR [--stage-dump on|off] [--raw-head-dump on|off]" >&2
}

NPZ=
OUTPUT_DIR=
STAGE_DUMP=on
RAW_HEAD_DUMP=off
while [[ $# -gt 0 ]]; do
  case "$1" in
    --npz) NPZ=${2:-}; shift 2 ;;
    --output-dir) OUTPUT_DIR=${2:-}; shift 2 ;;
    --stage-dump) STAGE_DUMP=${2:-}; shift 2 ;;
    --raw-head-dump) RAW_HEAD_DUMP=${2:-}; shift 2 ;;
    *) usage; exit 2 ;;
  esac
done
[[ -f "$NPZ" && -n "$OUTPUT_DIR" ]] || { usage; exit 2; }
[[ "$STAGE_DUMP" == on || "$STAGE_DUMP" == off ]] || { usage; exit 2; }
[[ "$RAW_HEAD_DUMP" == on || "$RAW_HEAD_DUMP" == off ]] || { usage; exit 2; }

WORKTREE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RUNTIME_ROOT=${AUTOWARE_RUNTIME_ROOT:-"$WORKTREE/.autoware_runtime"}
MODEL_DATA_ROOT=${AUTOWARE_MODEL_DATA_ROOT:-/home/didgang1203/models/autoware}
TENSORRT_ROOT=${TENSORRT_ROOT:?Set TENSORRT_ROOT to the local TensorRT root.}
LAUNCH_FILE="$RUNTIME_ROOT/src/autoware_lidar_centerpoint/launch/lidar_centerpoint.launch.xml"
MODEL_PARAM="$WORKTREE/tools/centerpoint_offline/configs/centerpoint_single_sweep.param.yaml"
ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-77}

for path in "$RUNTIME_ROOT/install/setup.bash" "$LAUNCH_FILE" "$MODEL_PARAM"; do
  [[ -e "$path" ]] || { echo "missing runtime input: $path" >&2; exit 2; }
done
[[ ! -e "$OUTPUT_DIR" ]] || {
  echo "refusing to overwrite output directory: $OUTPUT_DIR" >&2
  exit 2
}
mkdir -p "$OUTPUT_DIR"
if [[ "$RAW_HEAD_DUMP" == on ]]; then
  mkdir -p "$OUTPUT_DIR/raw_head"
fi

set +u
source /opt/ros/humble/setup.bash
source "$RUNTIME_ROOT/install/setup.bash"
set -u
export ROS_DOMAIN_ID
export LD_LIBRARY_PATH="$TENSORRT_ROOT/lib:/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# ``ros2 topic echo`` occasionally writes terminal control sequences when a
# WSL terminal setting leaks into redirected capture.  Stage YAML is a data
# artifact, so force non-interactive text output rather than stripping bytes.
export TERM=dumb
export NO_COLOR=1

pids=()
cleanup() {
  for pid in "${pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  for pid in "${pids[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT

ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 map av2_egovehicle \
  >"$OUTPUT_DIR/static_tf.log" 2>&1 &
pids+=("$!")

timeout 45s ros2 launch "$LAUNCH_FILE" \
  input/pointcloud:=/ad/perception/lidar/points_xyzirc \
  output/objects:=/ad/perception/objects/detected \
  data_path:="$MODEL_DATA_ROOT" node_name:=lidar_centerpoint \
  model_name:=centerpoint model_param_path:="$MODEL_PARAM" \
  enable_stage_dump:="$([[ "$STAGE_DUMP" == on ]] && echo true || echo false)" \
  enable_raw_head_dump:="$([[ "$RAW_HEAD_DUMP" == on ]] && echo true || echo false)" \
  raw_head_dump_directory:="$OUTPUT_DIR/raw_head" \
  build_only:=false >"$OUTPUT_DIR/node.log" 2>&1 &
pids+=("$!")

topics=(/ad/perception/objects/detected)
names=(final)
if [[ "$STAGE_DUMP" == on ]]; then
  topics+=(
    /lidar_centerpoint/debug/stages/post_score
    /lidar_centerpoint/debug/stages/post_circle_nms
    /lidar_centerpoint/debug/stages/pre_iou
    /lidar_centerpoint/debug/stages/post_iou
    /lidar_centerpoint/debug/stages/final
  )
  names+=(post_score post_circle_nms pre_iou post_iou final_stage)
fi

ready_topic=${topics[-1]}
for _ in $(seq 1 30); do
  if ros2 topic list | grep -Fxq "$ready_topic"; then
    break
  fi
  sleep 1
done
ros2 topic list | grep -Fxq "$ready_topic" || {
  echo "CenterPoint topic did not become ready: $ready_topic" >&2
  exit 1
}

echo_pids=()
for index in "${!topics[@]}"; do
  timeout 25s ros2 topic echo --once "${topics[$index]}" \
    >"$OUTPUT_DIR/${names[$index]}.yaml" 2>"$OUTPUT_DIR/${names[$index]}.err" &
  echo_pids+=("$!")
  pids+=("$!")
done

for _ in $(seq 1 20); do
  all_subscribed=true
  for topic in "${topics[@]}"; do
    if ! ros2 topic info "$topic" 2>/dev/null | grep -Eq 'Subscription count: [1-9]'; then
      all_subscribed=false
      break
    fi
  done
  [[ "$all_subscribed" == true ]] && break
  sleep 1
done
[[ "$all_subscribed" == true ]] || {
  echo "stage capture subscribers did not become ready" >&2
  exit 1
}
PYTHONPATH="$WORKTREE/tools/centerpoint_offline${PYTHONPATH:+:$PYTHONPATH}" /usr/bin/python3 \
  "$WORKTREE/tools/centerpoint_offline/publish_av2_xyzirc.py" "$NPZ" \
  --topic /ad/perception/lidar/points_xyzirc --frame-id av2_egovehicle --count 5 \
  >"$OUTPUT_DIR/publisher.log" 2>&1

for pid in "${echo_pids[@]}"; do
  wait "$pid"
done
for name in "${names[@]}"; do
  grep -q '^objects:' "$OUTPUT_DIR/$name.yaml" || {
    echo "missing DetectedObjects capture for $name" >&2
    exit 1
  }
done
if [[ "$RAW_HEAD_DUMP" == on ]]; then
  timestamp_ns=$(/usr/bin/python3 -c \
    'import numpy as np,sys; print(int(np.load(sys.argv[1], allow_pickle=False)["timestamp_ns"]))' \
    "$NPZ")
  for suffix in meta.json heatmap.f32 reg.f32 height.f32 dim.f32 rot.f32 vel.f32; do
    [[ -s "$OUTPUT_DIR/raw_head/${timestamp_ns}_${suffix}" ]] || {
      echo "missing raw-head capture: ${timestamp_ns}_${suffix}" >&2
      exit 1
    }
  done
fi
echo "CenterPoint frame run: PASS ($STAGE_DUMP)"
echo "Output: $OUTPUT_DIR"
