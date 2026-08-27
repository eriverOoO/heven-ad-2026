#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage:
  run_camera_lidar_tracking.sh [BAG_PATH [RATE]]
  run_camera_lidar_tracking.sh [options]

options:
  --mode N                         tracking preset 1..10 (default: 3)
  --bag PATH                       extracted MCAP bag directory
  --rate RATE                      positive playback rate (default: 0.5)
  --centerpoint-checkpoint PATH    required by modes 8..10
  --openpcdet-root PATH            default: references/openpcdet
  --centerpoint-device DEVICE      default: cuda:0
  --kalmannet-checkpoint PATH      required by mode 7
  --kalmannet-device DEVICE        default: cpu
  --list-modes                     print the ten presets and exit
  -h, --help                       show this help

The equivalent HEVEN_* environment variables may provide model paths/devices.
EOF
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repository_root="$(cd -- "$script_dir/.." && pwd -P)"
default_workspace="$(cd -- "$repository_root/../.." && pwd -P)"
workspace_root="${HEVEN_AD_WS_PATH:-$default_workspace}"
preset_config="$repository_root/ad_lidar_perception/config/tracking/camera_replay_presets.yaml"
python_source="$repository_root/ad_lidar_perception"

default_bag="$repository_root/morai_cam4_20260813_163222/morai_cam4_20260813_163222"
bag_path="$default_bag"
rate="0.5"
mode="${HEVEN_TRACKING_MODE:-3}"
centerpoint_checkpoint="${HEVEN_CENTERPOINT_CHECKPOINT:-$repository_root/models/experimental/centerpoint_t14_reproduction.pth}"
openpcdet_root="${HEVEN_OPENPCDET_ROOT:-$repository_root/references/openpcdet}"
centerpoint_device="${HEVEN_CENTERPOINT_DEVICE:-cuda:0}"
kalmannet_checkpoint="${HEVEN_KALMANNET_CHECKPOINT:-$repository_root/models/experimental/dense_kalmannet_v2.pt}"
kalmannet_device="${HEVEN_KALMANNET_DEVICE:-cpu}"
list_modes=false
positional=()
readonly centerpoint_sha256="466c8181a377682e032bb32579c8ddb65807b5feebd8625a750b1d5538ddbc95"
readonly kalmannet_sha256="956604975e5204b2c584c3e8fe16a7ba9346097980566ecbbb22c2001fbf7d48"

verify_sha256() {
  local label="$1"
  local path="$2"
  local expected="$3"
  local actual
  actual="$(sha256sum -- "$path")"
  actual="${actual%% *}"
  if [[ "$actual" != "$expected" ]]; then
    echo "$label checkpoint SHA-256 mismatch: $actual" >&2
    echo "Expected the frozen research artifact: $expected" >&2
    exit 1
  fi
}

while (($#)); do
  case "$1" in
    --mode|--bag|--rate|--centerpoint-checkpoint|--openpcdet-root|--centerpoint-device|--kalmannet-checkpoint|--kalmannet-device)
      if (($# < 2)); then
        echo "missing value for $1" >&2
        usage
        exit 2
      fi
      option="$1"
      value="$2"
      shift 2
      case "$option" in
        --mode) mode="$value" ;;
        --bag) bag_path="$value" ;;
        --rate) rate="$value" ;;
        --centerpoint-checkpoint) centerpoint_checkpoint="$value" ;;
        --openpcdet-root) openpcdet_root="$value" ;;
        --centerpoint-device) centerpoint_device="$value" ;;
        --kalmannet-checkpoint) kalmannet_checkpoint="$value" ;;
        --kalmannet-device) kalmannet_device="$value" ;;
      esac
      ;;
    --list-modes)
      list_modes=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --*)
      echo "unknown option: $1" >&2
      usage
      exit 2
      ;;
    *)
      positional+=("$1")
      shift
      ;;
  esac
done

if ((${#positional[@]} > 2)); then
  echo "at most BAG_PATH and RATE may be positional" >&2
  usage
  exit 2
fi
if ((${#positional[@]} >= 1)); then
  bag_path="${positional[0]}"
fi
if ((${#positional[@]} == 2)); then
  rate="${positional[1]}"
fi

export PYTHONPATH="$python_source${PYTHONPATH:+:$PYTHONPATH}"
if [[ "$list_modes" == true ]]; then
  python3 -m ad_lidar_perception.tracking_replay_presets --config "$preset_config"
  exit 0
fi
python3 -m ad_lidar_perception.tracking_replay_presets --config "$preset_config" --mode "$mode"

if [[ ! -f "$bag_path/metadata.yaml" ]]; then
  nested_bag="$bag_path/$(basename -- "$bag_path")"
  if [[ -f "$nested_bag/metadata.yaml" ]]; then
    bag_path="$nested_bag"
  fi
fi

if [[ ! -r /opt/ros/humble/setup.bash ]]; then
  echo "ROS 2 Humble is required: /opt/ros/humble/setup.bash" >&2
  exit 1
fi
if [[ ! -r "$workspace_root/install/setup.bash" ]]; then
  echo "Built workspace setup is missing: $workspace_root/install/setup.bash" >&2
  echo "Run ./scripts/bootstrap_workspace.sh first, or set HEVEN_AD_WS_PATH." >&2
  exit 1
fi
if [[ ! -d "$bag_path" || ! -f "$bag_path/metadata.yaml" ]]; then
  echo "Extracted MCAP bag folder is missing: $bag_path" >&2
  usage
  exit 1
fi
if [[ "$bag_path" != /* ]]; then
  bag_path="$(cd -- "$bag_path" && pwd -P)"
fi
required_topics=(
  /ad/sensors/camera/front/compressed
  /ad/sensors/lidar/points
)
for required_topic in "${required_topics[@]}"; do
  if ! grep -Fq -- "name: $required_topic" "$bag_path/metadata.yaml"; then
    echo "Required bag topic is missing: $required_topic" >&2
    exit 1
  fi
done

case "$mode" in
  7)
    if [[ ! -f "$kalmannet_checkpoint" ]]; then
      echo "Mode 7 requires --kalmannet-checkpoint FILE." >&2
      exit 1
    fi
    verify_sha256 "KalmanNet" "$kalmannet_checkpoint" "$kalmannet_sha256"
    ;;
  8|9|10)
    if [[ ! -f "$centerpoint_checkpoint" ]]; then
      echo "Mode $mode requires --centerpoint-checkpoint FILE." >&2
      exit 1
    fi
    verify_sha256 "CenterPoint" "$centerpoint_checkpoint" "$centerpoint_sha256"
    if [[ ! -d "$openpcdet_root" ]]; then
      echo "Mode $mode requires --openpcdet-root DIRECTORY." >&2
      exit 1
    fi
    ;;
esac

set +u
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
# shellcheck disable=SC1090
source "$workspace_root/install/setup.bash"
set -u

for package in ad_lidar_perception rosbag2_storage_mcap compressed_image_transport; do
  if ! ros2 pkg prefix "$package" >/dev/null 2>&1; then
    echo "Required ROS package is missing: $package" >&2
    echo "Run ./scripts/bootstrap_workspace.sh to install and rebuild." >&2
    exit 1
  fi
done

if [[ "$mode" == 7 || "$mode" == 8 || "$mode" == 9 || "$mode" == 10 ]]; then
  if ! python3 -c 'import torch' >/dev/null 2>&1; then
    echo "Mode $mode requires a torch-enabled Python environment." >&2
    exit 1
  fi
fi
if [[ "$mode" == 8 || "$mode" == 9 || "$mode" == 10 ]]; then
  if ! python3 -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1; then
    echo "Mode $mode requires CUDA-enabled PyTorch and an available GPU." >&2
    exit 1
  fi
fi

launch_arguments=(
  "bag_path:=$bag_path"
  "rate:=$rate"
  "mode:=$mode"
  "centerpoint_checkpoint:=$centerpoint_checkpoint"
  "openpcdet_root:=$openpcdet_root"
  "centerpoint_device:=$centerpoint_device"
  "kalmannet_checkpoint:=$kalmannet_checkpoint"
  "kalmannet_device:=$kalmannet_device"
)
exec ros2 launch ad_lidar_perception camera_lidar_tracking_replay.launch.py "${launch_arguments[@]}"
