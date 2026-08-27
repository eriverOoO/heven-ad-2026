#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repository_root="$(cd -- "$script_dir/.." && pwd -P)"
default_workspace="$(cd -- "$repository_root/../.." && pwd -P)"
workspace_root="${HEVEN_AD_WS_PATH:-$default_workspace}"

default_bag="$repository_root/morai_cam4_20260813_163222/morai_cam4_20260813_163222"
bag_path="${1:-$default_bag}"
rate="${2:-0.5}"

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
  echo "Usage: $0 /absolute/path/to/extracted_bag [rate]" >&2
  exit 1
fi
if [[ "$bag_path" != /* ]]; then
  bag_path="$(cd -- "$bag_path" && pwd -P)"
fi
for required_topic in \
  /ad/sensors/camera/front/compressed \
  /ad/sensors/lidar/points
do
  if ! grep -Fq -- "name: $required_topic" "$bag_path/metadata.yaml"; then
    echo "Required bag topic is missing: $required_topic" >&2
    exit 1
  fi
done

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

exec ros2 launch ad_lidar_perception \
  camera_lidar_tracking_replay.launch.py \
  bag_path:="$bag_path" \
  rate:="$rate"
