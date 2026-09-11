#!/usr/bin/env bash
# Replay one validated MORAI bag through legacy or opt-in CenterPoint backend.
set -euo pipefail

runtime_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
usage() {
  printf 'usage: %s --backend legacy|centerpoint --bag ABS_PATH --scene NAME --route NAME --traffic NAME --spawn NAME [--extra-launch-arg key:=value]\n' "$0" >&2
  exit 64
}
backend=""; bag=""; scene=""; route=""; traffic=""; spawn=""; extra=()
while (($#)); do
  case "$1" in
    --backend) backend="${2:-}"; shift 2;;
    --bag) bag="${2:-}"; shift 2;;
    --scene) scene="${2:-}"; shift 2;;
    --route) route="${2:-}"; shift 2;;
    --traffic) traffic="${2:-}"; shift 2;;
    --spawn) spawn="${2:-}"; shift 2;;
    --extra-launch-arg) extra+=("${2:-}"); shift 2;;
    *) usage;;
  esac
done
[[ -n "$backend" && -n "$bag" && -n "$scene" && -n "$route" && -n "$traffic" && -n "$spawn" ]] || usage
command -v ros2 >/dev/null || {
  printf 'ros2 unavailable: source the persistent HEVEN/Autoware overlays before running the A/B harness\n' >&2
  exit 2
}

python3 "$runtime_root/tools/centerpoint_offline/validate_morai_audit_bag.py" \
  "$bag" --scene "$scene" --route "$route" \
  --traffic-configuration "$traffic" --spawn-configuration "$spawn" \
  --used-for-training no --deep

if [[ "$backend" == legacy ]]; then
  exec ros2 launch ad_lidar_perception lidar_bag_replay.launch.py \
    bag_path:="$bag" detector_backend:=euclidean \
    composition_config:="$runtime_root/ad_lidar_perception/config/lidar_perception.yaml" \
    "${extra[@]}"
elif [[ "$backend" == centerpoint ]]; then
  export HEVEN_CENTERPOINT_BAG_PATH="$bag"
  exec "$runtime_root/tools/centerpoint_runtime/run_heven_centerpoint_candidate.sh" \
    use_sim_time:=true "${extra[@]}"
fi
printf 'unsupported backend: %s\n' "$backend" >&2
exit 64
