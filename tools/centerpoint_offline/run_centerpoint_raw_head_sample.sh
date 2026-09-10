#!/usr/bin/env bash
# Sequentially replay the fixed native/source-ring sample with raw-head capture enabled.
set -euo pipefail

usage() {
  echo "usage: $0 --derived-root DIR --output-root DIR [--max-frames N]" >&2
}

DERIVED_ROOT=
OUTPUT_ROOT=
MAX_FRAMES=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --derived-root) DERIVED_ROOT=${2:-}; shift 2 ;;
    --output-root) OUTPUT_ROOT=${2:-}; shift 2 ;;
    --max-frames) MAX_FRAMES=${2:-}; shift 2 ;;
    *) usage; exit 2 ;;
  esac
done
[[ -d "$DERIVED_ROOT/native" && -d "$DERIVED_ROOT/source_ring_vlp16_v2" ]] || {
  usage
  exit 2
}
[[ -n "$OUTPUT_ROOT" && "$MAX_FRAMES" =~ ^[0-9]+$ ]] || { usage; exit 2; }
[[ ! -e "$OUTPUT_ROOT" ]] || {
  echo "refusing to overwrite output root: $OUTPUT_ROOT" >&2
  exit 2
}

WORKTREE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RUNNER="$WORKTREE/tools/centerpoint_offline/run_autoware_stage_frame.sh"
mkdir -p "$OUTPUT_ROOT"
mapfile -t frames < <(find "$DERIVED_ROOT/native" -maxdepth 1 -type f -name '*.npz' -printf '%f\n' | sort)
[[ ${#frames[@]} -gt 0 ]] || { echo "no native NPZ frames" >&2; exit 2; }

completed=0
for filename in "${frames[@]}"; do
  timestamp=${filename%.npz}
  for mode in native source_ring_vlp16_v2; do
    npz="$DERIVED_ROOT/$mode/$filename"
    [[ -f "$npz" ]] || { echo "missing paired NPZ: $npz" >&2; exit 2; }
    "$RUNNER" --npz "$npz" --output-dir "$OUTPUT_ROOT/$timestamp/$mode" \
      --stage-dump on --raw-head-dump on
  done
  completed=$((completed + 1))
  if [[ "$MAX_FRAMES" -gt 0 && "$completed" -ge "$MAX_FRAMES" ]]; then
    break
  fi
done
echo "Raw-head paired replay: PASS ($completed frames, $((completed * 2)) runs)"
