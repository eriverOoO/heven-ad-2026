#!/usr/bin/env bash
# Sequentially replay only newly-created sampling-control modes with R0 capture.
set -euo pipefail

usage() {
  echo "usage: $0 --derived-root DIR --output-root DIR --modes MODE [MODE ...] [--max-frames N] [--resume]" >&2
}

DERIVED_ROOT=
OUTPUT_ROOT=
MAX_FRAMES=0
MODES=()
RESUME=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --derived-root) DERIVED_ROOT=${2:-}; shift 2 ;;
    --output-root) OUTPUT_ROOT=${2:-}; shift 2 ;;
    --max-frames) MAX_FRAMES=${2:-}; shift 2 ;;
    --resume) RESUME=true; shift ;;
    --modes)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do MODES+=("$1"); shift; done
      ;;
    *) usage; exit 2 ;;
  esac
done
[[ -d "$DERIVED_ROOT/native" && -n "$OUTPUT_ROOT" && ${#MODES[@]} -gt 0 ]] || { usage; exit 2; }
[[ "$MAX_FRAMES" =~ ^[0-9]+$ ]] || { usage; exit 2; }
[[ ! -e "$OUTPUT_ROOT" || "$RESUME" == true ]] || { echo "refusing to overwrite output root: $OUTPUT_ROOT" >&2; exit 2; }
for mode in "${MODES[@]}"; do [[ -d "$DERIVED_ROOT/$mode" ]] || { echo "missing mode: $mode" >&2; exit 2; }; done

WORKTREE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RUNNER="$WORKTREE/tools/centerpoint_offline/run_autoware_stage_frame.sh"
mkdir -p "$OUTPUT_ROOT"
mapfile -t frames < <(find "$DERIVED_ROOT/native" -maxdepth 1 -type f -name '*.npz' -printf '%f\n' | sort)
[[ ${#frames[@]} -gt 0 ]] || { echo "no native frames" >&2; exit 2; }

is_complete() {
  local directory=$1 timestamp=$2 stage
  [[ -s "$directory/raw_head/${timestamp}_meta.json" ]] || return 1
  for stage in final post_score post_circle_nms pre_iou post_iou final_stage; do
    grep -q '^objects:' "$directory/${stage}.yaml" 2>/dev/null || return 1
    grep -q $'\033' "$directory/${stage}.yaml" 2>/dev/null && return 1
  done
}

completed=0
for filename in "${frames[@]}"; do
  timestamp=${filename%.npz}
  for mode in "${MODES[@]}"; do
    npz="$DERIVED_ROOT/$mode/$filename"
    output_dir="$OUTPUT_ROOT/$timestamp/$mode"
    [[ -f "$npz" ]] || { echo "missing NPZ: $npz" >&2; exit 2; }
    if is_complete "$output_dir" "$timestamp"; then
      echo "Skipping verified completed frame: $timestamp/$mode"
      continue
    fi
    if [[ -e "$output_dir" ]]; then
      [[ "$RESUME" == true ]] || { echo "partial output requires --resume: $output_dir" >&2; exit 2; }
      attempt=1
      while [[ -e "${output_dir}.failed_attempt_${attempt}" ]]; do attempt=$((attempt + 1)); done
      mv "$output_dir" "${output_dir}.failed_attempt_${attempt}"
      echo "Preserved incomplete output: ${output_dir}.failed_attempt_${attempt}" >&2
    fi
    succeeded=false
    for attempt in 1 2 3; do
      if bash "$RUNNER" --npz "$npz" --output-dir "$output_dir" --stage-dump on --raw-head-dump on; then
        succeeded=true
        break
      fi
      failed_dir="${output_dir}.failed_retry_${attempt}"
      [[ ! -e "$failed_dir" ]] || { echo "refusing to overwrite $failed_dir" >&2; exit 2; }
      mv "$output_dir" "$failed_dir"
      echo "Retry $attempt/3 after preserving $failed_dir" >&2
    done
    [[ "$succeeded" == true ]] || { echo "frame failed after retries: $timestamp/$mode" >&2; exit 1; }
  done
  completed=$((completed + 1))
  if [[ "$MAX_FRAMES" -gt 0 && "$completed" -ge "$MAX_FRAMES" ]]; then break; fi
done
echo "Sampling-control replay: PASS ($completed frames, $((completed * ${#MODES[@]})) runs)"
