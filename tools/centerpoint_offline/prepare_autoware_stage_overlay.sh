#!/usr/bin/env bash
# Prepare an ignored, isolated lidar_centerpoint source overlay with opt-in stage topics.
set -euo pipefail

WORKTREE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
AUTOWARE_SRC=${AUTOWARE_SRC:-/home/didgang1203/projects/autoware_tracker_ws/src}
RUNTIME_ROOT=${AUTOWARE_RUNTIME_ROOT:-"$WORKTREE/.autoware_runtime"}
SOURCE_PACKAGE="$AUTOWARE_SRC/autoware_universe/perception/autoware_lidar_centerpoint"
OVERLAY_PACKAGE="$RUNTIME_ROOT/src/autoware_lidar_centerpoint"
PATCH_FILE="$WORKTREE/tools/centerpoint_offline/patches/autoware_lidar_centerpoint_stage_dump.patch"
EXPECTED_SOURCE_SHA=68277081c14ef841cd43e16c0f5f821f367b004490b3369dfbfd29f0f8f62c7c
MARKER=.heven_centerpoint_stage_overlay

[[ -d "$SOURCE_PACKAGE" ]] || { echo "missing pinned source package: $SOURCE_PACKAGE" >&2; exit 2; }
[[ -f "$PATCH_FILE" ]] || { echo "missing stage patch: $PATCH_FILE" >&2; exit 2; }
actual_sha=$(sha256sum "$SOURCE_PACKAGE/src/node.cpp" | awk '{print $1}')
[[ "$actual_sha" == "$EXPECTED_SOURCE_SHA" ]] || {
  echo "pinned node.cpp SHA mismatch: expected $EXPECTED_SOURCE_SHA, got $actual_sha" >&2
  exit 2
}

if [[ -f "$OVERLAY_PACKAGE/$MARKER" ]]; then
  patch --dry-run --reverse --silent -d "$OVERLAY_PACKAGE" -p1 <"$PATCH_FILE" || {
    echo "existing stage overlay does not match the pinned patch" >&2
    exit 2
  }
  echo "stage overlay already prepared: $OVERLAY_PACKAGE"
  exit 0
fi
[[ ! -e "$OVERLAY_PACKAGE" ]] || {
  echo "refusing to replace unmarked runtime source: $OVERLAY_PACKAGE" >&2
  exit 2
}

mkdir -p "$RUNTIME_ROOT/src"
cp -a "$SOURCE_PACKAGE" "$OVERLAY_PACKAGE"
patch --forward --batch -d "$OVERLAY_PACKAGE" -p1 <"$PATCH_FILE"
touch "$OVERLAY_PACKAGE/$MARKER"
echo "stage overlay prepared: $OVERLAY_PACKAGE"
