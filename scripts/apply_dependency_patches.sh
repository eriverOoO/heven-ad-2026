#!/usr/bin/env bash

set -euo pipefail

readonly KALMAN_BASE_SHA="fc1f4d39c942813ea83dc4f017eb0892756ea94d"
readonly KALMAN_REPOSITORY="kalman-filter-localization-ros2"
readonly -a KALMAN_PATCH_RELATIVE_PATHS=(
  "patches/kalman-filter-localization-ros2/0001-large-imu-gap-recovery.patch"
  "patches/kalman-filter-localization-ros2/0002-stationary-accel-bias-initialization.patch"
  "patches/kalman-filter-localization-ros2/0003-wheel-confirmed-zupt.patch"
  "patches/kalman-filter-localization-ros2/0004-gate-preinitialization-output.patch"
)

if [[ "$#" -ne 1 ]]; then
  echo "usage: $0 <workspace-src-directory>" >&2
  exit 2
fi

readonly SCRIPT_DIRECTORY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPOSITORY_ROOT="$(cd "$SCRIPT_DIRECTORY/.." && pwd)"
readonly WORKSPACE_SOURCE_DIRECTORY="$(cd "$1" && pwd)"
readonly DEPENDENCY_DIRECTORY="$WORKSPACE_SOURCE_DIRECTORY/$KALMAN_REPOSITORY"
declare -a PATCH_PATHS=()
for relative_path in "${KALMAN_PATCH_RELATIVE_PATHS[@]}"; do
  patch_path="$REPOSITORY_ROOT/$relative_path"
  if [[ ! -f "$patch_path" ]]; then
    echo "missing dependency patch: $patch_path" >&2
    exit 1
  fi
  PATCH_PATHS+=("$patch_path")
done
if ! git -C "$DEPENDENCY_DIRECTORY" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "missing git checkout: $DEPENDENCY_DIRECTORY" >&2
  exit 1
fi

readonly ACTUAL_HEAD="$(git -C "$DEPENDENCY_DIRECTORY" rev-parse HEAD)"
if [[ "$ACTUAL_HEAD" != "$KALMAN_BASE_SHA" ]]; then
  echo "unexpected $KALMAN_REPOSITORY HEAD: $ACTUAL_HEAD (expected $KALMAN_BASE_SHA)" >&2
  exit 1
fi

readonly VERIFICATION_DIRECTORY="$(mktemp -d)"
readonly VERIFICATION_INDEX="$VERIFICATION_DIRECTORY/index"
cleanup_verification_index() {
  rm -f -- "$VERIFICATION_INDEX"
  rmdir -- "$VERIFICATION_DIRECTORY"
}
trap cleanup_verification_index EXIT

overlay_is_applied() {
  # Patches are ordered and can extend the context of earlier patches. Verify
  # the complete overlay by reversing it in a private temporary Git index.
  GIT_INDEX_FILE="$VERIFICATION_INDEX" git -C "$DEPENDENCY_DIRECTORY" read-tree HEAD
  GIT_INDEX_FILE="$VERIFICATION_INDEX" git -C "$DEPENDENCY_DIRECTORY" add -A
  for ((index=${#PATCH_PATHS[@]} - 1; index >= 0; index--)); do
    patch_path="${PATCH_PATHS[$index]}"
    if ! GIT_INDEX_FILE="$VERIFICATION_INDEX" git -C "$DEPENDENCY_DIRECTORY" \
      apply --cached --reverse --check "$patch_path" >/dev/null 2>&1
    then
      return 1
    fi
    GIT_INDEX_FILE="$VERIFICATION_INDEX" git -C "$DEPENDENCY_DIRECTORY" \
      apply --cached --reverse "$patch_path"
  done
}

if overlay_is_applied; then
  echo "$KALMAN_REPOSITORY dependency patches are already applied"
  exit 0
fi

individually_applied_patch_count=0
for patch_path in "${PATCH_PATHS[@]}"; do
  if git -C "$DEPENDENCY_DIRECTORY" apply --reverse --check "$patch_path" >/dev/null 2>&1; then
    individually_applied_patch_count=$((individually_applied_patch_count + 1))
  fi
done
if [[ "$individually_applied_patch_count" -ne 0 ]]; then
  echo "refusing partially applied $KALMAN_REPOSITORY dependency overlay" >&2
  exit 1
fi

if [[ -n "$(git -C "$DEPENDENCY_DIRECTORY" status --porcelain)" ]]; then
  echo "refusing to patch a dirty $KALMAN_REPOSITORY checkout" >&2
  git -C "$DEPENDENCY_DIRECTORY" status --short >&2
  exit 1
fi

for patch_path in "${PATCH_PATHS[@]}"; do
  git -C "$DEPENDENCY_DIRECTORY" apply --check "$patch_path"
  git -C "$DEPENDENCY_DIRECTORY" apply --whitespace=error "$patch_path"
done

if ! overlay_is_applied; then
  echo "dependency overlay verification failed after application" >&2
  exit 1
fi

echo "applied ${KALMAN_PATCH_RELATIVE_PATHS[*]} to $KALMAN_REPOSITORY@$KALMAN_BASE_SHA"
