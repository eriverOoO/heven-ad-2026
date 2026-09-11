#!/usr/bin/env bash
# Downloads the CenterPoint T-14 reproduction checkpoint from its GitHub
# Release asset and verifies its SHA-256 against
# models/experimental/manifest.yaml before placing it at the exact path
# scripts/run_camera_lidar_tracking.sh (modes 8-10) already expects.
#
# The checkpoint binary is never committed as a git blob -- see
# models/experimental/manifest.yaml for why.
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: download_centerpoint_checkpoint.sh [--dest PATH] [--force]

options:
  --dest PATH   destination file path
                (default: models/experimental/centerpoint_t14_reproduction.pth)
  --force       re-download even if a file already exists at --dest
  -h, --help    show this help
EOF
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repository_root="$(cd -- "$script_dir/.." && pwd -P)"

readonly download_url="https://github.com/eriverOoO/heven-ad-2026/releases/download/centerpoint-t14-reproduction-v1/centerpoint_t14_reproduction.pth"
readonly expected_sha256="466c8181a377682e032bb32579c8ddb65807b5feebd8625a750b1d5538ddbc95"

dest="$repository_root/models/experimental/centerpoint_t14_reproduction.pth"
force=false

while (($#)); do
  case "$1" in
    --dest)
      [[ $# -ge 2 ]] || { echo "missing value for --dest" >&2; usage; exit 2; }
      dest="$2"
      shift 2
      ;;
    --force)
      force=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

verify_sha256() {
  local path="$1"
  local actual
  actual="$(sha256sum -- "$path")"
  actual="${actual%% *}"
  [[ "$actual" == "$expected_sha256" ]]
}

if [[ -f "$dest" ]] && [[ "$force" != true ]]; then
  if verify_sha256 "$dest"; then
    echo "already present and verified: $dest"
    exit 0
  fi
  echo "existing file at $dest has a mismatched SHA-256; re-downloading (or pass --force to always re-download)" >&2
fi

mkdir -p -- "$(dirname -- "$dest")"

tmp_dest="$(mktemp "${dest}.download.XXXXXX")"
trap 'rm -f -- "$tmp_dest"' EXIT

if command -v curl >/dev/null 2>&1; then
  curl -fL --retry 3 -o "$tmp_dest" "$download_url"
elif command -v wget >/dev/null 2>&1; then
  wget -O "$tmp_dest" "$download_url"
else
  echo "neither curl nor wget is available" >&2
  exit 1
fi

if ! verify_sha256 "$tmp_dest"; then
  actual="$(sha256sum -- "$tmp_dest")"
  actual="${actual%% *}"
  echo "downloaded checkpoint SHA-256 mismatch: $actual" >&2
  echo "expected: $expected_sha256" >&2
  exit 1
fi

mv -- "$tmp_dest" "$dest"
trap - EXIT
echo "downloaded and verified: $dest"
