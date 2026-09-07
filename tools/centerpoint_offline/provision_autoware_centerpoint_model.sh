#!/usr/bin/env bash
# Download only the Autoware 1.8.0-pinned full CenterPoint v3 artifacts.
set -euo pipefail

if [[ "${AD_AUTOWARE_MODEL_LICENSE_REVIEWED:-}" != "1" ]]; then
  printf '%s\n' 'Refusing download: set AD_AUTOWARE_MODEL_LICENSE_REVIEWED=1 after reviewing the upstream model license.' >&2
  exit 2
fi

model_root="${AUTOWARE_CENTERPOINT_MODEL_ROOT:-/home/didgang1203/models/autoware/lidar_centerpoint}"
base_url="https://awf.ml.dev.web.auto/perception/models/centerpoint/v3"
mkdir -p "$model_root"

download_verified() {
  local name="$1"
  local expected="$2"
  local target="$model_root/$name"
  local actual=""
  if [[ -f "$target" ]]; then
    actual="$(sha256sum "$target" | awk '{print $1}')"
    if [[ "$actual" == "$expected" ]]; then
      printf 'verified existing %s\n' "$target"
      return
    fi
    printf 'refusing to overwrite hash-mismatched artifact: %s\n' "$target" >&2
    exit 3
  fi
  local temporary="$target.download-$$"
  trap 'rm -f "$temporary"' RETURN
  curl --fail --location --retry 3 --output "$temporary" "$base_url/$name"
  actual="$(sha256sum "$temporary" | awk '{print $1}')"
  if [[ "$actual" != "$expected" ]]; then
    printf 'hash mismatch for %s: expected %s got %s\n' "$name" "$expected" "$actual" >&2
    exit 4
  fi
  mv "$temporary" "$target"
  trap - RETURN
  printf 'downloaded and verified %s\n' "$target"
}

download_verified pts_voxel_encoder_centerpoint.onnx dc1a876580d86ee7a341d543f8ade2ede7f43bd032dc5b44155b1f0175405764
download_verified pts_backbone_neck_head_centerpoint.onnx 3fe7e128955646740c41a25be0c8f141d5a94594fe79d7405fe2a859e391542e
download_verified centerpoint_ml_package.param.yaml 9bbc16e521dd87c91cbadf1cb89c8b81393d1f8e1069af385aaba677576f0e27
download_verified detection_class_remapper.param.yaml c711f8875ece9b527dfe31ffc75f8c0de2e77945ef67860a959a4e04c36772d5
