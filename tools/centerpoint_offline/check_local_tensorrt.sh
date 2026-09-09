#!/usr/bin/env bash
# Validate a user-local TensorRT tar installation without modifying the host.
set -euo pipefail

TENSORRT_ROOT=${TENSORRT_ROOT:?Set TENSORRT_ROOT to an unpacked TensorRT root.}
CUDA_ROOT=${CUDA_ROOT:-/usr/local/cuda}
for path in include/NvInfer.h include/NvOnnxParser.h lib/libnvinfer.so lib/libnvonnxparser.so bin/trtexec; do
  [[ -e "$TENSORRT_ROOT/$path" ]] || { echo "missing: $TENSORRT_ROOT/$path" >&2; exit 2; }
done
[[ -f "$CUDA_ROOT/include/cuda_runtime_api.h" ]] || {
  echo "missing CUDA runtime header: $CUDA_ROOT/include/cuda_runtime_api.h" >&2; exit 2;
}

export LD_LIBRARY_PATH="$TENSORRT_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# TensorRT 10.8's trtexec prints a version banner for --version but then
# returns non-zero because it still expects a model. --help both emits the
# banner and exits successfully, so use it for this no-model smoke check.
"$TENSORRT_ROOT/bin/trtexec" --help >/tmp/heven_trtexec_help.$$ 2>&1
grep -q 'TensorRT v100800' /tmp/heven_trtexec_help.$$
head -n 1 /tmp/heven_trtexec_help.$$
rm -f /tmp/heven_trtexec_help.$$
if ldd "$TENSORRT_ROOT/lib/libnvinfer.so" | grep -q 'not found'; then
  echo 'TensorRT has unresolved shared-library dependencies.' >&2; exit 2;
fi

TMPDIR_LOCAL=$(mktemp -d /tmp/heven-tensorrt-smoke.XXXXXX)
trap 'rm -rf "$TMPDIR_LOCAL"' EXIT
cat > "$TMPDIR_LOCAL/smoke.cpp" <<'EOF'
#include <NvInfer.h>
#include <NvOnnxParser.h>
int main() { return NV_TENSORRT_MAJOR == 10 && NV_TENSORRT_MINOR == 8 ? 0 : 1; }
EOF
g++ -std=c++17 -Wno-deprecated-declarations "$TMPDIR_LOCAL/smoke.cpp" -I"$TENSORRT_ROOT/include" -I"$CUDA_ROOT/include" \
  -L"$TENSORRT_ROOT/lib" -Wl,-rpath,"$TENSORRT_ROOT/lib" \
  -lnvinfer -lnvonnxparser -o "$TMPDIR_LOCAL/smoke"
"$TMPDIR_LOCAL/smoke"
echo "TensorRT 10.8 header/library compile-link smoke: PASS"
