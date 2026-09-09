#!/usr/bin/env bash
# Validate a user-local TensorRT tar installation without modifying the host.
set -euo pipefail

TENSORRT_ROOT=${TENSORRT_ROOT:?Set TENSORRT_ROOT to an unpacked TensorRT root.}
for path in include/NvInfer.h include/NvOnnxParser.h lib/libnvinfer.so lib/libnvonnxparser.so bin/trtexec; do
  [[ -e "$TENSORRT_ROOT/$path" ]] || { echo "missing: $TENSORRT_ROOT/$path" >&2; exit 2; }
done

export LD_LIBRARY_PATH="$TENSORRT_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$TENSORRT_ROOT/bin/trtexec" --version
ldd "$TENSORRT_ROOT/lib/libnvinfer.so" | tee /dev/stderr | grep -q 'not found' && {
  echo 'TensorRT has unresolved shared-library dependencies.' >&2; exit 2;
}

TMPDIR_LOCAL=$(mktemp -d /tmp/heven-tensorrt-smoke.XXXXXX)
trap 'rm -rf "$TMPDIR_LOCAL"' EXIT
cat > "$TMPDIR_LOCAL/smoke.cpp" <<'EOF'
#include <NvInfer.h>
#include <NvOnnxParser.h>
int main() { return NV_TENSORRT_MAJOR == 10 && NV_TENSORRT_MINOR == 8 ? 0 : 1; }
EOF
g++ -std=c++17 "$TMPDIR_LOCAL/smoke.cpp" -I"$TENSORRT_ROOT/include" \
  -L"$TENSORRT_ROOT/lib" -Wl,-rpath,"$TENSORRT_ROOT/lib" \
  -lnvinfer -lnvonnxparser -o "$TMPDIR_LOCAL/smoke"
"$TMPDIR_LOCAL/smoke"
echo "TensorRT 10.8 header/library compile-link smoke: PASS"
