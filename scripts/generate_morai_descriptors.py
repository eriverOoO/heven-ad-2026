#!/usr/bin/env python3
"""Regenerate ad_morai_bridge_dev/data/morai_api.desc.

The MORAI gRPC API ships as generated C# sources whose FileDescriptorProtos
are embedded as base64 blobs. This script extracts them once, orders them by
dependency, and writes a single FileDescriptorSet that the bridge loads at
runtime. Run it whenever the MORAI-DriveExample_GRPC dependency is bumped:

    python3 scripts/generate_morai_descriptors.py
"""

import base64
from pathlib import Path
import re
import sys

from google.protobuf import descriptor_pool
from google.protobuf.descriptor_pb2 import FileDescriptorProto, FileDescriptorSet


REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE = (
    REPOSITORY.parent
    / "MORAI-DriveExample_GRPC"
    / "MoraiStandardProtos"
    / "Protos"
)
OUTPUT = REPOSITORY / "ad_morai_bridge_dev" / "data" / "morai_api.desc"

_DESCRIPTOR = re.compile(
    r"FromBase64String\(\s*(?:string\.Concat\((.*?)\)|"
    r'"([A-Za-z0-9+/=]+)"\s*)\)',
    re.DOTALL,
)
_CHUNK = re.compile(r'"([A-Za-z0-9+/=]+)"')


def main() -> int:
    descriptors: dict[str, bytes] = {}
    for path in sorted(SOURCE.rglob("*.cs")):
        text = path.read_text(encoding="utf-8")
        for match in _DESCRIPTOR.finditer(text):
            chunks = (
                _CHUNK.findall(match.group(1))
                if match.group(1) is not None
                else [match.group(2)]
            )
            raw = base64.b64decode("".join(chunks), validate=True)
            descriptors[FileDescriptorProto.FromString(raw).name] = raw
    if not descriptors:
        print(f"no descriptors found under {SOURCE}", file=sys.stderr)
        return 1

    # Order files so every dependency precedes its dependents.
    pool = descriptor_pool.DescriptorPool()
    ordered: list[bytes] = []
    pending = dict(descriptors)
    while pending:
        progressed = False
        for name, raw in list(pending.items()):
            try:
                pool.AddSerializedFile(raw)
            except TypeError:
                continue
            ordered.append(raw)
            del pending[name]
            progressed = True
        if not progressed:
            print(f"unresolved dependencies: {sorted(pending)}", file=sys.stderr)
            return 1

    descriptor_set = FileDescriptorSet()
    for raw in ordered:
        descriptor_set.file.append(FileDescriptorProto.FromString(raw))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_bytes(descriptor_set.SerializeToString())
    print(f"wrote {OUTPUT} ({len(ordered)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
