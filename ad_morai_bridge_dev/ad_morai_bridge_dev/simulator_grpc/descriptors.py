from dataclasses import dataclass
from pathlib import Path

from google.protobuf import descriptor_pool
from google.protobuf.descriptor import MethodDescriptor
from google.protobuf.descriptor_pb2 import FileDescriptorSet


@dataclass(frozen=True)
class MoraiApi:
    """The MORAI gRPC API loaded from the committed descriptor set.

    data/morai_api.desc is generated from the pinned MORAI C# sources by
    scripts/generate_morai_descriptors.py; regenerate it when the
    MORAI-DriveExample_GRPC dependency is bumped.
    """

    pool: descriptor_pool.DescriptorPool
    methods: dict[str, MethodDescriptor]

    @classmethod
    def load(cls, source: Path | None = None) -> "MoraiApi":
        source = source or _installed_descriptor_set()
        descriptor_set = FileDescriptorSet.FromString(source.read_bytes())

        pool = descriptor_pool.DescriptorPool()
        methods: dict[str, MethodDescriptor] = {}
        for file_proto in descriptor_set.file:
            pool.Add(file_proto)
            file_descriptor = pool.FindFileByName(file_proto.name)
            for service in file_descriptor.services_by_name.values():
                for method in service.methods:
                    methods[f"{service.full_name}/{method.name}"] = method
        return cls(pool, methods)

    def resolve_method(self, name: str) -> MethodDescriptor:
        key = name.lstrip("/")
        try:
            return self.methods[key]
        except KeyError as exc:
            raise ValueError(f"unknown MORAI gRPC method: {name}") from exc


def _installed_descriptor_set() -> Path:
    from ament_index_python.packages import get_package_share_directory

    return (
        Path(get_package_share_directory("ad_morai_bridge_dev"))
        / "data"
        / "morai_api.desc"
    )
