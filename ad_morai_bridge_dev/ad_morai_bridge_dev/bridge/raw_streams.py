from dataclasses import dataclass


@dataclass(frozen=True)
class RawStreamSpec:
    name: str
    bind_ip: str
    port: int
    frame_id: str


def read_raw_stream_specs(node) -> tuple[RawStreamSpec, ...]:
    """Extra developer-defined UDP taps published as raw packets (default none).

    A duplicate or invalid port fails loudly at socket bind time.
    """
    names = node.declare_parameter("extra_raw_streams.names", []).value or []
    specs = []
    for name in [str(value) for value in names]:
        prefix = f"extra_raw_streams.{name}"
        specs.append(
            RawStreamSpec(
                name,
                str(
                    node.declare_parameter(
                        f"{prefix}.bind_ip", "127.0.0.1"
                    ).value
                ),
                int(node.declare_parameter(f"{prefix}.port", 0).value),
                str(node.declare_parameter(f"{prefix}.frame_id", name).value),
            )
        )
    return tuple(specs)
