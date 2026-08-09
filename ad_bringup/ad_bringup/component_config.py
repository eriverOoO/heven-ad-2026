from collections.abc import Collection, Mapping
from pathlib import Path

import yaml


def load_components(
    path: str | Path,
    known_components: Collection[str],
) -> dict[str, bool]:
    config_path = Path(path)
    try:
        document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(
            f"failed to read component config {config_path}: {error}"
        ) from error

    if not isinstance(document, Mapping):
        raise ValueError(
            "component config must contain a 'components' mapping"
        )
    components = document.get("components")
    if not isinstance(components, Mapping):
        raise ValueError(
            "component config must contain a 'components' mapping"
        )

    if any(not isinstance(name, str) for name in components):
        raise ValueError("component names must be strings")

    known = tuple(known_components)
    unknown = set(components) - set(known)
    if unknown:
        raise ValueError("unknown component: " + ", ".join(sorted(unknown)))
    if any(type(value) is not bool for value in components.values()):
        raise ValueError("component values must be booleans")

    return {name: components.get(name, False) for name in known}
