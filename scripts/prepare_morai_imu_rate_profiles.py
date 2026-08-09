#!/usr/bin/env python3
"""Create immutable MORAI IMU-rate profile clones from one sensor JSON."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence


RATE_PERIODS = {
    "20hz": 0.05000000074505806,
    "30hz": 0.03333333507180214,
    "50hz": 0.019999999552965164,
}
IMU_IDENTIFIER_KEYS = ("m_SensorUniqueID", "UNIQUEID")


def _json_pointer_token(value: object) -> str:
    return str(value).replace("~", "~0").replace("/", "~1")


def _changed_json_pointers(
    before: object, after: object, pointer: str = ""
) -> list[str]:
    if isinstance(before, dict) and isinstance(after, dict):
        changed = []
        for key in sorted(set(before) | set(after)):
            child = f"{pointer}/{_json_pointer_token(key)}"
            if key not in before or key not in after:
                changed.append(child)
            else:
                changed.extend(_changed_json_pointers(before[key], after[key], child))
        return changed
    if isinstance(before, list) and isinstance(after, list):
        changed = []
        for index, (old, new) in enumerate(zip(before, after)):
            changed.extend(_changed_json_pointers(old, new, f"{pointer}/{index}"))
        return changed if len(before) == len(after) else changed + [pointer]
    return [] if before == after else [pointer]


def _validate_finite_values(value: object, pointer: str = "") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite JSON number at {pointer or '/'}")
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_finite_values(item, f"{pointer}/{_json_pointer_token(key)}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_finite_values(item, f"{pointer}/{index}")


def _load_document(source: Path) -> tuple[bytes, dict[str, Any]]:
    try:
        source_bytes = source.read_bytes()
        document = json.loads(source_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid source JSON {source}: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError("source JSON root must be an object")
    _validate_finite_values(document)
    return source_bytes, document


def _select_imu(document: dict[str, Any], imu_unique_id: int) -> tuple[int, float]:
    if isinstance(imu_unique_id, bool) or not isinstance(imu_unique_id, int):
        raise ValueError("IMU unique ID must be an integer")
    imu_list = document.get("IMUList")
    if not isinstance(imu_list, list) or not imu_list:
        raise ValueError("IMUList must be a non-empty list")

    matches = []
    for index, imu in enumerate(imu_list):
        if not isinstance(imu, dict):
            raise ValueError(f"IMUList/{index} must be an object")
        identifiers = {
            key: imu[key] for key in IMU_IDENTIFIER_KEYS if key in imu
        }
        if not identifiers:
            raise ValueError(f"IMUList/{index} must contain a sensor identifier")
        for key, value in identifiers.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"IMUList/{index}/{key} must be an integer")
        if len(set(identifiers.values())) != 1:
            raise ValueError(f"IMUList/{index} sensor identifiers disagree")
        unique_id = next(iter(identifiers.values()))
        configuration = imu.get("ic")
        if not isinstance(configuration, dict):
            raise ValueError(f"IMUList/{index}/ic must be an object")
        period = configuration.get("sensorPeriod")
        if isinstance(period, bool) or not isinstance(period, (int, float)):
            raise ValueError(
                f"IMUList/{index}/ic/sensorPeriod must be a finite number"
            )
        try:
            period_value = float(period)
        except OverflowError as exc:
            raise ValueError(
                f"IMUList/{index}/ic/sensorPeriod must be a positive finite number"
            ) from exc
        if not math.isfinite(period_value) or period_value <= 0.0:
            raise ValueError(
                f"IMUList/{index}/ic/sensorPeriod must be a positive finite number"
            )
        if unique_id == imu_unique_id:
            matches.append((index, period_value))

    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one IMU with UNIQUEID {imu_unique_id}, found {len(matches)}"
        )
    return matches[0]


def _json_bytes(document: object) -> bytes:
    return (json.dumps(document, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _require_safe_output_paths(source: Path, paths: Sequence[Path]) -> None:
    source_path = source.resolve()
    for path in paths:
        if path.resolve() == source_path:
            raise ValueError("refusing to use the source path as generated output")


def _refuse_conflicts(expected: dict[Path, bytes]) -> None:
    for path, content in expected.items():
        if not path.exists():
            continue
        try:
            current = path.read_bytes()
        except OSError as exc:
            raise FileExistsError(f"conflicting output {path}") from exc
        if current != content:
            raise FileExistsError(f"conflicting output {path}")


def _write_exclusive(path: Path, content: bytes) -> None:
    if path.exists():
        try:
            if path.read_bytes() == content:
                return
        except OSError:
            pass
        raise FileExistsError(f"conflicting output {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            try:
                if path.read_bytes() == content:
                    return
            except OSError:
                pass
            raise FileExistsError(f"conflicting output {path}")
    finally:
        temporary.unlink(missing_ok=True)


def prepare_profiles(
    source: Path, output_dir: Path, imu_unique_id: int
) -> dict[str, object]:
    """Prepare 20/30/50 Hz clones without changing ``source``."""
    source = Path(source)
    output_dir = Path(output_dir)
    source_bytes, document = _load_document(source)
    selected_index, old_period = _select_imu(document, imu_unique_id)
    pointer = f"/IMUList/{selected_index}/ic/sensorPeriod"

    profile_metadata: dict[str, dict[str, object]] = {}
    expected: dict[Path, bytes] = {}
    for label, period in RATE_PERIODS.items():
        prepared = copy.deepcopy(document)
        prepared["IMUList"][selected_index]["ic"]["sensorPeriod"] = period
        changed = _changed_json_pointers(document, prepared)
        expected_changes = [] if period == old_period else [pointer]
        if changed != expected_changes:
            raise RuntimeError(
                "profile semantic-diff verification failed: "
                f"expected {expected_changes}, found {changed}"
            )
        _validate_finite_values(prepared)
        output_path = output_dir / f"{source.stem}__imu_{label}.json"
        output_bytes = _json_bytes(prepared)
        expected[output_path] = output_bytes
        profile_metadata[label] = {
            "json_pointer": pointer,
            "sensor_period": period,
            "sha256": hashlib.sha256(output_bytes).hexdigest(),
            "semantic_diff": {
                "changed_json_pointers": changed,
                "only_selected_sensor_period_changed": True,
            },
        }

    manifest: dict[str, object] = {
        "schema_version": 1,
        "source": {
            "bytes": len(source_bytes),
            "path": str(source),
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
        },
        "selected_imu": {
            "json_pointer": pointer,
            "old_sensor_period": old_period,
            "unique_id": imu_unique_id,
        },
        "profiles": profile_metadata,
    }
    manifest_path = output_dir / "imu_rate_profiles_manifest.json"
    expected[manifest_path] = _json_bytes(manifest)

    _require_safe_output_paths(source, tuple(expected))
    _refuse_conflicts(expected)
    for path, content in expected.items():
        _write_exclusive(path, content)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--imu-unique-id", type=int, required=True)
    arguments = parser.parse_args(argv)
    print(
        json.dumps(
            prepare_profiles(
                arguments.source, arguments.output_dir, arguments.imu_unique_id
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
