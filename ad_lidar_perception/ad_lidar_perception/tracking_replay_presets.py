"""Validated configuration for the ten camera/LiDAR replay presets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import argparse

import yaml


SUPPORTED_DETECTORS = {"euclidean", "centerpoint"}
SUPPORTED_ASSOCIATIONS = {"giou_3d", "euclidean", "mahalanobis"}
SUPPORTED_MATCHERS = {"greedy", "hungarian"}
SUPPORTED_ESTIMATORS = {"linear_kf", "ekf", "imm", "kalmannet"}


class PresetConfigError(ValueError):
    """Raised when the checked-in preset matrix is invalid."""


@dataclass(frozen=True)
class TrackingReplayPreset:
    mode: str
    label: str
    detector: str
    association: str
    matcher: str
    estimator: str
    yaw_measurement_mode: str
    euclidean_gate_m: float
    mahalanobis_gate: float
    mahalanobis_max_distance_m: float

    @property
    def needs_centerpoint(self) -> bool:
        return self.detector == "centerpoint"

    @property
    def needs_kalmannet(self) -> bool:
        return self.estimator == "kalmannet"


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PresetConfigError(f"{name} must be a mapping")
    return value


def load_tracking_replay_presets(
    path: str | Path,
) -> tuple[str, dict[str, TrackingReplayPreset]]:
    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise PresetConfigError(
            f"cannot load preset config {config_path}: {exc}"
        ) from exc
    root = _mapping(raw, "preset config")
    if root.get("schema_version") != 1:
        raise PresetConfigError("preset schema_version must be 1")
    common = _mapping(root.get("common"), "common")
    raw_presets = _mapping(root.get("presets"), "presets")
    expected_modes = {str(index) for index in range(1, 11)}
    actual_modes = {str(mode) for mode in raw_presets}
    if actual_modes != expected_modes:
        raise PresetConfigError(
            "presets must define each mode from 1 through 10 exactly once"
        )

    presets: dict[str, TrackingReplayPreset] = {}
    for mode in sorted(actual_modes, key=int):
        values = dict(common)
        values.update(_mapping(raw_presets[mode], f"preset {mode}"))
        try:
            preset = TrackingReplayPreset(
                mode=mode,
                label=str(values["label"]),
                detector=str(values["detector"]),
                association=str(values["association"]),
                matcher=str(values["matcher"]),
                estimator=str(values["estimator"]),
                yaw_measurement_mode=str(values["yaw_measurement_mode"]),
                euclidean_gate_m=float(values["euclidean_gate_m"]),
                mahalanobis_gate=float(values["mahalanobis_gate"]),
                mahalanobis_max_distance_m=float(values["mahalanobis_max_distance_m"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PresetConfigError(f"preset {mode} has invalid fields: {exc}") from exc
        if preset.detector not in SUPPORTED_DETECTORS:
            raise PresetConfigError(f"preset {mode} has unsupported detector")
        if preset.association not in SUPPORTED_ASSOCIATIONS:
            raise PresetConfigError(f"preset {mode} has unsupported association")
        if preset.matcher not in SUPPORTED_MATCHERS:
            raise PresetConfigError(f"preset {mode} has unsupported matcher")
        if preset.estimator not in SUPPORTED_ESTIMATORS:
            raise PresetConfigError(f"preset {mode} has unsupported estimator")
        if preset.yaw_measurement_mode not in {"detector", "unobserved"}:
            raise PresetConfigError(f"preset {mode} has unsupported yaw mode")
        if preset.euclidean_gate_m <= 0 or preset.mahalanobis_gate <= 0:
            raise PresetConfigError(f"preset {mode} gates must be positive")
        expected_cap = 10.0 if preset.association == "mahalanobis" else 0.0
        if preset.mahalanobis_max_distance_m != expected_cap:
            raise PresetConfigError(
                f"preset {mode} must use Mahalanobis 10m hybrid cap only "
                "for Mahalanobis"
            )
        presets[mode] = preset

    default_mode = str(root.get("default_mode", ""))
    if default_mode not in presets:
        raise PresetConfigError("default_mode must name one of the ten presets")
    return default_mode, presets


def select_tracking_replay_preset(
    path: str | Path, mode: str | int
) -> TrackingReplayPreset:
    _, presets = load_tracking_replay_presets(path)
    normalized = str(mode).strip()
    try:
        return presets[normalized]
    except KeyError as exc:
        raise PresetConfigError(
            f"mode must be an integer from 1 through 10, not {normalized!r}"
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode")
    args = parser.parse_args()
    default_mode, presets = load_tracking_replay_presets(args.config)
    if args.mode is not None:
        preset = select_tracking_replay_preset(args.config, args.mode)
        print(f"{preset.mode}: {preset.label}")
        return
    for mode, preset in presets.items():
        marker = " (default)" if mode == default_mode else ""
        print(f"{mode:>2}: {preset.label}{marker}")


if __name__ == "__main__":
    main()
