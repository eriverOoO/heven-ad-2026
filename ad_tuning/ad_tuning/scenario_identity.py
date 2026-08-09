from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path


@dataclass(frozen=True)
class ScenarioIdentity:
    actual_path: Path
    expected_path: Path
    sha256: str


def default_morai_save_root() -> Path:
    return Path(
        os.environ.get(
            "AD_MORAI_SAVE_ROOT",
            str(
                Path.home()
                / "MoraiLauncher_Lin"
                / "MoraiLauncher_Lin_Data"
                / "SaveFile"
            ),
        )
    )


def _scenario_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_scenario_identity(
    *,
    scenario_file: str | Path,
    experiment_scenario: str,
    save_root: Path,
) -> ScenarioIdentity:
    actual_text = str(scenario_file).strip()
    if not actual_text:
        raise ValueError(
            "scenario_file must be explicit for DWA tuning"
        )
    scenario_text = str(experiment_scenario).strip()
    if not scenario_text or scenario_text == "unspecified":
        raise ValueError(
            "experiment.scenario must explicitly identify scenario_file"
        )

    actual_path = Path(actual_text).expanduser()
    if not actual_path.is_file():
        raise ValueError(
            f"scenario_file does not exist or is not a file: {actual_path}"
        )
    actual_path = actual_path.resolve()

    expected_path = Path(scenario_text).expanduser()
    if expected_path.suffix != ".json":
        expected_path = expected_path.with_suffix(".json")
    if not expected_path.is_absolute():
        expected_path = save_root.expanduser() / "Scenario" / expected_path
    if not expected_path.is_file():
        raise ValueError(
            "experiment.scenario file does not exist or is not a file: "
            f"{expected_path}"
        )
    expected_path = expected_path.resolve()

    actual_sha256 = _scenario_sha256(actual_path)
    expected_sha256 = _scenario_sha256(expected_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "scenario_file does not match experiment.scenario: "
            f"{actual_path} != {expected_path}"
        )
    return ScenarioIdentity(
        actual_path=actual_path,
        expected_path=expected_path,
        sha256=actual_sha256,
    )
