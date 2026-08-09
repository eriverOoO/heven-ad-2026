from pathlib import Path

import pytest

from ad_tuning.scenario_identity import validate_scenario_identity


SCENARIO_BYTES = b'{"scene":1}\n'
SCENARIO_SHA256 = (
    "31f3006969720059cd0f5e6bdc3b39963caf627402cd57207528ba7c5be9cf23"
)


def _write_scenario(path: Path, content: bytes = SCENARIO_BYTES) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_empty_actual_scenario_file_is_rejected_before_launch():
    with pytest.raises(ValueError, match="scenario_file must be explicit"):
        validate_scenario_identity(
            scenario_file="",
            experiment_scenario="checkpoint3_6",
            save_root=Path("/unused"),
        )


def test_actual_scenario_and_experiment_label_must_have_same_bytes(tmp_path):
    scenario = _write_scenario(
        tmp_path / "Scenario" / "checkpoint3_6.json"
    )

    identity = validate_scenario_identity(
        scenario_file=scenario,
        experiment_scenario="checkpoint3_6",
        save_root=tmp_path,
    )

    assert identity.actual_path == scenario.resolve()
    assert identity.expected_path == scenario.resolve()
    assert identity.sha256 == SCENARIO_SHA256


def test_mismatched_reset_file_and_experiment_scenario_are_rejected(tmp_path):
    actual = _write_scenario(
        tmp_path / "actual.json", b'{"scene":"actual"}\n'
    )
    _write_scenario(
        tmp_path / "Scenario" / "checkpoint3_6.json",
        b'{"scene":"labeled"}\n',
    )

    with pytest.raises(
        ValueError,
        match="scenario_file does not match experiment.scenario",
    ):
        validate_scenario_identity(
            scenario_file=actual,
            experiment_scenario="checkpoint3_6",
            save_root=tmp_path,
        )
