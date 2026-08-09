from pathlib import Path

import pytest

from ad_tuning import tuner_node


def test_tuning_output_dir_prefers_explicit_path(tmp_path):
    explicit = tmp_path / "explicit"

    assert tuner_node.resolve_tuning_output_dir(
        explicit,
        algorithm="dwa",
        environ={"AD_DATA_DIR": str(tmp_path / "ignored")},
    ) == explicit


def test_tuning_output_dir_uses_canonical_data_root(tmp_path):
    data_root = tmp_path / "ad_data"

    assert tuner_node.resolve_tuning_output_dir(
        "",
        algorithm="profile_stanley",
        environ={"AD_DATA_DIR": str(data_root)},
    ) == data_root / "tuning" / "profile_stanley"


def test_tuning_output_dir_requires_explicit_root():
    with pytest.raises(ValueError, match="AD_DATA_DIR"):
        tuner_node.resolve_tuning_output_dir(
            "", algorithm="dwa", environ={}
        )
