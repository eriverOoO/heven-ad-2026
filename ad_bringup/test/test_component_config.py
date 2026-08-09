from pathlib import Path

import pytest

from ad_bringup.component_config import load_components


KNOWN = ("description", "localization", "planner")


def write_config(path: Path, source: str) -> Path:
    path.write_text(source, encoding="utf-8")
    return path


def test_omitted_known_components_default_false_and_extra_sections_are_allowed(
    tmp_path,
):
    path = write_config(
        tmp_path / "components.yaml",
        "components:\n  description: true\nnotes:\n  owner: bringup\n",
    )

    assert load_components(path, KNOWN) == {
        "description": True,
        "localization": False,
        "planner": False,
    }


def test_unknown_component_is_rejected_as_a_probable_typo(tmp_path):
    path = write_config(
        tmp_path / "unknown.yaml",
        "components:\n  description: true\n  localizaton: true\n",
    )

    with pytest.raises(ValueError, match="unknown component: localizaton"):
        load_components(path, KNOWN)


def test_non_string_component_name_is_rejected_clearly(tmp_path):
    path = write_config(
        tmp_path / "non-string-name.yaml",
        "components:\n  1: true\n",
    )

    with pytest.raises(ValueError, match="component names must be strings"):
        load_components(path, KNOWN)


@pytest.mark.parametrize("value", ["true", 1, None, []])
def test_component_values_must_be_yaml_booleans(tmp_path, value):
    import yaml

    path = tmp_path / "invalid-value.yaml"
    path.write_text(
        yaml.safe_dump({"components": {"description": value}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="component values must be booleans"):
        load_components(path, KNOWN)


@pytest.mark.parametrize(
    "source",
    [
        "[]\n",
        "metadata:\n  owner: bringup\n",
        "components: true\n",
    ],
)
def test_document_requires_a_components_mapping(tmp_path, source):
    path = write_config(tmp_path / "invalid-document.yaml", source)

    with pytest.raises(ValueError, match="components.*mapping"):
        load_components(path, KNOWN)


def test_file_and_yaml_errors_include_the_config_path(tmp_path):
    missing = tmp_path / "missing.yaml"
    with pytest.raises(ValueError, match=str(missing)):
        load_components(missing, KNOWN)

    malformed = write_config(tmp_path / "malformed.yaml", "components: [\n")
    with pytest.raises(ValueError, match=str(malformed)):
        load_components(malformed, KNOWN)
