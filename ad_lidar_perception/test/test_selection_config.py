from pathlib import Path

import pytest

from ad_lidar_perception.selection import (
    SelectionError,
    load_selection,
)


VALID = """\
schema_version: 1
detector:
  backend: centerpoint_tiny
  model_subdir: models/autoware
  build_only: false
tracker:
  backend: autoware
occupancy:
  static_enabled: true
  dynamic_enabled: true
  publish_combined: true
"""


def write_config(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "selection.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_loads_every_supported_detector_and_exact_scalar_types(tmp_path):
    for backend in (
        "none",
        "euclidean_cluster",
        "centerpoint_tiny",
        "centerpoint",
        "transfusion",
        "bevfusion_lidar",
    ):
        tracker = "none" if backend == "none" else "autoware"
        dynamic = "false" if tracker == "none" else "true"
        config = VALID.replace("centerpoint_tiny", backend).replace(
            "backend: autoware", f"backend: {tracker}"
        ).replace("dynamic_enabled: true", f"dynamic_enabled: {dynamic}")
        selection = load_selection(write_config(tmp_path, config))

        assert selection.schema_version == 1
        assert selection.detector.backend == backend
        assert selection.detector.model_subdir == Path("models/autoware")
        assert selection.detector.build_only is False
        assert selection.tracker.backend == tracker
        assert selection.occupancy.static_enabled is True
        assert selection.occupancy.dynamic_enabled is (tracker == "autoware")
        assert selection.occupancy.publish_combined is True


@pytest.mark.parametrize(
    "text, diagnostic",
    [
        ("[]\n", "root must be a mapping"),
        (
            VALID.replace("detector:\n", "detector: false\nignored:\n"),
            "detector must be a mapping",
        ),
        (
            VALID.replace("schema_version: 1\n", ""),
            "root is missing keys: schema_version",
        ),
        (
            VALID.replace(
                "schema_version: 1\n",
                "schema_version: 1\nextra: 2\n",
            ),
            "root has unknown keys: extra",
        ),
        (
            VALID.replace("  build_only: false\n", ""),
            "detector is missing keys: build_only",
        ),
        (
            VALID.replace(
                "  build_only: false\n", "  build_only: false\n  mystery: 1\n"
            ),
            "detector has unknown keys: mystery",
        ),
        (
            VALID.replace("schema_version: 1", "schema_version: true"),
            "schema_version must be an integer",
        ),
        (
            VALID.replace("schema_version: 1", "schema_version: 2"),
            "schema_version must equal 1",
        ),
        (
            VALID.replace("  build_only: false", '  build_only: "false"'),
            "detector.build_only must be a boolean",
        ),
        (
            VALID.replace("  static_enabled: true", "  static_enabled: 1"),
            "occupancy.static_enabled must be a boolean",
        ),
        (
            VALID.replace("centerpoint_tiny", "unknown_detector"),
            "detector.backend must be one of",
        ),
        (
            VALID.replace("backend: autoware", "backend: cv_kf"),
            "tracker.backend must be one of",
        ),
    ],
)
def test_rejects_unknown_missing_or_wrong_typed_values(
    tmp_path, text, diagnostic
):
    with pytest.raises(SelectionError, match=diagnostic):
        load_selection(write_config(tmp_path, text))


@pytest.mark.parametrize(
    "text",
    [
        VALID.replace(
            "schema_version: 1", "schema_version: 1\nschema_version: 1"
        ),
        VALID.replace(
            "  backend: centerpoint_tiny",
            "  backend: centerpoint_tiny\n  backend: centerpoint",
        ),
        VALID.replace(
            "  static_enabled: true",
            "  static_enabled: true\n  static_enabled: false",
        ),
    ],
)
def test_rejects_duplicate_keys_at_every_depth(tmp_path, text):
    with pytest.raises(SelectionError, match="duplicate key"):
        load_selection(write_config(tmp_path, text))


@pytest.mark.parametrize(
    "subdir",
    [
        '""',
        '"."',
        '".."',
        '"/models/autoware"',
        '"models/../autoware"',
        '"models\\\\autoware"',
        '"C:models/autoware"',
        '"models/\\0autoware"',
    ],
)
def test_rejects_unsafe_model_subdirectories(tmp_path, subdir):
    text = VALID.replace("models/autoware", subdir)
    with pytest.raises(SelectionError, match="detector.model_subdir"):
        load_selection(write_config(tmp_path, text))


def test_rejects_dependency_inversions(tmp_path):
    tracker_without_detector = (
        VALID.replace("centerpoint_tiny", "none")
        .replace("dynamic_enabled: true", "dynamic_enabled: false")
    )
    with pytest.raises(
        SelectionError, match="tracker requires a non-none detector"
    ):
        load_selection(write_config(tmp_path, tracker_without_detector))

    dynamic_without_tracker = VALID.replace(
        "backend: autoware", "backend: none"
    )
    with pytest.raises(
        SelectionError, match="dynamic occupancy requires tracker autoware"
    ):
        load_selection(write_config(tmp_path, dynamic_without_tracker))

    combined_without_static = VALID.replace(
        "static_enabled: true", "static_enabled: false"
    )
    with pytest.raises(
        SelectionError, match="combined occupancy requires static occupancy"
    ):
        load_selection(write_config(tmp_path, combined_without_static))


def test_none_none_default_is_valid(tmp_path):
    text = (
        VALID.replace("centerpoint_tiny", "none")
        .replace("backend: autoware", "backend: none")
        .replace("dynamic_enabled: true", "dynamic_enabled: false")
    )
    selection = load_selection(write_config(tmp_path, text))
    assert selection.detector.backend == "none"
    assert selection.tracker.backend == "none"


def test_morai_classical_profile_is_only_cluster_tracker_and_imm():
    profile = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "lidar_perception_morai_classical.yaml"
    )
    selection = load_selection(profile)

    assert selection.detector.backend == "euclidean_cluster"
    assert selection.detector.build_only is False
    assert selection.tracker.backend == "autoware"
    assert selection.occupancy.static_enabled is False
    assert selection.occupancy.dynamic_enabled is False
    assert selection.occupancy.publish_combined is False
