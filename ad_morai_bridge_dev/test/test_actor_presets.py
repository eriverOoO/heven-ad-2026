import importlib
from pathlib import Path

import pytest


REPOSITORY = Path(__file__).resolve().parents[2]
CONFIG = REPOSITORY / "ad_morai_bridge_dev" / "config" / "actor_presets.yaml"


def _module():
    return importlib.import_module("ad_morai_bridge_dev.actors.presets")


def test_repository_presets_are_strict_distinct_map_derived_routes():
    presets = _module().load_presets(CONFIG)

    assert set(presets) == {"kcity-highway", "kcity-roundabout-loop"}
    assert presets["kcity-roundabout-loop"].route_links == (
        ("A2256W000133", 0),
        ("A2256W000134", 0),
        ("A2256W000135", 0),
        ("A2256W000132", 0),
    )
    assert presets["kcity-highway"].route_links == (
        ("A2256W000409", 0),
        ("A2256W000420", 0),
        ("A2256W000179", 0),
        ("A2256W001105", 0),
    )
    assert len({preset.request_id for preset in presets.values()}) == 2


@pytest.mark.parametrize(
    "document,match",
    [
        (
            {"schema_version": 2, "client_key": "heven-task4", "presets": []},
            "schema_version",
        ),
        (
            {
                "schema_version": 1,
                "client_key": "heven-task4",
                "presets": [
                    {
                        "name": "same",
                        "map_id": "map",
                        "actor_type": "vehicle",
                        "behavior": "physical_ai",
                        "request_id": "one",
                        "model_name": "model",
                        "label": "label",
                        "xyz": [0, 0, 0],
                        "rpy_deg": [0, 0, 0],
                        "velocity": 1,
                        "decision_range": 1,
                        "route_links": [{"id": "L1", "waypoint_idx": 0}],
                    },
                    {
                        "name": "same",
                        "map_id": "map",
                        "actor_type": "vehicle",
                        "behavior": "physical_ai",
                        "request_id": "two",
                        "model_name": "model",
                        "label": "label",
                        "xyz": [0, 0, 0],
                        "rpy_deg": [0, 0, 0],
                        "velocity": 1,
                        "decision_range": 1,
                        "route_links": [{"id": "L1", "waypoint_idx": 0}],
                    },
                ],
            },
            "duplicate preset name",
        ),
    ],
)
def test_invalid_documents_are_rejected(tmp_path, document, match):
    import yaml

    path = tmp_path / "presets.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        _module().load_presets(path)


def test_forbidden_operation_keys_and_nonfinite_values_are_rejected(tmp_path):
    path = tmp_path / "presets.yaml"
    path.write_text(
        """
schema_version: 1
client_key: heven-task4
presets:
  - name: unsafe
    map_id: R_KR_PR_K-city_2025
    actor_type: vehicle
    behavior: physical_ai
    request_id: npc-1
    model_name: model
    label: label
    xyz: [.nan, 0, 0]
    rpy_deg: [0, 0, 0]
    velocity: 1
    decision_range: 1
    route_links: [{id: L1, waypoint_idx: 0}]
    scenario_load: forbidden
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="forbidden|finite"):
        _module().load_presets(path)
