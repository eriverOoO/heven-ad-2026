import hashlib
import importlib
import json
from pathlib import Path

import pytest
import yaml


REPOSITORY = Path(__file__).resolve().parents[2]
PRESETS = REPOSITORY / "ad_morai_bridge_dev" / "config" / "actor_presets.yaml"
PROVENANCE = (
    REPOSITORY
    / "ad_morai_bridge_dev"
    / "config"
    / "actor_presets.provenance.yaml"
)
SOURCE_LINK_SET = REPOSITORY / "ad_data" / "map" / "link_set.json"


def _module():
    return importlib.import_module("ad_morai_bridge_dev.actors.provenance")


def _write_single_link_fixture(
    tmp_path,
    *,
    waypoint_idx,
    points,
    spawn_xyz,
    heading_deg,
):
    source = tmp_path / "link_set.json"
    source.write_text(
        json.dumps(
            [
                {
                    "idx": "L1",
                    "from_node_idx": "N0",
                    "to_node_idx": "N0",
                    "points": points,
                }
            ],
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    presets = tmp_path / "presets.yaml"
    presets.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "client_key": "test-client",
                "presets": [
                    {
                        "name": "loop",
                        "map_id": "test-map",
                        "actor_type": "vehicle",
                        "behavior": "physical_ai",
                        "request_id": "npc-1",
                        "model_name": "model",
                        "label": "npc-1",
                        "xyz": spawn_xyz,
                        "rpy_deg": [0.0, 0.0, heading_deg],
                        "velocity": 1.0,
                        "decision_range": 1.0,
                        "route_links": [
                            {"id": "L1", "waypoint_idx": waypoint_idx}
                        ],
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    provenance = tmp_path / "provenance.yaml"
    provenance.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "source": {
                    "map_id": "test-map",
                    "link_set_path_hint": "ignored/link_set.json",
                    "link_set_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                },
                "routes": {
                    "loop": {
                        "closed": True,
                        "links": [
                            {
                                "id": "L1",
                                "waypoint_idx": waypoint_idx,
                                "from_node": "N0",
                                "to_node": "N0",
                                "first_point": spawn_xyz,
                                "initial_heading_deg": heading_deg,
                            }
                        ],
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return presets, provenance, source


def test_repository_actor_presets_have_deterministic_route_provenance():
    summary = _module().validate_actor_preset_provenance(PRESETS, PROVENANCE)

    assert summary == {
        "map_id": "R_KR_PR_K-city_2025",
        "preset_count": 2,
        "routes": {
            "kcity-highway": {
                "closed": False,
                "link_count": 4,
            },
            "kcity-roundabout-loop": {
                "closed": True,
                "link_count": 4,
            },
        },
        "source_link_set_sha256": (
            "5ce0fd57da07b83221c0a4d117b0606b2d9c094c8de476a7d1176c78a4dc5b0f"
        ),
    }


@pytest.mark.parametrize("source_backed", [False, True])
def test_waypoint_only_preset_drift_is_rejected(tmp_path, source_backed):
    if source_backed and not SOURCE_LINK_SET.is_file():
        pytest.skip("ignored workstation source link_set is unavailable")
    preset_document = yaml.safe_load(PRESETS.read_text(encoding="utf-8"))
    preset_document["presets"][0]["route_links"][0]["waypoint_idx"] = 999
    changed_presets = tmp_path / "actor_presets.yaml"
    changed_presets.write_text(
        yaml.safe_dump(preset_document, sort_keys=False),
        encoding="utf-8",
    )

    arguments = {}
    if source_backed:
        arguments["source_link_set"] = SOURCE_LINK_SET
    with pytest.raises(ValueError, match="waypoint"):
        _module().validate_actor_preset_provenance(
            changed_presets,
            PROVENANCE,
            **arguments,
        )


def test_source_geometry_uses_the_selected_nonzero_waypoint(tmp_path):
    presets, provenance, source = _write_single_link_fixture(
        tmp_path,
        waypoint_idx=1,
        points=[[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [1.0, 3.0, 3.0]],
        spawn_xyz=[1.0, 2.0, 3.0],
        heading_deg=90.0,
    )

    result = _module().validate_actor_preset_provenance(
        presets, provenance, source_link_set=source
    )
    assert result["routes"] == {"loop": {"closed": True, "link_count": 1}}


def test_source_rejects_a_matching_but_out_of_range_waypoint(tmp_path):
    presets, provenance, source = _write_single_link_fixture(
        tmp_path,
        waypoint_idx=999,
        points=[[1.0, 2.0, 3.0], [2.0, 2.0, 3.0]],
        spawn_xyz=[1.0, 2.0, 3.0],
        heading_deg=0.0,
    )

    with pytest.raises(ValueError, match="waypoint.*range"):
        _module().validate_actor_preset_provenance(
            presets, provenance, source_link_set=source
        )


def test_source_preserves_repeated_link_occurrences_in_route_order(tmp_path):
    presets, provenance, source = _write_single_link_fixture(
        tmp_path,
        waypoint_idx=0,
        points=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
        spawn_xyz=[0.0, 0.0, 0.0],
        heading_deg=0.0,
    )
    preset_document = yaml.safe_load(presets.read_text(encoding="utf-8"))
    preset_document["presets"][0]["route_links"].append(
        {"id": "L1", "waypoint_idx": 1}
    )
    presets.write_text(
        yaml.safe_dump(preset_document, sort_keys=False),
        encoding="utf-8",
    )
    provenance_document = yaml.safe_load(provenance.read_text(encoding="utf-8"))
    provenance_document["routes"]["loop"]["links"].append(
        {
            "id": "L1",
            "waypoint_idx": 1,
            "from_node": "N0",
            "to_node": "N0",
            "first_point": [1.0, 0.0, 0.0],
            "initial_heading_deg": 90.0,
        }
    )
    provenance.write_text(
        yaml.safe_dump(provenance_document, sort_keys=False),
        encoding="utf-8",
    )

    result = _module().validate_actor_preset_provenance(
        presets, provenance, source_link_set=source
    )
    assert result["routes"] == {"loop": {"closed": True, "link_count": 2}}


@pytest.mark.parametrize("invalid_waypoint", [False, 0.0, -1, "0"])
def test_provenance_waypoint_requires_a_nonnegative_integer(
    tmp_path, invalid_waypoint
):
    presets, provenance, _source = _write_single_link_fixture(
        tmp_path,
        waypoint_idx=0,
        points=[[1.0, 2.0, 3.0], [2.0, 2.0, 3.0]],
        spawn_xyz=[1.0, 2.0, 3.0],
        heading_deg=0.0,
    )
    document = yaml.safe_load(provenance.read_text(encoding="utf-8"))
    document["routes"]["loop"]["links"][0]["waypoint_idx"] = invalid_waypoint
    provenance.write_text(
        yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="waypoint.*nonnegative integer"):
        _module().validate_actor_preset_provenance(presets, provenance)


def test_source_link_set_hash_and_link_geometry_are_verified(tmp_path):
    presets, provenance, source = _write_single_link_fixture(
        tmp_path,
        waypoint_idx=0,
        points=[[1.0, 2.0, 3.0], [2.0, 2.0, 3.0]],
        spawn_xyz=[1.0, 2.0, 3.0],
        heading_deg=0.0,
    )

    source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        _module().validate_actor_preset_provenance(
            presets, provenance, source_link_set=source
        )


def test_provenance_cli_prints_stable_json(capsys):
    code = _module().main([str(PRESETS), str(PROVENANCE)])

    assert code == 0
    output = capsys.readouterr().out
    assert output.endswith("\n")
    assert json.loads(output)["preset_count"] == 2
    assert output == json.dumps(json.loads(output), sort_keys=True) + "\n"


def test_setup_declares_yaml_runtime_dependency():
    setup_text = (REPOSITORY / "ad_morai_bridge_dev" / "setup.py").read_text(
        encoding="utf-8"
    )

    assert '"PyYAML>=5.4"' in setup_text
