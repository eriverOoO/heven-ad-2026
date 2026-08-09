import json
from pathlib import Path

from ad_morai_bridge.stream_defaults import STREAMS as COMPETITION_STREAMS
from ad_morai_bridge_dev.bridge.endpoints import (
    OUTPUTS,
    STREAMS as DEVELOPMENT_STREAMS,
)


ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "docs" / "morai" / "protocol_coverage.json"
DOCUMENTATION = ROOT / "docs" / "morai" / "protocol-coverage.md"
README = ROOT / "README.md"

REQUIRED_TOPICS = (
    "/ad/sensors/gps/rmc",
    "/ad/sensors/gps/gga",
    "/ad/sensors/imu/full",
    "/ad/dev/vehicle/ego_status/full",
    "/ad/dev/lidar2d/full",
    "/ad/dev/camera/front/bboxes",
    "/ad/dev/camera/traffic_light/bboxes",
    "/ad/dev/npc/collisions",
    "/ad/dev/command/scenario_load",
)


def _catalog():
    return json.loads(CATALOG.read_text(encoding="utf-8"))


def test_protocol_catalog_is_versioned_and_has_unique_endpoint_names():
    data = _catalog()

    assert data["schema_version"] == 1
    assert data["raw_audit"] == {
        "parameter": "publish_raw_packets",
        "default": False,
        "topics_are_capabilities_not_default_publishers": True,
    }
    assert data["references"]["morai_msgs_24r2_commit"] == (
        "176610aeaecad28cc31e62e1cd2548ba14c6fde7"
    )
    assert data["references"]["morai_msgs_26r1_commit"] == (
        "c84d648638510d28b691b959b03502009f4e2070"
    )
    names = [entry["name"] for entry in data["endpoints"]]
    assert len(names) == len(set(names))
    assert {entry["support"] for entry in data["endpoints"]} <= {
        "typed",
        "raw_only",
        "unsupported_build",
    }


def test_catalog_covers_exact_runtime_tables():
    data = _catalog()
    runtime = {
        (
            entry["profile"],
            entry["direction"],
            entry["runtime_key"],
            entry["port"],
        )
        for entry in data["endpoints"]
        if entry.get("runtime_key")
    }
    expected = {
        ("competition", "input", name, defaults.port)
        for name, defaults in COMPETITION_STREAMS.items()
    }
    expected |= {
        ("development", "input", name, values[0])
        for name, values in DEVELOPMENT_STREAMS.items()
    }
    expected |= {
        ("development", "output", name, port)
        for name, port in OUTPUTS.items()
    }

    assert runtime == expected


def test_catalog_keeps_competition_and_development_surfaces_separate():
    data = _catalog()
    competition = [
        entry for entry in data["endpoints"] if entry["profile"] == "competition"
    ]
    development = [
        entry for entry in data["endpoints"] if entry["profile"] == "development"
    ]

    assert all(
        not topic.startswith("/ad/dev/")
        for entry in competition
        for topic in entry["topics"]
    )
    assert all(
        topic.startswith("/ad/dev/")
        for entry in development
        for topic in entry["topics"]
    )


def test_catalog_has_four_competition_camera_roles():
    data = _catalog()
    cameras = {
        entry["name"]: entry
        for entry in data["endpoints"]
        if entry["profile"] == "competition"
        and entry["direction"] == "input"
        and entry["name"].startswith("camera_")
    }

    assert set(cameras) == {
        "camera_front",
        "camera_left",
        "camera_right",
        "camera_traffic_light",
    }
    assert cameras["camera_traffic_light"]["port"] == 9307
    assert cameras["camera_traffic_light"]["topics"][0] == (
        "/ad/sensors/camera/traffic_light/compressed"
    )


def test_human_documentation_covers_full_topics_and_reference_limits():
    documentation = DOCUMENTATION.read_text(encoding="utf-8")
    readme = README.read_text(encoding="utf-8")

    assert all(topic in documentation for topic in REQUIRED_TOPICS)
    assert "docs/morai/protocol-coverage.md" in readme
    assert "176610aeaecad28cc31e62e1cd2548ba14c6fde7" in documentation
    assert all(level in documentation for level in ("typed", "raw_only", "unsupported_build"))
    assert "UDP-byte compatible" in documentation
