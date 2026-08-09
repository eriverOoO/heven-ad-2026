import ast
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
COMPETITION_PACKAGE_NAMES = {
    "ad_control",
    "ad_description",
    "ad_interfaces",
    "ad_localization",
    "ad_morai_bridge",
    "ad_lidar_perception",
    "ad_camera_perception",
    "ad_planner",
}
COMPETITION_PACKAGE_ROOTS = tuple(
    REPO_ROOT / name for name in sorted(COMPETITION_PACKAGE_NAMES)
)
COMPETITION_LAUNCH = REPO_ROOT / "ad_morai_bridge" / "launch" / "bridge.launch.py"
COMPETITION_PROFILE = REPO_ROOT / "ad_morai_bridge" / "config" / "competition.yaml"
DEVELOPMENT_PACKAGE_NAMES = {"ad_morai_interfaces_dev", "ad_morai_bridge_dev"}
DEVELOPMENT_ACTOR_TOOL_SURFACES = {
    "ad_morai_actor",
    "actor_presets",
    "morai_actor_control",
}
PROTOCOL_CATALOG = REPO_ROOT / "docs" / "morai" / "protocol_coverage.json"
SENSOR_MOUNT_CONFIG = REPO_ROOT / "ad_description" / "config" / "sensor_mounts.yaml"
DEVELOPMENT_ONLY_BRINGUP_LAUNCH = (
    REPO_ROOT / "ad_bringup" / "launch" / "morai_global_path.launch.py"
)
COMPETITION_BRINGUP_RUNTIME_PATHS = {
    REPO_ROOT / "ad_bringup" / "launch" / "bringup.launch.py",
    REPO_ROOT / "ad_bringup" / "ad_bringup" / "bringup_stack.py",
    REPO_ROOT / "ad_bringup" / "config" / "components.yaml",
    REPO_ROOT / "ad_bringup" / "launch" / "course_trial.launch.py",
    REPO_ROOT / "ad_bringup" / "ad_bringup" / "course_stack.py",
}
CAMERA_SIGNAL_RUNTIME_PATHS = {
    (
        REPO_ROOT
        / "ad_camera_perception"
        / "ad_camera_perception"
        / "traffic_light"
        / "__init__.py"
    ),
    (
        REPO_ROOT
        / "ad_camera_perception"
        / "ad_camera_perception"
        / "traffic_light"
        / "traffic.py"
    ),
    (
        REPO_ROOT
        / "ad_camera_perception"
        / "ad_camera_perception"
        / "inference"
        / "yolov7_backend.py"
    ),
    (
        REPO_ROOT
        / "ad_camera_perception"
        / "ad_camera_perception"
        / "nodes"
        / "traffic_signal_node.py"
    ),
    (
        REPO_ROOT
        / "ad_camera_perception"
        / "ad_camera_perception"
        / "nodes"
        / "traffic_light_detector_node.py"
    ),
    (
        REPO_ROOT
        / "ad_camera_perception"
        / "ad_camera_perception"
        / "nodes"
        / "traffic_light_evaluator_node.py"
    ),
    (
        REPO_ROOT
        / "ad_camera_perception"
        / "ad_camera_perception"
        / "nodes"
        / "traffic_light_visualizer_node.py"
    ),
    (
        REPO_ROOT
        / "ad_camera_perception"
        / "ad_camera_perception"
        / "visualization"
        / "traffic_light_overlay.py"
    ),
    REPO_ROOT / "ad_camera_perception" / "config" / "traffic_light.yaml",
    (
        REPO_ROOT
        / "ad_camera_perception"
        / "launch"
        / "camera_perception.launch.py"
    ),
    (
        REPO_ROOT
        / "ad_camera_perception"
        / "launch"
        / "traffic_signal.launch.py"
    ),
}

COMPETITION_INPUT_PORTS = {
    "competition_status": 1909,
    "collisions": 9092,
    "camera_front": 9291,
    "camera_left": 9293,
    "camera_right": 9295,
    "camera_traffic_light": 9307,
    "gps": 9297,
    "imu": 9299,
    "velodyne": 2368,
}
INPUT_TOPICS = {
    "competition_status": "/ad/vehicle/status",
    "collisions": "/ad/safety/collisions",
    "camera_front": "/ad/sensors/camera/front/compressed",
    "camera_left": "/ad/sensors/camera/left/compressed",
    "camera_right": "/ad/sensors/camera/right/compressed",
    "camera_traffic_light": "/ad/sensors/camera/traffic_light/compressed",
    "gps": "/ad/sensors/gps/fix",
    "imu": "/ad/sensors/imu/data",
    "velodyne": "/ad/sensors/lidar/raw",
}
CONTROL_CONTRACT = ("/ad/control/command", 9093)

RUNTIME_SUFFIXES = {
    ".c",
    ".cc",
    ".cfg",
    ".cpp",
    ".cxx",
    ".h",
    ".hh",
    ".hpp",
    ".hxx",
    ".msg",
    ".py",
    ".xml",
    ".yaml",
    ".yml",
}
CONVENTIONAL_CPP_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hh",
    ".hpp",
    ".hxx",
}
IGNORED_PARTS = {"test", "tests", "__pycache__", ".pytest_cache"}
FORBIDDEN_RUNTIME_PATTERNS = {
    "ground-truth object surface": re.compile(
        r"\bobject(?:info|status)\b", re.IGNORECASE
    ),
    "camera bounding-box surface": re.compile(
        r"\bcamera(?:[\w/-]*bbox|[\w /_-]*bounding.?box)\b",
        re.IGNORECASE,
    ),
    "traffic-light surface": re.compile(
        r"(?<!camera[_/])\btraffic.?light\b", re.IGNORECASE
    ),
    "intersection surface": re.compile(r"\bintersection\b", re.IGNORECASE),
    "V2X surface": re.compile(r"\bv2x\b", re.IGNORECASE),
    "simulator command/output surface": re.compile(
        r"\bsimulator.?(?:command|output)\b", re.IGNORECASE
    ),
    "scenario-load surface": re.compile(r"\bscenario.?load\b", re.IGNORECASE),
    "ghost-control surface": re.compile(r"\bghost(?:.?control)?\b", re.IGNORECASE),
    "legacy project prefix": re.compile(r"\bmm_|/mm/"),
    "absolute user path": re.compile(r"/(?:home|users)/", re.IGNORECASE),
}


def _runtime_files():
    for package_root in COMPETITION_PACKAGE_ROOTS:
        for path in sorted(package_root.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(package_root)
            if any(part in IGNORED_PARTS for part in relative.parts):
                continue
            if path.name == "CMakeLists.txt" or path.suffix in RUNTIME_SUFFIXES:
                yield path
    yield COMPETITION_LAUNCH
    yield COMPETITION_PROFILE


def _profile(name: str) -> dict:
    path = REPO_ROOT / "ad_morai_bridge" / "config" / name
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert set(document) == {"/**"}
    assert set(document["/**"]) == {"ros__parameters"}
    parameters = document["/**"]["ros__parameters"]
    assert isinstance(parameters, dict)
    return parameters


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _keyword(call: ast.Call, name: str) -> ast.AST | None:
    return next((item.value for item in call.keywords if item.arg == name), None)


def test_runtime_scanner_covers_conventional_cpp_sources():
    scanned = set(_runtime_files())

    assert CONVENTIONAL_CPP_SUFFIXES <= RUNTIME_SUFFIXES
    assert REPO_ROOT / "ad_control" / "src" / "lateral" / "stanley.cpp" in scanned
    assert (
        REPO_ROOT / "ad_control" / "include" / "ad_control" / "common" / "types.hpp"
        in scanned
    )
    assert (
        REPO_ROOT / "ad_planner" / "src" / "planner" / "planner_node.cpp"
        in scanned
    )
    assert (
        REPO_ROOT
        / "ad_planner"
        / "include"
        / "ad_planner"
        / "common"
        / "types.hpp"
        in scanned
    )


def test_camera_box_rule_does_not_reject_lidar_object_geometry():
    pattern = FORBIDDEN_RUNTIME_PATTERNS["camera bounding-box surface"]

    assert pattern.search("camera_bbox")
    assert pattern.search("camera bounding-box")
    assert not pattern.search("Shape::BOUNDING_BOX")
    assert not pattern.search("bounding-box tracked object")


def test_legacy_prefix_rule_does_not_reject_imm_identifiers():
    pattern = FORBIDDEN_RUNTIME_PATTERNS["legacy project prefix"]

    assert pattern.search("mm_legacy_node")
    assert pattern.search("/mm/status")
    assert not pattern.search("imm_predictor")


def test_normal_bringup_paths_contain_no_development_network_surfaces():
    forbidden = (*DEVELOPMENT_PACKAGE_NAMES, "grpc", "7789", "/ad/dev/")
    violations = []

    for path in sorted(COMPETITION_BRINGUP_RUNTIME_PATHS):
        text = path.read_text(encoding="utf-8").lower()
        for token in forbidden:
            if token.lower() in text:
                violations.append(
                    f"{path.relative_to(REPO_ROOT)} references {token}"
                )

    assert DEVELOPMENT_ONLY_BRINGUP_LAUNCH.is_file()
    assert DEVELOPMENT_ONLY_BRINGUP_LAUNCH not in COMPETITION_BRINGUP_RUNTIME_PATHS
    assert violations == []


def test_runtime_sources_contain_no_forbidden_or_legacy_surfaces():
    violations = []
    scanned = []
    for path in _runtime_files():
        scanned.append(path)
        text = path.read_text(encoding="utf-8")
        for label, pattern in FORBIDDEN_RUNTIME_PATTERNS.items():
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                # The competition permits camera-based signal perception. Keep
                # the exemption path-specific so simulator traffic-light
                # status/control surfaces remain forbidden everywhere.
                matched_line = text.splitlines()[line - 1].strip()
                if (
                    label == "traffic-light surface"
                    and (
                        path in CAMERA_SIGNAL_RUNTIME_PATHS
                        or (
                            path == SENSOR_MOUNT_CONFIG
                            and matched_line == "traffic_light:"
                        )
                    )
                ):
                    continue
                violations.append(
                    f"{path.relative_to(REPO_ROOT)}:{line}: {label}: {match.group(0)!r}"
                )

    assert scanned
    assert violations == [], "\n" + "\n".join(violations)


def test_project_owned_packages_nodes_and_executables_use_ad_prefix():
    package_names = {
        ET.parse(root / "package.xml").getroot().findtext("name")
        for root in COMPETITION_PACKAGE_ROOTS
    }
    assert package_names == COMPETITION_PACKAGE_NAMES
    assert all(name.startswith("ad_") for name in package_names)

    console_scripts = []
    for package_root in COMPETITION_PACKAGE_ROOTS:
        setup_path = package_root / "setup.py"
        if not setup_path.is_file():
            continue
        tree = ast.parse(setup_path.read_text(encoding="utf-8"), setup_path.as_posix())
        console_scripts.extend(
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and " = " in node.value
        )
    assert console_scripts
    assert all(script.partition(" = ")[0].startswith("ad_") for script in console_scripts)

    launch_path = COMPETITION_LAUNCH
    launch_tree = ast.parse(
        launch_path.read_text(encoding="utf-8"), launch_path.as_posix()
    )
    launched_nodes = [
        node
        for node in ast.walk(launch_tree)
        if isinstance(node, ast.Call) and _call_name(node) == "Node"
    ]
    assert len(launched_nodes) == 5
    for call in launched_nodes:
        package = ast.literal_eval(_keyword(call, "package"))
        executable = ast.literal_eval(_keyword(call, "executable"))
        name = ast.literal_eval(_keyword(call, "name"))
        assert name.startswith("ad_")
        if package.startswith("ad_"):
            assert executable.startswith("ad_")


def test_profiles_are_exactly_nine_inputs_plus_one_control_endpoint():
    expected_top_level = {
        "queue_capacity",
        "timestamp_mode",
        "source_stamp_tolerance_sec",
        "publish_raw_packets",
        *COMPETITION_INPUT_PORTS,
        "control",
    }
    parameters = _profile("competition.yaml")
    assert set(parameters) == expected_top_level
    assert parameters["publish_raw_packets"] is False
    endpoints = []
    for stream_name, expected_port in COMPETITION_INPUT_PORTS.items():
        stream = parameters[stream_name]
        assert stream["enabled"] is True
        assert stream["bind_ip"] == "127.0.0.1"
        assert stream["topic"] == INPUT_TOPICS[stream_name]
        assert stream["port"] == expected_port
        assert stream["topic"].startswith("/ad/")
        endpoints.append((stream["bind_ip"], stream["port"]))

    control = parameters["control"]
    assert control["enabled"] is False
    assert control["target_ip"] == "127.0.0.1"
    assert control["topic"] == CONTROL_CONTRACT[0]
    assert control["target_port"] == CONTROL_CONTRACT[1]
    assert control["topic"].startswith("/ad/")
    endpoints.append((control["target_ip"], control["target_port"]))
    assert len(endpoints) == len(set(endpoints)) == 10


def test_protocol_catalog_matches_competition_network_allowlist():
    catalog = json.loads(PROTOCOL_CATALOG.read_text(encoding="utf-8"))
    competition_inputs = {
        entry["runtime_key"]: entry["port"]
        for entry in catalog["endpoints"]
        if entry["profile"] == "competition"
        and entry["direction"] == "input"
        and entry.get("runtime_key")
    }
    assert competition_inputs == COMPETITION_INPUT_PORTS

    control = next(
        entry
        for entry in catalog["endpoints"]
        if entry["profile"] == "competition"
        and entry["direction"] == "output"
        and entry["name"] == "ctrl_cmd"
    )
    assert (control["topics"][0], control["port"]) == CONTROL_CONTRACT

    development_ports = {
        entry["port"]
        for entry in catalog["endpoints"]
        if entry["profile"] == "development" and entry.get("runtime_key")
    }
    competition_runtime = "\n".join(
        path.read_text(encoding="utf-8") for path in _runtime_files()
    )
    assert all(
        re.search(rf"(?<!\d){port}(?!\d)", competition_runtime) is None
        for port in development_ports
    )


def test_competition_launch_defaults_to_live_profile_without_control_or_point_conversion():
    launch_path = COMPETITION_LAUNCH
    tree = ast.parse(launch_path.read_text(encoding="utf-8"), launch_path.as_posix())
    declarations = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _call_name(node) != "DeclareLaunchArgument":
            continue
        name = ast.literal_eval(node.args[0])
        declarations[name] = _keyword(node, "default_value")

    assert set(declarations) == {
        "config",
        "control_enabled",
        "enable_traffic_light_camera",
        "enable_velodyne_points",
        "measurement_compatibility_config",
        "measurement_compatibility_enabled",
        "platform_profile",
        "velodyne_point_timing_mode",
        "velodyne_organize_cloud",
    }
    assert isinstance(declarations["config"], ast.Name)
    assert declarations["config"].id == "default_config"
    assert ast.literal_eval(declarations["control_enabled"]) == "false"
    assert (
        ast.literal_eval(declarations["enable_traffic_light_camera"])
        == "false"
    )
    assert ast.literal_eval(declarations["enable_velodyne_points"]) == "false"
    assert (
        ast.literal_eval(declarations["measurement_compatibility_enabled"])
        == "false"
    )
    assert isinstance(declarations["measurement_compatibility_config"], ast.Call)
    assert ast.literal_eval(declarations["platform_profile"]) == "morai"
    assert (
        ast.literal_eval(declarations["velodyne_point_timing_mode"])
        == "auto"
    )
    assert ast.literal_eval(declarations["velodyne_organize_cloud"]) == "true"

    string_literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert "competition.yaml" in string_literals
    assert "loopback.yaml" not in string_literals


def test_competition_runtime_does_not_import_or_depend_on_development_packages():
    violations = []
    for path in _runtime_files():
        text = path.read_text(encoding="utf-8")
        for package_name in DEVELOPMENT_PACKAGE_NAMES:
            if package_name in text:
                violations.append(
                    f"{path.relative_to(REPO_ROOT)} references {package_name}"
                )
    assert violations == []


def test_competition_runtime_has_no_development_actor_tool_surface():
    violations = []
    for path in _runtime_files():
        text = path.read_text(encoding="utf-8").lower()
        for token in DEVELOPMENT_ACTOR_TOOL_SURFACES:
            if token in text:
                violations.append(
                    f"{path.relative_to(REPO_ROOT)} references {token}"
                )
    assert violations == []


def test_runtime_packages_contain_no_generated_or_local_data():
    forbidden_parts = {
        "ad_data",
        "build",
        "data",
        "devel",
        "external",
        "install",
        "log",
        "logs",
        "results",
    }
    forbidden_suffixes = {
        ".bag",
        ".db3",
        ".mcap",
        ".onnx",
        ".pcap",
        ".pt",
        ".pth",
    }
    violations = []
    for package_root in COMPETITION_PACKAGE_ROOTS:
        for path in package_root.rglob("*"):
            relative = path.relative_to(package_root)
            if any(part in {"__pycache__", ".pytest_cache"} for part in relative.parts):
                continue
            if forbidden_parts.intersection(relative.parts):
                violations.append(path.relative_to(REPO_ROOT).as_posix())
            elif path.is_file() and path.suffix.lower() in forbidden_suffixes:
                violations.append(path.relative_to(REPO_ROOT).as_posix())
    assert violations == []


def test_readme_links_detailed_udp_documentation():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "docs/morai/protocol-coverage.md" in readme


def test_udp_docs_record_morai_persistence_and_port_contract():
    protocol = (REPO_ROOT / "docs" / "morai" / "protocol-coverage.md").read_text(
        encoding="utf-8"
    )
    persistence = (
        REPO_ROOT / "docs" / "morai" / "network-persistence.md"
    ).read_text(encoding="utf-8")

    for fragment in (
        "Competition Status",
        "1909",
        "Ego Status",
        "1911",
        "network-persistence.md",
    ):
        assert fragment in protocol
    for fragment in (
        "1908",
        "1909",
        "1910",
        "1911",
        "Disconnect",
        "Save",
        "Connect",
    ):
        assert fragment in persistence
