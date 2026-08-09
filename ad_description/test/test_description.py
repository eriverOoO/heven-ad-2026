"""Contracts for the IONIQ 5 xacro, vehicle data, and sensor profiles."""

import importlib.util
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest
import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
XACRO_PATH = PACKAGE_ROOT / "urdf" / "ad_ioniq5.urdf.xacro"
SENSOR_CONFIG = PACKAGE_ROOT / "config" / "sensor_mounts.yaml"
VEHICLE_CONFIG = PACKAGE_ROOT / "config" / "vehicle_parameters.yaml"
MESH_DIR = PACKAGE_ROOT / "meshes" / "ioniq5"
LAUNCH_PATH = PACKAGE_ROOT / "launch" / "description.launch.py"

EXPECTED_SENSOR_PARENTS = {
    "gps_link": "rear_axle_link",
    "imu_link": "rear_axle_link",
    "lidar_link": "rear_axle_link",
    "camera_front_link": "rear_axle_link",
    "camera_left_link": "rear_axle_link",
    "camera_right_link": "rear_axle_link",
    "camera_traffic_light_link": "rear_axle_link",
    "camera_front_optical_frame": "camera_front_link",
    "camera_left_optical_frame": "camera_left_link",
    "camera_right_optical_frame": "camera_right_link",
    "camera_traffic_light_optical_frame": "camera_traffic_light_link",
}


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _non_null_leaf_paths(value, prefix: str):
    """Yield source-classifiable parameter paths, treating sequences as values."""
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _non_null_leaf_paths(child, f"{prefix}.{key}")
    elif value is not None:
        yield prefix


def _render(profile: str | None = None) -> ET.Element:
    import xacro

    sensor_data = _load_yaml(SENSOR_CONFIG)
    selected = profile or sensor_data["active_profile"]
    document = xacro.process_file(
        str(XACRO_PATH),
        mappings={
            "sensor_config": str(SENSOR_CONFIG),
            "sensor_profile": selected,
        },
    )
    return ET.fromstring(document.toxml())


def _joints_by_child(root: ET.Element) -> dict[str, ET.Element]:
    return {
        joint.find("child").attrib["link"]: joint
        for joint in root.findall("joint")
    }


def _origin_values(joint: ET.Element) -> tuple[tuple[float, ...], tuple[float, ...]]:
    origin = joint.find("origin")
    assert origin is not None
    return (
        tuple(float(value) for value in origin.attrib["xyz"].split()),
        tuple(float(value) for value in origin.attrib["rpy"].split()),
    )


def _configured_origin(mount: dict) -> tuple[tuple[float, ...], tuple[float, ...]]:
    position = mount["position_m"]
    rotation = mount["rpy_rad"]
    return (
        (position["x"], position["y"], position["z"]),
        (rotation["roll"], rotation["pitch"], rotation["yaw"]),
    )


def _transform_point(
    matrix: tuple[float, ...], point: tuple[float, float, float]
) -> tuple[float, float, float]:
    x, y, z = point
    homogeneous = (x, y, z, 1.0)
    return tuple(
        sum(matrix[row * 4 + column] * homogeneous[column] for column in range(4))
        for row in range(3)
    )


def _rotate_rpy(
    point: tuple[float, float, float], rpy: tuple[float, float, float]
) -> tuple[float, float, float]:
    """Apply the URDF fixed-axis RPY rotation Rz(yaw) * Ry(pitch) * Rx(roll)."""
    x, y, z = point
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    after_roll = (x, cr * y - sr * z, sr * y + cr * z)
    after_pitch = (
        cp * after_roll[0] + sp * after_roll[2],
        after_roll[1],
        -sp * after_roll[0] + cp * after_roll[2],
    )
    return (
        cy * after_pitch[0] - sy * after_pitch[1],
        sy * after_pitch[0] + cy * after_pitch[1],
        after_pitch[2],
    )


def _dae_scene_points(dae_root: ET.Element) -> list[tuple[float, float, float]]:
    namespace = {"c": "http://www.collada.org/2005/11/COLLADASchema"}
    geometries = {}
    for geometry in dae_root.findall(".//c:library_geometries/c:geometry", namespace):
        position_source = geometry.find(
            ".//c:vertices/c:input[@semantic='POSITION']", namespace
        )
        assert position_source is not None
        source_id = position_source.attrib["source"].removeprefix("#")
        source = geometry.find(f".//c:source[@id='{source_id}']", namespace)
        assert source is not None
        values = tuple(
            float(value)
            for value in source.find("c:float_array", namespace).text.split()
        )
        accessor = source.find("c:technique_common/c:accessor", namespace)
        assert accessor is not None and int(accessor.attrib["stride"]) == 3
        geometries[geometry.attrib["id"]] = list(zip(*[iter(values)] * 3))

    scene_points = []
    for node in dae_root.findall(".//c:library_visual_scenes//c:node", namespace):
        matrix_element = node.find("c:matrix", namespace)
        assert matrix_element is not None
        matrix = tuple(float(value) for value in matrix_element.text.split())
        assert len(matrix) == 16
        for instance in node.findall("c:instance_geometry", namespace):
            geometry_id = instance.attrib["url"].removeprefix("#")
            scene_points.extend(
                _transform_point(matrix, point) for point in geometries[geometry_id]
            )
    return scene_points


def test_vehicle_parameters_are_coherent_and_source_classified():
    document = _load_yaml(VEHICLE_CONFIG)
    vehicle = document["vehicle"]
    geometry = vehicle["geometry"]
    dynamics = vehicle["dynamics"]
    footprint = vehicle["collision_footprint"]

    assert vehicle["model"] == "2023_Hyundai_Ioniq5"
    assert geometry["length_m"] == pytest.approx(4.635)
    assert geometry["width_m"] == pytest.approx(1.890)
    assert geometry["wheelbase_m"] == pytest.approx(3.0)
    assert geometry["front_overhang_m"] == pytest.approx(0.845)
    assert geometry["rear_overhang_m"] == pytest.approx(0.790)
    assert footprint["center_offset_x_m"] == pytest.approx(1.5275)
    assert footprint["half_length_m"] == pytest.approx(2.3175)
    assert footprint["half_width_m"] == pytest.approx(0.945)
    assert dynamics["cg_to_front_axle_m"] + dynamics["cg_to_rear_axle_m"] \
        == pytest.approx(geometry["wheelbase_m"])
    sources = document["sources"]
    provenance = document["provenance"]
    assert sources
    assert provenance
    assert all(
        item["status"] in {"manufacturer_confirmed", "measured", "derived", "provisional"}
        for item in provenance.values()
    )
    assert all(item["source"] in sources for item in provenance.values())
    assert all(item["covers"] for item in provenance.values())

    covered_roots = {
        covered
        for item in provenance.values()
        for covered in item["covers"]
    }
    required_roots = {
        "vehicle.control",
        "vehicle.geometry",
        "vehicle.mass",
        "vehicle.dynamics",
        "vehicle.tires",
        "vehicle.steering",
        "vehicle.aerodynamics",
        "vehicle.collision_footprint",
    }
    parameter_paths = []
    for root in required_roots:
        value = document
        for component in root.split("."):
            value = value[component]
        parameter_paths.extend(_non_null_leaf_paths(value, root))

    uncovered = [
        path
        for path in parameter_paths
        if not any(path == covered or path.startswith(f"{covered}.") for covered in covered_roots)
    ]
    assert not uncovered
    assert provenance["control_geometry"]["source"] == (
        "front_axle_control_point_derivation"
    )


def test_sensor_profiles_prepare_rear_axle_navigation_and_center_lidar():
    document = _load_yaml(SENSOR_CONFIG)
    assert document["active_profile"] == "current_front_sensor_mounts"
    assert document["morai_source"] == {
        "simulator_build": "25.S4.MolitComp03",
        "vehicle": "2023_Hyundai_Ioniq5",
        "active_file": "SensorInfo_2023_Hyundai_Ioniq5.json",
        "final_sha256": "8349697afaebe21d3c4bf0f85de961b8ae208a2f83780ae3783120a85fdda1c3",
    }
    convention = document["coordinate_convention"]
    assert convention["sensor_parent_frame"] == "rear_axle_link"
    assert convention["static_frames"] == {
        "rear_axle": {
            "frame_id": "rear_axle_link",
            "parent_frame": "base_link",
            "position_m": {"x": 0.0, "y": 0.0, "z": 0.3685},
            "rpy_rad": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
        },
        "front_axle": {
            "frame_id": "front_axle_link",
            "parent_frame": "rear_axle_link",
            "position_m": {"x": 3.0, "y": 0.0, "z": 0.0},
            "rpy_rad": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
        },
    }
    vehicle = _load_yaml(VEHICLE_CONFIG)["vehicle"]
    assert convention["static_frames"]["front_axle"]["position_m"]["x"] \
        == vehicle["geometry"]["wheelbase_m"]
    profiles = document["profiles"]
    assert set(profiles) == {
        "current_front_sensor_mounts",
        "planned_centered_sensor_mounts",
    }

    current = profiles["current_front_sensor_mounts"]["sensors"]
    planned = profiles["planned_centered_sensor_mounts"]["sensors"]
    assert _configured_origin(current["gps"])[0] == (0.0, 0.0, 1.2)
    assert _configured_origin(current["imu"])[0] == (0.0, 0.0, 1.2)
    assert _configured_origin(current["lidar"])[0] == (1.15, 0.0, 1.4)
    assert current["gps"]["maximum_rate_hz"] == 20.0
    assert current["gps"]["noise"] == {
        "gaussian_enabled": True,
        "gaussian_mean_m": 0.0,
        "gaussian_stdev_m": 0.5,
    }
    assert current["imu"]["maximum_rate_hz"] == 50.0
    assert current["imu"]["noise"] == {
        "white_noise_enabled": True,
        "linear_acceleration_stdev_mps2": {
            "x": 0.041,
            "y": 0.041,
            "z": 0.041,
        },
        "angular_velocity_stdev_radps": {
            "roll": 0.003,
            "pitch": 0.003,
            "yaw": 0.00314,
        },
        "random_walk_enabled": False,
        "bias_instability_enabled": False,
    }
    assert current["lidar"]["maximum_rate_hz"] == 10.0
    assert _configured_origin(planned["gps"])[0] == (0.0, 0.0, 0.7)
    assert _configured_origin(planned["imu"])[0] == (0.0, 0.0, 0.7)
    assert _configured_origin(planned["lidar"])[0] == (1.5275, 0.0, 1.0)

    cameras = current["cameras"]
    assert set(cameras) == {"front", "left", "right", "traffic_light"}
    assert cameras["front"]["image_size"] == {"width": 1280, "height": 720}
    assert cameras["front"]["maximum_rate_hz"] == 20.0
    assert cameras["left"]["maximum_rate_hz"] == 20.0
    assert cameras["right"]["maximum_rate_hz"] == 20.0
    assert cameras["traffic_light"]["maximum_rate_hz"] == 20.0
    assert cameras["traffic_light"]["horizontal_fov_deg"] == 90.0
    assert _configured_origin(cameras["traffic_light"])[1] == pytest.approx(
        (0.0, -math.pi / 6.0, 0.0)
    )


@pytest.mark.parametrize(
    "profile", ["current_front_sensor_mounts", "planned_centered_sensor_mounts"]
)
def test_xacro_renders_the_selected_profile_and_all_four_optical_frames(profile):
    root = _render(profile)
    links = {link.attrib["name"] for link in root.findall("link")}
    joints = _joints_by_child(root)
    assert links == {
        "base_link", "rear_axle_link", "front_axle_link", *EXPECTED_SENSOR_PARENTS
    }
    expected_parents = {
        "rear_axle_link": "base_link",
        "front_axle_link": "rear_axle_link",
        **EXPECTED_SENSOR_PARENTS,
    }
    assert set(joints) == set(expected_parents)
    assert {
        child: joint.find("parent").attrib["link"]
        for child, joint in joints.items()
    } == expected_parents
    assert all(joint.attrib["type"] == "fixed" for joint in joints.values())
    assert "map" not in links and "odom" not in links

    rear_xyz, rear_rpy = _origin_values(joints["rear_axle_link"])
    front_xyz, front_rpy = _origin_values(joints["front_axle_link"])
    assert rear_xyz == pytest.approx((0.0, 0.0, 0.3685), abs=1e-12)
    assert rear_rpy == pytest.approx((0.0, 0.0, 0.0), abs=1e-12)
    assert front_xyz == pytest.approx((3.0, 0.0, 0.0), abs=1e-12)
    assert front_rpy == pytest.approx((0.0, 0.0, 0.0), abs=1e-12)

    configured = _load_yaml(SENSOR_CONFIG)["profiles"][profile]["sensors"]
    for sensor, frame in (
        ("gps", "gps_link"), ("imu", "imu_link"), ("lidar", "lidar_link")
    ):
        xyz, rpy = _origin_values(joints[frame])
        expected_xyz, expected_rpy = _configured_origin(configured[sensor])
        assert xyz == pytest.approx(expected_xyz, abs=1e-12)
        assert rpy == pytest.approx(expected_rpy, abs=1e-12)

    expected_optical_rpy = (-math.pi / 2.0, 0.0, -math.pi / 2.0)
    for name in ("front", "left", "right", "traffic_light"):
        xyz, rpy = _origin_values(joints[f"camera_{name}_optical_frame"])
        assert xyz == (0.0, 0.0, 0.0)
        assert rpy == pytest.approx(expected_optical_rpy, abs=1e-12)


def test_xacro_uses_the_installed_textured_dae_without_machine_paths():
    root = _render()
    mesh = root.find("./link[@name='base_link']/visual/geometry/mesh")
    assert mesh is not None
    assert mesh.attrib["filename"] == (
        "package://ad_description/meshes/ioniq5/hyundai_ioniq5.dae"
    )
    assert tuple(float(value) for value in mesh.attrib["scale"].split()) \
        == pytest.approx((0.9925888599244829,) * 3)

    dae = MESH_DIR / "hyundai_ioniq5.dae"
    assert dae.stat().st_size > 500_000
    dae_root = ET.parse(dae).getroot()
    namespace = {"c": "http://www.collada.org/2005/11/COLLADASchema"}
    texture_names = {
        element.text for element in dae_root.findall(".//c:image/c:init_from", namespace)
    }
    assert len(texture_names) == 6
    assert all((MESH_DIR / name).is_file() for name in texture_names)
    assert (MESH_DIR / "ATTRIBUTION.md").is_file()
    assert "CC BY 4.0" in (MESH_DIR / "ATTRIBUTION.md").read_text(encoding="utf-8")
    manifest = ET.parse(PACKAGE_ROOT / "package.xml").getroot()
    assert {license_tag.text for license_tag in manifest.findall("license")} == {
        "Apache-2.0",
        "CC-BY-4.0",
    }

    for element in root.iter():
        for value in element.attrib.values():
            assert not value.startswith("/home/")
            assert not value.startswith("file://")


def test_visual_transform_places_assimp_scene_above_ground_at_rear_axle_origin():
    root = _render()
    visual = root.find("./link[@name='base_link']/visual")
    assert visual is not None
    origin = visual.find("origin")
    mesh = visual.find("geometry/mesh")
    assert origin is not None and mesh is not None
    translation = tuple(float(value) for value in origin.attrib["xyz"].split())
    rpy = tuple(float(value) for value in origin.attrib["rpy"].split())
    scale = tuple(float(value) for value in mesh.attrib["scale"].split())

    dae_root = ET.parse(MESH_DIR / "hyundai_ioniq5.dae").getroot()
    assert dae_root.findtext(
        ".//{http://www.collada.org/2005/11/COLLADASchema}up_axis"
    ) == "Y_UP"
    transformed = []
    for point in _dae_scene_points(dae_root):
        scaled = tuple(point[index] * scale[index] for index in range(3))
        rotated = _rotate_rpy(scaled, rpy)
        transformed.append(
            tuple(rotated[index] + translation[index] for index in range(3))
        )

    bounds = tuple(
        (min(point[axis] for point in transformed), max(point[axis] for point in transformed))
        for axis in range(3)
    )
    assert bounds[0] == pytest.approx((-0.7592, 3.8636), abs=0.002)
    assert bounds[2][0] == pytest.approx(0.0, abs=2e-6)
    assert bounds[2][1] == pytest.approx(1.6157, abs=0.002)


def test_launch_installs_and_processes_xacro_with_selectable_sensor_config():
    cmake_text = (PACKAGE_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "install(DIRECTORY config launch meshes urdf" in cmake_text
    package_text = (PACKAGE_ROOT / "package.xml").read_text(encoding="utf-8")
    assert "<exec_depend>xacro</exec_depend>" in package_text

    source = LAUNCH_PATH.read_text(encoding="utf-8")
    assert "xacro.process_file" in source
    assert 'DeclareLaunchArgument("sensor_config"' in source
    assert 'DeclareLaunchArgument("sensor_profile"' in source
    assert "robot_state_publisher" in source
    assert "static_transform_publisher" not in source

    spec = importlib.util.spec_from_file_location("ad_description_launch", LAUNCH_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.get_package_share_directory = lambda _package: str(PACKAGE_ROOT)
    description = module.generate_launch_description()

    from launch import LaunchContext
    from launch.actions import DeclareLaunchArgument, OpaqueFunction
    from launch_ros.actions import Node

    arguments = {
        entity.name
        for entity in description.entities
        if isinstance(entity, DeclareLaunchArgument)
    }
    assert arguments == {"sensor_config", "sensor_profile"}
    opaque = next(
        entity
        for entity in description.entities
        if isinstance(entity, OpaqueFunction)
    )
    context = LaunchContext()
    context.launch_configurations.update(
        {
            "sensor_config": str(SENSOR_CONFIG),
            "sensor_profile": "",
        }
    )
    nodes = opaque.execute(context)
    assert len(nodes) == 1
    assert isinstance(nodes[0], Node)
    assert nodes[0]._Node__node_namespace is None
