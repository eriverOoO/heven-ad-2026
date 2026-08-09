from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parent
LAUNCH_PATH = PACKAGE_ROOT / "launch" / "perception_validation.launch.py"


def _load_module():
    spec = spec_from_file_location("perception_validation_launch", LAUNCH_PATH)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_morai_classical_launch_has_exact_fixed_pipeline_switches():
    module = _load_module()
    package_share = REPOSITORY_ROOT / "ad_lidar_perception"
    description_share = REPOSITORY_ROOT / "ad_description"

    assert module._lidar_arguments(package_share, description_share) == {
        "platform_profile": "morai",
        "composition_config": str(
            package_share
            / "config"
            / "lidar_perception_morai_classical.yaml"
        ),
        "start_ground_segmentation": "true",
        "deskew_enabled": "false",
        "deskew_mode": "3d",
        "self_crop_enabled": "true",
        "patchwork_leveling_enabled": "false",
        "finite_filter_enabled": "false",
        "densifier_enabled": "false",
        "point_layout_adapter_enabled": "false",
        "ground_config": str(
            package_share
            / "config"
            / "preprocessing"
            / "ground_segmentation.yaml"
        ),
        "sensor_config": str(
            description_share / "config" / "sensor_mounts.yaml"
        ),
        "sensor_profile": "",
    }


def test_validation_launch_includes_only_lidar_perception(monkeypatch):
    module = _load_module()
    package_shares = {
        "ad_lidar_perception": REPOSITORY_ROOT / "ad_lidar_perception",
        "ad_description": REPOSITORY_ROOT / "ad_description",
    }
    monkeypatch.setattr(
        module,
        "get_package_share_directory",
        lambda package: str(package_shares[package]),
    )

    description = module.generate_launch_description()

    assert len(description.entities) == 1
    include = description.entities[0]
    arguments = dict(include.launch_arguments)
    assert arguments == module._lidar_arguments(
        package_shares["ad_lidar_perception"],
        package_shares["ad_description"],
    )
