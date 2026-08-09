from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

from launch import LaunchContext
from launch.actions import DeclareLaunchArgument, OpaqueFunction
import yaml


PACKAGE = Path(__file__).resolve().parents[1]
LAUNCH = PACKAGE / "launch" / "object_detection.launch.py"

BACKENDS = {
    "centerpoint_tiny": {
        "package": "autoware_lidar_centerpoint",
        "launch": "lidar_centerpoint.launch.xml",
        "model_dir": "lidar_centerpoint",
        "ml": "centerpoint_tiny_ml_package.param.yaml",
        "remapper": "detection_class_remapper.param.yaml",
    },
    "centerpoint": {
        "package": "autoware_lidar_centerpoint",
        "launch": "lidar_centerpoint.launch.xml",
        "model_dir": "lidar_centerpoint",
        "ml": "centerpoint_ml_package.param.yaml",
        "remapper": "detection_class_remapper.param.yaml",
    },
    "transfusion": {
        "package": "autoware_lidar_transfusion",
        "launch": "lidar_transfusion.launch.xml",
        "model_dir": "lidar_transfusion",
        "ml": "transfusion_ml_package.param.yaml",
        "remapper": "detection_class_remapper.param.yaml",
    },
    "bevfusion_lidar": {
        "package": "autoware_bevfusion",
        "launch": "bevfusion.launch.xml",
        "model_dir": "bevfusion",
        "ml": "ml_package_bevfusion_lidar.param.yaml",
        "remapper": "detection_class_remapper.param.yaml",
    },
}


def load_launch_module():
    spec = spec_from_file_location("object_detection_launch", LAUNCH)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def runtime(backend: str, tmp_path: Path, build_only: bool = False):
    contract = BACKENDS[backend]
    model = tmp_path / "data" / contract["model_dir"]
    return SimpleNamespace(
        backend=backend,
        package=contract["package"],
        executable=f"{contract['package']}_node",
        launch_path=tmp_path / "install" / contract["launch"],
        model_path=model,
        ml_package_path=model / contract["ml"],
        class_remapper_path=model / contract["remapper"],
        build_only=build_only,
    )


def ros_parameters(name: str):
    document = yaml.safe_load(
        (PACKAGE / "config" / "detectors" / f"{name}.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert set(document) == {"/**"}
    assert set(document["/**"]) == {"ros__parameters"}
    return document["/**"]["ros__parameters"]


def test_complete_centerpoint_overlays_keep_upstream_schema_and_odom():
    expected = {
        "encoder_onnx_path":
            "$(var model_path)/pts_voxel_encoder_$(var model_name).onnx",
        "encoder_engine_path":
            "$(var model_path)/pts_voxel_encoder_$(var model_name).engine",
        "head_onnx_path":
            "$(var model_path)/pts_backbone_neck_head_$(var model_name).onnx",
        "head_engine_path":
            "$(var model_path)/pts_backbone_neck_head_"
            "$(var model_name).engine",
        "trt_precision": "fp16",
        "cloud_capacity": 2000000,
        "post_process_params": {
            "circle_nms_dist_threshold": 0.5,
            "iou_nms_search_distance_2d": 10.0,
            "iou_nms_threshold": 0.1,
            "yaw_norm_thresholds": [0.3, 0.3, 0.3, 0.3, 0.0],
        },
        "densification_params": {
            "world_frame_id": "odom",
            "num_past_frames": 1,
        },
    }
    assert ros_parameters("centerpoint_tiny") == expected
    assert ros_parameters("centerpoint") == expected
    assert "has_twist" not in expected


def test_complete_transfusion_overlay_keeps_upstream_schema_and_odom():
    assert ros_parameters("transfusion") == {
        "trt_precision": "fp16",
        "cloud_capacity": 2000000,
        "onnx_path": "$(var model_path)/transfusion.onnx",
        "engine_path": "$(var model_path)/transfusion.engine",
        "densification_num_past_frames": 1,
        "densification_world_frame_id": "odom",
        "circle_nms_dist_threshold": 0.5,
        "iou_nms_search_distance_2d": 10.0,
        "iou_nms_threshold": 0.1,
        "yaw_norm_thresholds": [0.3, 0.3, 0.3, 0.3, 0.0],
        "score_threshold": 0.1,
    }


def test_complete_bevfusion_lidar_overlay_keeps_upstream_schema_and_odom():
    assert ros_parameters("bevfusion_lidar") == {
        "max_camera_lidar_delay": 0.0,
        "plugins_path":
            "$(find-pkg-share autoware_tensorrt_plugins)/plugins/"
            "libautoware_tensorrt_plugins.so",
        "trt_precision": "fp16",
        "cloud_capacity": 2000000,
        "onnx_path": "$(var model_path)/bevfusion_lidar.onnx",
        "engine_path": "$(var model_path)/bevfusion_lidar.engine",
        "image_backbone_onnx_path": "",
        "image_backbone_engine_path": "",
        "image_backbone_trt_precision": "",
        "densification_num_past_frames": 1,
        "densification_world_frame_id": "odom",
        "run_image_undistortion": False,
        "circle_nms_dist_threshold": 0.5,
        "iou_nms_search_distance_2d": 10.0,
        "iou_nms_threshold": 0.1,
        "yaw_norm_thresholds": [0.3, 0.3, 0.3, 0.3, 0.0],
        "score_threshold": 0.1,
    }


def test_exact_official_include_arguments_for_every_backend(tmp_path):
    module = load_launch_module()
    package_share = tmp_path / "share" / "ad_lidar_perception"
    for backend, contract in BACKENDS.items():
        verified = runtime(backend, tmp_path)
        arguments = module._detector_arguments(verified, package_share)
        assert arguments == {
            "input/pointcloud": "/ad/perception/lidar/points_xyzirc",
            "output/objects": "/ad/perception/objects/detected",
            "model_name": backend,
            "model_path": str(
                tmp_path / "data" / contract["model_dir"]
            ),
            "model_param_path": str(
                package_share / "config" / "detectors" / f"{backend}.yaml"
            ),
            "ml_package_param_path": str(
                tmp_path
                / "data"
                / contract["model_dir"]
                / contract["ml"]
            ),
            "class_remapper_param_path": str(
                tmp_path
                / "data"
                / contract["model_dir"]
                / contract["remapper"]
            ),
            "build_only": "false",
        }
        assert set(arguments) == {
            "input/pointcloud",
            "output/objects",
            "model_name",
            "model_path",
            "model_param_path",
            "ml_package_param_path",
            "class_remapper_param_path",
            "build_only",
        }

    assert module._detector_arguments(
        runtime("centerpoint_tiny", tmp_path, build_only=True), package_share
    )["build_only"] == "true"


def test_launch_is_one_preflight_then_pinned_xml_include(
    tmp_path, monkeypatch
):
    module = load_launch_module()
    verified_detector = runtime("centerpoint_tiny", tmp_path)
    selection = SimpleNamespace(
        detector=SimpleNamespace(backend="centerpoint_tiny")
    )
    calls = []

    monkeypatch.setattr(module, "load_selection", lambda path: selection)
    monkeypatch.setattr(
        module,
        "verify_selection",
        lambda selected, **kwargs: (
            calls.append((selected, kwargs))
            or SimpleNamespace(detector=verified_detector)
        ),
    )
    monkeypatch.setattr(
        module,
        "get_package_share_directory",
        lambda name: str(tmp_path / "share" / name),
    )
    monkeypatch.setattr(
        module,
        "AnyLaunchDescriptionSource",
        lambda location: SimpleNamespace(location=location),
    )
    monkeypatch.setattr(
        module,
        "IncludeLaunchDescription",
        lambda source, **kwargs: SimpleNamespace(
            launch_description_source=source,
            launch_arguments=list(kwargs["launch_arguments"]),
        ),
    )

    context = LaunchContext()
    context.launch_configurations.update(
        {
            "selection_config": str(tmp_path / "selection.yaml"),
            "data_root": str(tmp_path / "data-root"),
        }
    )
    actions = module._launch_setup(context)
    assert len(calls) == 1
    assert calls[0][1] == {
        "lock_path": (
            tmp_path
            / "share"
            / "ad_lidar_perception"
            / "config"
            / "autoware_perception.lock.yaml"
        ),
        "data_root": tmp_path / "data-root",
    }
    assert len(actions) == 1
    include = actions[0]
    assert include.launch_description_source.location == str(
        verified_detector.launch_path
    )
    assert dict(include.launch_arguments)["input/pointcloud"] == (
        "/ad/perception/lidar/points_xyzirc"
    )


def test_launch_interface_uses_installed_config_and_opaque_setup():
    module = load_launch_module()
    description = module.generate_launch_description()
    arguments = {
        entity.name
        for entity in description.entities
        if isinstance(entity, DeclareLaunchArgument)
    }
    assert arguments == {"selection_config", "data_root"}
    assert sum(
        isinstance(entity, OpaqueFunction)
        for entity in description.entities
    ) == 1
