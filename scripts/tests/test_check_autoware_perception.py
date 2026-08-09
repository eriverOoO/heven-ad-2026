import hashlib
import os
from pathlib import Path
import stat
from contextlib import contextmanager

import pytest
import yaml

import ad_lidar_perception.autoware_provenance as provenance
from ad_lidar_perception.autoware_provenance import (
    LockError,
    VerificationError,
    load_lock,
    main,
    resolve_data_root,
    verify_selection,
)
from ad_lidar_perception.selection import load_selection


REPO = Path(__file__).resolve().parents[2]
OFFICIAL_LOCK = (
    REPO / "ad_lidar_perception" / "config" / "autoware_perception.lock.yaml"
)

BACKENDS = {
    "centerpoint_tiny": {
        "package": "autoware_lidar_centerpoint",
        "executable": "autoware_lidar_centerpoint_node",
        "launch": "launch/lidar_centerpoint.launch.xml",
        "installed_files": ["config/centerpoint_common.param.yaml"],
        "model_dir": "lidar_centerpoint",
        "model_name": "centerpoint_tiny",
        "ml_package": "centerpoint_tiny_ml_package.param.yaml",
        "class_remapper": "detection_class_remapper.param.yaml",
        "artifacts": {
            "encoder_onnx": "pts_voxel_encoder_centerpoint_tiny.onnx",
            "head_onnx": "pts_backbone_neck_head_centerpoint_tiny.onnx",
            "ml_package": "centerpoint_tiny_ml_package.param.yaml",
            "class_remapper": "detection_class_remapper.param.yaml",
        },
        "engines": [
            "pts_voxel_encoder_centerpoint_tiny.engine",
            "pts_backbone_neck_head_centerpoint_tiny.engine",
        ],
    },
    "centerpoint": {
        "package": "autoware_lidar_centerpoint",
        "executable": "autoware_lidar_centerpoint_node",
        "launch": "launch/lidar_centerpoint.launch.xml",
        "installed_files": ["config/centerpoint_common.param.yaml"],
        "model_dir": "lidar_centerpoint",
        "model_name": "centerpoint",
        "ml_package": "centerpoint_ml_package.param.yaml",
        "class_remapper": "detection_class_remapper.param.yaml",
        "artifacts": {
            "encoder_onnx": "pts_voxel_encoder_centerpoint.onnx",
            "head_onnx": "pts_backbone_neck_head_centerpoint.onnx",
            "ml_package": "centerpoint_ml_package.param.yaml",
            "class_remapper": "detection_class_remapper.param.yaml",
        },
        "engines": [
            "pts_voxel_encoder_centerpoint.engine",
            "pts_backbone_neck_head_centerpoint.engine",
        ],
    },
    "transfusion": {
        "package": "autoware_lidar_transfusion",
        "executable": "autoware_lidar_transfusion_node",
        "launch": "launch/lidar_transfusion.launch.xml",
        "installed_files": ["config/transfusion_common.param.yaml"],
        "model_dir": "lidar_transfusion",
        "model_name": "transfusion",
        "ml_package": "transfusion_ml_package.param.yaml",
        "class_remapper": "detection_class_remapper.param.yaml",
        "artifacts": {
            "onnx": "transfusion.onnx",
            "ml_package": "transfusion_ml_package.param.yaml",
            "class_remapper": "detection_class_remapper.param.yaml",
        },
        "engines": ["transfusion.engine"],
    },
    "bevfusion_lidar": {
        "package": "autoware_bevfusion",
        "executable": "autoware_bevfusion_node",
        "launch": "launch/bevfusion.launch.xml",
        "installed_files": ["config/common_bevfusion.param.yaml"],
        "model_dir": "bevfusion",
        "model_name": "bevfusion_lidar",
        "ml_package": "ml_package_bevfusion_lidar.param.yaml",
        "class_remapper": "detection_class_remapper.param.yaml",
        "artifacts": {
            "onnx": "bevfusion_lidar.onnx",
            "ml_package": "ml_package_bevfusion_lidar.param.yaml",
            "class_remapper": "detection_class_remapper.param.yaml",
        },
        "engines": ["bevfusion_lidar.engine"],
    },
}

OFFICIAL_HASHES = {
    ("centerpoint_tiny", "encoder_onnx"):
        "2c53465715c1fd2e9dc5727ef3fca74f4cdf0538f74286b0946e219d0ca5693b",
    ("centerpoint_tiny", "head_onnx"):
        "9bb0b634f3664bd098ce7d6a3d8a9fb7cc8d9b8252b27f302c71e43316bab551",
    ("centerpoint_tiny", "ml_package"):
        "e0846b08fbd023d6a7c085f5389d0ccaef8ad9cd2c5e1eb7dbc6583618e38424",
    ("centerpoint_tiny", "class_remapper"):
        "c711f8875ece9b527dfe31ffc75f8c0de2e77945ef67860a959a4e04c36772d5",
    ("centerpoint", "encoder_onnx"):
        "dc1a876580d86ee7a341d543f8ade2ede7f43bd032dc5b44155b1f0175405764",
    ("centerpoint", "head_onnx"):
        "3fe7e128955646740c41a25be0c8f141d5a94594fe79d7405fe2a859e391542e",
    ("centerpoint", "ml_package"):
        "9bbc16e521dd87c91cbadf1cb89c8b81393d1f8e1069af385aaba677576f0e27",
    ("centerpoint", "class_remapper"):
        "c711f8875ece9b527dfe31ffc75f8c0de2e77945ef67860a959a4e04c36772d5",
    ("transfusion", "onnx"):
        "1d8f0ee6d59ccc3cca914f9892f6ac8f0a9e35082abb91da183c00e3e2c2718a",
    ("transfusion", "ml_package"):
        "476f7727adc17a823962f2e09ba23d40f3116c50be48361d98179d054cd131b6",
    ("transfusion", "class_remapper"):
        "c711f8875ece9b527dfe31ffc75f8c0de2e77945ef67860a959a4e04c36772d5",
    ("bevfusion_lidar", "onnx"):
        "5c29087963bf2c4dc02bf45c29d459303be602d63f9b6adff22a75c9cfb459a6",
    ("bevfusion_lidar", "ml_package"):
        "866265b9f0fc8c17805c0974339d3c7c4e601c1aa212818971fc650d71782181",
    ("bevfusion_lidar", "class_remapper"):
        "928f9eb14ac042d725909f12b2be1532c16b09a683485c5936cf04fb04728520",
}

UPSTREAM_LEAF_HASHES = {
    ("centerpoint_tiny", "launch"):
        "be57b86ff6f9aa4a70ee77f469f3f9b9e1533f098b186cab3ebd455e66e93797",
    ("centerpoint_tiny", "config/centerpoint_common.param.yaml"):
        "3a8585a54ec786464a4b601cc6b5ccbf5db51301177d3349368dc63094ea5277",
    ("centerpoint", "launch"):
        "be57b86ff6f9aa4a70ee77f469f3f9b9e1533f098b186cab3ebd455e66e93797",
    ("centerpoint", "config/centerpoint_common.param.yaml"):
        "3a8585a54ec786464a4b601cc6b5ccbf5db51301177d3349368dc63094ea5277",
    ("transfusion", "launch"):
        "e7653ea2410491b1121c06122d88534ef945d24f27f9b65b97a9e143b5f72751",
    ("transfusion", "config/transfusion_common.param.yaml"):
        "7f6a6d74e7ac15b3ce09c4b1428fd5e0b4c2dc7e3c50691b2b157c9b8a6d70d8",
    ("bevfusion_lidar", "launch"):
        "33e87f8d6adfaca7b3a30da0de7f6ccc7d4df656301167ab885c993ce95d3b60",
    ("bevfusion_lidar", "config/common_bevfusion.param.yaml"):
        "3a8585a54ec786464a4b601cc6b5ccbf5db51301177d3349368dc63094ea5277",
    ("tracker", "launch"):
        "7f0da73e7341aca828d8e2c2bec5ab9c5f823060b8ecbce18dbcdd686b9a50d4",
    ("tracker", "config/data_association_matrix.param.yaml"):
        "fdc0bbb7654fe08f0ab246f7e356de2e93c8205742f2fbad95b65fa8c7af2d65",
    ("tracker", "config/input_channels.param.yaml"):
        "ecc470f23ba8d42910d0a2bd483c013ec9dfda150e1ca3d76575a1acc2af683a",
}


def regular(path: Path, content: bytes = b"fixture\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def replace_with_install_symlink(path: Path, target: Path) -> None:
    content = path.read_bytes()
    mode = stat.S_IMODE(path.stat().st_mode)
    regular(target, content)
    target.chmod(mode)
    path.unlink()
    path.symlink_to(target)


def package(prefix: Path, name: str, version: str, executable: str, leaves):
    regular(
        prefix / "share" / name / "package.xml",
        (
            '<?xml version="1.0"?>'
            f"<package><name>{name}</name>"
            f"<version>{version}</version></package>"
        ).encode(),
    )
    executable_path = prefix / "lib" / name / executable
    regular(executable_path, b"executable\n")
    executable_path.chmod(0o755)
    for leaf in leaves:
        regular(prefix / "share" / name / leaf)


@contextmanager
def trust_synthetic_lock(path: Path):
    """Narrow test-only replacement for the production trust anchor."""
    original = provenance._PINNED_LOCK_SHA256
    provenance._PINNED_LOCK_SHA256 = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    try:
        yield
    finally:
        provenance._PINNED_LOCK_SHA256 = original


def load_synthetic_lock(path: Path):
    with trust_synthetic_lock(path):
        return load_lock(path)


def selection_file(
    tmp_path: Path,
    backend: str = "centerpoint_tiny",
    *,
    tracker: bool = False,
    build_only: bool = False,
) -> Path:
    path = tmp_path / f"selection-{backend}.yaml"
    path.write_text(
        f"""\
schema_version: 1
detector:
  backend: {backend}
  model_subdir: models/autoware
  build_only: {str(build_only).lower()}
tracker:
  backend: {"autoware" if tracker else "none"}
occupancy:
  static_enabled: true
  dynamic_enabled: {str(tracker).lower()}
  publish_combined: true
""",
        encoding="utf-8",
    )
    return path


def synthetic_lock(tmp_path: Path):
    backends = {}
    payloads = {}
    for backend, contract in BACKENDS.items():
        artifacts = {}
        for role, relative in contract["artifacts"].items():
            payload = f"{backend}:{role}\n".encode()
            payloads[(backend, role)] = payload
            artifacts[role] = {
                "path": relative,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        backends[backend] = {
            "package": contract["package"],
            "version": "0.51.0",
            "executable": contract["executable"],
            "launch": {
                "path": contract["launch"],
                "sha256": hashlib.sha256(b"fixture\n").hexdigest(),
            },
            "installed_files": [
                {
                    "path": path,
                    "sha256": hashlib.sha256(b"fixture\n").hexdigest(),
                }
                for path in contract["installed_files"]
            ],
            "model_dir": contract["model_dir"],
            "model_name": contract["model_name"],
            "ml_package": contract["ml_package"],
            "class_remapper": contract["class_remapper"],
            "artifacts": artifacts,
            "engines": contract["engines"],
        }
    document = {
        "schema_version": 1,
        "autoware": {
            "meta_release": "1.8.0",
            "universe": "0.51.0",
            "autoware_msgs": "1.13.0",
            "manifest_url":
                "https://github.com/autowarefoundation/autoware/blob/1.8.0/"
                "repositories/autoware.repos",
            "artifact_role_url":
                "https://github.com/autowarefoundation/autoware/blob/1.8.0/"
                "ansible/roles/artifacts/tasks/main.yaml",
        },
        "weights_policy": {
            "license_status": "unresolved_upstream_artifact_license",
            "redistribution_allowed_by_project": False,
            "required_ack_env": "AD_AUTOWARE_MODEL_LICENSE_REVIEWED",
        },
        "messages": {
            "package": "autoware_perception_msgs",
            "version": "1.13.0",
        },
        "detectors": backends,
        "tracker": {
            "package": "autoware_multi_object_tracker",
            "version": "0.51.0",
            "executable": "multi_object_tracker_node",
            "launch": {
                "path": "launch/multi_object_tracker.launch.xml",
                "sha256": hashlib.sha256(b"fixture\n").hexdigest(),
            },
            "installed_files": [
                {
                    "path": "config/data_association_matrix.param.yaml",
                    "sha256": hashlib.sha256(b"fixture\n").hexdigest(),
                },
                {
                    "path": "config/input_channels.param.yaml",
                    "sha256": hashlib.sha256(b"fixture\n").hexdigest(),
                },
            ],
            "prediction_package": "ad_lidar_perception",
            "prediction_executable": "ad_autoware_prediction_node",
        },
    }
    path = tmp_path / "lock.yaml"
    path.write_text(
        yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
    )
    return path, payloads


def fixture(
    tmp_path: Path,
    backend: str = "centerpoint_tiny",
    *,
    tracker: bool = False,
    build_only: bool = False,
):
    lock_path, payloads = synthetic_lock(tmp_path)
    prefix = tmp_path / "install"
    data_root = tmp_path / "data"
    contract = BACKENDS[backend]
    package(
        prefix,
        contract["package"],
        "0.51.0",
        contract["executable"],
        [contract["launch"], *contract["installed_files"]],
    )
    package(
        prefix,
        "autoware_perception_msgs",
        "1.13.0",
        "unused",
        [],
    )
    model_dir = (
        data_root / "models" / "autoware" / contract["model_dir"]
    )
    for role, relative in contract["artifacts"].items():
        regular(model_dir / relative, payloads[(backend, role)])
    for engine in contract["engines"]:
        regular(model_dir / engine, b"target gpu engine\n")

    if tracker:
        package(
            prefix,
            "autoware_multi_object_tracker",
            "0.51.0",
            "multi_object_tracker_node",
            [
                "launch/multi_object_tracker.launch.xml",
                "config/data_association_matrix.param.yaml",
                "config/input_channels.param.yaml",
            ],
        )
        package(
            prefix,
            "ad_lidar_perception",
            "0.1.0",
            "ad_autoware_prediction_node",
            [],
        )

    selected = load_selection(
        selection_file(
            tmp_path,
            backend,
            tracker=tracker,
            build_only=build_only,
        )
    )
    return selected, lock_path, prefix, data_root, model_dir


def verify(selected, lock_path, prefix, data_root, **kwargs):
    environment = {"AD_AUTOWARE_MODEL_LICENSE_REVIEWED": "1"}
    environment.update(kwargs.pop("environ", {}))
    with trust_synthetic_lock(lock_path):
        return verify_selection(
            selected,
            lock_path=lock_path,
            prefixes=[prefix],
            data_root=data_root,
            environ=environment,
            **kwargs,
        )


def test_official_lock_contains_exact_release_contract_and_hashes():
    lock = load_lock(OFFICIAL_LOCK)
    assert lock["autoware"] == {
        "meta_release": "1.8.0",
        "universe": "0.51.0",
        "autoware_msgs": "1.13.0",
        "manifest_url":
            "https://github.com/autowarefoundation/autoware/blob/1.8.0/"
            "repositories/autoware.repos",
        "artifact_role_url":
            "https://github.com/autowarefoundation/autoware/blob/1.8.0/"
            "ansible/roles/artifacts/tasks/main.yaml",
    }
    assert lock["weights_policy"] == {
        "license_status": "unresolved_upstream_artifact_license",
        "redistribution_allowed_by_project": False,
        "required_ack_env": "AD_AUTOWARE_MODEL_LICENSE_REVIEWED",
    }
    for backend, expected in BACKENDS.items():
        actual = lock["detectors"][backend]
        for key in (
            "package",
            "executable",
            "model_dir",
            "model_name",
            "ml_package",
            "class_remapper",
            "engines",
        ):
            assert actual[key] == expected[key]
        assert actual["launch"] == {
            "path": expected["launch"],
            "sha256": UPSTREAM_LEAF_HASHES[(backend, "launch")],
        }
        assert actual["installed_files"] == [
            {
                "path": path,
                "sha256": UPSTREAM_LEAF_HASHES[(backend, path)],
            }
            for path in expected["installed_files"]
        ]
        assert actual["version"] == "0.51.0"
        for role, relative in expected["artifacts"].items():
            assert actual["artifacts"][role] == {
                "path": relative,
                "sha256": OFFICIAL_HASHES[(backend, role)],
            }
    assert lock["tracker"]["executable"] == "multi_object_tracker_node"
    assert lock["tracker"]["launch"] == {
        "path": "launch/multi_object_tracker.launch.xml",
        "sha256": UPSTREAM_LEAF_HASHES[("tracker", "launch")],
    }
    assert lock["tracker"]["installed_files"] == [
        {
            "path": path,
            "sha256": UPSTREAM_LEAF_HASHES[("tracker", path)],
        }
        for path in (
            "config/data_association_matrix.param.yaml",
            "config/input_channels.param.yaml",
        )
    ]


@pytest.mark.parametrize(
    "path,value",
    [
        (("detectors", "centerpoint_tiny", "package"), "other_package"),
        (("detectors", "centerpoint_tiny", "executable"), "other_node"),
        (("detectors", "centerpoint_tiny", "model_dir"), "other_models"),
        (
            ("detectors", "centerpoint_tiny", "engines"),
            ["other.engine"],
        ),
        (
            (
                "detectors",
                "centerpoint_tiny",
                "artifacts",
                "encoder_onnx",
                "sha256",
            ),
            "0" * 64,
        ),
        (("tracker", "executable"), "other_tracker_node"),
    ],
)
def test_caller_selected_lock_cannot_change_pinned_contract(
    tmp_path, path, value
):
    document = yaml.safe_load(OFFICIAL_LOCK.read_text(encoding="utf-8"))
    target = document
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    candidate = tmp_path / "caller-selected.lock.yaml"
    candidate.write_text(
        yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
    )
    with pytest.raises(LockError, match="trusted checked-in lock"):
        load_lock(candidate)


def test_lock_hash_and_parse_use_the_same_single_read_payload(
    tmp_path, monkeypatch
):
    trusted = OFFICIAL_LOCK.read_bytes()
    modified = trusted.replace(
        b"executable: autoware_lidar_centerpoint_node",
        b"executable: substituted_node",
        1,
    )
    assert modified != trusted
    candidate = tmp_path / "swapped-after-trusted-read.lock.yaml"
    candidate.write_bytes(trusted)

    original_read_bytes = Path.read_bytes
    original_read_text = Path.read_text
    reads = {"bytes": 0, "text": 0}

    def swapping_read_bytes(path):
        payload = original_read_bytes(path)
        if path == candidate:
            reads["bytes"] += 1
            candidate.write_bytes(modified)
        return payload

    def counted_read_text(path, *args, **kwargs):
        if path == candidate:
            reads["text"] += 1
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", swapping_read_bytes)
    monkeypatch.setattr(Path, "read_text", counted_read_text)

    lock = load_lock(candidate)

    assert reads == {"bytes": 1, "text": 0}
    assert (
        lock["detectors"]["centerpoint_tiny"]["executable"]
        == "autoware_lidar_centerpoint_node"
    )
    assert b"executable: substituted_node" in original_read_bytes(candidate)


def test_lock_rejects_root_and_nested_duplicate_keys(tmp_path):
    lock_path, _ = synthetic_lock(tmp_path)
    text = lock_path.read_text(encoding="utf-8")
    lock_path.write_text(text + "\nschema_version: 1\n", encoding="utf-8")
    with pytest.raises(LockError, match="duplicate key"):
        load_synthetic_lock(lock_path)

    lock_path, _ = synthetic_lock(tmp_path)
    text = lock_path.read_text(encoding="utf-8")
    text = text.replace(
        "      encoder_onnx:\n"
        "        path: pts_voxel_encoder_centerpoint_tiny.onnx\n",
        "      encoder_onnx:\n"
        "        path: pts_voxel_encoder_centerpoint_tiny.onnx\n"
        "        path: duplicate.onnx\n",
        1,
    )
    lock_path.write_text(text, encoding="utf-8")
    with pytest.raises(LockError, match="duplicate key"):
        load_synthetic_lock(lock_path)


def test_lock_rejects_unknown_missing_and_wrong_exact_types(tmp_path):
    lock_path, _ = synthetic_lock(tmp_path)
    document = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    document["unknown"] = True
    lock_path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(LockError, match="root has unknown keys: unknown"):
        load_synthetic_lock(lock_path)

    lock_path, _ = synthetic_lock(tmp_path)
    document = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    del document["tracker"]["executable"]
    lock_path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(
        LockError, match="tracker is missing keys: executable"
    ):
        load_synthetic_lock(lock_path)

    lock_path, _ = synthetic_lock(tmp_path)
    document = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    document["detectors"]["centerpoint_tiny"]["unexpected"] = "value"
    lock_path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(
        LockError,
        match="detectors.centerpoint_tiny has unknown keys: unexpected",
    ):
        load_synthetic_lock(lock_path)

    for field, value, diagnostic in (
        (("schema_version",), True, "schema_version must be an integer"),
        (
            ("autoware", "meta_release"),
            "1.9.0",
            "autoware.meta_release must equal 1.8.0",
        ),
        (
            ("weights_policy", "redistribution_allowed_by_project"),
            0,
            "redistribution_allowed_by_project must be a boolean",
        ),
        (
            ("detectors", "centerpoint_tiny", "installed_files"),
            "config/file.yaml",
            "installed_files must be a list",
        ),
    ):
        lock_path, _ = synthetic_lock(tmp_path)
        document = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
        target = document
        for key in field[:-1]:
            target = target[key]
        target[field[-1]] = value
        lock_path.write_text(yaml.safe_dump(document), encoding="utf-8")
        with pytest.raises(LockError, match=diagnostic):
            load_synthetic_lock(lock_path)


def test_lock_rejects_bad_hash_and_every_unsafe_relative_path_role(tmp_path):
    lock_path, _ = synthetic_lock(tmp_path)
    document = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    document["detectors"]["centerpoint_tiny"]["artifacts"]["encoder_onnx"][
        "sha256"
    ] = "abc123"
    lock_path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(LockError, match="64 lowercase hexadecimal"):
        load_synthetic_lock(lock_path)

    path_fields = (
        ("executable", "../node"),
        ("launch", {"path": "/absolute.launch.xml", "sha256": "0" * 64}),
        (
            "installed_files",
            [{"path": "config/../escape.yaml", "sha256": "0" * 64}],
        ),
        ("model_dir", ".."),
        ("ml_package", "nested\\escape.yaml"),
        ("class_remapper", "C:escape.yaml"),
        ("engines", ["../escape.engine"]),
    )
    for field, unsafe in path_fields:
        lock_path, _ = synthetic_lock(tmp_path)
        document = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
        document["detectors"]["centerpoint_tiny"][field] = unsafe
        lock_path.write_text(yaml.safe_dump(document), encoding="utf-8")
        with pytest.raises(LockError, match=field):
            load_synthetic_lock(lock_path)

    lock_path, _ = synthetic_lock(tmp_path)
    document = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    document["detectors"]["centerpoint_tiny"]["artifacts"]["encoder_onnx"][
        "path"
    ] = "../escape.onnx"
    lock_path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(LockError, match="artifacts.encoder_onnx.path"):
        load_synthetic_lock(lock_path)

    tracker_path_fields = (
        ("executable", "../tracker_node"),
        (
            "launch",
            {"path": "/tracker.launch.xml", "sha256": "0" * 64},
        ),
        (
            "installed_files",
            [{"path": "../input_channels.yaml", "sha256": "0" * 64}],
        ),
        ("prediction_executable", "nested/predictor"),
    )
    for field, unsafe in tracker_path_fields:
        lock_path, _ = synthetic_lock(tmp_path)
        document = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
        document["tracker"][field] = unsafe
        lock_path.write_text(yaml.safe_dump(document), encoding="utf-8")
        with pytest.raises(LockError, match=f"tracker.{field}"):
            load_synthetic_lock(lock_path)


@pytest.mark.parametrize(
    "field,mutation,diagnostic",
    [
        (
            "launch",
            lambda record: record.update({"unexpected": True}),
            "launch has unknown keys: unexpected",
        ),
        (
            "launch",
            lambda record: record.__setitem__("sha256", "A" * 64),
            "launch.sha256 must be 64 lowercase hexadecimal",
        ),
        (
            "installed_files",
            lambda record: record.__setitem__("sha256", 123),
            "installed_files\\[0\\].sha256 must be a string",
        ),
    ],
)
def test_lock_rejects_unsafe_upstream_leaf_records(
    tmp_path, field, mutation, diagnostic
):
    lock_path, _ = synthetic_lock(tmp_path)
    document = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    value = document["detectors"]["centerpoint_tiny"][field]
    record = value if field == "launch" else value[0]
    mutation(record)
    lock_path.write_text(
        yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
    )
    with pytest.raises(LockError, match=diagnostic):
        load_synthetic_lock(lock_path)


def test_none_none_returns_without_package_or_license_probe(
    tmp_path, monkeypatch
):
    selected = load_selection(
        selection_file(tmp_path, backend="none", tracker=False)
    )
    monkeypatch.setattr(
        "ad_lidar_perception.autoware_provenance._find_package_prefix",
        lambda *_args, **_kwargs: pytest.fail("package probe was called"),
    )
    result = verify_selection(
        selected,
        lock_path=tmp_path / "not-needed.yaml",
        prefixes=[tmp_path / "missing"],
        data_root=tmp_path / "missing-data",
        environ={},
    )
    assert result.detector is None
    assert result.tracker is None


def test_resolves_data_root_only_from_explicit_or_environment(tmp_path):
    explicit = tmp_path / "explicit"
    environment = tmp_path / "environment"
    assert resolve_data_root(explicit, environ={}) == explicit.resolve()
    assert resolve_data_root(
        None, environ={"AD_DATA_DIR": str(environment)}
    ) == environment.resolve()
    with pytest.raises(ValueError, match="AD_DATA_DIR"):
        resolve_data_root(None, environ={})


@pytest.mark.parametrize("backend", BACKENDS)
def test_accepts_every_exact_backend_mapping(tmp_path, backend):
    selected, lock_path, prefix, data_root, model_dir = fixture(
        tmp_path, backend
    )
    result = verify(selected, lock_path, prefix, data_root)
    assert result.detector is not None
    assert result.detector.backend == backend
    assert result.detector.package == BACKENDS[backend]["package"]
    assert result.detector.executable == BACKENDS[backend]["executable"]
    assert result.detector.model_path == model_dir


@pytest.mark.parametrize(
    "backend,role",
    [
        (backend, role)
        for backend, contract in BACKENDS.items()
        for role in contract["artifacts"]
    ],
)
def test_rejects_one_byte_corruption_for_every_artifact_role(
    tmp_path, backend, role
):
    selected, lock_path, prefix, data_root, model_dir = fixture(
        tmp_path, backend
    )
    artifact = model_dir / BACKENDS[backend]["artifacts"][role]
    artifact.write_bytes(artifact.read_bytes() + b"x")
    with pytest.raises(VerificationError) as error:
        verify(selected, lock_path, prefix, data_root)
    assert f"{backend} artifact {role} SHA-256 mismatch" in str(error.value)


@pytest.mark.parametrize("backend", BACKENDS)
def test_engine_gate_distinguishes_build_only_from_runtime(tmp_path, backend):
    selected, lock_path, prefix, data_root, model_dir = fixture(
        tmp_path, backend
    )
    for engine in BACKENDS[backend]["engines"]:
        (model_dir / engine).unlink()
    with pytest.raises(VerificationError, match="engine"):
        verify(selected, lock_path, prefix, data_root)

    selected = load_selection(
        selection_file(tmp_path, backend, build_only=True)
    )
    result = verify(selected, lock_path, prefix, data_root)
    assert result.detector is not None
    assert result.detector.build_only is True


@pytest.mark.parametrize("kind", ["empty", "fifo", "symlink"])
def test_build_only_rejects_invalid_existing_engine(tmp_path, kind):
    selected, lock_path, prefix, data_root, model_dir = fixture(
        tmp_path, build_only=True
    )
    target = model_dir / BACKENDS["centerpoint_tiny"]["engines"][0]
    target.unlink()
    if kind == "empty":
        target.touch()
    elif kind == "fifo":
        os.mkfifo(target)
    else:
        regular(tmp_path / "outside-build-engine")
        target.symlink_to(tmp_path / "outside-build-engine")
    with pytest.raises(VerificationError) as error:
        verify(selected, lock_path, prefix, data_root)
    assert "centerpoint_tiny engine" in str(error.value)


@pytest.mark.parametrize("ack", [None, "", "true", "yes", " 1", "1 "])
def test_license_acknowledgement_must_be_exactly_one(tmp_path, ack):
    selected, lock_path, prefix, data_root, _ = fixture(tmp_path)
    environment = {} if ack is None else {
        "AD_AUTOWARE_MODEL_LICENSE_REVIEWED": ack
    }
    with trust_synthetic_lock(lock_path):
        with pytest.raises(
            VerificationError,
            match="AD_AUTOWARE_MODEL_LICENSE_REVIEWED must equal exactly 1",
        ):
            verify_selection(
                selected,
                lock_path=lock_path,
                prefixes=[prefix],
                data_root=data_root,
                environ=environment,
            )


def test_reports_version_executable_and_leaf_failures_together(tmp_path):
    selected, lock_path, prefix, data_root, _ = fixture(tmp_path)
    detector = BACKENDS["centerpoint_tiny"]
    package_xml = prefix / "share" / detector["package"] / "package.xml"
    package_xml.write_text(
        package_xml.read_text().replace("0.51.0", "0.50.0"),
        encoding="utf-8",
    )
    (prefix / "lib" / detector["package"] / detector["executable"]).unlink()
    (prefix / "share" / detector["package"] / detector["launch"]).unlink()
    messages_xml = (
        prefix / "share" / "autoware_perception_msgs" / "package.xml"
    )
    messages_xml.write_text(
        messages_xml.read_text().replace("1.13.0", "1.11.0"),
        encoding="utf-8",
    )

    with pytest.raises(VerificationError) as error:
        verify(selected, lock_path, prefix, data_root)
    lines = str(error.value).splitlines()[1:]
    assert lines == sorted(lines)
    diagnostic = str(error.value)
    assert "autoware_lidar_centerpoint version 0.50.0 != 0.51.0" in diagnostic
    assert "autoware_perception_msgs version 1.11.0 != 1.13.0" in diagnostic
    assert "autoware_lidar_centerpoint executable" in diagnostic
    assert "lidar_centerpoint.launch.xml" in diagnostic


def test_rejects_non_executable_binary(tmp_path):
    selected, lock_path, prefix, data_root, _ = fixture(tmp_path)
    contract = BACKENDS["centerpoint_tiny"]
    executable = prefix / "lib" / contract["package"] / contract["executable"]
    executable.chmod(0o644)
    with pytest.raises(VerificationError, match="is not executable"):
        verify(selected, lock_path, prefix, data_root)


def test_accepts_standard_symlink_install_regular_targets(tmp_path):
    selected, lock_path, prefix, data_root, _ = fixture(
        tmp_path, tracker=True
    )
    detector = BACKENDS["centerpoint_tiny"]
    installed_paths = [
        prefix / "share" / detector["package"] / "package.xml",
        prefix / "lib" / detector["package"] / detector["executable"],
        prefix / "share" / detector["package"] / detector["launch"],
        *[
            prefix / "share" / detector["package"] / leaf
            for leaf in detector["installed_files"]
        ],
        prefix / "share" / "autoware_perception_msgs" / "package.xml",
        (
            prefix
            / "share"
            / "autoware_multi_object_tracker"
            / "package.xml"
        ),
        (
            prefix
            / "lib"
            / "autoware_multi_object_tracker"
            / "multi_object_tracker_node"
        ),
        (
            prefix
            / "share"
            / "autoware_multi_object_tracker"
            / "launch"
            / "multi_object_tracker.launch.xml"
        ),
        (
            prefix
            / "share"
            / "autoware_multi_object_tracker"
            / "config"
            / "data_association_matrix.param.yaml"
        ),
        (
            prefix
            / "share"
            / "autoware_multi_object_tracker"
            / "config"
            / "input_channels.param.yaml"
        ),
        (
            prefix
            / "lib"
            / "ad_lidar_perception"
            / "ad_autoware_prediction_node"
        ),
    ]
    build_root = tmp_path / "build"
    for index, installed in enumerate(installed_paths):
        replace_with_install_symlink(
            installed, build_root / str(index) / installed.name
        )

    result = verify(selected, lock_path, prefix, data_root)
    assert result.detector is not None
    assert result.tracker is not None
    assert installed_paths[-1].is_symlink()


@pytest.mark.parametrize(
    "leaf",
    [
        "launch/lidar_centerpoint.launch.xml",
        "config/centerpoint_common.param.yaml",
    ],
)
def test_rejects_detector_upstream_leaf_content_mismatch(tmp_path, leaf):
    selected, lock_path, prefix, data_root, _ = fixture(tmp_path)
    target = prefix / "share" / "autoware_lidar_centerpoint" / leaf
    target.write_bytes(target.read_bytes() + b"tampered")
    with pytest.raises(VerificationError, match="SHA-256 mismatch"):
        verify(selected, lock_path, prefix, data_root)


@pytest.mark.parametrize(
    "leaf",
    [
        "launch/multi_object_tracker.launch.xml",
        "config/data_association_matrix.param.yaml",
        "config/input_channels.param.yaml",
    ],
)
def test_rejects_tracker_upstream_leaf_content_mismatch(tmp_path, leaf):
    selected, lock_path, prefix, data_root, _ = fixture(
        tmp_path, tracker=True
    )
    target = prefix / "share" / "autoware_multi_object_tracker" / leaf
    target.write_bytes(target.read_bytes() + b"tampered")
    with pytest.raises(VerificationError, match="SHA-256 mismatch"):
        verify(selected, lock_path, prefix, data_root)


def test_hashes_final_symlink_install_target_content(tmp_path):
    selected, lock_path, prefix, data_root, _ = fixture(tmp_path)
    installed = (
        prefix
        / "share"
        / "autoware_lidar_centerpoint"
        / "config"
        / "centerpoint_common.param.yaml"
    )
    target = tmp_path / "build" / "centerpoint_common.param.yaml"
    replace_with_install_symlink(installed, target)
    target.write_bytes(target.read_bytes() + b"tampered")
    with pytest.raises(VerificationError, match="SHA-256 mismatch"):
        verify(selected, lock_path, prefix, data_root)


def test_tracker_requires_exact_package_leaves_and_prediction_executable(
    tmp_path,
):
    selected, lock_path, prefix, data_root, _ = fixture(
        tmp_path, tracker=True
    )
    result = verify(selected, lock_path, prefix, data_root)
    assert result.tracker is not None
    assert result.tracker.executable == "multi_object_tracker_node"

    tracker_root = prefix / "share" / "autoware_multi_object_tracker"
    (tracker_root / "config/input_channels.param.yaml").unlink()
    prediction = (
        prefix
        / "lib"
        / "ad_lidar_perception"
        / "ad_autoware_prediction_node"
    )
    prediction.unlink()
    with pytest.raises(VerificationError) as error:
        verify(selected, lock_path, prefix, data_root)
    diagnostic = str(error.value)
    assert "input_channels.param.yaml" in diagnostic
    assert "ad_autoware_prediction_node" in diagnostic


def test_rejects_tracker_version_and_executable_mismatch(tmp_path):
    selected, lock_path, prefix, data_root, _ = fixture(
        tmp_path, tracker=True
    )
    tracker_xml = (
        prefix
        / "share"
        / "autoware_multi_object_tracker"
        / "package.xml"
    )
    tracker_xml.write_text(
        tracker_xml.read_text().replace("0.51.0", "0.52.0"),
        encoding="utf-8",
    )
    (
        prefix
        / "lib"
        / "autoware_multi_object_tracker"
        / "multi_object_tracker_node"
    ).unlink()
    with pytest.raises(VerificationError) as error:
        verify(selected, lock_path, prefix, data_root)
    diagnostic = str(error.value)
    assert (
        "autoware_multi_object_tracker version 0.52.0 != 0.51.0"
        in diagnostic
    )
    assert "multi_object_tracker_node" in diagnostic


@pytest.mark.parametrize(
    "target_kind",
    ["artifact_final", "artifact_intermediate"],
)
def test_rejects_final_and_intermediate_symlinks(tmp_path, target_kind):
    selected, lock_path, prefix, data_root, model_dir = fixture(tmp_path)
    if target_kind == "artifact_final":
        target = model_dir / "pts_voxel_encoder_centerpoint_tiny.onnx"
        payload = target.read_bytes()
        target.unlink()
        regular(tmp_path / "outside", payload)
        target.symlink_to(tmp_path / "outside")
    elif target_kind == "artifact_intermediate":
        real = data_root / "real-autoware"
        (data_root / "models" / "autoware").rename(real)
        (data_root / "models" / "autoware").symlink_to(real)
    with pytest.raises(VerificationError, match="symlink"):
        verify(selected, lock_path, prefix, data_root)


@pytest.mark.parametrize("kind", ["broken", "empty", "fifo"])
def test_rejects_invalid_installed_symlink_targets(tmp_path, kind):
    selected, lock_path, prefix, data_root, _ = fixture(tmp_path)
    installed = (
        prefix
        / "share"
        / "autoware_lidar_centerpoint"
        / "config"
        / "centerpoint_common.param.yaml"
    )
    installed.unlink()
    target = tmp_path / f"installed-target-{kind}"
    if kind == "empty":
        target.touch()
    elif kind == "fifo":
        os.mkfifo(target)
    installed.symlink_to(target)

    with pytest.raises(VerificationError) as error:
        verify(selected, lock_path, prefix, data_root)
    diagnostic = str(error.value)
    if kind == "broken":
        assert "broken symlink" in diagnostic
    else:
        assert "symlink target must be a non-empty regular file" in diagnostic


@pytest.mark.parametrize("kind", ["empty", "fifo"])
def test_rejects_empty_and_special_artifacts(tmp_path, kind):
    selected, lock_path, prefix, data_root, model_dir = fixture(tmp_path)
    target = model_dir / "pts_voxel_encoder_centerpoint_tiny.onnx"
    target.unlink()
    if kind == "empty":
        target.touch()
    else:
        os.mkfifo(target)
        assert stat.S_ISFIFO(target.lstat().st_mode)
    with pytest.raises(VerificationError, match="non-empty regular file"):
        verify(selected, lock_path, prefix, data_root)


@pytest.mark.parametrize("kind", ["empty", "fifo"])
def test_rejects_empty_and_special_installed_leaves(tmp_path, kind):
    selected, lock_path, prefix, data_root, _ = fixture(tmp_path)
    target = (
        prefix
        / "share"
        / "autoware_lidar_centerpoint"
        / "config"
        / "centerpoint_common.param.yaml"
    )
    target.unlink()
    if kind == "empty":
        target.touch()
    else:
        os.mkfifo(target)
    with pytest.raises(VerificationError, match="non-empty regular file"):
        verify(selected, lock_path, prefix, data_root)


@pytest.mark.parametrize("kind", ["empty", "fifo", "symlink"])
def test_rejects_invalid_runtime_engines(tmp_path, kind):
    selected, lock_path, prefix, data_root, model_dir = fixture(tmp_path)
    target = model_dir / BACKENDS["centerpoint_tiny"]["engines"][0]
    target.unlink()
    if kind == "empty":
        target.touch()
    elif kind == "fifo":
        os.mkfifo(target)
    else:
        regular(tmp_path / "outside-engine")
        target.symlink_to(tmp_path / "outside-engine")
    with pytest.raises(VerificationError) as error:
        verify(selected, lock_path, prefix, data_root)
    diagnostic = str(error.value)
    assert "centerpoint_tiny engine" in diagnostic
    if kind == "symlink":
        assert "symlink" in diagnostic
    else:
        assert "non-empty regular file" in diagnostic


def test_cli_emits_one_deterministic_failure_and_success(tmp_path, capsys):
    selected, lock_path, prefix, data_root, _ = fixture(tmp_path)
    selection_path = selection_file(tmp_path)
    with trust_synthetic_lock(lock_path):
        exit_code = main(
            [
                "--selection",
                str(selection_path),
                "--lock",
                str(lock_path),
                "--data-root",
                str(data_root),
                "--prefix",
                str(prefix),
            ],
            environ={"AD_AUTOWARE_MODEL_LICENSE_REVIEWED": "1"},
        )
    assert exit_code == 0
    assert "verified" in capsys.readouterr().out.lower()

    with trust_synthetic_lock(lock_path):
        exit_code = main(
            [
                "--selection",
                str(selection_path),
                "--lock",
                str(lock_path),
                "--data-root",
                str(data_root),
                "--prefix",
                str(tmp_path / "missing-prefix"),
            ],
            environ={"AD_AUTOWARE_MODEL_LICENSE_REVIEWED": "1"},
        )
    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.err.count("Autoware perception verification failed:") == 1


def test_checker_contains_no_downloader_network_or_subprocess_code():
    source = (
        REPO
        / "ad_lidar_perception"
        / "ad_lidar_perception"
        / "autoware_provenance.py"
    ).read_text(encoding="utf-8")
    forbidden = (
        "import subprocess",
        "from subprocess",
        "import requests",
        "import urllib",
        "from urllib",
        "os.system",
        "Popen(",
        "run(",
        "wget",
        "curl",
    )
    for token in forbidden:
        assert token not in source
