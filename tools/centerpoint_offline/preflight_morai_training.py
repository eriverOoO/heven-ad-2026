#!/usr/bin/env python3
"""CenterPoint MORAI training preflight - audit + gate only, never trains.

Validates a derived ``centerpoint_morai_adapter_v1`` export against the
repository's current CenterPoint / OpenPCDet contracts (point features,
class list, 3D box convention, point-cloud range, voxel grid, loader),
refuses evaluation-readiness when no independent validation group exists,
records model / data / config / checkpoint / code provenance, and audits
the PyTorch / CUDA / OpenPCDet environment.

It performs NO optimizer step, NO backward pass, NO epoch loop. The
optional model smoke is a single ``torch.no_grad()`` forward for shape
verification only.

    python3 tools/centerpoint_offline/preflight_morai_training.py \
        --dataset <adapter_export_root> \
        --config tools/centerpoint_offline/configs/centerpoint_morai_v1.yaml \
        --output-dir <dir outside the repo>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from morai_dataset import (
    BOX_FIELDS,
    CLASS_NAMES as LOADER_CLASS_NAMES,
    MoraiHevenDatasetCore,
    collate_openpcdet_contract,
)

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_CONFIG = HERE / "configs" / "centerpoint_morai_v1.yaml"
ADAPTER_SCHEMA_VERSION = "centerpoint_morai_adapter_v1"
TARGET_POINT_FEATURES = ("x", "y", "z", "intensity")
REPORTED_POINT_CLOUD_RANGE = (-4.0, -25.0, -3.0, 100.0, 25.0, 5.0)

# Precedence: config errors first, then dataset, then evaluator, then env.
STATUS_BLOCKED_CONFIG = "BLOCKED_CONFIG"
STATUS_BLOCKED_DATASET = "BLOCKED_DATASET"
STATUS_BLOCKED_EVALUATOR = "BLOCKED_EVALUATOR"
STATUS_BLOCKED_ENVIRONMENT = "BLOCKED_ENVIRONMENT"
STATUS_READY = "READY"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"config must be a mapping: {path}")
    if "_BASE_CONFIG_" in data:
        raise ValueError(
            f"{path} uses _BASE_CONFIG_; the v1 experiment configs must be "
            "self-contained (the offline loaders do not resolve bases)"
        )
    return data


# --------------------------------------------------------------------------
# dataset + split
# --------------------------------------------------------------------------
def run_adapter_validator(dataset_root: Path) -> dict[str, Any]:
    """Invoke ``ad_morai_dataset_validate_centerpoint`` (reuse, not reimplement)."""
    env = dict(os.environ)
    pkg_dir = REPO_ROOT / "ad_morai_bridge_dev"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(pkg_dir), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "ad_morai_bridge_dev.dataset.centerpoint_adapter_validate",
                str(dataset_root),
                "--json",
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=600,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {"invoked": False, "error": f"{type(error).__name__}: {error}"}
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {
            "invoked": True,
            "error": "validator did not emit JSON",
            "stderr": completed.stderr[-2000:],
            "returncode": completed.returncode,
        }
    report["invoked"] = True
    report["returncode"] = completed.returncode
    return report


def audit_dataset(dataset_root: Path, assert_real: bool) -> dict[str, Any]:
    result: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "exists": dataset_root.is_dir(),
        "blockers": [],
    }
    if not dataset_root.is_dir():
        result["blockers"].append("dataset root does not exist")
        result["real_dataset_available"] = False
        result["fixture_only"] = True
        return result

    resolved = str(dataset_root.resolve())
    looks_temporary = (
        resolved.startswith(tempfile.gettempdir() + os.sep)
        or "pytest-of-" in resolved
        or "/tmp/" in resolved
    )
    result["path_looks_temporary"] = looks_temporary
    result["real_dataset_available"] = bool(assert_real) and not looks_temporary
    result["fixture_only"] = not result["real_dataset_available"]
    # The "no real dataset" gate is global, not a structural defect of the
    # export: keep it out of `blockers` (which drives dataset_status) so a
    # fixture can still be reported as structurally READY.
    result["real_dataset_blockers"] = []
    if assert_real and looks_temporary:
        result["real_dataset_blockers"].append(
            "--assert-real-dataset was passed but the path is under a temp dir; "
            "treating as fixture"
        )
    if not result["real_dataset_available"]:
        result["real_dataset_blockers"].append(
            "no real (human-asserted, non-temporary) adapter export; "
            "training/evaluation stays blocked"
        )

    marker = dataset_root / ".centerpoint_morai_adapter"
    result["adapter_marker_present"] = marker.is_file()
    if not marker.is_file():
        result["blockers"].append("adapter marker absent (not an adapter export)")

    for name in ("export_manifest.json", "metadata.json", "split_manifest.json"):
        path = dataset_root / name
        if path.is_file():
            result[name.split(".")[0]] = json.loads(path.read_text("utf-8"))
        else:
            result["blockers"].append(f"{name} missing")

    export_manifest = result.get("export_manifest", {})
    result["adapter_schema_version"] = export_manifest.get("adapter_schema_version")
    result["source_dataset_id"] = export_manifest.get("source_dataset_id")
    result["source_dataset_manifest_sha256"] = export_manifest.get(
        "source_dataset_manifest_sha256"
    )
    result["content_fingerprint"] = export_manifest.get("content_fingerprint")
    result["adapter_repository_commit"] = export_manifest.get(
        "adapter_repository_commit"
    )
    if result["adapter_schema_version"] != ADAPTER_SCHEMA_VERSION:
        result["blockers"].append(
            f"adapter schema {result['adapter_schema_version']!r} != "
            f"{ADAPTER_SCHEMA_VERSION!r}"
        )
    if export_manifest.get("status") != "complete":
        result["blockers"].append("export status is not 'complete'")

    validator = run_adapter_validator(dataset_root)
    result["adapter_validator"] = validator
    if not validator.get("invoked"):
        result["blockers"].append(
            f"could not invoke adapter validator: {validator.get('error')}"
        )
    elif not validator.get("ok"):
        result["blockers"].append(
            f"adapter validator failed: {validator.get('errors', [])[:3]}"
        )
    return result


def audit_splits(dataset: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    split_manifest = dataset.get("split_manifest", {})
    group_splits: dict[str, str] = split_manifest.get("group_splits", {})
    frame_counts: dict[str, int] = split_manifest.get("frame_counts", {})
    negatives: dict[str, int] = split_manifest.get("negative_frame_counts", {})
    class_counts = split_manifest.get("class_counts_by_split", {})

    group_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    for split in group_splits.values():
        group_counts[split] = group_counts.get(split, 0) + 1

    min_train = int(config.get("minimum_train_groups", 1))
    min_val = int(config.get("minimum_val_groups", 1))
    min_test = int(config.get("minimum_test_groups", 1))

    warnings: list[str] = []
    blockers: list[str] = []

    # Leakage: trust the adapter validator's own check_split_leakage result.
    validator = dataset.get("adapter_validator", {})
    leakage_errors = [
        error
        for error in validator.get("errors", [])
        if "leakage" in error or "multiple splits" in error
    ]
    if leakage_errors:
        blockers.append(f"split leakage detected: {leakage_errors}")
    if not split_manifest.get("leakage_free", False):
        warnings.append("split_manifest.leakage_free is not true")

    if group_counts["train"] < 1 or frame_counts.get("train", 0) < 1:
        blockers.append("train split is empty")
    if group_counts["val"] < 1 or frame_counts.get("val", 0) < 1:
        blockers.append(
            "no independent validation group (val groups == 0): NOT evaluation ready"
        )
    if group_counts["train"] < min_train:
        warnings.append(
            f"train has {group_counts['train']} independent groups "
            f"(< recommended {min_train})"
        )
    if 0 < group_counts["val"] < min_val + 1:
        warnings.append(
            f"val has {group_counts['val']} independent group(s); small val group "
            "count may yield unstable validation metrics"
        )
    if 0 < group_counts["test"] < min_test:
        warnings.append(
            f"test has {group_counts['test']} independent group(s) "
            f"(< recommended {min_test})"
        )

    evaluation_ready = not blockers
    return {
        "group_splits": group_splits,
        "group_counts": group_counts,
        "frame_counts": frame_counts,
        "negative_frame_counts": negatives,
        "class_counts_by_split": class_counts,
        "minimum_groups": {"train": min_train, "val": min_val, "test": min_test},
        "independent_val_group_exists": group_counts["val"] >= 1,
        "test_split_reserved": bool(config.get("test_split_reserved", True)),
        "evaluation_ready": evaluation_ready,
        "warnings": warnings,
        "blockers": blockers,
    }


# --------------------------------------------------------------------------
# contract checks
# --------------------------------------------------------------------------
def check_contracts(
    dataset: dict[str, Any],
    data_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    experiment: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    metadata = dataset.get("metadata", {})
    split_manifest = dataset.get("split_manifest", {})

    # --- classes -------------------------------------------------------
    model_classes = list(model_cfg.get("CLASS_NAMES", []))
    experiment_classes = list(experiment.get("class_names", []))
    loader_accepted = list(metadata.get("class_names", []))
    if model_classes != experiment_classes:
        errors.append(
            f"model CLASS_NAMES {model_classes} != experiment class_names "
            f"{experiment_classes}"
        )
    for name in model_classes:
        if name not in loader_accepted:
            errors.append(
                f"class {name!r} not in the export's accepted class set "
                f"{loader_accepted}"
            )
    heads = model_cfg.get("MODEL", {}).get("DENSE_HEAD", {}).get(
        "CLASS_NAMES_EACH_HEAD", []
    )
    flat_heads = [name for head in heads for name in head]
    if sorted(flat_heads) != sorted(model_classes):
        errors.append(
            f"CLASS_NAMES_EACH_HEAD {heads} does not cover CLASS_NAMES "
            f"{model_classes}"
        )
    present = {
        name
        for split in split_manifest.get("class_counts_by_split", {}).values()
        for name in split
    }
    extra = sorted(present - set(model_classes))
    if extra:
        errors.append(
            f"export contains classes {extra} not in model CLASS_NAMES "
            f"{model_classes} (MoraiHevenDatasetCore would raise 'unknown class')"
        )
    # vehicle -> class id 1 under the fixed 1-based order
    if model_classes and model_classes[0] != "vehicle":
        warnings.append(
            f"first class is {model_classes[0]!r}; loader assigns it id 1"
        )
    class_id_map = {name: i + 1 for i, name in enumerate(model_classes)}

    # --- point features ---------------------------------------------
    used = list(
        data_cfg.get("POINT_FEATURE_ENCODING", {}).get("used_feature_list", [])
    )
    src = list(
        data_cfg.get("POINT_FEATURE_ENCODING", {}).get("src_feature_list", [])
    )
    meta_features = list(metadata.get("point_feature_names", []))
    if tuple(used) != TARGET_POINT_FEATURES:
        errors.append(f"used_feature_list {used} != {list(TARGET_POINT_FEATURES)}")
    if tuple(src) != TARGET_POINT_FEATURES:
        errors.append(f"src_feature_list {src} != {list(TARGET_POINT_FEATURES)}")
    if meta_features and tuple(meta_features) != TARGET_POINT_FEATURES:
        errors.append(
            f"export point_feature_names {meta_features} != "
            f"{list(TARGET_POINT_FEATURES)}"
        )
    num_point_features = len(used)

    # --- box convention -----------------------------------------------
    meta_box_fields = list(metadata.get("box_fields", []))
    if meta_box_fields and tuple(meta_box_fields) != BOX_FIELDS:
        errors.append(f"export box_fields {meta_box_fields} != {list(BOX_FIELDS)}")
    if metadata.get("gt_frame") and metadata["gt_frame"] != "lidar_link":
        errors.append(f"export gt_frame {metadata['gt_frame']!r} != 'lidar_link'")
    if metadata.get("intensity_transform") not in (None, "identity"):
        warnings.append(
            f"export intensity_transform is {metadata['intensity_transform']!r}"
        )
    if metadata.get("point_range_cropping_applied"):
        errors.append("export applied point-range cropping (adapter must not crop)")
    if metadata.get("box_range_filtering_applied"):
        errors.append("export applied box-range filtering (adapter must not filter)")

    # --- range consistency -----------------------------------------
    data_range = [float(v) for v in data_cfg.get("POINT_CLOUD_RANGE", [])]
    head_range = [
        float(v)
        for v in model_cfg.get("MODEL", {})
        .get("DENSE_HEAD", {})
        .get("POST_PROCESSING", {})
        .get("POST_CENTER_LIMIT_RANGE", [])
    ]
    meta_range = [float(v) for v in metadata.get("reported_point_cloud_range", [])]
    experiment_range = [float(v) for v in experiment.get("point_cloud_range", [])]
    if tuple(data_range) != REPORTED_POINT_CLOUD_RANGE:
        errors.append(
            f"data POINT_CLOUD_RANGE {data_range} != {list(REPORTED_POINT_CLOUD_RANGE)}"
        )
    if head_range and head_range != data_range:
        errors.append(
            f"head POST_CENTER_LIMIT_RANGE {head_range} != POINT_CLOUD_RANGE "
            f"{data_range}"
        )
    if meta_range and meta_range != data_range:
        errors.append(
            f"export reported_point_cloud_range {meta_range} != config "
            f"{data_range}"
        )
    if experiment_range and experiment_range != data_range:
        warnings.append("experiment manifest point_cloud_range differs from data cfg")

    # --- voxel grid --------------------------------------------------
    voxel = None
    for processor in data_cfg.get("DATA_PROCESSOR", []):
        if processor.get("NAME") == "transform_points_to_voxels":
            voxel = [float(v) for v in processor.get("VOXEL_SIZE", [])]
    grid_info: dict[str, Any] = {"voxel_size": voxel}
    if voxel and len(data_range) == 6:
        span = np.asarray(data_range[3:]) - np.asarray(data_range[:3])
        raw = span / np.asarray(voxel)
        grid = np.round(raw).astype(int)
        grid_info["grid_size_xyz"] = grid.tolist()
        if not np.allclose(raw, grid, atol=1e-6):
            errors.append(
                f"point-cloud span {span.tolist()} is not an integer multiple of "
                f"voxel size {voxel} (raw grid {raw.tolist()})"
            )
        stride = int(
            model_cfg.get("MODEL", {})
            .get("DENSE_HEAD", {})
            .get("TARGET_ASSIGNER_CONFIG", {})
            .get("FEATURE_MAP_STRIDE", 8)
        )
        grid_info["feature_map_stride"] = stride
        for axis, size in zip("xy", grid[:2]):
            if size % stride:
                errors.append(
                    f"grid_{axis} {int(size)} not divisible by FEATURE_MAP_STRIDE "
                    f"{stride}"
                )
        feat = (grid[:2] // stride).tolist()
        grid_info["bev_feature_map_xy"] = feat
        strides_2d = model_cfg.get("MODEL", {}).get("BACKBONE_2D", {}).get(
            "LAYER_STRIDES", []
        )
        for s in strides_2d:
            if s and (feat[0] % s or feat[1] % s):
                errors.append(
                    f"BEV feature map {feat} not divisible by BACKBONE_2D stride {s}"
                )

    return {
        "class_names": model_classes,
        "class_id_map": class_id_map,
        "num_point_features": num_point_features,
        "box_fields": list(BOX_FIELDS),
        "grid": grid_info,
        "loader_class_names": list(LOADER_CLASS_NAMES),
        "errors": errors,
        "warnings": warnings,
        "passed": not errors,
    }


# --------------------------------------------------------------------------
# environment + checkpoint
# --------------------------------------------------------------------------
def audit_environment(openpcdet_root: Path | None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
    }
    for module in ("numpy", "yaml", "easydict", "numba", "SharedArray"):
        try:
            __import__(module)
            result[f"{module}_available"] = True
        except ImportError:
            result[f"{module}_available"] = False
    result["numpy_version"] = np.__version__

    try:
        import torch

        result["torch_available"] = True
        result["torch_version"] = torch.__version__
        result["cuda_available"] = bool(torch.cuda.is_available())
        result["torch_cuda_version"] = torch.version.cuda
        result["gpu"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        )
    except (ImportError, OSError) as error:
        result["torch_available"] = False
        result["torch_error"] = f"{type(error).__name__}: {error}"
        result["cuda_available"] = False

    for module in ("spconv", "spconv.pytorch"):
        try:
            __import__(module)
            result[f"{module}_available"] = True
        except (ImportError, OSError) as error:
            result[f"{module}_available"] = False
            result[f"{module}_error"] = f"{type(error).__name__}: {error}"

    try:
        import pcdet

        result["pcdet_available"] = True
        result["pcdet_version"] = getattr(pcdet, "__version__", None)
        result["pcdet_location"] = getattr(pcdet, "__file__", None)
    except (ImportError, OSError) as error:
        result["pcdet_available"] = False
        result["pcdet_error"] = f"{type(error).__name__}: {error}"

    candidates = []
    seen: set[str] = set()
    roots = [openpcdet_root, REPO_ROOT / "references" / "openpcdet"]
    if result.get("pcdet_location"):
        roots.append(Path(result["pcdet_location"]).resolve().parents[1])
    for root in filter(None, roots):
        root = Path(root).resolve()
        if str(root) in seen:
            continue
        seen.add(str(root))
        candidates.append(
            {
                "path": str(root),
                "has_centerpoint": (
                    root / "pcdet" / "models" / "detectors" / "centerpoint.py"
                ).is_file(),
                "is_resolved_pcdet": result.get("pcdet_location", "").startswith(
                    str(root) + os.sep
                ),
            }
        )
    result["openpcdet_candidates"] = candidates

    result["environment_ready"] = bool(
        result.get("torch_available")
        and result.get("cuda_available")
        and result.get("spconv.pytorch_available")
        and result.get("pcdet_available")
    )
    result["blockers"] = []
    if not result.get("torch_available"):
        result["blockers"].append("torch not importable")
    if result.get("torch_available") and not result.get("cuda_available"):
        result["blockers"].append("CUDA not available to torch")
    if not result.get("spconv.pytorch_available"):
        result["blockers"].append("spconv.pytorch not importable")
    if not result.get("pcdet_available"):
        result["blockers"].append("pcdet (OpenPCDet) not importable")
    return result


EXPECTED_EVALUATOR_SCHEMA_VERSION = "morai_centerpoint_eval_v1"


def audit_evaluator(model_cfg: dict[str, Any], experiment: dict[str, Any]) -> dict[str, Any]:
    """Detect whether a real MORAI detection evaluator is wired in."""
    result: dict[str, Any] = {"expected_schema": EXPECTED_EVALUATOR_SCHEMA_VERSION}
    try:
        from morai_evaluator import (  # noqa: WPS433
            MORAI_EVALUATOR_SCHEMA_VERSION,
            PRIMARY_VALIDATION_METRIC,
        )

        result["evaluator_module_importable"] = True
        result["evaluator_schema_version"] = MORAI_EVALUATOR_SCHEMA_VERSION
        result["evaluator_primary_metric"] = PRIMARY_VALIDATION_METRIC
    except ImportError as error:
        result["evaluator_module_importable"] = False
        result["error"] = f"{type(error).__name__}: {error}"
        result["evaluation_metric_implemented"] = False
        return result

    eval_metric = (
        model_cfg.get("MODEL", {}).get("POST_PROCESSING", {}).get("EVAL_METRIC")
    )
    result["model_config_eval_metric"] = eval_metric
    result["experiment_primary_metric"] = experiment.get("primary_validation_metric")

    schema_ok = (
        result["evaluator_schema_version"] == EXPECTED_EVALUATOR_SCHEMA_VERSION
    )
    config_ok = eval_metric == EXPECTED_EVALUATOR_SCHEMA_VERSION
    result["schema_matches_expected"] = schema_ok
    result["model_config_selects_evaluator"] = config_ok
    result["evaluation_metric_implemented"] = bool(schema_ok and config_ok)
    if not config_ok:
        result["note"] = (
            f"MODEL.POST_PROCESSING.EVAL_METRIC is {eval_metric!r}; set it to "
            f"{EXPECTED_EVALUATOR_SCHEMA_VERSION!r} to use the MORAI evaluator"
        )
    return result


def audit_checkpoint(
    checkpoint_path: Path | None, model_classes: list[str], torch_available: bool
) -> dict[str, Any]:
    if checkpoint_path is None:
        return {
            "checkpoint_available": False,
            "policy": "train_from_scratch",
            "note": "no --init-checkpoint given; v1 default is train-from-scratch",
        }
    if not checkpoint_path.is_file():
        return {
            "checkpoint_available": False,
            "requested_path": str(checkpoint_path),
            "error": "file not found",
        }
    result: dict[str, Any] = {
        "checkpoint_available": True,
        "path": str(checkpoint_path),
        "sha256": _sha256_file(checkpoint_path),
        "size_bytes": checkpoint_path.stat().st_size,
    }
    if not torch_available:
        result["state_dict_probe"] = "skipped (torch unavailable)"
        return result
    try:
        import torch

        blob = torch.load(checkpoint_path, map_location="cpu")
        state = blob.get("model_state", blob.get("model_state_dict", blob))
        head_classes = None
        for key, value in state.items():
            if key.endswith("hm.1.weight") and hasattr(value, "shape"):
                head_classes = int(value.shape[0])
                break
        point_features = None
        for key, value in state.items():
            if key.endswith("conv_input.0.weight") and hasattr(value, "shape"):
                point_features = int(value.shape[-1])
                break
        result["checkpoint_epoch"] = blob.get("epoch")
        result["checkpoint_iteration"] = blob.get("iteration")
        result["heatmap_head_classes"] = head_classes
        result["point_features"] = point_features
        compatible = head_classes == len(model_classes)
        result["strict_load_compatible_with_v1"] = compatible
        result["policy"] = (
            "train_from_scratch"
            if not compatible
            else "strict_load_possible_but_not_performed"
        )
        if not compatible:
            result["note"] = (
                f"checkpoint heatmap head is {head_classes}-class; v1 is "
                f"{len(model_classes)}-class -> strict state_dict load fails by "
                "construction. A future selective backbone-only load is a "
                "separate explicit decision, not part of training prep."
            )
    except (RuntimeError, OSError, KeyError) as error:
        result["state_dict_probe_error"] = f"{type(error).__name__}: {error}"
    return result


# --------------------------------------------------------------------------
# loader / model smokes (no training)
# --------------------------------------------------------------------------
def loader_smoke(dataset_root: Path) -> dict[str, Any]:
    """Torch-free: always run MoraiHevenDatasetCore over the export."""
    result: dict[str, Any] = {"attempted": True}
    try:
        train = MoraiHevenDatasetCore(dataset_root, split="train")
        result["train_samples"] = len(train)
        pos_index = train.first_nonempty_index()
        positive = train[pos_index]
        result["positive_sample"] = {
            "sample_id": positive["sample_id"],
            "points_shape": list(positive["points"].shape),
            "gt_boxes_shape": list(positive["gt_boxes"].shape),
            "gt_names": [str(n) for n in positive["gt_names"]],
            "coordinate_frame": positive["coordinate_frame"],
            "points_finite": bool(np.isfinite(positive["points"]).all()),
        }
        negative = None
        for index in range(len(train)):
            sample = train[index]
            if len(sample["gt_boxes"]) == 0:
                negative = sample
                break
        if negative is not None:
            result["negative_sample"] = {
                "sample_id": negative["sample_id"],
                "points_shape": list(negative["points"].shape),
                "gt_boxes_shape": list(negative["gt_boxes"].shape),
            }
        else:
            result["negative_sample"] = "none present in train split"

        collated = collate_openpcdet_contract([positive])
        result["pre_voxel_collate"] = {
            "points_shape": list(collated["points"].shape),
            "gt_boxes_shape": list(collated["gt_boxes"].shape),
            "vehicle_class_id": (
                float(collated["gt_boxes"][0, 0, 7])
                if collated["gt_boxes"].shape[1]
                else None
            ),
        }
        try:
            val = MoraiHevenDatasetCore(dataset_root, split="val")
            result["val_samples"] = len(val)
            if len(val):
                vs = val[0]
                result["validation_sample"] = {
                    "sample_id": vs["sample_id"],
                    "points_shape": list(vs["points"].shape),
                    "gt_boxes_shape": list(vs["gt_boxes"].shape),
                }
        except (FileNotFoundError, ValueError) as error:
            result["val_sample_error"] = f"{type(error).__name__}: {error}"
        result["success"] = True
    except (FileNotFoundError, ValueError, KeyError) as error:
        result["success"] = False
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def model_smoke(
    dataset_root: Path,
    data_cfg_path: Path,
    model_cfg_path: Path,
    openpcdet_root: Path,
    run_forward: bool,
) -> dict[str, Any]:
    """Construct the model and (optionally) one no_grad forward. NEVER trains."""
    result: dict[str, Any] = {"attempted": True, "forward_requested": run_forward}
    try:
        import torch
        from easydict import EasyDict

        from openpcdet_runtime import import_smoke_components, load_batch_to_cuda
        from morai_dataset import make_openpcdet_dataset

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        dataset_template, centerpoint_class = import_smoke_components(openpcdet_root)
        import pcdet as _pcdet

        result["openpcdet_root_arg"] = str(openpcdet_root)
        result["pcdet_resolved_from"] = getattr(_pcdet, "__file__", None)
        dataset_class = make_openpcdet_dataset(dataset_template)
        data_cfg = EasyDict(_load_yaml(data_cfg_path))
        model_cfg = EasyDict(_load_yaml(model_cfg_path))
        data_cfg.DATA_PATH = str(dataset_root.resolve())
        dataset = dataset_class(
            dataset_cfg=data_cfg,
            class_names=list(model_cfg.CLASS_NAMES),
            training=True,
            root_path=dataset_root.resolve(),
            logger=None,
        )
        index = dataset.morai_core.first_nonempty_index()
        prepared = dataset[index]
        result["data_processor"] = {
            "sample_id": str(prepared.get("frame_id")),
            "voxels_shape": list(np.asarray(prepared["voxels"]).shape),
            "voxel_coords_shape": list(np.asarray(prepared["voxel_coords"]).shape),
            "gt_boxes_shape": list(np.asarray(prepared["gt_boxes"]).shape),
        }
        batch = dataset.collate_batch([prepared])
        result["batch_collation"] = {
            "batch_size": int(batch["batch_size"]),
            "points_or_voxels": sorted(
                k for k in batch if k in ("voxels", "points", "voxel_coords")
            ),
            "gt_boxes_shape": list(np.asarray(batch["gt_boxes"]).shape),
        }
        model = centerpoint_class(
            model_cfg=model_cfg.MODEL,
            num_class=len(model_cfg.CLASS_NAMES),
            dataset=dataset,
        ).cuda().eval()
        result["model_class"] = type(model).__name__
        result["parameter_count"] = int(sum(p.numel() for p in model.parameters()))
        result["device"] = "cuda"
        result["model_construction_success"] = True

        if run_forward:
            load_batch_to_cuda(batch)
            with torch.no_grad():
                predictions, _ = model(batch)
            result["forward_smoke"] = {
                "ran": True,
                "kind": "shape_smoke_test_only",
                "pred_boxes_shape": list(predictions[0]["pred_boxes"].shape),
                "note": "no backward, no optimizer.step, no metric",
            }
        result["success"] = True
    except (ImportError, OSError, RuntimeError, KeyError, ValueError) as error:
        result["success"] = False
        result["error"] = f"{type(error).__name__}: {error}"
    return result


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument(
        "--openpcdet-root",
        type=Path,
        default=REPO_ROOT / "references" / "openpcdet",
    )
    parser.add_argument(
        "--assert-real-dataset",
        action="store_true",
        help="operator asserts --dataset is a real, non-fixture MORAI export",
    )
    parser.add_argument("--attempt-loader-smoke", action="store_true")
    parser.add_argument("--attempt-model-smoke", action="store_true")
    parser.add_argument(
        "--attempt-forward-smoke",
        action="store_true",
        help="one torch.no_grad() forward for shape verification (implies model smoke)",
    )
    return parser.parse_args(argv)


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    experiment = _load_yaml(args.config)
    config_dir = args.config.resolve().parent
    data_cfg_path = config_dir / experiment.get(
        "data_config", "centerpoint_morai_v1_dataset.yaml"
    )
    model_cfg_path = config_dir / experiment.get(
        "model_config", "centerpoint_morai_v1_model.yaml"
    )
    config_errors: list[str] = []
    try:
        data_cfg = _load_yaml(data_cfg_path)
    except (OSError, ValueError) as error:
        data_cfg = {}
        config_errors.append(f"data config: {error}")
    try:
        model_cfg = _load_yaml(model_cfg_path)
    except (OSError, ValueError) as error:
        model_cfg = {}
        config_errors.append(f"model config: {error}")

    dataset = audit_dataset(args.dataset, args.assert_real_dataset)
    splits = audit_splits(dataset, experiment)
    contracts = check_contracts(dataset, data_cfg, model_cfg, experiment)
    environment = audit_environment(args.openpcdet_root)

    default_ckpt = os.environ.get("HEVEN_CENTERPOINT_T14_CKPT")
    checkpoint_path = args.init_checkpoint or (
        Path(default_ckpt) if default_ckpt else None
    )
    checkpoint = audit_checkpoint(
        checkpoint_path,
        contracts["class_names"],
        bool(environment.get("torch_available")),
    )

    loader = {"attempted": False}
    if args.attempt_loader_smoke or args.attempt_model_smoke or args.attempt_forward_smoke:
        loader = loader_smoke(args.dataset)

    model = {"attempted": False}
    if args.attempt_model_smoke or args.attempt_forward_smoke:
        if environment.get("environment_ready"):
            model = model_smoke(
                args.dataset,
                data_cfg_path,
                model_cfg_path,
                args.openpcdet_root,
                run_forward=args.attempt_forward_smoke,
            )
        else:
            model = {
                "attempted": True,
                "success": False,
                "error": "environment not ready; model smoke skipped",
            }

    config_fingerprint = _canonical_hash(
        {
            "experiment": experiment,
            "data_config": data_cfg,
            "model_config": model_cfg,
        }
    )

    evaluator = audit_evaluator(model_cfg, experiment)
    evaluation_metric_implemented = evaluator["evaluation_metric_implemented"]

    structurally_ready = (
        not dataset["blockers"]
        and not splits["blockers"]
        and splits["evaluation_ready"]
    )
    dataset_status = "READY" if structurally_ready else "BLOCKED"

    # ---- final status (precedence) ---------------------------------
    if config_errors or not contracts["passed"]:
        final_status = STATUS_BLOCKED_CONFIG
    elif (
        not structurally_ready
        or not dataset["real_dataset_available"]
        or dataset["real_dataset_blockers"]
    ):
        final_status = STATUS_BLOCKED_DATASET
    elif not evaluation_metric_implemented:
        final_status = STATUS_BLOCKED_EVALUATOR
    elif not environment["environment_ready"]:
        final_status = STATUS_BLOCKED_ENVIRONMENT
    else:
        final_status = STATUS_READY

    return {
        "preflight_schema": "heven.centerpoint_morai_training_prep.v1",
        "experiment_id": experiment.get("experiment_id"),
        "created_at_utc": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "repository_commit": _git_commit(),
        "config_path": str(args.config),
        "data_config_path": str(data_cfg_path),
        "model_config_path": str(model_cfg_path),
        "config_fingerprint": config_fingerprint,
        "config_errors": config_errors,
        "dataset_checks": dataset,
        "split_checks": splits,
        "model_config_checks": contracts,
        "environment_checks": environment,
        "checkpoint_checks": checkpoint,
        "evaluator_checks": evaluator,
        "loader_checks": loader,
        "model_smoke_checks": model,
        "seed": experiment.get("seed"),
        "determinism_level": experiment.get("determinism_level", "seeded"),
        "real_dataset_available": dataset["real_dataset_available"],
        "fixture_only": dataset["fixture_only"],
        "independent_val_group_exists": splits["independent_val_group_exists"],
        "evaluation_ready": splits["evaluation_ready"],
        "evaluation_metric_implemented": evaluation_metric_implemented,
        "train_env_ready": environment["environment_ready"],
        "dataset_status": dataset_status,
        "config_status": "PASS" if contracts["passed"] and not config_errors else "FAIL",
        "env_status": "READY" if environment["environment_ready"] else "BLOCKED",
        "final_status": final_status,
        "training_started": False,
        "optimizer_step_executed": False,
        "future_train_command": _future_train_command(args, data_cfg_path, model_cfg_path),
        "future_validation_command": _future_validation_command(args),
    }


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def _future_train_command(
    args: argparse.Namespace, data_cfg_path: Path, model_cfg_path: Path
) -> dict[str, Any]:
    return {
        "status": "NOT RUN IN THIS PR - blocked (no real dataset, no evaluator)",
        "command": [
            "python3",
            "tools/centerpoint_offline/train_morai_centerpoint.py",
            "--dataset",
            "<REAL centerpoint_morai_adapter_v1 export>",
            "--openpcdet-root",
            str(args.openpcdet_root),
            "--data-config",
            str(data_cfg_path),
            "--model-config",
            str(model_cfg_path),
            "--epochs",
            "<decide from real dataset size; historical smoke used 3>",
            "--batch-size",
            "1",
            "--workers",
            "0",
            "--seed",
            "2026",
            "--output-dir",
            "<dir outside the repo>",
        ],
    }


def _future_validation_command(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "status": (
            "NOT RUN - additionally blocked: MoraiHevenDataset.evaluation() raises "
            "NotImplementedError; no MORAI validation metric exists yet"
        ),
        "intended": "evaluate ONLY splits/val.txt (DATA_SPLIT.test -> val); "
        "splits/test.txt stays reserved for a single final evaluation",
    }


def _write_atomic(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(f".{destination.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(destination)


def _print_summary(report: dict[str, Any]) -> None:
    splits = report["split_checks"]
    env = report["environment_checks"]
    ck = report["checkpoint_checks"]
    print("=" * 68)
    print(f"CenterPoint MORAI training preflight - {report['experiment_id']}")
    print("=" * 68)
    print(
        f"DATASET:        {report['dataset_status']}  "
        f"(real={report['real_dataset_available']}, fixture_only={report['fixture_only']})"
    )
    gc = splits["group_counts"]
    print(
        f"SPLITS:         train {gc['train']} groups / val {gc['val']} / test {gc['test']}"
        f"  frames {splits['frame_counts']}"
    )
    print(
        f"MODEL CONTRACT: {report['config_status']}"
        + (
            ""
            if report["model_config_checks"]["passed"]
            else f"  {report['model_config_checks']['errors'][:2]}"
        )
    )
    print(
        f"ENV:            {report['env_status']}  "
        f"(torch={env.get('torch_available')}, cuda={env.get('cuda_available')}, "
        f"pcdet={env.get('pcdet_available')})"
    )
    print(
        "CHECKPOINT:     "
        + (
            "available"
            if ck.get("checkpoint_available")
            else "missing / not required"
        )
        + (
            f"  strict_load_compatible={ck.get('strict_load_compatible_with_v1')}"
            if "strict_load_compatible_with_v1" in ck
            else ""
        )
    )
    print(f"EVALUATOR:      implemented={report['evaluation_metric_implemented']}")
    print("FULL TRAINING:  NOT STARTED")
    print("-" * 68)
    print(f"FINAL STATUS:   {report['final_status']}")
    if splits["blockers"]:
        print(f"  split blockers:  {splits['blockers']}")
    if report["dataset_checks"]["blockers"]:
        print(f"  export blockers: {report['dataset_checks']['blockers']}")
    if report["dataset_checks"]["real_dataset_blockers"]:
        print(f"  dataset gate:    {report['dataset_checks']['real_dataset_blockers']}")
    if report["config_errors"]:
        print(f"  config errors:   {report['config_errors']}")
    print("=" * 68)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    try:
        output_dir.relative_to(REPO_ROOT)
        print(
            f"ERROR: --output-dir {output_dir} is inside the repository; "
            "preflight artifacts must live outside git",
            file=sys.stderr,
        )
        return 2
    except ValueError:
        pass

    report = build_report(args)
    _write_atomic(report, output_dir / "preflight_report.json")
    _print_summary(report)
    print(f"\nwrote {output_dir / 'preflight_report.json'}")
    return 0 if report["final_status"] == STATUS_READY else 1


if __name__ == "__main__":
    raise SystemExit(main())
