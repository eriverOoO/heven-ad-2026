"""Minimal OpenPCDet imports for the offline CenterPoint smoke test."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np


def _namespace_package(name: str, path: Path) -> None:
    """Install a package shell without executing that package's ``__init__``."""
    if name in sys.modules:
        raise RuntimeError(f"{name} was imported before minimal OpenPCDet setup")
    package = types.ModuleType(name)
    package.__file__ = str(path / "__init__.py")
    package.__package__ = name
    package.__path__ = [str(path)]
    package.__spec__ = importlib.machinery.ModuleSpec(
        name=name, loader=None, is_package=True
    )
    package.__spec__.submodule_search_locations = package.__path__
    sys.modules[name] = package


def import_smoke_components(openpcdet_root: Path) -> tuple[type, type]:
    """Import only the dataset base and configured CenterPoint detector."""
    root = openpcdet_root.resolve()
    pcdet_root = root / "pcdet"
    required = (
        pcdet_root / "__init__.py",
        pcdet_root / "datasets" / "dataset.py",
        pcdet_root / "models" / "detectors" / "centerpoint.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"invalid OpenPCDet checkout; missing: {missing}")

    root_string = str(root)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)
    importlib.import_module("pcdet")
    _namespace_package("pcdet.datasets", pcdet_root / "datasets")
    _namespace_package("pcdet.models", pcdet_root / "models")
    _namespace_package("pcdet.models.detectors", pcdet_root / "models" / "detectors")

    dataset_template = importlib.import_module(
        "pcdet.datasets.dataset"
    ).DatasetTemplate
    centerpoint = importlib.import_module(
        "pcdet.models.detectors.centerpoint"
    ).CenterPoint
    return dataset_template, centerpoint


def load_batch_to_cuda(batch: dict[str, Any]) -> None:
    """Move the LiDAR-only batch fields used by CenterPoint to CUDA."""
    import torch

    for key, value in batch.items():
        if not isinstance(value, np.ndarray) or key == "frame_id":
            continue
        batch[key] = torch.from_numpy(value).float().cuda()
