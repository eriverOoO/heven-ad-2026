"""MORAI Tracking Dataset Factory.

Deterministic scenario / reset / capture pipeline that records the raw
point cloud plus source-grounded ego, TF and persistent simulator actor
ground truth required for later object-tracking evaluation, CenterPoint
data export and KalmanNet trajectory extraction.

No model is trained here and no production detector / tracker / planner
default is changed. See ``docs/perception/morai_tracking_dataset_factory_v1.md``.
"""

from ad_morai_bridge_dev.dataset.schema import (
    SCHEMA_VERSION,
    DatasetPaths,
    FrameRecord,
    frame_validity,
)

__all__ = [
    "SCHEMA_VERSION",
    "DatasetPaths",
    "FrameRecord",
    "frame_validity",
]
