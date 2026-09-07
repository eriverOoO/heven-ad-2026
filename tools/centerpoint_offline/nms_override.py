"""Validated, opt-in NMS override handling for the historical OpenPCDet run."""
from __future__ import annotations

def effective_nms_threshold(configured: float, override: float | None) -> float:
    value = configured if override is None else override
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError('NMS threshold must be in [0, 1]')
    return float(value)

def apply_nms_threshold(model, override: float | None) -> float:
    cfg = model.dense_head.model_cfg.POST_PROCESSING.NMS_CONFIG
    value = effective_nms_threshold(float(cfg.NMS_THRESH), override)
    cfg.NMS_THRESH = value
    return value
