"""Loader + coordinate-alignment for the frozen MORAI GT-based
state-estimator evaluation stream (T-11/T-12 lineage), reused read-only
as a zero-shot transfer evaluation target for AV2-pretrained KalmanNet
checkpoints.

**Provenance, verified directly against real local files, not assumed
(see docs/perception/kalmannet_av2_stage1_v1.md's "MORAI frozen stream
provenance" section for the full audit):**

- Source: ``~/heven_presentation_assets/state_estimator_gt_comparison/
  sequences.pkl`` -- 29 real MORAI GT sequences (18 actors), built from
  the already-exported ``~/datasets/morai_heven/`` + T-11's canonical
  dataset (``~/heven_presentation_assets/motion_gt_expansion/canonical/``).
  No new MORAI capture in this task.
- Split (frozen BEFORE any T-12 training, per
  ``motion_gt_expansion/canonical/split_policy.json``'s own
  ``"policy_frozen_before_t12": true``): train actors
  ``[1,13,15,18,34,35,36,38,40,44,46,48]``, val actors
  ``[21,23,27,31]``, **test actors ``[16,20,30]``** (3 sequences, 157
  total frames, 24 missing measurements -- 15.3% missing rate).
- Frame: ``lidar_link`` (ego-relative), ordered by real
  ``ground_truth.boxes[]`` ``header_stamp_ns``.
- Measurement: real matched Euclidean-detector output only
  (``z=[x,y]``), missing frames preserved as ``None`` (predict-only),
  never interpolated or GT-filled.
- **Verified never used to train or tune anything this task reuses**:
  `kalmannet_training_provenance.json` (T-12's own KalmanNet) and
  `frozen_model_v2_manifest.json`/T-12.2's own STATUS record ("Split/
  dense data: T-12's frozen split reused unmodified") both confirm test
  actors 16/20/30 were never in `train_actors`/`val_actors` for either
  T-12's own checkpoint or DENSE-KALMANNET-v2. `selected_kf_config.json`
  confirms the Tuned Linear KF was selected on `val_actors` only.
  T-13's own ROS replay window happened to contain only actors 2/3
  (both on the separate `excluded_actor_ids` list, never train/val/test)
  -- test actors 16/20/30 were never exercised there either.

**Coordinate alignment**: this stream's raw positions are ego-relative
`lidar_link` meters (tens of meters, e.g. actor 16's first sample is
`(-2.80, 10.39)`) -- NOT AV2's `first_state_relative` convention (every
AV2 training sequence's own frame-0 position is exactly `(0,0)`). To
give an AV2-pretrained checkpoint a fair, apples-to-apples input (the
same relative-motion representation it was trained on, not an unfamiliar
absolute offset), this module applies the identical
first-state-relative shift AV2 sequences already carry: subtract each
sequence's own first GT position from every GT and measurement position
in that sequence. This is a coordinate-frame equalization only -- no
GT is used as a measurement, no future information leaks backward (the
shift uses only each sequence's OWN frame 0, same as AV2's own
construction), and velocities are untouched (a constant position offset
does not change any velocity).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

DEFAULT_SEQUENCES_PATH = Path.home() / "heven_presentation_assets/state_estimator_gt_comparison/sequences.pkl"
FROZEN_TEST_ACTOR_IDS = (16, 20, 30)


def load_morai_frozen_test_sequences(
    sequences_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Loads the raw pickle, filters to ``role == "test"`` (the 3
    sequences for actors 16/20/30), and re-shapes each into the exact
    ``{"actor_id","frames","dt","x_true","z_meas"}`` contract
    ``trainer_core.run_sequence``/``batched_kalmannet.build_padded_batch``
    already expect -- no adapter beyond a dict-key subset + the
    coordinate shift below."""
    path = Path(sequences_path) if sequences_path is not None else DEFAULT_SEQUENCES_PATH
    with open(path, "rb") as f:
        raw = pickle.load(f)

    test_raw = [s for s in raw if s.get("role") == "test"]
    actor_ids = sorted({s["actor_id"] for s in test_raw})
    if actor_ids != sorted(FROZEN_TEST_ACTOR_IDS):
        raise ValueError(
            f"frozen MORAI test actor set changed: expected {sorted(FROZEN_TEST_ACTOR_IDS)}, "
            f"found {actor_ids} -- this stream must not be silently substituted"
        )

    out = []
    for s in test_raw:
        out.append(_to_first_state_relative(s))
    return out


def _to_first_state_relative(seq: dict[str, Any]) -> dict[str, Any]:
    x0, y0 = seq["x_true"][0][0], seq["x_true"][0][1]
    x_true = [[x - x0, y - y0, vx, vy] for x, y, vx, vy in seq["x_true"]]
    z_meas = [None if z is None else [z[0] - x0, z[1] - y0] for z in seq["z_meas"]]
    return {
        "actor_id": seq["actor_id"],
        "frames": list(seq["frames"]),
        "dt": list(seq["dt"]),
        "x_true": x_true,
        "z_meas": z_meas,
        "morai_origin_xy": (x0, y0),  # kept for provenance/debugging only, never read by run_sequence
    }
