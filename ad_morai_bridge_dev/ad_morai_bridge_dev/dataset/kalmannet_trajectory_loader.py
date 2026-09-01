"""Lightweight NumPy-only reader for a ``kalmannet_morai_trajectory_v1`` export.

No torch, no ROS, no OpenPCDet. Consumers (Linear KF replay, a future
KalmanNet trainer) use this to pull one trajectory's time-major arrays and,
if they want, the exact ``[x, y, vx, vy]`` / ``[x, y]`` shapes the research
``kalmannet_core.Sequence`` expects.

This module trains nothing and makes no estimator claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

STATE_DIM = 4  # [x, y, vx, vy] - mirrors kalmannet_core.STATE_DIM
MEAS_DIM = 2   # [x, y]         - mirrors kalmannet_core.MEAS_DIM


@dataclass
class TrajectoryArrays:
    trajectory_id: str
    timestamps_ns: np.ndarray       # int64  [T]
    dt_s: np.ndarray                # float64 [T]  (dt_s[0] is nan)
    gt_state: np.ndarray            # float64 [T, 4]  [x, y, vx, vy]
    gt_position_map: np.ndarray     # float64 [T, 3]
    gt_velocity_map: np.ndarray     # float64 [T, 3]
    gt_yaw: np.ndarray              # float64 [T]
    measurement: np.ndarray         # float64 [T, 2]  (all nan in v1)
    measurement_valid: np.ndarray   # bool    [T]     (all False in v1)
    source_frame_indices: np.ndarray  # int64 [T]
    source_sample_ids: np.ndarray   # <U     [T]
    raw: dict[str, np.ndarray]      # every stored array, verbatim

    @property
    def length(self) -> int:
        return int(self.timestamps_ns.shape[0])

    def to_kalmannet_sequence(self) -> dict[str, Any]:
        """A dict matching ``kalmannet_core.Sequence``'s field contract.

        Returned rather than the dataclass itself so this module keeps no
        dependency on ``ad_lidar_perception``. ``dt[0]`` is ``None`` and
        ``z_meas[k]`` is ``None`` wherever ``measurement_valid[k]`` is
        False (every frame, in a v1 export).
        """

        length = self.length
        dt: list[Any] = [None]
        dt.extend(float(v) for v in self.dt_s[1:])
        x_true = [[float(v) for v in row] for row in self.gt_state]
        z_meas: list[Any] = []
        for k in range(length):
            if bool(self.measurement_valid[k]):
                z_meas.append([float(v) for v in self.measurement[k]])
            else:
                z_meas.append(None)
        return {
            "actor_id": self.trajectory_id,
            "frames": list(range(length)),
            "dt": dt,
            "x_true": x_true,
            "z_meas": z_meas,
        }


def load_trajectory(path: Path | str) -> TrajectoryArrays:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        raw = {name: np.array(data[name]) for name in data.files}
    return TrajectoryArrays(
        trajectory_id=path.stem,
        timestamps_ns=raw["timestamps_ns"],
        dt_s=raw["dt_s"],
        gt_state=raw["gt_state"],
        gt_position_map=raw["gt_position_map"],
        gt_velocity_map=raw["gt_velocity_map"],
        gt_yaw=raw["gt_yaw"],
        measurement=raw["measurement"],
        measurement_valid=raw["measurement_valid"].astype(bool),
        source_frame_indices=raw["source_frame_indices"],
        source_sample_ids=raw["source_sample_ids"],
        raw=raw,
    )


def iter_split(export_root: Path | str, split: str) -> list[TrajectoryArrays]:
    export_root = Path(export_root)
    split_file = export_root / "splits" / f"{split}.txt"
    ids = [
        line.strip()
        for line in split_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [
        load_trajectory(export_root / "trajectories" / f"{tid}.npz") for tid in ids
    ]
