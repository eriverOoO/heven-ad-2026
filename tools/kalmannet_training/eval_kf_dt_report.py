#!/usr/bin/env python3
"""LinearCVKF baseline on the EXACT SAME temporally-thinned streams the
KalmanNet checkpoints are evaluated on (task section 20). LinearCVKF's
analytical F(dt)/Q(dt) already consume each retained transition's real
variable dt, so no KF change is needed -- this just measures whether the
classical filter stays stable across variable dt where the learned gain
might not (outcome E).

AV2-tuned KF params (sigma_a=5.0, r_std=0.300095) are reused from the
Stage-1 pilot's own `calibrate_kf_on_av2` result (GENERIC-ROBUST, AV2
TRAIN+VAL only, held-out test never touched). Every prior AV2 KalmanNet
task converged on sigma_a=5.0 / r_std~=0.30 -- the residual sigma_z is
essentially identical across AV2 splits, so re-calibrating on the frozen
10k train set (a ~15 GB load on this host) would give the same numbers.

usage: eval_kf_dt_report.py <shard_root> <split_manifest> <out.json> <aug_seed> <policy>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from av2_split import load_split_manifest  # noqa: E402
from variable_dt import (  # noqa: E402
    POLICY_CHOICES,
    load_split_sequences_with_thinning,
    evaluate_with_dt_buckets,
)
from evaluate_kalmannet import run_kf  # noqa: E402
from ad_morai_bridge_dev.dataset.av2_motion_forecasting_adapter import CorruptionConfig  # noqa: E402

AV2_TUNED_KF = dict(sigma_a=5.0, r_std=0.300095, p0_scale=10.0)

GENERIC_ROBUST = CorruptionConfig(
    mode="custom",
    gaussian_noise_std_m=0.3,
    dropout_prob=0.1,
    dropout_burst_enabled=True,
    dropout_burst_prob=0.02,
    dropout_burst_min_len=2,
    dropout_burst_max_len=5,
    seed=20260905,
)


def main() -> int:
    shard_root = Path(sys.argv[1])
    split_manifest_path = Path(sys.argv[2])
    output_path = Path(sys.argv[3])
    augmentation_seed = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    only_policy = sys.argv[5] if len(sys.argv) > 5 else None

    split_manifest = load_split_manifest(split_manifest_path)
    policies = {only_policy: POLICY_CHOICES[only_policy]} if only_policy else POLICY_CHOICES

    report: dict = {}
    for name, policy in policies.items():
        seqs = load_split_sequences_with_thinning(
            shard_root, split_manifest, policy, augmentation_seed, GENERIC_ROBUST,
        )["val"]
        report[name] = evaluate_with_dt_buckets(seqs, lambda s: run_kf(s, **AV2_TUNED_KF))
        o = report[name]["overall"]
        print(f"KF {name}: overall n={o['n']} pos_rmse={o['position_rmse_m']:.4f} "
              f"vel_rmse={o['velocity_rmse_mps']:.4f}")
        del seqs

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
