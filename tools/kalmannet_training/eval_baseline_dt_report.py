#!/usr/bin/env python3
"""Evaluate an EXISTING frozen checkpoint (no retraining) under all three
dt policies (fixed/mild/strong), using the exact same
``variable_dt.load_split_sequences_with_thinning`` + ``evaluate_with_dt_buckets``
path ``train_kalmannet_variable_dt.py`` uses for its own
``--output-internal-val-dt-report`` -- so baseline A (the frozen FIXED-DT
2k checkpoint) and screening B/C are evaluated identically, INTERNAL VAL
only, never official AV2 VAL.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import checkpoint_utils  # noqa: E402
from av2_split import load_split_manifest  # noqa: E402
from variable_dt import (  # noqa: E402
    POLICY_CHOICES,
    load_split_sequences_with_thinning,
)
from ad_lidar_perception.kalmannet_core import KalmanNetGRU  # noqa: E402
from evaluate_kalmannet import run_knet  # noqa: E402
from variable_dt import evaluate_with_dt_buckets  # noqa: E402


def main() -> int:
    checkpoint_path = Path(sys.argv[1])
    shard_root = Path(sys.argv[2])
    split_manifest_path = Path(sys.argv[3])
    output_path = Path(sys.argv[4])
    augmentation_seed = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    # Optional 7th arg: a single policy name (fixed/mild/strong). On a
    # RAM-constrained host the full 3-policy loop OOMs on a 10k val split
    # (~15 GB), so the caller can run this once per policy in separate
    # processes and merge the per-policy JSON files afterward.
    only_policy = sys.argv[6] if len(sys.argv) > 6 else None

    state_dict, manifest = checkpoint_utils.load_checkpoint(checkpoint_path)
    hidden_size = manifest["model_config"]["hidden_size"]
    corruption_config_dict = manifest["corruption_config"]

    from ad_morai_bridge_dev.dataset.av2_motion_forecasting_adapter import CorruptionConfig

    corruption_config = CorruptionConfig(
        mode=corruption_config_dict.get("mode", "custom"),
        gaussian_noise_std_m=corruption_config_dict.get("gaussian_noise_std_m", 0.0),
        dropout_prob=corruption_config_dict.get("dropout_prob", 0.0),
        dropout_burst_enabled=corruption_config_dict.get("dropout_burst_enabled", False),
        dropout_burst_prob=corruption_config_dict.get("dropout_burst_prob", 0.0),
        dropout_burst_min_len=corruption_config_dict.get("dropout_burst_min_len", 2),
        dropout_burst_max_len=corruption_config_dict.get("dropout_burst_max_len", 5),
        seed=corruption_config_dict.get("seed", 20260905),
    )

    split_manifest = load_split_manifest(split_manifest_path)

    device = "cpu"
    net = KalmanNetGRU(hidden_size=hidden_size).to(device)
    net.load_state_dict(state_dict)
    net.eval()

    policies = {only_policy: POLICY_CHOICES[only_policy]} if only_policy else POLICY_CHOICES

    report: dict = {}
    for eval_policy_name, eval_policy in policies.items():
        eval_sequences = load_split_sequences_with_thinning(
            shard_root, split_manifest, eval_policy, augmentation_seed, corruption_config,
        )["val"]
        report[eval_policy_name] = evaluate_with_dt_buckets(
            eval_sequences, lambda s: run_knet(net, s, device)
        )
        print(f"{eval_policy_name}: overall n={report[eval_policy_name]['overall']['n']} "
              f"pos_rmse={report[eval_policy_name]['overall']['position_rmse_m']:.4f}")
        del eval_sequences

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
