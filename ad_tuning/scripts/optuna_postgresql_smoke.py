#!/usr/bin/env python3
"""Exercise shared-study initialization and one concurrent worker claim."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

import optuna

from ad_tuning.search import (
    create_profile_stanley_study,
    suggest_profile_stanley_parameters,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fingerprint", required=True)
    parser.add_argument("--claim", action="store_true")
    parser.add_argument("--sleep-sec", type=float, default=2.0)
    arguments = parser.parse_args()

    worker_id = os.environ["AD_TUNING_WORKER_ID"]
    study = create_profile_stanley_study(
        Path("/tmp/ad_tuning_postgresql_smoke"),
        arguments.fingerprint,
        20260728,
        storage_url="",
        worker_id=worker_id,
        heartbeat_interval_s=1,
        heartbeat_grace_period_s=10,
        heartbeat_retry_count=0,
        connect_timeout_s=3,
        experiment_metadata={
            "purpose": "tailscale_three_worker_postgresql_smoke",
            "version": 1,
        },
    )

    if arguments.claim:

        def objective(trial: optuna.Trial) -> float:
            parameters = suggest_profile_stanley_parameters(trial)
            trial.set_user_attr("smoke_worker", worker_id)
            trial.set_user_attr("constraint_values", [0.0, -1.0])
            time.sleep(arguments.sleep_sec)
            print(
                f"worker={worker_id} claimed={trial.number} "
                f"speed_kp={parameters['profile_stanley.speed_pid.kp']}"
            )
            return float(trial.number)

        study.optimize(objective, n_trials=1, show_progress_bar=False)

    trials = [
        {
            "number": trial.number,
            "state": trial.state.name,
            "worker": trial.user_attrs.get("smoke_worker", ""),
        }
        for trial in study.trials
    ]
    print(
        f"worker={worker_id} coordinator="
        f"{study.user_attrs.get('warm_start_coordinator')} "
        f"warm_starts={study.user_attrs.get('warm_start_count')} "
        f"trials={trials}"
    )


if __name__ == "__main__":
    main()
