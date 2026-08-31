"""``ad_morai_dataset_batch`` - manifest-driven sequential capture.

One machine, sequential deterministic runs. A ``dataset_plan.yaml``::

    dataset_id: morai_tracking_v1
    frames: 200
    settle_sec: 1.0
    runs:
      - scenario: lead_constant
        seeds: [0, 1, 2]
      - scenario: cut_in
        seeds: [0, 1]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from ad_morai_bridge_dev.dataset.capture_cli import main as capture_main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--no-reset", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args(argv)

    plan = yaml.safe_load(args.plan.read_text(encoding="utf-8"))
    dataset_id = str(plan.get("dataset_id", "morai_tracking_v1"))
    frames = plan.get("frames")
    duration_sec = plan.get("duration_sec")
    settle_sec = plan.get("settle_sec", 1.0)

    failures = 0
    for entry in plan.get("runs", []):
        scenario = str(entry["scenario"])
        for seed in entry.get("seeds", [0]):
            call = [
                "--scenario", scenario,
                "--seed", str(int(seed)),
                "--output", str(args.output),
                "--dataset-id", dataset_id,
                "--settle-sec", str(settle_sec),
            ]
            if frames is not None:
                call += ["--frames", str(int(frames))]
            if duration_sec is not None:
                call += ["--duration-sec", str(float(duration_sec))]
            if args.catalog is not None:
                call += ["--catalog", str(args.catalog)]
            if args.no_reset:
                call.append("--no-reset")
            if args.dry_run:
                call.append("--dry-run")
            print(f"=== {scenario} seed {seed} ===")
            code = capture_main(call)
            if code != 0:
                failures += 1
                if not args.continue_on_error:
                    return code
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
