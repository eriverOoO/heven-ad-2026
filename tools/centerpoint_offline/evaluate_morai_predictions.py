#!/usr/bin/env python3
"""Standalone CenterPoint MORAI detector evaluator (no OpenPCDet, no torch).

    python3 tools/centerpoint_offline/evaluate_morai_predictions.py \
        --dataset <centerpoint_morai_adapter_v1 export> \
        --split val \
        --predictions <predictions.jsonl> \
        --output <evaluation.json>

Predictions: one JSON object per line. Either a bare
``{"sample_id": ..., "detections": [{"class_name", "score", "box_lidar":[x,y,z,l,w,h,yaw]}]}``
or a ``heven.offline_detection.v1`` record (same shape, plus a ``schema`` field).
Invalid prediction boxes are excluded and counted, not fatal.

Defaults to the ``val`` split. Reading ``train`` or ``test`` requires an
explicit opt-in flag - the historical 100%-overlap failure mode must not be
reachable through this tool's normal path.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from morai_dataset import MoraiHevenDatasetCore
from morai_evaluator import (
    MORAI_EVALUATOR_SCHEMA_VERSION,
    MoraiEvaluatorError,
    evaluate_detections,
    sample_gt_from_label,
    sample_pred_from_arrays,
)

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
HEVEN_OFFLINE_SCHEMA = "heven.offline_detection.v1"


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


def run_adapter_validator(dataset_root: Path) -> dict[str, Any]:
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
        report = json.loads(completed.stdout)
        report["invoked"] = True
        return report
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        return {"invoked": False, "error": f"{type(error).__name__}: {error}"}


def read_predictions(path: Path) -> dict[str, dict[str, list]]:
    """Lenient per-sample prediction reader (see module docstring)."""
    by_sample: dict[str, dict[str, list]] = {}
    for line_number, raw in enumerate(path.read_text("utf-8").splitlines(), start=1):
        raw = raw.strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as error:
            raise MoraiEvaluatorError(f"predictions line {line_number}: {error}")
        schema = payload.get("schema")
        if schema is not None and schema != HEVEN_OFFLINE_SCHEMA:
            raise MoraiEvaluatorError(
                f"predictions line {line_number}: unknown schema {schema!r}"
            )
        sample_id = str(payload["sample_id"])
        if sample_id in by_sample:
            raise MoraiEvaluatorError(
                f"duplicate prediction record for sample {sample_id!r}"
            )
        names, scores, boxes = [], [], []
        for det in payload.get("detections", []):
            names.append(str(det["class_name"]))
            scores.append(float(det["score"]))
            boxes.append([float(v) for v in det["box_lidar"][:7]])
        by_sample[sample_id] = {"names": names, "scores": scores, "boxes": boxes}
    return by_sample


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--class-name", default="vehicle")
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="permit --split test (normally reserved for a single final eval)",
    )
    parser.add_argument(
        "--allow-train",
        action="store_true",
        help="permit --split train (diagnostic only; never a validation path)",
    )
    parser.add_argument(
        "--skip-leakage-check",
        action="store_true",
        help="do not invoke the adapter split-leakage validator (not recommended)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.split == "test" and not args.allow_test:
        print(
            "refusing to evaluate the reserved 'test' split; pass --allow-test",
            file=sys.stderr,
        )
        return 2
    if args.split == "train" and not args.allow_train:
        print(
            "refusing to evaluate 'train' as validation; pass --allow-train "
            "(diagnostic only)",
            file=sys.stderr,
        )
        return 2

    if not args.skip_leakage_check:
        report = run_adapter_validator(args.dataset)
        if not report.get("invoked"):
            print(f"adapter validator could not run: {report.get('error')}", file=sys.stderr)
            return 2
        if not report.get("ok"):
            leak = [e for e in report.get("errors", []) if "leakage" in e or "multiple splits" in e]
            print(
                f"adapter export invalid ({len(report.get('errors', []))} errors"
                + (f"; leakage: {leak}" if leak else "")
                + "); refusing to evaluate",
                file=sys.stderr,
            )
            return 2

    core = MoraiHevenDatasetCore(args.dataset, split=args.split)
    metadata = core.metadata
    gt_by_sample = {
        sample_id: sample_gt_from_label(core._load_label(sample_id))
        for sample_id in core.sample_ids
    }
    raw_preds = read_predictions(args.predictions)
    unknown = sorted(set(raw_preds) - set(gt_by_sample))
    if unknown:
        print(
            f"{len(unknown)} prediction sample(s) not in split {args.split!r}: "
            f"{unknown[:5]}",
            file=sys.stderr,
        )
        return 2
    pred_by_sample = {
        sample_id: sample_pred_from_arrays(
            entry["names"], entry["scores"], np.asarray(entry["boxes"]).reshape(-1, 7)
            if entry["boxes"]
            else np.zeros((0, 7)),
        )
        for sample_id, entry in raw_preds.items()
    }

    started = time.perf_counter()
    try:
        result = evaluate_detections(
            gt_by_sample,
            pred_by_sample,
            class_name=args.class_name,
            provenance={
                "evaluator_commit": _git_commit(),
                "dataset_root": str(args.dataset.resolve()),
                "dataset_version": metadata.get("dataset_version"),
                "adapter_schema_version": metadata.get("adapter_schema_version"),
                "split": args.split,
                "split_sample_count": len(core),
                "prediction_file": str(args.predictions),
                "prediction_sample_count": len(pred_by_sample),
            },
        )
    except MoraiEvaluatorError as error:
        print(f"evaluation failed: {error}", file=sys.stderr)
        return 1
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    payload = result.to_dict()
    payload["evaluation_wall_ms"] = elapsed_ms
    payload["ms_per_frame"] = elapsed_ms / max(len(core), 1)
    print(result.result_string())
    print(
        f"\n[{MORAI_EVALUATOR_SCHEMA_VERSION}] {elapsed_ms:.1f} ms "
        f"({payload['ms_per_frame']:.2f} ms/frame over {len(core)} frames)"
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.output.with_name(f".{args.output.name}.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(args.output)
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
