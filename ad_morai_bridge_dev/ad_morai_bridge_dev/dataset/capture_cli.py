"""``ad_morai_dataset_capture`` - one deterministic scenario capture run.

Sequence: load catalog -> derive seed parameters -> (optionally) reset the
MORAI scenario over gRPC -> settle -> capture N LiDAR-anchored frames ->
finalise the run and dataset manifests.

MORAI is optional: ``--no-reset`` captures from an already-running graph and
``--dry-run`` builds and records the reset plan / run config without ROS.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ad_morai_bridge_dev.dataset.scenario import (
    ScenarioCatalog,
    build_reset_plan,
    default_catalog_path,
    derive_run_parameters,
)
from ad_morai_bridge_dev.dataset.schema import (
    ACTOR_GT_TOPIC,
    ACTOR_GT_TYPE,
    EGO_GT_TOPIC,
    EGO_GT_TYPE,
    LIDAR_TOPIC,
    LIDAR_TYPE,
    CLASS_MAP_VERSION,
    DatasetPaths,
    RUN_STATUS_ABORTED,
    SkewConfig,
)
from ad_morai_bridge_dev.dataset.capture import CaptureLimits
from ad_morai_bridge_dev.dataset.writer import RunWriter, repo_provenance, write_dataset_documents


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _next_run_id(paths: DatasetPaths, scenario_id: str, seed: int, overwrite: bool) -> str:
    base = f"{scenario_id}__seed_{seed}__run_"
    scenario_dir = paths.runs_dir / scenario_id
    existing = (
        {p.name for p in scenario_dir.iterdir() if p.is_dir()}
        if scenario_dir.is_dir()
        else set()
    )
    for index in range(1000):
        candidate = f"{base}{index:03d}"
        if candidate not in existing or overwrite:
            return candidate
    raise RuntimeError("exhausted run indices")


def _reset_plan_summary(plan: Any) -> dict[str, Any]:
    return {
        "ego": {
            "actor_id": plan.ego.actor_id,
            "location": list(plan.ego.pose.location),
            "rotation": list(plan.ego.pose.rotation),
            "velocity": plan.ego.velocity,
        },
        "actors": [
            {
                "actor_id": a.actor_id,
                "object_type": a.object_type,
                "location": list(a.pose.location),
                "rotation": list(a.pose.rotation),
                "velocity": a.velocity,
            }
            for a in plan.actors
        ],
    }


def _build_run_config(
    scenario: Any,
    parameters: Any,
    plan: Any,
    limits: CaptureLimits,
    skew: SkewConfig,
    dataset_id: str,
    reset_performed: bool,
) -> dict[str, Any]:
    return {
        "dataset_id": dataset_id,
        "class_map_version": CLASS_MAP_VERSION,
        "scenario": {
            "scenario_id": scenario.scenario_id,
            "purpose": scenario.purpose,
            "range_regime": scenario.range_regime,
            "scenario_file": str(scenario.scenario_file),
            "scenario_sha256": scenario.scenario_sha256,
        },
        "seed": parameters.as_dict(),
        "reset_plan": _reset_plan_summary(plan),
        "reset_performed": reset_performed,
        "limits": {
            "frame_count": limits.frame_count,
            "duration_ns": limits.duration_ns,
            "settle_ns": limits.settle_ns,
        },
        "skew": skew.as_dict(),
        "topic_contract": {
            LIDAR_TOPIC: LIDAR_TYPE,
            EGO_GT_TOPIC: EGO_GT_TYPE,
            ACTOR_GT_TOPIC: ACTOR_GT_TYPE,
        },
        **repo_provenance(_repository_root()),
    }


def _collect_run_manifests(paths: DatasetPaths) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    if not paths.runs_dir.is_dir():
        return manifests
    for scenario_dir in sorted(p for p in paths.runs_dir.iterdir() if p.is_dir()):
        for run_dir in sorted(p for p in scenario_dir.iterdir() if p.is_dir()):
            manifest_path = run_dir / "run_manifest.json"
            if manifest_path.is_file():
                data = json.loads(manifest_path.read_text("utf-8"))
                if data.get("status") != "running":
                    manifests.append(data)
    return manifests


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dataset-id", default="morai_tracking_v1")
    parser.add_argument("--catalog", type=Path, default=default_catalog_path())
    parser.add_argument("--frames", type=int)
    parser.add_argument("--duration-sec", type=float)
    parser.add_argument("--settle-sec", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-reset", action="store_true", help="capture without a gRPC reset")
    parser.add_argument("--dry-run", action="store_true", help="build the plan, do not capture")
    parser.add_argument("--grpc-target", default="127.0.0.1:7789")
    parser.add_argument("--grpc-timeout-sec", type=float, default=5.0)
    parser.add_argument("--max-ego-skew-ms", type=float, default=30.0)
    parser.add_argument("--max-actor-gt-skew-ms", type=float, default=30.0)
    parser.add_argument("--max-tf-skew-ms", type=float, default=30.0)
    parser.add_argument("--list-scenarios", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    catalog = ScenarioCatalog.load(args.catalog)

    if args.list_scenarios:
        for scenario in catalog:
            print(
                f"{scenario.scenario_id:24s} {scenario.range_regime:6s} "
                f"{scenario.purpose}"
            )
        return 0

    if not args.scenario or not args.output:
        print("--scenario and --output are required", file=sys.stderr)
        return 2
    if not args.dry_run and args.frames is None and args.duration_sec is None:
        print("--frames or --duration-sec is required", file=sys.stderr)
        return 2

    scenario = catalog.get(args.scenario)
    parameters = derive_run_parameters(scenario, args.seed)
    plan = build_reset_plan(scenario, parameters)
    if args.dry_run:
        limits = CaptureLimits(frame_count=1, settle_ns=0)
    else:
        limits = CaptureLimits(
            frame_count=args.frames,
            duration_ns=(
                None if args.duration_sec is None else round(args.duration_sec * 1e9)
            ),
            settle_ns=round(args.settle_sec * 1e9),
        )
    skew = SkewConfig(
        max_ego_skew_ns=round(args.max_ego_skew_ms * 1e6),
        max_actor_gt_skew_ns=round(args.max_actor_gt_skew_ms * 1e6),
        max_tf_skew_ns=round(args.max_tf_skew_ms * 1e6),
    )
    paths = DatasetPaths(args.output.resolve())
    run_id = _next_run_id(paths, scenario.scenario_id, args.seed, args.overwrite)

    reset_performed = False
    if not args.dry_run and not args.no_reset:
        reset_performed = _perform_reset(args, scenario, plan)

    run_config = _build_run_config(
        scenario, parameters, plan, limits, skew, args.dataset_id, reset_performed
    )
    writer = RunWriter(
        paths, scenario.scenario_id, run_id, run_config, overwrite=args.overwrite
    )

    if args.dry_run:
        writer.finalize(
            RUN_STATUS_ABORTED,
            "dry_run",
            {"rejected_frames": {}, "final_phase": "dry_run", "point_field_schema": []},
        )
        write_dataset_documents(
            paths, args.dataset_id, repo_provenance(_repository_root()),
            _collect_run_manifests(paths),
        )
        print(json.dumps(run_config, indent=2, sort_keys=True))
        return 0

    status = _run_capture(writer, limits, skew, scenario)
    write_dataset_documents(
        paths, args.dataset_id, repo_provenance(_repository_root()),
        _collect_run_manifests(paths),
    )
    print(f"run {run_id}: {status}")
    return 0 if status in ("complete",) else 1


def _perform_reset(args: argparse.Namespace, scenario: Any, plan: Any) -> bool:
    try:
        from ad_morai_bridge_dev.scenarios.reset import reset_scenario
        from ad_morai_bridge_dev.scenarios.setup import load_scenario_and_resume
        from ad_morai_bridge_dev.simulator_grpc.client import MoraiGrpcClient
        from ad_morai_bridge_dev.simulator_grpc.descriptors import MoraiApi
    except ImportError as exc:
        raise RuntimeError(
            f"gRPC reset requested but MORAI client unavailable: {exc}; use --no-reset"
        ) from exc
    client = MoraiGrpcClient.connect(
        MoraiApi.load(), args.grpc_target, default_timeout=args.grpc_timeout_sec
    )
    try:
        load_scenario_and_resume(client, str(scenario.scenario_file), args.grpc_timeout_sec)
        reset_scenario(client, plan, args.grpc_timeout_sec)
    finally:
        client.close()
    return True


def _run_capture(writer: RunWriter, limits: CaptureLimits, skew: SkewConfig, scenario: Any) -> str:
    import rclpy

    from ad_morai_bridge_dev.dataset.capture_node import DatasetCaptureNode, spin_capture

    rclpy.init()
    node = None
    try:
        node = DatasetCaptureNode(
            writer, limits, skew, str(scenario.scenario_file), scenario.scenario_sha256
        )
        return spin_capture(node)
    finally:
        if node is not None:
            node.abort("shutdown")
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
