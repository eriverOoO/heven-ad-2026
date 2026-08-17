#!/usr/bin/env python3
"""Minimal CUDA training path for the MORAI HEVEN CenterPoint adapter."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

from morai_dataset import make_openpcdet_dataset
from openpcdet_runtime import import_smoke_components, load_batch_to_cuda


HERE = Path(__file__).resolve().parent
DEFAULT_DATA_CONFIG = HERE / "configs" / "morai_heven_dataset.yaml"
DEFAULT_MODEL_CONFIG = HERE / "configs" / "morai_centerpoint_train.yaml"


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--openpcdet-root", type=Path, required=True)
    parser.add_argument("--epochs", type=positive_int, default=1)
    parser.add_argument("--batch-size", type=positive_int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--data-config", type=Path, default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--resume", type=Path, help="Resume all training state from a checkpoint.")
    parser.add_argument(
        "--summary-json",
        type=Path,
        help="Write a machine-readable dry-run summary atomically.",
    )
    parser.add_argument(
        "--max-iterations",
        type=positive_int,
        help="Optional global cap for a short pipeline smoke; omit for all epochs.",
    )
    args = parser.parse_args(argv)
    if args.workers < 0:
        parser.error("--workers must be non-negative")
    return args


def load_yaml_config(path: Path) -> dict[str, Any]:
    """Load a YAML mapping and resolve a local _BASE_CONFIG_ recursively."""
    path = path.resolve()
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    base_name = loaded.pop("_BASE_CONFIG_", None)
    if base_name is None:
        return loaded
    base_path = Path(base_name)
    if not base_path.is_absolute():
        base_path = path.parent / base_path
    merged = load_yaml_config(base_path)
    _merge_dict(merged, loaded)
    return merged


def _merge_dict(target: dict[str, Any], update: dict[str, Any]) -> None:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge_dict(target[key], value)
        else:
            target[key] = value


def checkpoint_paths(output_dir: Path, epoch: int) -> tuple[Path, Path]:
    if epoch < 1:
        raise ValueError("epoch must be positive")
    return output_dir / "checkpoint_latest.pth", output_dir / f"checkpoint_epoch_{epoch:03d}.pth"


def save_checkpoint_atomic(state: dict[str, Any], destination: Path) -> None:
    import torch

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        torch.save(state, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_atomic(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def restore_checkpoint(
    checkpoint_path: Path, model: Any, optimizer: Any, scheduler: Any
) -> dict[str, Any]:
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cuda")
    required = {
        "epoch", "iteration", "model_state", "optimizer_state", "scheduler_state"
    }
    missing = required.difference(checkpoint)
    if missing:
        raise KeyError(f"checkpoint is missing keys: {sorted(missing)}")
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    return checkpoint


def create_dataloader(dataset: Any, batch_size: int, workers: int, seed: int) -> Any:
    import torch

    generator = torch.Generator()
    generator.manual_seed(seed)

    def seed_worker(worker_id: int) -> None:
        worker_seed = (seed + worker_id) % (2**32)
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        collate_fn=dataset.collate_batch,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    import torch
    from easydict import EasyDict

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the CenterPoint training smoke")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    data_cfg = EasyDict(load_yaml_config(args.data_config))
    train_cfg = EasyDict(load_yaml_config(args.model_config))
    data_cfg.DATA_PATH = str(args.dataset.resolve())
    dataset_template, centerpoint_class = import_smoke_components(args.openpcdet_root)
    dataset_class = make_openpcdet_dataset(dataset_template)
    dataset = dataset_class(
        dataset_cfg=data_cfg,
        class_names=list(train_cfg.CLASS_NAMES),
        training=True,
        root_path=args.dataset.resolve(),
        logger=None,
    )
    dataloader = create_dataloader(dataset, args.batch_size, args.workers, args.seed)
    if len(dataloader) == 0:
        raise RuntimeError("training dataloader is empty")

    model = centerpoint_class(
        model_cfg=train_cfg.MODEL,
        num_class=len(train_cfg.CLASS_NAMES),
        dataset=dataset,
    ).cuda().train()
    optim_cfg = train_cfg.OPTIMIZATION
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(optim_cfg.LR), weight_decay=float(optim_cfg.WEIGHT_DECAY)
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(step) for step in optim_cfg.DECAY_STEP_LIST],
        gamma=float(optim_cfg.LR_DECAY),
    )
    start_epoch = 0
    global_iteration = 0
    losses: list[float] = []
    iteration_times: list[float] = []
    prior_wall_time = 0.0
    resume_success = False
    optimizer_state_restored = False
    scheduler_state_restored = False
    if args.resume is not None:
        restored = restore_checkpoint(args.resume.resolve(), model, optimizer, scheduler)
        start_epoch = int(restored["epoch"])
        global_iteration = int(restored["iteration"])
        history = restored.get("dry_run_history", {})
        losses = [float(value) for value in history.get("losses", [])]
        iteration_times = [float(value) for value in history.get("iteration_times", [])]
        prior_wall_time = float(history.get("wall_clock_seconds", 0.0))
        resume_success = True
        optimizer_state_restored = bool(optimizer.state_dict()["state"])
        scheduler_state_restored = (
            scheduler.state_dict() == restored["scheduler_state"]
        )
        print(
            f"resumed checkpoint={args.resume} epoch={start_epoch} "
            f"iteration={global_iteration}",
            flush=True,
        )
    if start_epoch >= args.epochs:
        raise ValueError(
            f"--epochs ({args.epochs}) must exceed restored epoch ({start_epoch})"
        )
    torch.cuda.reset_peak_memory_stats()
    run_started = time.perf_counter()
    post_resume_optimizer_step_success = False

    for epoch_index in range(start_epoch, args.epochs):
        epoch = epoch_index + 1
        for iteration, batch in enumerate(dataloader, start=1):
            if args.max_iterations is not None and global_iteration >= args.max_iterations:
                break
            load_batch_to_cuda(batch)
            optimizer.zero_grad(set_to_none=True)
            iteration_started = time.perf_counter()
            result, _, _ = model(batch)
            loss = result["loss"].mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch}, iteration {iteration}")
            loss.backward()
            parameter_before = None
            updated_parameter = None
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter_before = parameter.detach().clone()
                    updated_parameter = parameter
                    break
            optimizer.step()
            if updated_parameter is None or torch.equal(parameter_before, updated_parameter.detach()):
                raise RuntimeError(
                    f"optimizer did not update the sampled parameter at iteration {iteration}"
                )
            model.update_global_step()
            global_iteration += 1
            iteration_time = time.perf_counter() - iteration_started
            loss_value = float(loss.item())
            losses.append(loss_value)
            iteration_times.append(iteration_time)
            if args.resume is not None:
                post_resume_optimizer_step_success = True
            allocated = torch.cuda.memory_allocated() / (1024**2)
            peak = torch.cuda.max_memory_allocated() / (1024**2)
            print(
                f"epoch={epoch} iteration={iteration} loss={loss_value:.6f} "
                f"lr={optimizer.param_groups[0]['lr']:.8g} "
                f"gpu_allocated_mib={allocated:.1f} gpu_peak_mib={peak:.1f} "
                f"iteration_seconds={iteration_time:.3f}",
                flush=True,
            )
        scheduler.step()
        wall_clock_seconds = prior_wall_time + (time.perf_counter() - run_started)
        state = {
            "epoch": epoch,
            "iteration": global_iteration,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "seed": args.seed,
            "dry_run_history": {
                "losses": losses,
                "iteration_times": iteration_times,
                "wall_clock_seconds": wall_clock_seconds,
            },
        }
        latest, per_epoch = checkpoint_paths(args.output_dir, epoch)
        save_checkpoint_atomic(state, per_epoch)
        save_checkpoint_atomic(state, latest)
        if args.max_iterations is not None and global_iteration >= args.max_iterations:
            break

    if global_iteration == 0:
        raise RuntimeError("no training iteration was executed")
    if args.summary_json is not None:
        peak_mib = torch.cuda.max_memory_allocated() / (1024**2)
        summary = {
            "purpose": "training_pipeline_dry_run_only",
            "dataset_version": dataset.morai_core.dataset_version,
            "epochs_requested": args.epochs,
            "epochs_completed": epoch,
            "iterations_completed": global_iteration,
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "min_loss": min(losses),
            "max_loss": max(losses),
            "nan_or_inf_detected": False,
            "wall_clock_training_seconds": wall_clock_seconds,
            "average_iteration_seconds": sum(iteration_times) / len(iteration_times),
            "peak_vram_mib": peak_mib,
            "checkpoint_save_success": latest.is_file() and per_epoch.is_file(),
            "checkpoint_resume_success": resume_success,
            "optimizer_state_restored": optimizer_state_restored,
            "scheduler_state_restored": scheduler_state_restored,
            "post_resume_optimizer_step_success": post_resume_optimizer_step_success,
            "optimizer_parameter_update_verified": True,
            "exact_command": [sys.executable, *sys.argv],
            "output_directory": str(args.output_dir.resolve()),
        }
        write_json_atomic(summary, args.summary_json.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
