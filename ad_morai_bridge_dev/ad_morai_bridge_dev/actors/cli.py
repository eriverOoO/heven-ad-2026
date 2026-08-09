"""Command-line interface for the development-only safe MORAI actor API."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

from ad_morai_bridge_dev.actors.control import (
    ActorRef,
    DESTROY_ACTOR,
    GET_ALL_ACTORS_STATE,
    GET_AVAILABLE_OBJECT,
    MoraiActorController,
    SET_PAUSE,
    SET_TRANSFORM,
    SET_VEHICLE_ROUTE,
    SET_VELOCITY,
    SPAWN_VEHICLE,
    SpawnedActor,
    SpawnVerificationError,
    VEHICLE_OBJECT_TYPE,
    build_route_request,
    build_spawn_request,
)
from ad_morai_bridge_dev.actors.presets import ActorPreset, load_presets


SOURCE_PRESETS = Path(__file__).resolve().parents[1] / "config" / "actor_presets.yaml"


class ManifestWriteError(RuntimeError):
    def __init__(self, spawned_actor: SpawnedActor, detail: str) -> None:
        super().__init__(f"spawn manifest write failed: {detail}")
        self.spawned_actor = spawned_actor


@dataclass
class _ManifestDestination:
    target: Path
    temporary: Path

    @classmethod
    def prepare(cls, target: Path) -> "_ManifestDestination":
        target = target.resolve()
        if target.exists():
            raise ValueError(f"spawn manifest destination already exists: {target}")
        if not target.parent.is_dir():
            raise ValueError(
                f"spawn manifest parent is not a directory: {target.parent}"
            )
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
        )
        os.close(descriptor)
        return cls(target=target, temporary=Path(temporary))

    def write(self, manifest: dict[str, object]) -> None:
        with self.temporary.open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(self.temporary, self.target)
        self.temporary.unlink()

    def cancel(self) -> None:
        try:
            self.temporary.unlink()
        except FileNotFoundError:
            pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grpc-target", default="127.0.0.1:7789")
    parser.add_argument("--timeout-sec", type=float, default=5.0)
    parser.add_argument("--client-key", default="heven-task4-actor-toolkit")
    parser.add_argument("--ego-id", default="Ego")
    parser.add_argument("--ego-client-key", default="")
    parser.add_argument("--dry-run", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("models", help="list available NPC vehicle models")
    list_parser = subparsers.add_parser("list", help="list actor states")
    list_parser.add_argument("--owned-only", action="store_true")

    state = subparsers.add_parser("state", help="query one exact vehicle state")
    state.add_argument("--actor-id", required=True)

    subparsers.add_parser("hold-ego", help="zero and pause configured Ego actor")

    teleport = subparsers.add_parser(
        "teleport-ego", help="hold and move exact Ego actor"
    )
    _add_transform_arguments(teleport)

    spawn = subparsers.add_parser("spawn-npc", help="spawn one owned NPC")
    spawn_source = spawn.add_mutually_exclusive_group(required=True)
    spawn_source.add_argument("--preset")
    spawn_source.add_argument("--request-id")
    spawn.add_argument("--preset-config", type=Path, default=_default_presets())
    spawn.add_argument("--model-name")
    spawn.add_argument("--label")
    spawn.add_argument("--xyz", nargs=3, type=float)
    spawn.add_argument("--rpy-deg", nargs=3, type=float)
    spawn.add_argument("--velocity", type=float)
    spawn.add_argument("--paused", action="store_true")
    spawn.add_argument("--manifest-out", type=Path)

    route = subparsers.add_parser("route-npc", help="route one created NPC")
    route.add_argument("--manifest", type=Path, required=True)
    route_source = route.add_mutually_exclusive_group(required=True)
    route_source.add_argument("--preset")
    route_source.add_argument("--link", action="append")
    route.add_argument("--preset-config", type=Path, default=_default_presets())
    route.add_argument("--decision-range", type=float)

    pause = subparsers.add_parser("pause-actor", help="pause one created actor")
    pause.add_argument("--manifest", type=Path, required=True)
    resume = subparsers.add_parser("resume-actor", help="resume one created actor")
    resume.add_argument("--manifest", type=Path, required=True)
    destroy = subparsers.add_parser(
        "destroy-created", help="destroy exactly one spawn manifest actor"
    )
    destroy.add_argument("--manifest", type=Path, required=True)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    connect: Callable[[str, float], Any] | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    if args.dry_run:
        _print_json({"dry_run": True, "operations": _dry_run_operations(args)})
        return 0

    _validate_runtime_config(args)
    connector = connect or _connect
    client = connector(args.grpc_target, args.timeout_sec)
    controller = MoraiActorController(
        client,
        client_key=args.client_key,
        ego_ref=ActorRef(
            args.ego_id,
            VEHICLE_OBJECT_TYPE,
            args.ego_client_key,
        ),
        timeout_sec=args.timeout_sec,
    )
    try:
        try:
            output = _execute(controller, args)
        except SpawnVerificationError as exc:
            _print_json(
                {
                    "error": "spawn_verification_failed",
                    "detail": str(exc),
                    "manifest": exc.spawned_actor.to_manifest(),
                },
                stream=sys.stderr,
            )
            return 2
        except ManifestWriteError as exc:
            _print_json(
                {
                    "error": "manifest_write_failed",
                    "detail": str(exc),
                    "manifest": exc.spawned_actor.to_manifest(),
                },
                stream=sys.stderr,
            )
            return 3
        _print_json(output)
    finally:
        controller.close()
    return 0


def _execute(controller: MoraiActorController, args: argparse.Namespace) -> object:
    if args.command == "models":
        return {"models": list(controller.list_vehicle_models())}
    if args.command == "list":
        states = controller.list_actors()
        if args.owned_only:
            states = tuple(
                state
                for state in states
                if state.actor.client_key == controller.client_key
            )
        return {"actors": [state.to_dict() for state in states]}
    if args.command == "state":
        actor = controller.discover_vehicle(args.actor_id)
        return controller.get_state(actor).to_dict()
    if args.command == "hold-ego":
        actor = controller.discover_ego()
        controller.hold_ego(actor)
        return {"held": actor.to_dict()}
    if args.command == "teleport-ego":
        actor = controller.discover_ego()
        controller.hold_ego(actor)
        state = controller.set_transform(
            actor, xyz=args.xyz, rpy_deg=args.rpy_deg
        )
        return state.to_dict()
    if args.command == "spawn-npc":
        spec = _spawn_spec(args)
        _require_cli_client_key(args.client_key, spec.get("client_key"))
        destination = (
            _ManifestDestination.prepare(args.manifest_out)
            if args.manifest_out is not None
            else None
        )
        try:
            created = controller.spawn_npc(
                request_id=spec["request_id"],
                model_name=spec["model_name"],
                label=spec["label"],
                xyz=spec["xyz"],
                rpy_deg=spec["rpy_deg"],
                velocity=spec["velocity"],
                paused=args.paused,
            )
        except SpawnVerificationError as exc:
            if destination is not None:
                try:
                    destination.write(exc.spawned_actor.to_manifest())
                except OSError:
                    destination.cancel()
            raise
        except Exception:
            if destination is not None:
                destination.cancel()
            raise
        manifest = created.to_manifest()
        if destination is not None:
            try:
                destination.write(manifest)
            except OSError as exc:
                destination.cancel()
                raise ManifestWriteError(created, str(exc)) from exc
        return manifest
    if args.command in {"route-npc", "pause-actor", "resume-actor"}:
        created = _load_manifest(args.manifest)
        _require_cli_client_key(args.client_key, created.actor.client_key)
        controller.register_created(created)
        if args.command == "route-npc":
            decision_range, links = _route_spec(args)
            controller.route_actor(
                created.actor,
                decision_range=decision_range,
                links=links,
            )
            return {"routed": created.actor.to_dict()}
        if args.command == "pause-actor":
            controller.pause_actor(created.actor)
            return {"paused": created.actor.to_dict()}
        controller.resume_actor(created.actor)
        return {"resumed": created.actor.to_dict()}
    if args.command == "destroy-created":
        created = _load_manifest(args.manifest)
        _require_cli_client_key(args.client_key, created.actor.client_key)
        return {
            "actor": created.actor.to_dict(),
            "destroyed": controller.destroy_created(created),
        }
    raise AssertionError(f"unhandled command: {args.command}")


def _dry_run_operations(args: argparse.Namespace) -> list[dict[str, object]]:
    if args.command == "models":
        return [
            _operation(
                GET_AVAILABLE_OBJECT,
                {
                    "vehicle": True,
                    "pedestrian": False,
                    "obstacle": False,
                    "spawn_point": False,
                    "map_object": False,
                },
            )
        ]
    if args.command == "list":
        return [
            _operation(
                GET_ALL_ACTORS_STATE,
                {"vehicle": True, "pedestrian": True, "obstacle": True},
            )
        ]
    if args.command == "state":
        return [_actor_discovery_operation(args.actor_id)]
    if args.command in {"hold-ego", "teleport-ego"}:
        operations = [
            _configured_ego_discovery_operation(
                ActorRef(
                    args.ego_id,
                    VEHICLE_OBJECT_TYPE,
                    args.ego_client_key,
                )
            )
        ]
        operations.extend(
            [
                _deferred_actor_operation(
                    SET_VELOCITY,
                    {"velocity": 0.0},
                ),
                _deferred_actor_operation(
                    SET_PAUSE,
                    {"enable": True},
                ),
            ]
        )
        if args.command == "teleport-ego":
            operations.append(
                _deferred_actor_operation(
                    SET_TRANSFORM,
                    {
                        "transform": {
                            "location": _xyz_dict(args.xyz),
                            "rotation": _xyz_dict(args.rpy_deg),
                        },
                    },
                )
            )
        return operations
    if args.command == "spawn-npc":
        spec = _spawn_spec(args)
        _require_cli_client_key(args.client_key, spec.get("client_key"))
        actor = ActorRef(spec["request_id"], VEHICLE_OBJECT_TYPE, args.client_key)
        request = build_spawn_request(
            actor,
            model_name=spec["model_name"],
            label=spec["label"],
            xyz=spec["xyz"],
            rpy_deg=spec["rpy_deg"],
            velocity=spec["velocity"],
            paused=args.paused,
        )
        return [_operation(SPAWN_VEHICLE, request)]
    created = _load_manifest(args.manifest)
    _require_cli_client_key(args.client_key, created.actor.client_key)
    if args.command == "route-npc":
        decision_range, links = _route_spec(args)
        return [
            _operation(
                SET_VEHICLE_ROUTE,
                build_route_request(
                    created.actor,
                    decision_range=decision_range,
                    links=links,
                ),
            )
        ]
    if args.command == "pause-actor":
        return [
            _operation(
                SET_PAUSE,
                {"actor_info": created.actor.to_grpc(), "enable": True},
            )
        ]
    if args.command == "resume-actor":
        return [
            _operation(
                SET_PAUSE,
                {"actor_info": created.actor.to_grpc(), "enable": False},
            )
        ]
    if args.command == "destroy-created":
        return [_operation(DESTROY_ACTOR, created.actor.to_grpc())]
    raise AssertionError(f"unhandled command: {args.command}")


def _spawn_spec(args: argparse.Namespace) -> dict[str, object]:
    if args.preset:
        preset = _preset(args.preset_config, args.preset)
        return {
            "client_key": preset.client_key,
            "request_id": preset.request_id,
            "model_name": preset.model_name,
            "label": preset.label,
            "xyz": preset.xyz,
            "rpy_deg": preset.rpy_deg,
            "velocity": preset.velocity,
        }
    missing = [
        name
        for name in ("model_name", "label", "xyz", "rpy_deg", "velocity")
        if getattr(args, name) is None
    ]
    if missing:
        raise ValueError(
            "manual spawn requires " + ", ".join(name.replace("_", "-") for name in missing)
        )
    return {
        "request_id": args.request_id,
        "model_name": args.model_name,
        "label": args.label,
        "xyz": args.xyz,
        "rpy_deg": args.rpy_deg,
        "velocity": args.velocity,
    }


def _route_spec(args: argparse.Namespace) -> tuple[float, tuple[tuple[str, int], ...]]:
    if args.preset:
        preset = _preset(args.preset_config, args.preset)
        _require_cli_client_key(args.client_key, preset.client_key)
        return preset.decision_range, preset.route_links
    if args.decision_range is None:
        raise ValueError("manual route requires --decision-range")
    return args.decision_range, tuple(_parse_link(value) for value in args.link)


def _preset(path: Path, name: str) -> ActorPreset:
    presets = load_presets(path)
    try:
        return presets[name]
    except KeyError as exc:
        raise ValueError(f"unknown actor preset: {name}") from exc


def _parse_link(value: str) -> tuple[str, int]:
    link_id, separator, raw_index = value.partition(":")
    if not link_id:
        raise ValueError("route link ID must be nonempty")
    try:
        waypoint_idx = int(raw_index) if separator else 0
    except ValueError as exc:
        raise ValueError(f"invalid route waypoint index: {value}") from exc
    return link_id, waypoint_idx


def _load_manifest(path: Path) -> SpawnedActor:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load spawn manifest {path}: {exc}") from exc
    return SpawnedActor.from_manifest(value)


def _require_cli_client_key(configured: str, expected: object) -> None:
    if expected is not None and configured != expected:
        raise ValueError(
            "configured --client-key does not match the preset/spawn manifest"
        )


def _operation(method: str, request: dict[str, object]) -> dict[str, object]:
    return {"method": method, "request": request}


def _actor_discovery_operation(actor_id: str) -> dict[str, object]:
    operation = _operation(
        GET_ALL_ACTORS_STATE,
        {
            "vehicle": True,
            "pedestrian": True,
            "obstacle": True,
        },
    )
    operation["select_exact_vehicle_id"] = actor_id
    return operation


def _configured_ego_discovery_operation(actor: ActorRef) -> dict[str, object]:
    operation = _operation(
        GET_ALL_ACTORS_STATE,
        {"vehicle": True, "pedestrian": True, "obstacle": True},
    )
    operation["select_exact_ego_identity"] = actor.to_grpc()
    return operation


def _deferred_actor_operation(
    method: str, remaining_request: dict[str, object]
) -> dict[str, object]:
    return {
        "method": method,
        "actor_info_from_operation": 0,
        "request_after_exact_match": remaining_request,
    }


def _xyz_dict(values: Sequence[float]) -> dict[str, float]:
    return dict(zip(("x", "y", "z"), map(float, values)))


def _add_transform_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--xyz", nargs=3, type=float, required=True)
    parser.add_argument("--rpy-deg", nargs=3, type=float, required=True)


def _print_json(value: object, *, stream=None) -> None:
    print(
        json.dumps(value, sort_keys=True, separators=(",", ":")),
        file=stream,
    )


def _validate_runtime_config(args: argparse.Namespace) -> None:
    if not math.isfinite(args.timeout_sec) or args.timeout_sec <= 0.0:
        raise ValueError("--timeout-sec must be finite and positive")
    if not args.client_key.strip():
        raise ValueError("--client-key must be nonempty")
    if not args.grpc_target.strip():
        raise ValueError("--grpc-target must be nonempty")


def _connect(target: str, timeout_sec: float):
    from ad_morai_bridge_dev.simulator_grpc.client import MoraiGrpcClient
    from ad_morai_bridge_dev.simulator_grpc.descriptors import MoraiApi

    return MoraiGrpcClient.connect(
        MoraiApi.load(), target, default_timeout=timeout_sec
    )


def _default_presets() -> Path:
    if SOURCE_PRESETS.is_file():
        return SOURCE_PRESETS
    from ament_index_python.packages import get_package_share_directory

    return (
        Path(get_package_share_directory("ad_morai_bridge_dev"))
        / "config"
        / "actor_presets.yaml"
    )


if __name__ == "__main__":
    raise SystemExit(main())
