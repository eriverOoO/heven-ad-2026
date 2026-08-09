"""Fail-closed, development-only MORAI actor operations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import json
import math
from typing import Any

from ad_morai_bridge_dev.simulator_grpc.client import MoraiGrpcClient


GET_AVAILABLE_OBJECT = "morai_sim_api.simualtor.Simulator/GetAvailableObject"
GET_ALL_ACTORS_STATE = "morai_sim_api.actor.Actor/GetAllActorsState"
GET_ACTOR_STATE = "morai_sim_api.actor.Actor/GetActorState"
SPAWN_VEHICLE = "morai_sim_api.actor.Actor/SpawnVehicle"
DESTROY_ACTOR = "morai_sim_api.actor.Actor/DestroyActor"
SET_PAUSE = "morai_sim_api.actor.Actor/SetPause"
SET_TRANSFORM = "morai_sim_api.actor.Actor/SetTransform"
SET_VELOCITY = "morai_sim_api.actor.Actor/SetVelocity"
SET_VEHICLE_ROUTE = "morai_sim_api.actor.Actor/SetVehicleRoute"

VEHICLE_OBJECT_TYPE = "OBJECT_TYPE_VEHICLE"
ALLOWED_ACTOR_METHODS = frozenset(
    {
        GET_AVAILABLE_OBJECT,
        GET_ALL_ACTORS_STATE,
        GET_ACTOR_STATE,
        SPAWN_VEHICLE,
        DESTROY_ACTOR,
        SET_PAUSE,
        SET_TRANSFORM,
        SET_VELOCITY,
        SET_VEHICLE_ROUTE,
    }
)

_ALL_ACTORS_FILTER = {
    "vehicle": True,
    "pedestrian": True,
    "obstacle": True,
}


@dataclass(frozen=True)
class ActorRef:
    actor_id: str
    object_type: str
    client_key: str

    def __post_init__(self) -> None:
        _nonempty_string(self.actor_id, "actor_id")
        if self.object_type not in {
            "OBJECT_TYPE_VEHICLE",
            "OBJECT_TYPE_PEDESTRIAN",
            "OBJECT_TYPE_OBSTACLE",
        }:
            raise ValueError(f"unsupported actor object_type: {self.object_type!r}")
        if not isinstance(self.client_key, str):
            raise ValueError("client_key must be a string")

    def to_grpc(self) -> dict[str, object]:
        return {
            "id": {"value": self.actor_id},
            "object_type": self.object_type,
            "client_key": self.client_key,
        }

    def to_dict(self) -> dict[str, str]:
        return {
            "actor_id": self.actor_id,
            "object_type": self.object_type,
            "client_key": self.client_key,
        }

    @classmethod
    def from_dict(cls, value: object) -> "ActorRef":
        mapping = _require_mapping(value, "actor manifest actor")
        if set(mapping) != {"actor_id", "object_type", "client_key"}:
            raise ValueError("actor manifest has unexpected actor fields")
        return cls(
            mapping["actor_id"],
            mapping["object_type"],
            mapping["client_key"],
        )


@dataclass(frozen=True)
class ActorState:
    actor: ActorRef
    xyz: tuple[float, float, float]
    rpy_deg: tuple[float, float, float]
    velocity_xyz: tuple[float, float, float]
    global_velocity_xyz: tuple[float, float, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "actor": self.actor.to_dict(),
            "xyz": list(self.xyz),
            "rpy_deg": list(self.rpy_deg),
            "velocity_xyz": list(self.velocity_xyz),
            "global_velocity_xyz": list(self.global_velocity_xyz),
        }


@dataclass(frozen=True)
class SpawnedActor:
    actor: ActorRef
    request_id: str
    model_name: str
    label: str
    verification_state: str = "verified"

    def __post_init__(self) -> None:
        if self.actor.object_type != VEHICLE_OBJECT_TYPE:
            raise ValueError("spawned actor must be a vehicle")
        _nonempty_string(self.actor.client_key, "spawned actor client_key")
        _nonempty_string(self.request_id, "request_id")
        _nonempty_string(self.model_name, "model_name")
        _nonempty_string(self.label, "label")
        if self.verification_state not in {"pending", "verified"}:
            raise ValueError("spawn verification_state must be pending or verified")

    def to_manifest(self) -> dict[str, object]:
        return {
            "schema_version": 2,
            "actor": self.actor.to_dict(),
            "request_id": self.request_id,
            "model_name": self.model_name,
            "label": self.label,
            "verification_state": self.verification_state,
            "reconciliation_required": self.verification_state != "verified",
        }

    @classmethod
    def from_manifest(cls, value: object) -> "SpawnedActor":
        mapping = _require_mapping(value, "spawn manifest")
        version = mapping.get("schema_version")
        version_one = {
            "schema_version",
            "actor",
            "request_id",
            "model_name",
            "label",
        }
        version_two = version_one | {
            "verification_state",
            "reconciliation_required",
        }
        if version == 1 and set(mapping) == version_one:
            verification_state = "verified"
        elif version == 2 and set(mapping) == version_two:
            verification_state = mapping.get("verification_state")
            expected_reconciliation = verification_state != "verified"
            if mapping.get("reconciliation_required") is not expected_reconciliation:
                raise ValueError("spawn manifest reconciliation flag is inconsistent")
        else:
            raise ValueError("invalid spawn manifest schema")
        return cls(
            actor=ActorRef.from_dict(mapping["actor"]),
            request_id=mapping["request_id"],
            model_name=mapping["model_name"],
            label=mapping["label"],
            verification_state=verification_state,
        )


class SpawnVerificationError(RuntimeError):
    """Spawn succeeded but exact post-spawn identity verification did not."""

    def __init__(self, spawned_actor: SpawnedActor, detail: str) -> None:
        super().__init__(
            f"spawned actor {spawned_actor.actor.actor_id!r} requires reconciliation: "
            f"{detail}"
        )
        self.spawned_actor = spawned_actor


class MoraiActorController:
    """Typed actor API whose transport is exclusively ``call_json``."""

    def __init__(
        self,
        client: MoraiGrpcClient,
        *,
        client_key: str,
        ego_ref: ActorRef,
        timeout_sec: float = 5.0,
    ) -> None:
        _nonempty_string(client_key, "client_key")
        _require_actor_ref(ego_ref)
        if ego_ref.object_type != VEHICLE_OBJECT_TYPE:
            raise ValueError("configured Ego must be a vehicle ActorRef")
        _finite_positive(timeout_sec, "timeout_sec")
        self._client = client
        self._client_key = client_key
        self._ego_ref = ego_ref
        self._timeout_sec = float(timeout_sec)
        self._created: dict[str, SpawnedActor] = {}
        self._verified_ego: ActorRef | None = None
        self._closed = False

    @property
    def client_key(self) -> str:
        return self._client_key

    @property
    def created_manifest(self) -> tuple[SpawnedActor, ...]:
        return tuple(self._created.values())

    def list_vehicle_models(self) -> tuple[str, ...]:
        response = self._rpc(
            GET_AVAILABLE_OBJECT,
            {
                "vehicle": True,
                "pedestrian": False,
                "obstacle": False,
                "spawn_point": False,
                "map_object": False,
            },
        )
        models = response.get("surround_vehicle")
        if not isinstance(models, list) or not all(
            isinstance(model, str) and model for model in models
        ):
            raise RuntimeError("MORAI vehicle model response is invalid")
        if len(models) != len(set(models)):
            raise RuntimeError("MORAI vehicle model response contains duplicates")
        return tuple(sorted(models))

    def list_actors(self) -> tuple[ActorState, ...]:
        response = self._rpc(GET_ALL_ACTORS_STATE, _ALL_ACTORS_FILTER)
        raw_states = response.get("states", [])
        if not isinstance(raw_states, list):
            raise RuntimeError("actor query response has no states array")
        states = tuple(_parse_state(value) for value in raw_states)
        actor_ids = [state.actor.actor_id for state in states]
        if len(actor_ids) != len(set(actor_ids)):
            raise RuntimeError("actor query contains duplicate actor ID")
        return states

    def discover_vehicle(self, actor_id: str) -> ActorRef:
        _nonempty_string(actor_id, "actor_id")
        matches = [
            state.actor
            for state in self.list_actors()
            if state.actor.actor_id == actor_id
            and state.actor.object_type == VEHICLE_OBJECT_TYPE
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"vehicle discovery requires one exact actor ID {actor_id!r}"
            )
        return matches[0]

    def discover_ego(self) -> ActorRef:
        matches = [
            state.actor
            for state in self.list_actors()
            if state.actor.actor_id == self._ego_ref.actor_id
        ]
        if matches != [self._ego_ref]:
            raise RuntimeError("MORAI actor list does not contain configured Ego identity")
        self._verified_ego = self._ego_ref
        return self._ego_ref

    def get_state(self, actor: ActorRef) -> ActorState:
        _require_actor_ref(actor)
        state = _parse_state(self._rpc(GET_ACTOR_STATE, actor.to_grpc()))
        if state.actor != actor:
            raise RuntimeError("actor state identity does not match requested actor")
        return state

    def spawn_npc(
        self,
        *,
        request_id: str,
        model_name: str,
        label: str,
        xyz: Sequence[float],
        rpy_deg: Sequence[float],
        velocity: float,
        paused: bool = False,
    ) -> SpawnedActor:
        _nonempty_string(request_id, "request_id")
        _nonempty_string(model_name, "model_name")
        _nonempty_string(label, "label")
        location = _finite_triplet(xyz, "xyz")
        rotation = _finite_triplet(rpy_deg, "rpy_deg")
        initial_velocity = _finite_number(velocity, "velocity")
        if initial_velocity < 0.0:
            raise ValueError("velocity must be nonnegative")
        if not isinstance(paused, bool):
            raise ValueError("paused must be a boolean")

        existing_ids = {state.actor.actor_id for state in self.list_actors()}
        if request_id in existing_ids:
            raise RuntimeError("spawn request ID already exists")
        request_ref = ActorRef(request_id, VEHICLE_OBJECT_TYPE, self._client_key)
        response = self._result_rpc(
            SPAWN_VEHICLE,
            build_spawn_request(
                request_ref,
                model_name=model_name,
                label=label,
                xyz=location,
                rpy_deg=rotation,
                velocity=initial_velocity,
                paused=paused,
            ),
        )
        actual_id = response.get("custom_message")
        if not isinstance(actual_id, str) or not actual_id:
            raise RuntimeError("spawn result did not return the actual actor ID")
        actual_ref = ActorRef(actual_id, VEHICLE_OBJECT_TYPE, self._client_key)
        pending = SpawnedActor(
            actual_ref,
            request_id,
            model_name,
            label,
            verification_state="pending",
        )
        if actual_id in self._created:
            raise SpawnVerificationError(
                pending, "runtime actor ID is already present in this manifest"
            )
        self._created[actual_id] = pending
        try:
            if actual_id in existing_ids:
                raise RuntimeError("spawn result reused a pre-existing actor ID")
            self.get_state(actual_ref)
        except Exception as exc:
            raise SpawnVerificationError(pending, str(exc)) from exc
        created = replace(pending, verification_state="verified")
        self._created[actual_id] = created
        return created

    def register_created(self, created: SpawnedActor) -> None:
        if not isinstance(created, SpawnedActor):
            raise ValueError("created must be a SpawnedActor")
        self._require_owned_ref(created.actor)
        self.get_state(created.actor)
        created = replace(created, verification_state="verified")
        current = self._created.get(created.actor.actor_id)
        if current is not None and current != created:
            raise RuntimeError("created manifest actor ID is already registered")
        self._created[created.actor.actor_id] = created

    def pause_actor(self, actor: ActorRef) -> None:
        self._require_mutable(actor)
        self._result_rpc(
            SET_PAUSE,
            {"actor_info": actor.to_grpc(), "enable": True},
        )
        self.get_state(actor)

    def resume_actor(self, actor: ActorRef) -> None:
        self._require_mutable(actor)
        self._result_rpc(
            SET_PAUSE,
            {"actor_info": actor.to_grpc(), "enable": False},
        )
        self.get_state(actor)

    def hold_ego(self, actor: ActorRef) -> None:
        self._require_configured_ego(actor)
        self._result_rpc(
            SET_VELOCITY,
            {"actor_info": actor.to_grpc(), "velocity": 0.0},
        )
        self.get_state(actor)
        self.pause_actor(actor)

    def set_transform(
        self,
        actor: ActorRef,
        *,
        xyz: Sequence[float],
        rpy_deg: Sequence[float],
    ) -> ActorState:
        self._require_mutable(actor)
        location = _finite_triplet(xyz, "xyz")
        rotation = _finite_triplet(rpy_deg, "rpy_deg")
        self._result_rpc(
            SET_TRANSFORM,
            {
                "actor_info": actor.to_grpc(),
                "transform": _transform_payload(location, rotation),
            },
        )
        state = self.get_state(actor)
        if state.xyz != location or state.rpy_deg != rotation:
            raise RuntimeError("actor transform postcondition was not observed")
        return state

    def route_actor(
        self,
        actor: ActorRef,
        *,
        decision_range: float,
        links: Sequence[tuple[str, int]],
    ) -> None:
        route = _route_payload(actor, decision_range, links)
        self._require_mutable(actor)
        self._result_rpc(SET_VEHICLE_ROUTE, route)
        self.get_state(actor)

    def destroy_created(self, created: SpawnedActor) -> bool:
        if not isinstance(created, SpawnedActor):
            raise ValueError("created must be a SpawnedActor")
        self._require_owned_ref(created.actor)
        matches = [
            state.actor
            for state in self.list_actors()
            if state.actor.actor_id == created.actor.actor_id
        ]
        if not matches:
            self._created.pop(created.actor.actor_id, None)
            return False
        if matches != [created.actor]:
            raise RuntimeError("created actor identity mismatch; refusing destroy")
        self._result_rpc(DESTROY_ACTOR, created.actor.to_grpc())
        remaining = [
            state.actor
            for state in self.list_actors()
            if state.actor.actor_id == created.actor.actor_id
        ]
        if remaining:
            raise RuntimeError("destroyed actor is still present")
        self._created.pop(created.actor.actor_id, None)
        return True

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._client.close()

    def _result_rpc(
        self, method: str, payload: Mapping[str, object]
    ) -> dict[str, Any]:
        response = self._rpc(method, payload)
        if not _is_success_status(response.get("status")):
            description = response.get("description")
            raise RuntimeError(
                str(description) if description else f"MORAI rejected {method}"
            )
        return response

    def _rpc(
        self, method: str, payload: Mapping[str, object]
    ) -> dict[str, Any]:
        if method not in ALLOWED_ACTOR_METHODS:
            raise RuntimeError("prohibited MORAI actor RPC")
        if self._closed:
            raise RuntimeError("MORAI actor controller is closed")
        result = self._client.call_json(
            method,
            json.dumps(payload, allow_nan=False, separators=(",", ":")),
            self._timeout_sec,
        )
        if not result.success:
            detail = result.error or result.status or "MORAI gRPC failed"
            raise RuntimeError(detail)
        try:
            response = json.loads(result.response_json or "{}")
        except (TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError("MORAI gRPC returned invalid JSON") from exc
        if not isinstance(response, dict):
            raise RuntimeError("MORAI gRPC response must be an object")
        return response

    def _require_owned_ref(self, actor: ActorRef) -> None:
        _require_actor_ref(actor)
        if actor.object_type != VEHICLE_OBJECT_TYPE:
            raise ValueError("only vehicle actors are supported")
        if actor.client_key != self._client_key:
            raise RuntimeError("actor is not owned by the configured client_key")

    def _require_configured_ego(self, actor: ActorRef) -> None:
        _require_actor_ref(actor)
        if actor != self._verified_ego or actor != self._ego_ref:
            raise RuntimeError("mutation requires the exact configured Ego identity")

    def _require_mutable(self, actor: ActorRef) -> None:
        if actor == self._verified_ego and actor == self._ego_ref:
            return
        self._require_owned_ref(actor)
        created = self._created.get(actor.actor_id)
        if created is None or created.actor != actor:
            raise RuntimeError("actor mutation requires a created manifest")


def build_spawn_request(
    actor: ActorRef,
    *,
    model_name: str,
    label: str,
    xyz: Sequence[float],
    rpy_deg: Sequence[float],
    velocity: float,
    paused: bool,
) -> dict[str, object]:
    location = _finite_triplet(xyz, "xyz")
    rotation = _finite_triplet(rpy_deg, "rpy_deg")
    speed = _finite_number(velocity, "velocity")
    if speed < 0.0:
        raise ValueError("velocity must be nonnegative")
    _nonempty_string(model_name, "model_name")
    _nonempty_string(label, "label")
    if actor.object_type != VEHICLE_OBJECT_TYPE or not actor.client_key:
        raise ValueError("spawn requires an owned vehicle ActorRef")
    if not isinstance(paused, bool):
        raise ValueError("paused must be a boolean")
    return {
        "spawn_info": {
            "actor_info": actor.to_grpc(),
            "transform": _transform_payload(location, rotation),
            "model_name": model_name,
            "label": label,
            "is_multi_object_one_mode": False,
        },
        "velocity": speed,
        "pause": paused,
        "multi_ego": False,
    }


def build_route_request(
    actor: ActorRef,
    *,
    decision_range: float,
    links: Sequence[tuple[str, int]],
) -> dict[str, object]:
    return _route_payload(actor, decision_range, links)


def _route_payload(
    actor: ActorRef,
    decision_range: float,
    links: Sequence[tuple[str, int]],
) -> dict[str, object]:
    _require_actor_ref(actor)
    distance = _finite_number(decision_range, "decision_range")
    if distance <= 0.0:
        raise ValueError("decision_range must be positive")
    if not isinstance(links, Sequence) or isinstance(links, (str, bytes)) or not links:
        raise ValueError("route links must be nonempty")
    canonical_links = []
    for item in links:
        if not isinstance(item, Sequence) or len(item) != 2:
            raise ValueError("each route link must contain ID and waypoint index")
        link_id, waypoint_idx = item
        _nonempty_string(link_id, "route link ID")
        if isinstance(waypoint_idx, bool) or not isinstance(waypoint_idx, int):
            raise ValueError("route waypoint_idx must be an integer")
        if waypoint_idx < 0:
            raise ValueError("route waypoint_idx must be nonnegative")
        canonical_links.append(
            {"id": {"value": link_id}, "waypoint_idx": waypoint_idx}
        )
    return {
        "actor_info": actor.to_grpc(),
        "decision_range": distance,
        "links": canonical_links,
    }


def _parse_state(value: object) -> ActorState:
    mapping = _require_mapping(value, "actor state")
    actor = _parse_ref(mapping.get("actor_info"))
    transform = _require_mapping(mapping.get("transform"), "actor transform")
    location = _require_mapping(transform.get("location"), "actor location")
    rotation = _require_mapping(transform.get("rotation"), "actor rotation")
    velocity = _require_mapping(mapping.get("velocity"), "actor velocity")
    global_velocity = _require_mapping(
        mapping.get("global_velocity"), "actor global velocity"
    )
    return ActorState(
        actor=actor,
        xyz=_vector_from_mapping(location, "actor location"),
        rpy_deg=_vector_from_mapping(rotation, "actor rotation"),
        velocity_xyz=_vector_from_mapping(velocity, "actor velocity"),
        global_velocity_xyz=_vector_from_mapping(
            global_velocity, "actor global velocity"
        ),
    )


def _parse_ref(value: object) -> ActorRef:
    mapping = _require_mapping(value, "actor_info")
    identifier = _require_mapping(mapping.get("id"), "actor identifier")
    return ActorRef(
        identifier.get("value"),
        mapping.get("object_type"),
        mapping.get("client_key", ""),
    )


def _transform_payload(
    xyz: tuple[float, float, float],
    rpy_deg: tuple[float, float, float],
) -> dict[str, dict[str, float]]:
    return {
        "location": dict(zip(("x", "y", "z"), xyz)),
        "rotation": dict(zip(("x", "y", "z"), rpy_deg)),
    }


def _vector_from_mapping(
    mapping: Mapping[str, object], name: str
) -> tuple[float, float, float]:
    if set(mapping) != {"x", "y", "z"}:
        raise ValueError(f"{name} must contain exactly x, y, and z")
    return tuple(_finite_number(mapping[axis], name) for axis in "xyz")


def _finite_triplet(value: Sequence[float], name: str) -> tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise ValueError(f"{name} must contain three finite values")
    return tuple(_finite_number(item, name) for item in value)


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _finite_positive(value: object, name: str) -> float:
    converted = _finite_number(value, name)
    if converted <= 0.0:
        raise ValueError(f"{name} must be positive")
    return converted


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _require_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _require_actor_ref(value: object) -> ActorRef:
    if not isinstance(value, ActorRef):
        raise ValueError("actor must be an ActorRef")
    return value


def _is_success_status(value: object) -> bool:
    return value == "STATUS_CODE_SUCCESS" or (type(value) is int and value == 1)
