from dataclasses import dataclass
import json
import math
from pathlib import Path


PAUSE = "morai_sim_api.simulation.Simulation/Pause"
RESUME = "morai_sim_api.simulation.Simulation/Resume"
GET_ACTORS = "morai_sim_api.actor.Actor/GetAllActorsState"
CONTROL_VEHICLE = "morai_sim_api.actor.Actor/ControlVehicle"
SET_VELOCITY = "morai_sim_api.actor.Actor/SetVelocity"
SET_TRANSFORM = "morai_sim_api.actor.Actor/SetTransform"


@dataclass(frozen=True)
class Pose:
    location: tuple[float, float, float]
    rotation: tuple[float, float, float]

    def as_grpc_transform(self):
        return {
            "location": dict(zip(("x", "y", "z"), self.location)),
            "rotation": dict(zip(("x", "y", "z"), self.rotation)),
        }


@dataclass(frozen=True)
class ResetActor:
    actor_id: str
    object_type: str
    pose: Pose
    velocity: float | None = None


@dataclass(frozen=True)
class ResetPlan:
    ego: ResetActor
    actors: tuple[ResetActor, ...]


def _pose(position, rotation):
    return Pose(
        tuple(float(position[key]) for key in ("x", "y", "z")),
        tuple(float(rotation[key]) for key in ("roll", "pitch", "yaw")),
    )


def _pedestrian_pose(item):
    pose = _pose(item["pos"], item["rot"])
    return Pose((pose.location[0], pose.location[1], 0.0), pose.rotation)


def _nonnegative_velocity(item, *names):
    for name in names:
        if name not in item:
            continue
        velocity = float(item[name])
        if not math.isfinite(velocity) or velocity < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
        return velocity
    raise KeyError(names[0])


def load_reset_plan(path: Path) -> ResetPlan:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        initial = data["egoVehicle"]["initPosition"]
        ego = ResetActor(
            "Ego",
            "OBJECT_TYPE_VEHICLE",
            _pose(initial["pos"], initial["rot"]),
            0.0,
        )
        vehicles = tuple(
            ResetActor(
                str(item["UNIQUEID"]),
                "OBJECT_TYPE_VEHICLE",
                _pose(
                    item["initPosition"]["pos"],
                    item["initPosition"]["rot"],
                ),
                _nonnegative_velocity(
                    item,
                    "currentVelocity",
                    "viewCurrentVelocity",
                    "desiredVelocity",
                    "constantVelocity",
                ),
            )
            for item in data.get("vehicleList", [])
        )
        pedestrians = tuple(
            ResetActor(
                str(item["UNIQUEID"]),
                "OBJECT_TYPE_PEDESTRIAN",
                _pedestrian_pose(item),
                _nonnegative_velocity(item, "speed"),
            )
            for item in data.get("pedestrianList", [])
        )
        obstacles = tuple(
            ResetActor(
                str(item["UNIQUEID"]),
                "OBJECT_TYPE_OBSTACLE",
                _pose(item["pos"], item["rot"]),
            )
            for item in data.get("objectList", [])
        )
        return ResetPlan(ego, vehicles + pedestrians + obstacles)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid MORAI scenario JSON: {exc}") from exc


def _actor_info(actor):
    return {
        "id": {"value": actor.actor_id},
        "object_type": actor.object_type,
    }


def _call_result(client, method, payload, timeout_sec):
    result = client.call_json(method, json.dumps(payload), timeout_sec)
    if not result.success:
        raise RuntimeError(result.error or result.status)
    response = json.loads(result.response_json or "{}")
    if response.get("status") not in ("STATUS_CODE_SUCCESS", 1):
        raise RuntimeError(
            response.get("description") or f"MORAI rejected {method}"
        )


def reset_scenario(client, plan: ResetPlan, timeout_sec: float) -> None:
    paused = False
    try:
        _call_result(client, PAUSE, {}, timeout_sec)
        paused = True
        result = client.call_json(
            GET_ACTORS,
            json.dumps(
                {"vehicle": True, "pedestrian": True, "obstacle": True}
            ),
            timeout_sec,
        )
        if not result.success:
            raise RuntimeError(result.error or result.status)
        states = json.loads(result.response_json).get("states", [])
        present = {
            state["actor_info"]["id"]["value"] for state in states
        }
        required = {plan.ego.actor_id} | {
            actor.actor_id for actor in plan.actors
        }
        missing = sorted(required - present)
        if missing:
            raise RuntimeError(f"missing MORAI actors: {', '.join(missing)}")

        ego_info = _actor_info(plan.ego)
        _call_result(
            client,
            CONTROL_VEHICLE,
            {
                "actor_info": ego_info,
                "long_cmd_type": "LONG_CMD_TYPE_ACCELERATION",
                "throttle": 0.0,
                "brake": 0.0,
                "steer": 0.0,
            },
            timeout_sec,
        )
        _call_result(
            client,
            SET_VELOCITY,
            {"actor_info": ego_info, "velocity": 0.0},
            timeout_sec,
        )
        _call_result(
            client,
            SET_TRANSFORM,
            {
                "actor_info": ego_info,
                "transform": plan.ego.pose.as_grpc_transform(),
            },
            timeout_sec,
        )

        for actor in plan.actors:
            info = _actor_info(actor)
            if actor.velocity is not None:
                _call_result(
                    client,
                    SET_VELOCITY,
                    {
                        "actor_info": info,
                        "velocity": actor.velocity,
                    },
                    timeout_sec,
                )
            _call_result(
                client,
                SET_TRANSFORM,
                {
                    "actor_info": info,
                    "transform": actor.pose.as_grpc_transform(),
                },
                timeout_sec,
            )
    finally:
        if paused:
            _call_result(client, RESUME, {}, timeout_sec)
