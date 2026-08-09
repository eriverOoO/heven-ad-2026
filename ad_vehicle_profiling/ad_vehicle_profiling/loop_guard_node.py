from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Optional

import rclpy
from rclpy.node import Node

from ad_morai_bridge_dev.simulator_grpc.client import MoraiGrpcClient
from ad_morai_bridge_dev.simulator_grpc.descriptors import MoraiApi


PAUSE = "morai_sim_api.simulation.Simulation/Pause"
RESUME = "morai_sim_api.simulation.Simulation/Resume"
GET_ACTORS = "morai_sim_api.actor.Actor/GetAllActorsState"
CONTROL_VEHICLE = "morai_sim_api.actor.Actor/ControlVehicle"
SET_VELOCITY = "morai_sim_api.actor.Actor/SetVelocity"
SET_TRANSFORM = "morai_sim_api.actor.Actor/SetTransform"
SAFE_REPOSITION_PHASES = {
    "PREFLIGHT",
    "REACH_TARGET_SPEED",
    "RECOVER",
    "SETTLE",
}
_EVENT_FIELDS = (
    "unix_time_sec",
    "reason",
    "profiler_phase",
    "before_x_m",
    "before_y_m",
    "before_z_m",
    "before_speed_mps",
    "restored_speed_mps",
    "target_x_m",
    "target_y_m",
    "target_z_m",
)


@dataclass(frozen=True)
class GuardLimits:
    maximum_abs_x_m: float = 0.8
    minimum_y_m: float = -1000.0
    minimum_z_m: float = -0.5


@dataclass(frozen=True)
class ActorState:
    x_m: float
    y_m: float
    z_m: float


def reset_reason(
    state: ActorState,
    phase: str,
    limits: GuardLimits,
) -> Optional[str]:
    if state.z_m < limits.minimum_z_m:
        return "emergency_fall"
    if phase not in SAFE_REPOSITION_PHASES:
        return None
    if abs(state.x_m) > limits.maximum_abs_x_m:
        return "lateral_boundary"
    if state.y_m < limits.minimum_y_m:
        return "longitudinal_loop_boundary"
    return None


def _require_success(result, action: str) -> dict:
    if not result.success:
        raise RuntimeError(result.error or f"{action}: {result.status}")
    response = json.loads(result.response_json or "{}")
    if response.get("status") not in ("STATUS_CODE_SUCCESS", 1, None):
        raise RuntimeError(
            response.get("description") or f"MORAI rejected {action}"
        )
    return response


def _actor_info() -> dict:
    return {
        "id": {"value": "Ego"},
        "object_type": "OBJECT_TYPE_VEHICLE",
    }


def query_ego_state(client, timeout_sec: float) -> ActorState:
    result = client.call_json(
        GET_ACTORS,
        json.dumps(
            {"vehicle": True, "pedestrian": False, "obstacle": False}
        ),
        timeout_sec,
    )
    response = _require_success(result, "get actors")
    for state in response.get("states", []):
        actor_id = (
            state.get("actor_info", {})
            .get("id", {})
            .get("value", "")
        )
        if actor_id == "Ego":
            location = state["transform"]["location"]
            return ActorState(
                float(location["x"]),
                float(location["y"]),
                float(location["z"]),
            )
    raise RuntimeError("MORAI Ego actor is missing")


def reposition_ego(
    client,
    *,
    speed_mps: float,
    target_location: tuple[float, float, float],
    target_rotation_deg: tuple[float, float, float],
    timeout_sec: float,
) -> None:
    paused = False
    try:
        _require_success(
            client.call_json(PAUSE, "{}", timeout_sec), "pause"
        )
        paused = True
        actor_info = _actor_info()
        _require_success(
            client.call_json(
                CONTROL_VEHICLE,
                json.dumps(
                    {
                        "actor_info": actor_info,
                        "long_cmd_type": (
                            "LONG_CMD_TYPE_ACCELERATION"
                        ),
                        "throttle": 0.0,
                        "brake": 0.0,
                        "steer": 0.0,
                    }
                ),
                timeout_sec,
            ),
            "neutralize controls",
        )
        _require_success(
            client.call_json(
                SET_TRANSFORM,
                json.dumps(
                    {
                        "actor_info": actor_info,
                        "transform": {
                            "location": dict(
                                zip(
                                    ("x", "y", "z"),
                                    target_location,
                                    strict=True,
                                )
                            ),
                            "rotation": dict(
                                zip(
                                    ("x", "y", "z"),
                                    target_rotation_deg,
                                    strict=True,
                                )
                            ),
                        },
                    }
                ),
                timeout_sec,
            ),
            "set transform",
        )
        _require_success(
            client.call_json(
                SET_VELOCITY,
                json.dumps(
                    {
                        "actor_info": actor_info,
                        "velocity": max(0.0, speed_mps),
                    }
                ),
                timeout_sec,
            ),
            "restore velocity",
        )
    finally:
        if paused:
            _require_success(
                client.call_json(RESUME, "{}", timeout_sec), "resume"
            )


class VehicleProfileLoopGuard(Node):
    def __init__(self) -> None:
        super().__init__("ad_vehicle_profile_loop_guard")
        self._run_directory = Path(
            str(self.declare_parameter("run_directory", "").value)
        ).expanduser()
        self._grpc_target = str(
            self.declare_parameter(
                "grpc.target", "127.0.0.1:7789"
            ).value
        )
        self._timeout_sec = float(
            self.declare_parameter("grpc.timeout_sec", 3.0).value
        )
        self._limits = GuardLimits(
            maximum_abs_x_m=float(
                self.declare_parameter(
                    "maximum_abs_x_m", 0.8
                ).value
            ),
            minimum_y_m=float(
                self.declare_parameter("minimum_y_m", -1000.0).value
            ),
            minimum_z_m=float(
                self.declare_parameter("minimum_z_m", -0.5).value
            ),
        )
        self._target_location = tuple(
            float(value)
            for value in self.declare_parameter(
                "target_location_m", [0.0, 0.0, 0.36]
            ).value
        )
        self._target_rotation = tuple(
            float(value)
            for value in self.declare_parameter(
                "target_rotation_deg", [0.0, 0.0, -90.0]
            ).value
        )
        self._client = MoraiGrpcClient.connect(
            MoraiApi.load(),
            self._grpc_target,
            default_timeout=self._timeout_sec,
        )
        self._last_wait_log_sec = 0.0
        self._timer = self.create_timer(1.0, self._tick)

    def _tick(self) -> None:
        state_path = self._run_directory / "run_state.json"
        if not state_path.exists():
            self._log_waiting("waiting for profiler run_state.json")
            return
        try:
            profiler_state = json.loads(
                state_path.read_text(encoding="utf-8")
            )
            phase = str(profiler_state.get("phase", ""))
            actor = query_ego_state(self._client, self._timeout_sec)
            reason = reset_reason(actor, phase, self._limits)
            if reason is None:
                return
            before_speed = float(
                profiler_state.get("current_speed_mps") or 0.0
            )
            restored_speed = (
                0.0 if reason == "emergency_fall" else before_speed
            )
            reposition_ego(
                self._client,
                speed_mps=restored_speed,
                target_location=self._target_location,
                target_rotation_deg=self._target_rotation,
                timeout_sec=self._timeout_sec,
            )
            self._append_event(
                reason,
                phase,
                actor,
                before_speed,
                restored_speed,
            )
            self.get_logger().warning(
                f"{reason}: repositioned Ego from "
                f"({actor.x_m:.2f}, {actor.y_m:.2f}, {actor.z_m:.2f}) "
                f"at {restored_speed:.2f} m/s"
            )
        except Exception as exc:
            self.get_logger().error(f"loop guard check failed: {exc}")

    def _append_event(
        self,
        reason: str,
        phase: str,
        actor: ActorState,
        before_speed: float,
        restored_speed: float,
    ) -> None:
        path = self._run_directory / "loop_guard_events.csv"
        write_header = not path.exists()
        with path.open("a", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=_EVENT_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(
                {
                    "unix_time_sec": time.time(),
                    "reason": reason,
                    "profiler_phase": phase,
                    "before_x_m": actor.x_m,
                    "before_y_m": actor.y_m,
                    "before_z_m": actor.z_m,
                    "before_speed_mps": before_speed,
                    "restored_speed_mps": restored_speed,
                    "target_x_m": self._target_location[0],
                    "target_y_m": self._target_location[1],
                    "target_z_m": self._target_location[2],
                }
            )
            stream.flush()
            os.fsync(stream.fileno())

    def _log_waiting(self, message: str) -> None:
        now = time.monotonic()
        if now - self._last_wait_log_sec >= 30.0:
            self.get_logger().info(message)
            self._last_wait_log_sec = now

    def destroy_node(self):
        self._client.close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[VehicleProfileLoopGuard] = None
    try:
        node = VehicleProfileLoopGuard()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
