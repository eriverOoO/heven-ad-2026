"""Fail-closed MORAI actor access for the isolated ESKF experiment."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping

from ad_morai_bridge_dev.eskf_experiment.types import ActorIdentity, TruthSample, VehicleCommand
from ad_morai_bridge_dev.simulator_grpc.client import MoraiGrpcClient


_GET_ALL_ACTORS_STATE = "morai_sim_api.actor.Actor/GetAllActorsState"
_GET_ACTOR_STATE = "morai_sim_api.actor.Actor/GetActorState"
_GET_VEHICLE_CONTROL_MODE = (
    "morai_sim_api.actor.Actor/GetVehicleControlMode"
)
_SET_VEHICLE_CONTROL_MODE = (
    "morai_sim_api.actor.Actor/SetVehicleControlMode"
)
_CONTROL_VEHICLE = "morai_sim_api.actor.Actor/ControlVehicle"

_ALLOWED_METHODS = frozenset(
    {
        _GET_ALL_ACTORS_STATE,
        _GET_ACTOR_STATE,
        _GET_VEHICLE_CONTROL_MODE,
        _SET_VEHICLE_CONTROL_MODE,
        _CONTROL_VEHICLE,
    }
)

_VEHICLE_OBJECT_TYPE = "OBJECT_TYPE_VEHICLE"
_DRIVE_GEAR = "GEAR_MODE_D"
_COMMAND_CONTROL_MODE = "VEHICLE_CONTROL_AUTO_MODE"
_SUCCESS_STATUSES = {1, "STATUS_CODE_SUCCESS"}
_CONTROL_MODES = {
    "VEHICLE_CONTROL_KEYBOARD",
    "VEHICLE_CONTROL_GAME_WHEEL",
    "VEHICLE_CONTROL_AUTO_MODE",
    "VEHICLE_CONTROL_AUTO_MODE_LATERAL",
    "VEHICLE_CONTROL_AUTO_MODE_LONGITUDINAL",
    "VEHICLE_CONTROL_CRUISE_MODE",
    "VEHICLE_CONTROL_SYNCHRONOUS_MODE",
}


class SafeMoraiExperimentClient:
    """Expose only the five actor operations approved for the ESKF trial.

    ``MoraiGrpcClient`` remains responsible for resolving dynamic protobuf
    descriptors. This wrapper deliberately has no public generic dispatch API.
    """

    def __init__(
        self,
        client: MoraiGrpcClient,
        *,
        timeout_sec: float = 5.0,
        truth_timeout_sec: float | None = None,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        sleep: Callable[[float], None] = time.sleep,
        brake_attempts: int = 5,
        stopped_speed_mps: float = 0.05,
        stopped_stable_duration_sec: float = 1.0,
        command_entry_stable_duration_sec: float = 1.0,
        cleanup_poll_interval_sec: float = 0.25,
        on_truth: Callable[[str, TruthSample], None] | None = None,
    ) -> None:
        if not math.isfinite(timeout_sec) or timeout_sec <= 0.0:
            raise ValueError("timeout_sec must be finite and positive")
        if truth_timeout_sec is None:
            truth_timeout_sec = timeout_sec
        if (
            not math.isfinite(truth_timeout_sec)
            or truth_timeout_sec <= 0.0
            or truth_timeout_sec > timeout_sec
        ):
            raise ValueError(
                "truth_timeout_sec must be positive and no greater than timeout_sec"
            )
        if brake_attempts < 2:
            raise ValueError("brake_attempts must be at least 2")
        if not math.isfinite(stopped_speed_mps) or stopped_speed_mps < 0.0:
            raise ValueError("stopped_speed_mps must be finite and nonnegative")
        if (
            not math.isfinite(stopped_stable_duration_sec)
            or stopped_stable_duration_sec < 0.0
        ):
            raise ValueError(
                "stopped_stable_duration_sec must be finite and nonnegative"
            )
        if (
            not math.isfinite(command_entry_stable_duration_sec)
            or command_entry_stable_duration_sec < 0.0
        ):
            raise ValueError(
                "command_entry_stable_duration_sec must be finite and nonnegative"
            )
        if (
            not math.isfinite(cleanup_poll_interval_sec)
            or cleanup_poll_interval_sec <= 0.0
        ):
            raise ValueError(
                "cleanup_poll_interval_sec must be finite and positive"
            )

        self._client = client
        self._timeout_sec = float(timeout_sec)
        self._truth_timeout_sec = float(truth_timeout_sec)
        self._monotonic_ns = monotonic_ns
        self._sleep = sleep
        self._brake_attempts = brake_attempts
        self._stopped_speed_mps = float(stopped_speed_mps)
        self._stopped_stable_duration_ns = round(
            stopped_stable_duration_sec * 1_000_000_000
        )
        self._command_entry_stable_duration_ns = round(
            command_entry_stable_duration_sec * 1_000_000_000
        )
        self._cleanup_poll_interval_sec = float(cleanup_poll_interval_sec)
        self._on_truth = on_truth
        self._actor_identity: ActorIdentity | None = None
        self._initial_control_mode: str | None = None
        self._final_control_mode: str | None = None
        self._command_control_active = False
        # Observe-only runs must remain strictly read-only. This latch is set
        # only when an explicitly authorized command-control path is entered;
        # from then on, errors retain the fail-closed braking behavior.
        self._command_control_transition_attempted = False
        # Set pessimistically before issuing SetVehicleControlMode. A timeout
        # can mean that the simulator applied the request but the response was
        # lost, so cleanup must still attempt to restore the captured mode.
        self._restore_control_mode_required = False
        self._pre_waveform_stable_stop_status = "not_requested"
        self._cleanup_stable_stop_status = "not_started"
        self._restoration_status = "not_required"
        self._restore_skipped_reason: str | None = None
        self._post_restore_stop_status = "not_required"
        self._last_brake_rpc_status = "not_attempted"
        self._closed = False

    @property
    def actor_identity(self) -> ActorIdentity:
        return self._require_identity()

    @property
    def initial_control_mode(self) -> str:
        if self._initial_control_mode is None:
            raise RuntimeError("Ego actor has not been discovered")
        return self._initial_control_mode

    @property
    def final_control_mode(self) -> str:
        if self._final_control_mode is None:
            raise RuntimeError("final vehicle control mode has not been verified")
        return self._final_control_mode

    @property
    def safety_status(self) -> dict[str, object]:
        """Return a snapshot suitable for the immutable run manifest."""
        return {
            "command_control_mode": (
                _COMMAND_CONTROL_MODE
                if self._pre_waveform_stable_stop_status == "verified"
                else None
            ),
            "pre_waveform_stable_stop_status": (
                self._pre_waveform_stable_stop_status
            ),
            "cleanup_stable_stop_status": self._cleanup_stable_stop_status,
            "restoration_status": self._restoration_status,
            "restore_skipped_reason": self._restore_skipped_reason,
            "post_restore_stop_status": self._post_restore_stop_status,
            "last_brake_rpc_status": self._last_brake_rpc_status,
        }

    def discover_ego(self) -> ActorIdentity:
        self._ensure_open()
        try:
            response = self._rpc(
                _GET_ALL_ACTORS_STATE,
                {"vehicle": True, "pedestrian": False, "obstacle": False},
            )
            states = response.get("states")
            if not isinstance(states, list):
                raise RuntimeError("actor discovery response has no states list")
            identities = []
            for state in states:
                if not isinstance(state, Mapping):
                    raise RuntimeError("actor discovery returned an invalid state")
                identity = _parse_identity(state.get("actor_info"))
                if (
                    identity.id_value == "Ego"
                    and identity.object_type == _VEHICLE_OBJECT_TYPE
                ):
                    identities.append(identity)
            if len(identities) != 1:
                raise RuntimeError(
                    "actor discovery requires exactly one Ego vehicle"
                )

            self._actor_identity = identities[0]
            self._initial_control_mode = self._get_control_mode_once()
            return self._actor_identity
        except Exception:
            self._best_effort_brake()
            raise

    def get_control_mode(self) -> str:
        self._ensure_open()
        self._require_identity()
        try:
            return self._get_control_mode_once()
        except Exception:
            self._best_effort_brake()
            raise

    def enter_command_control(self) -> None:
        """Enter MORAI auto control only for an explicitly authorized drive run."""
        self._ensure_open()
        self._require_identity()
        self._command_control_transition_attempted = True
        if self._command_control_active:
            try:
                if self._get_control_mode_once() != _COMMAND_CONTROL_MODE:
                    raise RuntimeError(
                        "verified command control mode changed unexpectedly"
                    )
                return
            except Exception:
                self._command_control_active = False
                self._best_effort_brake()
                raise
        self._pre_waveform_stable_stop_status = "pending"
        try:
            current_mode = self._get_control_mode_once()
            if current_mode != self.initial_control_mode:
                raise RuntimeError(
                    "vehicle control mode changed before command authorization"
                )
            if current_mode != _COMMAND_CONTROL_MODE:
                self._restore_control_mode_required = True
                self._restoration_status = "pending"
                self._set_control_mode_once(_COMMAND_CONTROL_MODE)
            # Brake immediately after the mode transition. Verification is
            # deliberately second so a slow/failed GET cannot leave AUTO mode
            # without a safe longitudinal command.
            self._control_once(_full_brake_command())
            verified_mode = self._get_control_mode_once()
            if verified_mode != _COMMAND_CONTROL_MODE:
                raise RuntimeError("MORAI command control mode was not applied")
            self._brake_until_stably_stopped(
                stable_duration_ns=self._command_entry_stable_duration_ns,
                failure_context="command control entry",
                truth_phase="command_entry_stop",
            )
            self._pre_waveform_stable_stop_status = "verified"
            self._command_control_active = True
        except Exception:
            if self._pre_waveform_stable_stop_status == "pending":
                self._pre_waveform_stable_stop_status = "failed"
            self._command_control_active = False
            self._best_effort_brake()
            raise

    def get_truth(self) -> TruthSample:
        self._ensure_open()
        self._require_identity()
        try:
            return self._get_truth_once()
        except Exception:
            self._best_effort_brake()
            raise

    def send_command(self, command: VehicleCommand) -> None:
        self._ensure_open()
        self._require_identity()
        try:
            _validate_command(command)
            if not self._command_control_active:
                raise RuntimeError(
                    "vehicle command requires verified command control mode"
                )
            self._control_once(command)
        except Exception:
            self._best_effort_brake()
            raise

    def full_brake(self) -> None:
        self._ensure_open()
        self._require_identity()
        self._command_control_transition_attempted = True
        try:
            self._control_once(_full_brake_command())
        except Exception:
            self._best_effort_brake()
            raise

    def close(self) -> None:
        if self._closed:
            return

        cleanup_error: Exception | None = None
        try:
            if self._actor_identity is not None:
                if self._command_control_transition_attempted:
                    self._cleanup_stable_stop_status = "pending"
                    try:
                        self._brake_until_stably_stopped(
                            truth_phase="client_cleanup_stop"
                        )
                    except Exception as exc:
                        self._cleanup_stable_stop_status = "failed"
                        if self._restore_control_mode_required:
                            self._restoration_status = "skipped_unverified_stop"
                            self._restore_skipped_reason = str(exc)
                        raise
                    self._cleanup_stable_stop_status = "verified"
                    if self._restore_control_mode_required:
                        try:
                            self._set_control_mode_once(self.initial_control_mode)
                            self._final_control_mode = self._get_control_mode_once()
                            if self._final_control_mode != self.initial_control_mode:
                                raise RuntimeError(
                                    "cleanup did not restore the initial vehicle "
                                    "control mode"
                                )
                        except Exception:
                            self._restoration_status = "failed"
                            raise
                        self._restoration_status = "verified"
                        self._post_restore_stop_status = "pending"
                        try:
                            post_restore_truth = self._get_internal_truth(
                                "post_restore"
                            )
                            post_restore_speed = math.sqrt(
                                sum(
                                    component * component
                                    for component in (
                                        post_restore_truth.world_velocity_xyz
                                    )
                                )
                            )
                            if post_restore_speed > self._stopped_speed_mps:
                                raise RuntimeError(
                                    "vehicle moved after restoring the initial "
                                    "control mode"
                                )
                        except Exception:
                            self._post_restore_stop_status = "failed"
                            raise
                        self._post_restore_stop_status = "verified"
                    else:
                        self._final_control_mode = self._get_control_mode_once()
                        if self._final_control_mode != self.initial_control_mode:
                            raise RuntimeError(
                                "vehicle control mode changed during cleanup"
                            )
                else:
                    self._cleanup_stable_stop_status = "not_required"
                    self._final_control_mode = self._get_control_mode_once()
                    if self._final_control_mode != self.initial_control_mode:
                        raise RuntimeError(
                            "vehicle control mode changed during observe-only "
                            "cleanup"
                        )
        except Exception as exc:
            cleanup_error = exc
        finally:
            self._closed = True
            self._client.close()

        if cleanup_error is not None:
            raise cleanup_error

    def _rpc(
        self,
        method: str,
        payload: Mapping[str, object],
        timeout_sec: float | None = None,
    ) -> dict:
        if method not in _ALLOWED_METHODS:
            raise RuntimeError("prohibited MORAI experiment RPC")
        result = self._client.call_json(
            method,
            json.dumps(payload, allow_nan=False),
            self._timeout_sec if timeout_sec is None else timeout_sec,
        )
        if not result.success:
            raise RuntimeError(result.error or result.status or "MORAI RPC failed")
        try:
            response = json.loads(result.response_json or "{}")
        except (TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError("MORAI RPC returned invalid JSON") from exc
        if not isinstance(response, dict):
            raise RuntimeError("MORAI RPC response must be a JSON object")
        return response

    def _get_truth_once(self) -> TruthSample:
        identity = self._require_identity()
        rpc_start_ns = self._monotonic_ns()
        response = self._rpc(
            _GET_ACTOR_STATE,
            _identity_payload(identity),
            self._truth_timeout_sec,
        )
        receipt_ns = self._monotonic_ns()
        if receipt_ns < rpc_start_ns:
            raise RuntimeError("monotonic truth timestamps are out of order")

        response_identity = _parse_identity(response.get("actor_info"))
        if response_identity != identity:
            raise RuntimeError("actor truth identity does not match discovered Ego")

        transform = _mapping(response, "transform")
        location = _vector3(_mapping(transform, "location"), "position")
        rotation_deg = _vector3(_mapping(transform, "rotation"), "rotation")
        _vector3(_mapping(response, "velocity"), "body velocity")
        world_velocity_kph = _vector3(
            _mapping(response, "global_velocity"), "world velocity"
        )
        world_velocity = tuple(value / 3.6 for value in world_velocity_kph)
        actor_acceleration = _vector3(
            _mapping(response, "acceleration"), "actor acceleration"
        )
        if "angular_velocity" in response:
            _vector3(
                _mapping(response, "angular_velocity"), "angular velocity"
            )

        vehicle_state = _mapping(response, "vehicle_state")
        # Some installed MORAI builds omit command echoes from ActorState.
        # Missing echoes are unavailable diagnostics, while present invalid
        # values must still fail closed.
        throttle = _optional_finite_number(vehicle_state, "throttle", 0.0)
        brake = _optional_finite_number(vehicle_state, "brake", 0.0)
        steer = _optional_finite_number(vehicle_state, "steer", 0.0)
        try:
            _validate_control_values(
                throttle, brake, steer, reject_overlap=False
            )
        except ValueError as exc:
            raise RuntimeError(f"invalid actor truth: {exc}") from exc
        gear_mode = vehicle_state.get("gear_mode")
        if gear_mode != _DRIVE_GEAR:
            raise RuntimeError(f"actor truth requires drive gear, got {gear_mode!r}")
        collisions = vehicle_state.get("collision_objects", [])
        if not isinstance(collisions, list) or not all(
            isinstance(item, str) for item in collisions
        ):
            raise RuntimeError("actor truth collision IDs must be strings")

        orientation = _rpy_degrees_to_quaternion(rotation_deg)
        # The official schema distinguishes velocity from global_velocity but
        # provides no global_acceleration or explicit acceleration frame tag.
        # Paired live samples at non-zero yaw indicate acceleration is actor-
        # local/body. Record this as an inference in artifacts and rotate it by
        # the actor transform before exposing shared world-frame truth.
        world_acceleration = _rotate_vector_by_quaternion(
            orientation, actor_acceleration
        )

        return TruthSample(
            receipt_monotonic_ns=receipt_ns,
            rpc_start_monotonic_ns=rpc_start_ns,
            simulator_timestamp=None,
            position_xyz=location,
            orientation_xyzw=orientation,
            world_velocity_xyz=world_velocity,
            world_acceleration_xyz=world_acceleration,
            gear_mode=str(gear_mode),
            throttle=throttle,
            brake=brake,
            steer=steer,
            collision_object_ids=tuple(collisions),
        )

    def _get_control_mode_once(self) -> str:
        response = self._rpc(
            _GET_VEHICLE_CONTROL_MODE,
            _identity_payload(self._require_identity()),
        )
        mode = response.get("mode")
        if mode not in _CONTROL_MODES:
            raise RuntimeError(f"invalid vehicle control mode: {mode!r}")
        return str(mode)

    def _set_control_mode_once(self, mode: str) -> None:
        if mode not in _CONTROL_MODES or mode == "VEHICLE_CONTROL_UNSPECIFIED":
            raise RuntimeError(f"invalid requested vehicle control mode: {mode!r}")
        response = self._rpc(
            _SET_VEHICLE_CONTROL_MODE,
            {
                "actor_info": _identity_payload(self._require_identity()),
                "mode": mode,
            },
        )
        if response.get("status") not in _SUCCESS_STATUSES:
            description = response.get("description")
            raise RuntimeError(
                str(description) if description else "MORAI rejected vehicle control mode"
            )

    def _control_once(self, command: VehicleCommand) -> None:
        identity = self._require_identity()
        is_full_brake = command == _full_brake_command()
        if is_full_brake:
            # A missing response is ambiguous: MORAI may have applied the
            # command even when the client cannot verify it.
            self._last_brake_rpc_status = "attempted_unverified"
        response = self._rpc(
            _CONTROL_VEHICLE,
            {
                "actor_info": _identity_payload(identity),
                "long_cmd_type": "LONG_CMD_TYPE_THROTTLE",
                "throttle": command.throttle,
                "brake": command.brake,
                "steer": command.steer,
            },
        )
        if response.get("status") not in _SUCCESS_STATUSES:
            description = response.get("description")
            raise RuntimeError(
                str(description) if description else "MORAI rejected vehicle command"
            )
        if is_full_brake:
            self._last_brake_rpc_status = "verified"

    def _best_effort_brake(self) -> None:
        if (
            self._actor_identity is None
            or self._closed
            or not self._command_control_transition_attempted
        ):
            return
        command = _full_brake_command()
        for _ in range(self._brake_attempts):
            try:
                self._control_once(command)
            except Exception:
                continue

    def _brake_until_stably_stopped(
        self,
        *,
        stable_duration_ns: int | None = None,
        failure_context: str = "cleanup",
        truth_phase: str,
    ) -> None:
        required_duration_ns = (
            self._stopped_stable_duration_ns
            if stable_duration_ns is None
            else stable_duration_ns
        )
        stable_since_ns: int | None = None
        stable_samples = 0
        last_error: Exception | None = None
        command = _full_brake_command()

        for attempt in range(self._brake_attempts):
            try:
                self._control_once(command)
                truth = self._get_internal_truth(truth_phase)
                speed = math.sqrt(
                    sum(component * component for component in truth.world_velocity_xyz)
                )
                if speed <= self._stopped_speed_mps:
                    stable_samples += 1
                    if stable_since_ns is None:
                        stable_since_ns = truth.receipt_monotonic_ns
                    if (
                        stable_samples >= 2
                        and truth.receipt_monotonic_ns - stable_since_ns
                        >= required_duration_ns
                    ):
                        return
                else:
                    stable_since_ns = None
                    stable_samples = 0
            except Exception as exc:
                last_error = exc
                stable_since_ns = None
                stable_samples = 0
            if attempt + 1 < self._brake_attempts:
                self._sleep(self._cleanup_poll_interval_sec)

        message = (
            f"{failure_context} could not verify a stable stopped truth window"
        )
        if last_error is not None:
            raise RuntimeError(message) from last_error
        raise RuntimeError(message)

    def _get_internal_truth(self, phase: str) -> TruthSample:
        truth = self._get_truth_once()
        if self._on_truth is not None:
            self._on_truth(phase, truth)
        return truth

    def _require_identity(self) -> ActorIdentity:
        if self._actor_identity is None:
            raise RuntimeError("Ego actor has not been discovered")
        return self._actor_identity

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("MORAI experiment client is closed")


def _parse_identity(value: object) -> ActorIdentity:
    if not isinstance(value, Mapping):
        raise RuntimeError("actor info is missing or invalid")
    identifier = value.get("id")
    if not isinstance(identifier, Mapping):
        raise RuntimeError("actor ID is missing or invalid")
    id_value = identifier.get("value")
    object_type = value.get("object_type")
    client_key = value.get("client_key", "")
    if not isinstance(id_value, str) or not id_value:
        raise RuntimeError("actor ID must be a nonempty string")
    if not isinstance(object_type, str) or not object_type:
        raise RuntimeError("actor object type must be a nonempty string")
    if not isinstance(client_key, str):
        raise RuntimeError("actor client_key must be a string")
    return ActorIdentity(id_value, object_type, client_key)


def _identity_payload(identity: ActorIdentity) -> dict:
    return {
        "id": {"value": identity.id_value},
        "object_type": identity.object_type,
        "client_key": identity.client_key,
    }


def _mapping(parent: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise RuntimeError(f"actor truth {key} is missing or invalid")
    return value


def _finite_number(parent: Mapping[str, object], key: str) -> float:
    value = parent.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"actor truth {key} must be finite")
    converted = float(value)
    if not math.isfinite(converted):
        raise RuntimeError(f"actor truth {key} must be finite")
    return converted


def _optional_finite_number(
    parent: Mapping[str, object], key: str, default: float
) -> float:
    if key not in parent:
        return default
    return _finite_number(parent, key)


def _vector3(value: Mapping[str, object], name: str) -> tuple[float, float, float]:
    try:
        return tuple(_finite_number(value, axis) for axis in ("x", "y", "z"))
    except RuntimeError as exc:
        raise RuntimeError(f"actor truth {name} must be finite") from exc


def _validate_command(command: VehicleCommand) -> None:
    if not isinstance(command, VehicleCommand):
        raise ValueError("command must be a VehicleCommand")
    _validate_control_values(
        command.throttle,
        command.brake,
        command.steer,
        reject_overlap=True,
    )


def _validate_control_values(
    throttle: float,
    brake: float,
    steer: float,
    *,
    reject_overlap: bool,
) -> None:
    values = (("throttle", throttle), ("brake", brake), ("steer", steer))
    for name, value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be finite")
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if not 0.0 <= throttle <= 1.0:
        raise ValueError("throttle must be in [0, 1]")
    if not 0.0 <= brake <= 1.0:
        raise ValueError("brake must be in [0, 1]")
    if not -1.0 <= steer <= 1.0:
        raise ValueError("steer must be in [-1, 1]")
    if reject_overlap and throttle > 0.0 and brake > 0.0:
        raise ValueError("positive throttle and brake overlap is prohibited")


def _full_brake_command() -> VehicleCommand:
    return VehicleCommand(throttle=0.0, brake=1.0, steer=0.0)


def _rpy_degrees_to_quaternion(
    rpy_deg: tuple[float, float, float],
) -> tuple[float, float, float, float]:
    roll, pitch, yaw = (math.radians(value) for value in rpy_deg)
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def _rotate_vector_by_quaternion(
    quaternion_xyzw: tuple[float, float, float, float],
    vector_xyz: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Rotate a body-frame vector into world axes with a unit quaternion."""
    qx, qy, qz, qw = quaternion_xyzw
    vx, vy, vz = vector_xyz
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + qy * tz - qz * ty,
        vy + qw * ty + qz * tx - qx * tz,
        vz + qw * tz + qx * ty - qy * tx,
    )
