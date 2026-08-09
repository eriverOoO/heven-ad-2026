import inspect
import json
import math
from collections import defaultdict, deque

import pytest

from ad_morai_bridge_dev.eskf_experiment.grpc import (
    SafeMoraiExperimentClient,
    _ALLOWED_METHODS,
)
from ad_morai_bridge_dev.eskf_experiment.types import VehicleCommand
from ad_morai_bridge_dev.simulator_grpc.client import GrpcCallResult


GET_ALL = "morai_sim_api.actor.Actor/GetAllActorsState"
GET_STATE = "morai_sim_api.actor.Actor/GetActorState"
GET_CONTROL_MODE = "morai_sim_api.actor.Actor/GetVehicleControlMode"
SET_CONTROL_MODE = "morai_sim_api.actor.Actor/SetVehicleControlMode"
CONTROL = "morai_sim_api.actor.Actor/ControlVehicle"


def _ok(payload):
    return GrpcCallResult(
        success=True,
        code=0,
        status="OK",
        response_json=json.dumps(payload),
    )


def _actor_info(*, actor_id="Ego", object_type="OBJECT_TYPE_VEHICLE", client_key="key-7"):
    return {
        "id": {"value": actor_id},
        "object_type": object_type,
        "client_key": client_key,
    }


def _actor_state(
    *,
    speed=0.0,
    actor_id="Ego",
    object_type="OBJECT_TYPE_VEHICLE",
    client_key="key-7",
    throttle=0.0,
    brake=0.0,
    steer=0.0,
    rotation_xyz=(0.0, 0.0, 90.0),
    acceleration_xyz=(0.1, 0.2, 0.3),
    collision_object_ids=("Ego",),
):
    return {
        "actor_info": _actor_info(
            actor_id=actor_id,
            object_type=object_type,
            client_key=client_key,
        ),
        "transform": {
            "location": {"x": 1.0, "y": 2.0, "z": 3.0},
            "rotation": dict(zip(("x", "y", "z"), rotation_xyz)),
        },
        "velocity": {"x": speed, "y": 0.0, "z": 0.0},
        "global_velocity": {"x": speed, "y": 0.0, "z": 0.0},
        "acceleration": dict(zip(("x", "y", "z"), acceleration_xyz)),
        "vehicle_state": {
            "throttle": throttle,
            "brake": brake,
            "steer": steer,
            "gear_mode": "GEAR_MODE_D",
            "collision_objects": list(collision_object_ids),
        },
    }


class FakeMoraiGrpcClient:
    def __init__(self):
        self.results = defaultdict(deque)
        self.calls = []
        self.closed = False

    def enqueue(self, method, *results):
        self.results[method].extend(results)

    def call_json(self, method, request_json, timeout):
        self.calls.append((method, json.loads(request_json), timeout))
        if self.results[method]:
            return self.results[method].popleft()
        if method == CONTROL:
            return _ok({"status": "STATUS_CODE_SUCCESS"})
        raise AssertionError(f"no fake result queued for {method}")

    def close(self):
        self.closed = True


class FakeMonotonicClock:
    def __init__(self, *timestamps_ns):
        self.timestamps_ns = iter(timestamps_ns)

    def __call__(self):
        return next(self.timestamps_ns)


class FakeSleeper:
    def __init__(self):
        self.calls = []

    def __call__(self, duration_sec):
        self.calls.append(duration_sec)


def _discovered_client(fake, **kwargs):
    fake.enqueue(
        GET_ALL,
        _ok({"states": [{"actor_info": _actor_info()}], "timestamp": "81"}),
    )
    fake.enqueue(GET_CONTROL_MODE, _ok({"mode": "VEHICLE_CONTROL_KEYBOARD"}))
    client = SafeMoraiExperimentClient(fake, **kwargs)
    identity = client.discover_ego()
    assert identity.id_value == "Ego"
    return client


def test_private_allowlist_is_exact_and_public_api_has_no_rpc_dispatch_parameter():
    assert _ALLOWED_METHODS == frozenset(
        {
            GET_ALL,
            GET_STATE,
            GET_CONTROL_MODE,
            SET_CONTROL_MODE,
            CONTROL,
        }
    )

    public_methods = {
        name: member
        for name, member in inspect.getmembers(
            SafeMoraiExperimentClient, predicate=inspect.isfunction
        )
        if not name.startswith("_")
    }
    assert set(public_methods) == {
        "close",
        "discover_ego",
        "enter_command_control",
        "full_brake",
        "get_control_mode",
        "get_truth",
        "send_command",
    }
    for member in public_methods.values():
        parameters = set(inspect.signature(member).parameters)
        assert not ({"method", "rpc", "rpc_path", "path"} & parameters)


@pytest.mark.parametrize(
    "states",
    [
        [],
        [{"actor_info": _actor_info(actor_id="NotEgo")}],
        [
            {"actor_info": _actor_info()},
            {"actor_info": _actor_info(client_key="other-key")},
        ],
        [{"actor_info": _actor_info(object_type="OBJECT_TYPE_PEDESTRIAN")}],
    ],
)
def test_discovery_requires_exactly_one_ego_vehicle(states):
    fake = FakeMoraiGrpcClient()
    fake.enqueue(GET_ALL, _ok({"states": states, "timestamp": "81"}))
    client = SafeMoraiExperimentClient(fake)

    with pytest.raises(RuntimeError, match="exactly one.*Ego.*vehicle"):
        client.discover_ego()

    assert [call[0] for call in fake.calls] == [GET_ALL]


def test_discovery_retains_exact_actor_info_and_queries_but_never_changes_control_mode():
    fake = FakeMoraiGrpcClient()
    client = _discovered_client(fake)

    assert client.actor_identity.id_value == "Ego"
    assert client.actor_identity.object_type == "OBJECT_TYPE_VEHICLE"
    assert client.actor_identity.client_key == "key-7"
    assert client.initial_control_mode == "VEHICLE_CONTROL_KEYBOARD"
    assert fake.calls == [
        (
            GET_ALL,
            {"vehicle": True, "pedestrian": False, "obstacle": False},
            5.0,
        ),
        (GET_CONTROL_MODE, _actor_info(), 5.0),
    ]
    assert all("mode" not in request for _method, request, _timeout in fake.calls)


def test_final_control_mode_can_be_compared_to_initial_without_mutating_it():
    fake = FakeMoraiGrpcClient()
    client = _discovered_client(fake)
    fake.enqueue(
        GET_CONTROL_MODE,
        _ok({"mode": "VEHICLE_CONTROL_AUTO_MODE_LONGITUDINAL"}),
    )

    final_mode = client.get_control_mode()

    assert client.initial_control_mode == "VEHICLE_CONTROL_KEYBOARD"
    assert final_mode == "VEHICLE_CONTROL_AUTO_MODE_LONGITUDINAL"
    assert fake.calls[-1] == (GET_CONTROL_MODE, _actor_info(), 5.0)
    assert [method for method, _request, _timeout in fake.calls] == [
        GET_ALL,
        GET_CONTROL_MODE,
        GET_CONTROL_MODE,
    ]


def test_invalid_control_mode_response_is_read_only_before_command_control():
    fake = FakeMoraiGrpcClient()
    client = _discovered_client(fake, brake_attempts=3)
    fake.enqueue(GET_CONTROL_MODE, _ok({"mode": "VEHICLE_CONTROL_UNSPECIFIED"}))

    with pytest.raises(RuntimeError, match="invalid vehicle control mode"):
        client.get_control_mode()

    assert not [call for call in fake.calls if call[0] in {SET_CONTROL_MODE, CONTROL}]


def test_truth_preserves_monotonic_rpc_bounds_and_converts_pose_to_shared_type():
    fake = FakeMoraiGrpcClient()
    clock = FakeMonotonicClock(100, 160)
    observed_internal_truth = []
    client = _discovered_client(
        fake,
        monotonic_ns=clock,
        on_truth=lambda phase, truth: observed_internal_truth.append(
            (phase, truth)
        ),
    )
    fake.enqueue(GET_STATE, _ok(_actor_state(speed=3.6)))

    truth = client.get_truth()

    assert truth.rpc_start_monotonic_ns == 100
    assert truth.receipt_monotonic_ns == 160
    assert truth.simulator_timestamp is None
    assert truth.position_xyz == (1.0, 2.0, 3.0)
    assert truth.orientation_xyzw == pytest.approx(
        (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5))
    )
    assert truth.world_velocity_xyz == (1.0, 0.0, 0.0)
    assert truth.world_acceleration_xyz == pytest.approx((-0.2, 0.1, 0.3))
    assert truth.gear_mode == "GEAR_MODE_D"
    assert truth.collision_object_ids == ("Ego",)
    assert fake.calls[-1] == (GET_STATE, _actor_info(), 5.0)
    assert observed_internal_truth == []


def test_truth_converts_actor_velocity_from_kph_to_mps():
    fake = FakeMoraiGrpcClient()
    client = _discovered_client(
        fake,
        monotonic_ns=FakeMonotonicClock(100, 160),
    )
    fake.enqueue(GET_STATE, _ok(_actor_state(speed=36.0)))

    truth = client.get_truth()

    assert truth.world_velocity_xyz == pytest.approx((10.0, 0.0, 0.0))


def test_truth_rpc_uses_the_stricter_freshness_deadline():
    fake = FakeMoraiGrpcClient()
    clock = FakeMonotonicClock(100, 160)
    client = _discovered_client(
        fake,
        monotonic_ns=clock,
        timeout_sec=2.0,
        truth_timeout_sec=0.25,
    )
    fake.enqueue(GET_STATE, _ok(_actor_state()))

    client.get_truth()

    assert fake.calls[-1] == (GET_STATE, _actor_info(), 0.25)


def test_truth_rotates_actor_local_acceleration_into_world_frame():
    fake = FakeMoraiGrpcClient()
    clock = FakeMonotonicClock(100, 160)
    client = _discovered_client(fake, monotonic_ns=clock)
    fake.enqueue(
        GET_STATE,
        _ok(
            _actor_state(
                rotation_xyz=(0.0, 0.0, 90.0),
                acceleration_xyz=(1.0, 0.0, 0.0),
            )
        ),
    )

    truth = client.get_truth()

    assert truth.world_acceleration_xyz == pytest.approx((0.0, 1.0, 0.0))


def test_live_actor_shape_may_omit_control_echo_fields():
    """The installed MORAI build omits throttle/brake/steer at rest."""
    fake = FakeMoraiGrpcClient()
    clock = FakeMonotonicClock(100, 160)
    client = _discovered_client(fake, monotonic_ns=clock)
    state = _actor_state()
    del state["vehicle_state"]["throttle"]
    del state["vehicle_state"]["brake"]
    del state["vehicle_state"]["steer"]
    fake.enqueue(GET_STATE, _ok(state))

    truth = client.get_truth()

    assert (truth.throttle, truth.brake, truth.steer) == (0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    "state, message",
    [
        (_actor_state(speed=float("nan")), "finite"),
        (_actor_state(throttle=1.01), "throttle"),
        (_actor_state(brake=-0.01), "brake"),
        (_actor_state(steer=float("inf")), "steer"),
        (_actor_state(actor_id="Other"), "identity"),
    ],
)
def test_invalid_actor_truth_is_rejected_without_mutating_observe_only_vehicle(
    state, message
):
    fake = FakeMoraiGrpcClient()
    clock = FakeMonotonicClock(100, 160)
    client = _discovered_client(fake, monotonic_ns=clock, brake_attempts=3)
    fake.enqueue(GET_STATE, _ok(state))

    with pytest.raises(RuntimeError, match=message):
        client.get_truth()

    assert not [call for call in fake.calls if call[0] in {SET_CONTROL_MODE, CONTROL}]


@pytest.mark.parametrize(
    "command, message",
    [
        (VehicleCommand(float("nan"), 0.0), "finite"),
        (VehicleCommand(1.01, 0.0), "throttle"),
        (VehicleCommand(0.0, -0.01), "brake"),
        (VehicleCommand(0.0, 0.0, 1.01), "steer"),
        (VehicleCommand(0.1, 0.1), "overlap"),
    ],
)
def test_invalid_commands_after_control_transition_trigger_fail_closed_braking(
    command, message
):
    fake = FakeMoraiGrpcClient()
    client = _discovered_client(
        fake,
        brake_attempts=3,
        command_entry_stable_duration_sec=0.0,
    )
    fake.enqueue(
        GET_CONTROL_MODE,
        _ok({"mode": "VEHICLE_CONTROL_KEYBOARD"}),
        _ok({"mode": "VEHICLE_CONTROL_AUTO_MODE"}),
    )
    fake.enqueue(SET_CONTROL_MODE, _ok({"status": "STATUS_CODE_SUCCESS"}))
    fake.enqueue(
        GET_STATE,
        _ok(_actor_state(speed=0.0)),
        _ok(_actor_state(speed=0.0)),
    )
    client.enter_command_control()
    controls_before_failure = len(
        [call for call in fake.calls if call[0] == CONTROL]
    )

    with pytest.raises(ValueError, match=message):
        client.send_command(command)

    brake_requests = [request for method, request, _timeout in fake.calls if method == CONTROL]
    assert len(brake_requests) - controls_before_failure == 3
    assert all(
        request["throttle"] == 0.0 and request["brake"] == 1.0
        for request in brake_requests
    )


def test_vehicle_commands_use_throttle_mode_and_the_discovered_actor_info():
    fake = FakeMoraiGrpcClient()
    client = _discovered_client(
        fake,
        brake_attempts=2,
        command_entry_stable_duration_sec=0.0,
    )
    fake.enqueue(
        GET_CONTROL_MODE,
        _ok({"mode": "VEHICLE_CONTROL_KEYBOARD"}),
        _ok({"mode": "VEHICLE_CONTROL_AUTO_MODE"}),
    )
    fake.enqueue(
        SET_CONTROL_MODE,
        _ok({"status": "STATUS_CODE_SUCCESS"}),
    )
    fake.enqueue(
        GET_STATE,
        _ok(_actor_state(speed=0.0)),
        _ok(_actor_state(speed=0.0)),
    )
    client.enter_command_control()

    client.send_command(VehicleCommand(throttle=0.2, brake=0.0, steer=-0.1))

    assert fake.calls[-1] == (
        CONTROL,
        {
            "actor_info": _actor_info(),
            "long_cmd_type": "LONG_CMD_TYPE_THROTTLE",
            "throttle": 0.2,
            "brake": 0.0,
            "steer": -0.1,
        },
        5.0,
    )


def test_vehicle_commands_are_rejected_before_command_control_is_entered():
    fake = FakeMoraiGrpcClient()
    client = _discovered_client(fake, brake_attempts=3)

    with pytest.raises(RuntimeError, match="command control mode"):
        client.send_command(VehicleCommand(throttle=0.2, brake=0.0))

    assert not [call for call in fake.calls if call[0] in {SET_CONTROL_MODE, CONTROL}]


def test_command_control_mode_is_explicitly_entered_verified_and_braked():
    fake = FakeMoraiGrpcClient()
    observed_internal_truth = []
    client = _discovered_client(
        fake,
        brake_attempts=2,
        command_entry_stable_duration_sec=0.0,
        on_truth=lambda phase, truth: observed_internal_truth.append(
            (phase, truth)
        ),
    )
    fake.enqueue(
        GET_CONTROL_MODE,
        _ok({"mode": "VEHICLE_CONTROL_KEYBOARD"}),
        _ok({"mode": "VEHICLE_CONTROL_AUTO_MODE"}),
    )
    fake.enqueue(
        SET_CONTROL_MODE,
        _ok({"status": "STATUS_CODE_SUCCESS"}),
    )
    fake.enqueue(
        GET_STATE,
        _ok(_actor_state(speed=0.0)),
        _ok(_actor_state(speed=0.0)),
    )

    client.enter_command_control()

    methods = [method for method, _request, _timeout in fake.calls]
    set_index = methods.index(SET_CONTROL_MODE)
    assert methods[set_index:set_index + 4] == [
        SET_CONTROL_MODE,
        CONTROL,
        GET_CONTROL_MODE,
        CONTROL,
    ]
    assert methods[-3:] == [GET_STATE, CONTROL, GET_STATE]
    assert fake.calls[set_index:set_index + 3][0][0] == SET_CONTROL_MODE
    assert [method for method, _request, _timeout in fake.calls[2:6]] == [
        GET_CONTROL_MODE,
        SET_CONTROL_MODE,
        CONTROL,
        GET_CONTROL_MODE,
    ]
    assert fake.calls[set_index] == (
        SET_CONTROL_MODE,
        {
            "actor_info": _actor_info(),
            "mode": "VEHICLE_CONTROL_AUTO_MODE",
        },
        5.0,
    )
    assert fake.calls[set_index + 1][1]["brake"] == 1.0
    assert [phase for phase, _truth in observed_internal_truth] == [
        "command_entry_stop",
        "command_entry_stop",
    ]
    assert client.safety_status == {
        "command_control_mode": "VEHICLE_CONTROL_AUTO_MODE",
        "pre_waveform_stable_stop_status": "verified",
        "cleanup_stable_stop_status": "not_started",
        "restoration_status": "pending",
        "restore_skipped_reason": None,
        "post_restore_stop_status": "not_required",
        "last_brake_rpc_status": "verified",
    }


def test_failed_initial_brake_never_latches_command_control_active():
    fake = FakeMoraiGrpcClient()
    client = _discovered_client(fake, brake_attempts=3)
    fake.enqueue(
        GET_CONTROL_MODE,
        _ok({"mode": "VEHICLE_CONTROL_KEYBOARD"}),
    )
    fake.enqueue(
        SET_CONTROL_MODE,
        _ok({"status": "STATUS_CODE_SUCCESS"}),
    )
    fake.enqueue(
        CONTROL,
        GrpcCallResult(False, 14, "UNAVAILABLE", error="initial brake failed"),
    )

    with pytest.raises(RuntimeError, match="initial brake failed"):
        client.enter_command_control()
    with pytest.raises(RuntimeError, match="command control mode"):
        client.send_command(VehicleCommand(throttle=0.1, brake=0.0))


def test_reentering_command_control_revalidates_the_live_mode():
    fake = FakeMoraiGrpcClient()
    client = _discovered_client(
        fake,
        brake_attempts=2,
        command_entry_stable_duration_sec=0.0,
    )
    fake.enqueue(
        GET_CONTROL_MODE,
        _ok({"mode": "VEHICLE_CONTROL_KEYBOARD"}),
        _ok({"mode": "VEHICLE_CONTROL_AUTO_MODE"}),
        _ok({"mode": "VEHICLE_CONTROL_AUTO_MODE"}),
    )
    fake.enqueue(
        SET_CONTROL_MODE,
        _ok({"status": "STATUS_CODE_SUCCESS"}),
    )
    fake.enqueue(
        GET_STATE,
        _ok(_actor_state(speed=0.0)),
        _ok(_actor_state(speed=0.0)),
    )

    client.enter_command_control()
    call_count = len(fake.calls)
    client.enter_command_control()

    assert len(fake.calls) == call_count + 1
    assert fake.calls[-1][0] == GET_CONTROL_MODE


def test_observe_only_close_queries_mode_without_vehicle_control_rpc():
    fake = FakeMoraiGrpcClient()
    client = _discovered_client(fake)
    fake.enqueue(
        GET_CONTROL_MODE,
        _ok({"mode": "VEHICLE_CONTROL_KEYBOARD"}),
    )

    client.close()

    assert fake.closed
    assert [method for method, _request, _timeout in fake.calls] == [
        GET_ALL,
        GET_CONTROL_MODE,
        GET_CONTROL_MODE,
    ]
    assert client.final_control_mode == "VEHICLE_CONTROL_KEYBOARD"
    assert client.safety_status["cleanup_stable_stop_status"] == "not_required"

    call_count = len(fake.calls)
    client.close()
    assert len(fake.calls) == call_count


def test_cleanup_restores_original_control_mode_only_after_stable_stop():
    fake = FakeMoraiGrpcClient()
    observed_internal_truth = []
    clock = FakeMonotonicClock(
        0,
        10,
        20,
        30,
        40,
        50,
        500_000_050,
        500_000_060,
        1_500_000_060,
        1_500_000_070,
        1_500_000_080,
        1_500_000_090,
    )
    client = _discovered_client(
        fake,
        monotonic_ns=clock,
        sleep=lambda _duration: None,
        brake_attempts=3,
        stopped_stable_duration_sec=1.0,
        command_entry_stable_duration_sec=0.0,
        on_truth=lambda phase, truth: observed_internal_truth.append(
            (phase, truth)
        ),
    )
    fake.enqueue(
        GET_CONTROL_MODE,
        _ok({"mode": "VEHICLE_CONTROL_KEYBOARD"}),
        _ok({"mode": "VEHICLE_CONTROL_AUTO_MODE"}),
        _ok({"mode": "VEHICLE_CONTROL_KEYBOARD"}),
    )
    fake.enqueue(
        SET_CONTROL_MODE,
        _ok({"status": "STATUS_CODE_SUCCESS"}),
        _ok({"status": "STATUS_CODE_SUCCESS"}),
    )
    fake.enqueue(
        GET_STATE,
        _ok(_actor_state(speed=0.0)),
        _ok(_actor_state(speed=0.0)),
        _ok(_actor_state(speed=0.0)),
        _ok(_actor_state(speed=0.0)),
        _ok(_actor_state(speed=0.0)),
        _ok(_actor_state(speed=0.0)),
    )

    client.enter_command_control()
    client.close()

    methods = [method for method, _request, _timeout in fake.calls]
    restore_index = max(
        index for index, method in enumerate(methods) if method == SET_CONTROL_MODE
    )
    truth_indices = [
        index for index, method in enumerate(methods) if method == GET_STATE
    ]
    assert restore_index > truth_indices[-2]
    assert restore_index < truth_indices[-1]
    assert fake.calls[restore_index][1]["mode"] == "VEHICLE_CONTROL_KEYBOARD"
    assert client.final_control_mode == "VEHICLE_CONTROL_KEYBOARD"
    assert client.safety_status["cleanup_stable_stop_status"] == "verified"
    assert client.safety_status["restoration_status"] == "verified"
    assert client.safety_status["post_restore_stop_status"] == "verified"
    assert [phase for phase, _truth in observed_internal_truth] == [
        "command_entry_stop",
        "command_entry_stop",
        "client_cleanup_stop",
        "client_cleanup_stop",
        "client_cleanup_stop",
        "post_restore",
    ]
    assert fake.closed


def test_cleanup_rejects_motion_after_restoring_the_initial_control_mode():
    fake = FakeMoraiGrpcClient()
    clock = FakeMonotonicClock(
        0,
        10,
        20,
        30,
        40,
        50,
        500_000_050,
        500_000_060,
        1_500_000_060,
        1_500_000_070,
        1_500_000_080,
        1_500_000_090,
    )
    client = _discovered_client(
        fake,
        monotonic_ns=clock,
        sleep=lambda _duration: None,
        brake_attempts=3,
        stopped_stable_duration_sec=1.0,
        command_entry_stable_duration_sec=0.0,
    )
    fake.enqueue(
        GET_CONTROL_MODE,
        _ok({"mode": "VEHICLE_CONTROL_KEYBOARD"}),
        _ok({"mode": "VEHICLE_CONTROL_AUTO_MODE"}),
        _ok({"mode": "VEHICLE_CONTROL_KEYBOARD"}),
    )
    fake.enqueue(
        SET_CONTROL_MODE,
        _ok({"status": "STATUS_CODE_SUCCESS"}),
        _ok({"status": "STATUS_CODE_SUCCESS"}),
    )
    fake.enqueue(
        GET_STATE,
        _ok(_actor_state(speed=0.0)),
        _ok(_actor_state(speed=0.0)),
        _ok(_actor_state(speed=0.0)),
        _ok(_actor_state(speed=0.0)),
        _ok(_actor_state(speed=0.0)),
        _ok(_actor_state(speed=0.2)),
    )

    client.enter_command_control()
    with pytest.raises(RuntimeError, match="after restoring"):
        client.close()

    assert client.final_control_mode == "VEHICLE_CONTROL_KEYBOARD"
    assert client.safety_status["restoration_status"] == "verified"
    assert client.safety_status["post_restore_stop_status"] == "failed"
    assert fake.closed


def test_ambiguous_enter_failure_still_restores_original_mode_after_stable_stop():
    fake = FakeMoraiGrpcClient()
    clock = FakeMonotonicClock(
        0,
        10,
        500_000_010,
        500_000_020,
        1_500_000_020,
        1_500_000_030,
        1_500_000_040,
        1_500_000_050,
    )
    client = _discovered_client(
        fake,
        monotonic_ns=clock,
        sleep=lambda _duration: None,
        brake_attempts=3,
        stopped_stable_duration_sec=1.0,
    )
    fake.enqueue(
        GET_CONTROL_MODE,
        _ok({"mode": "VEHICLE_CONTROL_KEYBOARD"}),
        _ok({"mode": "VEHICLE_CONTROL_KEYBOARD"}),
    )
    fake.enqueue(
        SET_CONTROL_MODE,
        GrpcCallResult(False, 14, "UNAVAILABLE", error="ambiguous mode response"),
        _ok({"status": "STATUS_CODE_SUCCESS"}),
    )
    fake.enqueue(
        GET_STATE,
        _ok(_actor_state(speed=0.0)),
        _ok(_actor_state(speed=0.0)),
        _ok(_actor_state(speed=0.0)),
        _ok(_actor_state(speed=0.0)),
    )

    with pytest.raises(RuntimeError, match="ambiguous mode response"):
        client.enter_command_control()
    client.close()

    set_requests = [
        request for method, request, _timeout in fake.calls
        if method == SET_CONTROL_MODE
    ]
    assert [request["mode"] for request in set_requests] == [
        "VEHICLE_CONTROL_AUTO_MODE",
        "VEHICLE_CONTROL_KEYBOARD",
    ]
    assert client.final_control_mode == "VEHICLE_CONTROL_KEYBOARD"


def test_observe_only_truth_failure_never_sends_vehicle_control_rpc():
    fake = FakeMoraiGrpcClient()
    clock = FakeMonotonicClock(0, 10)
    client = _discovered_client(
        fake,
        monotonic_ns=clock,
    )
    fake.enqueue(
        GET_STATE,
        GrpcCallResult(False, 14, "UNAVAILABLE", error="simulator offline"),
    )

    with pytest.raises(RuntimeError, match="simulator offline"):
        client.get_truth()

    assert not [call for call in fake.calls if call[0] in {SET_CONTROL_MODE, CONTROL}]


def test_dynamic_cleanup_leaves_auto_after_brake_attempts_if_stop_is_unverified():
    fake = FakeMoraiGrpcClient()
    clock = FakeMonotonicClock(
        0,
        10,
        20,
        30,
        40,
        50,
        1_000_000_050,
        1_000_000_060,
    )
    client = _discovered_client(
        fake,
        monotonic_ns=clock,
        sleep=lambda _duration: None,
        brake_attempts=2,
        stopped_stable_duration_sec=1.0,
        command_entry_stable_duration_sec=0.0,
    )
    fake.enqueue(
        GET_CONTROL_MODE,
        _ok({"mode": "VEHICLE_CONTROL_KEYBOARD"}),
        _ok({"mode": "VEHICLE_CONTROL_AUTO_MODE"}),
    )
    fake.enqueue(
        SET_CONTROL_MODE,
        _ok({"status": "STATUS_CODE_SUCCESS"}),
    )
    fake.enqueue(
        GET_STATE,
        _ok(_actor_state(speed=0.0)),
        _ok(_actor_state(speed=0.0)),
        _ok(_actor_state(speed=0.2)),
        _ok(_actor_state(speed=0.2)),
    )

    client.enter_command_control()
    with pytest.raises(RuntimeError, match="stable stopped truth window"):
        client.close()

    set_requests = [
        request for method, request, _timeout in fake.calls
        if method == SET_CONTROL_MODE
    ]
    assert [request["mode"] for request in set_requests] == [
        "VEHICLE_CONTROL_AUTO_MODE"
    ]
    assert client.safety_status["cleanup_stable_stop_status"] == "failed"
    assert (
        client.safety_status["restoration_status"]
        == "skipped_unverified_stop"
    )
    assert client.safety_status["restore_skipped_reason"]
    assert client.safety_status["last_brake_rpc_status"] == "verified"
    assert fake.closed


def test_truth_failure_after_command_control_attempt_triggers_repeated_full_brake():
    fake = FakeMoraiGrpcClient()
    clock = FakeMonotonicClock(100, 160, 170, 180, 190)
    client = _discovered_client(
        fake,
        monotonic_ns=clock,
        brake_attempts=3,
        command_entry_stable_duration_sec=0.0,
    )
    fake.enqueue(
        GET_CONTROL_MODE,
        _ok({"mode": "VEHICLE_CONTROL_KEYBOARD"}),
        _ok({"mode": "VEHICLE_CONTROL_AUTO_MODE"}),
    )
    fake.enqueue(SET_CONTROL_MODE, _ok({"status": "STATUS_CODE_SUCCESS"}))
    fake.enqueue(
        GET_STATE,
        _ok(_actor_state(speed=0.0)),
        _ok(_actor_state(speed=0.0)),
    )
    client.enter_command_control()
    controls_before_failure = len(
        [call for call in fake.calls if call[0] == CONTROL]
    )
    fake.enqueue(
        GET_STATE,
        GrpcCallResult(False, 14, "UNAVAILABLE", error="simulator offline"),
    )

    with pytest.raises(RuntimeError, match="simulator offline"):
        client.get_truth()

    controls_after_failure = len(
        [call for call in fake.calls if call[0] == CONTROL]
    )
    assert controls_after_failure - controls_before_failure == 3
