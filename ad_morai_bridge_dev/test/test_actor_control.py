import importlib
import inspect
import json
from collections import defaultdict, deque

import pytest

from ad_morai_bridge_dev.simulator_grpc.client import GrpcCallResult


GET_MODELS = "morai_sim_api.simualtor.Simulator/GetAvailableObject"
GET_ALL = "morai_sim_api.actor.Actor/GetAllActorsState"
GET_STATE = "morai_sim_api.actor.Actor/GetActorState"
SPAWN = "morai_sim_api.actor.Actor/SpawnVehicle"
DESTROY = "morai_sim_api.actor.Actor/DestroyActor"
SET_PAUSE = "morai_sim_api.actor.Actor/SetPause"
SET_TRANSFORM = "morai_sim_api.actor.Actor/SetTransform"
SET_VELOCITY = "morai_sim_api.actor.Actor/SetVelocity"
SET_ROUTE = "morai_sim_api.actor.Actor/SetVehicleRoute"


def _types():
    module = importlib.import_module("ad_morai_bridge_dev.actors.control")
    return (
        module.ActorRef,
        module.ActorState,
        module.SpawnedActor,
        module.MoraiActorController,
        module.ALLOWED_ACTOR_METHODS,
    )


def _ok(payload):
    return GrpcCallResult(True, 0, "OK", response_json=json.dumps(payload))


def _transport_failure(status="DEADLINE_EXCEEDED"):
    return GrpcCallResult(False, 4, status, error="deadline")


def _info(actor_id="Ego", object_type="OBJECT_TYPE_VEHICLE", client_key="world"):
    return {
        "id": {"value": actor_id},
        "object_type": object_type,
        "client_key": client_key,
    }


def _state(actor_id="Ego", object_type="OBJECT_TYPE_VEHICLE", client_key="world"):
    return {
        "actor_info": _info(actor_id, object_type, client_key),
        "transform": {
            "location": {"x": 1.0, "y": 2.0, "z": 3.0},
            "rotation": {"x": 0.0, "y": 0.0, "z": 90.0},
        },
        "velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
        "global_velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
    }


class RecordingClient:
    def __init__(self):
        self.results = defaultdict(deque)
        self.calls = []
        self.closed = False

    def enqueue(self, method, *results):
        self.results[method].extend(results)

    def call_json(self, method, request_json, timeout):
        request = json.loads(request_json)
        self.calls.append((method, request, timeout))
        if not self.results[method]:
            raise AssertionError(f"no result queued for {method}")
        return self.results[method].popleft()

    def close(self):
        self.closed = True


def _controller(fake, **kwargs):
    actor_ref, *_unused, controller_type, _allowed = _types()
    return controller_type(
        fake,
        client_key="heven-task4",
        ego_ref=actor_ref("Ego", "OBJECT_TYPE_VEHICLE", "world"),
        timeout_sec=2.5,
        **kwargs,
    )


def _spawned():
    actor_ref, _state_type, spawned_type, _controller_type, _allowed = _types()
    return spawned_type(
        actor=actor_ref("runtime-42", "OBJECT_TYPE_VEHICLE", "heven-task4"),
        request_id="npc-request-1",
        model_name="30200003",
        label="roundabout-npc",
    )


def test_actor_control_module_and_typed_public_api_exist():
    actor_ref, actor_state, spawned_actor, controller, allowed = _types()

    assert actor_ref.__dataclass_params__.frozen
    assert actor_state.__dataclass_params__.frozen
    assert spawned_actor.__dataclass_params__.frozen
    assert allowed == frozenset(
        {
            GET_MODELS,
            GET_ALL,
            GET_STATE,
            SPAWN,
            DESTROY,
            SET_PAUSE,
            SET_TRANSFORM,
            SET_VELOCITY,
            SET_ROUTE,
        }
    )
    public = {
        name: member
        for name, member in inspect.getmembers(controller, inspect.isfunction)
        if not name.startswith("_")
    }
    assert set(public) == {
        "close",
        "destroy_created",
        "discover_ego",
        "discover_vehicle",
        "get_state",
        "hold_ego",
        "list_actors",
        "list_vehicle_models",
        "pause_actor",
        "register_created",
        "resume_actor",
        "route_actor",
        "set_transform",
        "spawn_npc",
    }
    assert all(
        not ({"method", "path", "rpc", "rpc_path"} & set(inspect.signature(fn).parameters))
        for fn in public.values()
    )


def test_models_and_actor_list_use_canonical_filters_and_reject_duplicate_ids():
    fake = RecordingClient()
    fake.enqueue(
        GET_MODELS,
        _ok(
            {
                "ego_vehicle": ["20200012"],
                "surround_vehicle": ["30200003", "30100014"],
                "pedestrian": [],
                "obstacle": [],
                "spawn_point": [],
                "map_object": [],
            }
        ),
    )
    fake.enqueue(GET_ALL, _ok({"states": [_state("npc-1"), _state("npc-1")]}))
    controller = _controller(fake)

    assert controller.list_vehicle_models() == ("30100014", "30200003")
    with pytest.raises(RuntimeError, match="duplicate actor ID"):
        controller.list_actors()

    assert fake.calls == [
        (
            GET_MODELS,
            {
                "vehicle": True,
                "pedestrian": False,
                "obstacle": False,
                "spawn_point": False,
                "map_object": False,
            },
            2.5,
        ),
        (
            GET_ALL,
            {
                "vehicle": True,
                "pedestrian": True,
                "obstacle": True,
            },
            2.5,
        ),
    ]


def test_get_state_requires_the_exact_returned_actor_identity():
    actor_ref, *_rest = _types()
    fake = RecordingClient()
    target = actor_ref("npc-7", "OBJECT_TYPE_VEHICLE", "heven-task4")
    fake.enqueue(GET_STATE, _ok(_state("npc-7", client_key="somebody-else")))
    controller = _controller(fake)

    with pytest.raises(RuntimeError, match="identity does not match"):
        controller.get_state(target)

    assert fake.calls == [(GET_STATE, _info("npc-7", client_key="heven-task4"), 2.5)]


@pytest.mark.parametrize(
    "vector_path,missing_axis",
    [
        (("transform", "location"), "z"),
        (("transform", "rotation"), "x"),
        (("velocity",), "y"),
        (("global_velocity",), "z"),
    ],
)
def test_actor_state_rejects_missing_vector_axes(vector_path, missing_axis):
    actor_ref, *_rest = _types()
    fake = RecordingClient()
    target = actor_ref("npc-7", "OBJECT_TYPE_VEHICLE", "heven-task4")
    malformed = _state("npc-7", client_key="heven-task4")
    vector = malformed
    for key in vector_path:
        vector = vector[key]
    del vector[missing_axis]
    fake.enqueue(GET_STATE, _ok(malformed))
    controller = _controller(fake)

    with pytest.raises((ValueError, RuntimeError), match="x|y|z|vector|finite"):
        controller.get_state(target)


def test_discover_vehicle_requires_one_exact_vehicle_and_retains_client_key():
    fake = RecordingClient()
    fake.enqueue(
        GET_ALL,
        _ok(
            {
                "states": [
                    _state("Ego", client_key="world-owned"),
                    _state("npc-1", client_key="heven-task4"),
                ]
            }
        ),
    )
    controller = _controller(fake)

    discovered = controller.discover_vehicle("Ego")

    assert discovered.actor_id == "Ego"
    assert discovered.object_type == "OBJECT_TYPE_VEHICLE"
    assert discovered.client_key == "world-owned"


def test_read_only_vehicle_lookup_never_grants_mutation_authority():
    actor_ref, _state_type, _spawned_type, controller_type, _allowed = _types()
    fake = RecordingClient()
    fake.enqueue(
        GET_ALL,
        _ok({"states": [_state("foreign-npc", client_key="foreign-client")]}),
    )
    controller = controller_type(
        fake,
        client_key="heven-task4",
        timeout_sec=2.5,
        ego_ref=actor_ref("Ego", "OBJECT_TYPE_VEHICLE", "world"),
    )
    foreign = controller.discover_vehicle("foreign-npc")
    calls_after_lookup = list(fake.calls)

    rejected_mutations = (
        lambda: controller.pause_actor(foreign),
        lambda: controller.hold_ego(foreign),
        lambda: controller.set_transform(
            foreign, xyz=(1.0, 2.0, 3.0), rpy_deg=(0.0, 0.0, 0.0)
        ),
        lambda: controller.route_actor(
            foreign, decision_range=10.0, links=(("L1", 0),)
        ),
    )
    for mutation in rejected_mutations:
        with pytest.raises(
            RuntimeError,
            match="created manifest|configured Ego|not owned",
        ):
            mutation()

    assert fake.calls == calls_after_lookup


def test_discover_ego_requires_the_exact_configured_identity():
    actor_ref, _state_type, _spawned_type, controller_type, _allowed = _types()
    fake = RecordingClient()
    fake.enqueue(GET_ALL, _ok({"states": [_state("Ego", client_key="foreign")]}))
    controller = controller_type(
        fake,
        client_key="heven-task4",
        timeout_sec=2.5,
        ego_ref=actor_ref("Ego", "OBJECT_TYPE_VEHICLE", "expected-world"),
    )

    with pytest.raises(RuntimeError, match="configured Ego identity"):
        controller.discover_ego()

    assert [method for method, _request, _timeout in fake.calls] == [GET_ALL]


@pytest.mark.parametrize(
    "bad_value",
    [float("nan"), float("inf"), -float("inf")],
)
def test_invalid_spawn_transform_is_rejected_before_any_call(bad_value):
    fake = RecordingClient()
    controller = _controller(fake)

    with pytest.raises(ValueError, match="finite"):
        controller.spawn_npc(
            request_id="npc-request-1",
            model_name="30200003",
            label="npc",
            xyz=(bad_value, 2.0, 3.0),
            rpy_deg=(0.0, 0.0, 90.0),
            velocity=20.0,
        )

    assert fake.calls == []


def test_spawn_captures_actual_id_requeries_exact_state_and_records_manifest():
    fake = RecordingClient()
    fake.enqueue(GET_ALL, _ok({"states": [_state("Ego", client_key="world")]}))
    fake.enqueue(
        SPAWN,
        _ok(
            {
                "status": "STATUS_CODE_SUCCESS",
                "description": "created",
                "custom_message": "runtime-42",
            }
        ),
    )
    fake.enqueue(GET_STATE, _ok(_state("runtime-42", client_key="heven-task4")))
    controller = _controller(fake)

    created = controller.spawn_npc(
        request_id="npc-request-1",
        model_name="30200003",
        label="roundabout-npc",
        xyz=(10.0, 20.0, 30.0),
        rpy_deg=(0.0, 1.0, 90.0),
        velocity=25.0,
        paused=True,
    )

    assert created.actor.actor_id == "runtime-42"
    assert created.actor.client_key == "heven-task4"
    assert controller.created_manifest == (created,)
    assert fake.calls == [
        (
            GET_ALL,
            {"vehicle": True, "pedestrian": True, "obstacle": True},
            2.5,
        ),
        (
            SPAWN,
            {
                "spawn_info": {
                    "actor_info": _info("npc-request-1", client_key="heven-task4"),
                    "transform": {
                        "location": {"x": 10.0, "y": 20.0, "z": 30.0},
                        "rotation": {"x": 0.0, "y": 1.0, "z": 90.0},
                    },
                    "model_name": "30200003",
                    "label": "roundabout-npc",
                    "is_multi_object_one_mode": False,
                },
                "velocity": 25.0,
                "pause": True,
                "multi_ego": False,
            },
            2.5,
        ),
        (GET_STATE, _info("runtime-42", client_key="heven-task4"), 2.5),
    ]


def test_spawn_transport_timeout_is_not_retried_or_recorded():
    fake = RecordingClient()
    fake.enqueue(GET_ALL, _ok({"states": []}))
    fake.enqueue(SPAWN, _transport_failure())
    controller = _controller(fake)

    with pytest.raises(RuntimeError, match="deadline"):
        controller.spawn_npc(
            request_id="npc-request-1",
            model_name="30200003",
            label="npc",
            xyz=(1.0, 2.0, 3.0),
            rpy_deg=(0.0, 0.0, 0.0),
            velocity=10.0,
        )

    assert [method for method, _request, _timeout in fake.calls].count(SPAWN) == 1
    assert controller.created_manifest == ()


def test_spawn_verification_timeout_surfaces_and_retains_pending_actual_id():
    module = importlib.import_module("ad_morai_bridge_dev.actors.control")
    fake = RecordingClient()
    fake.enqueue(GET_ALL, _ok({"states": []}))
    fake.enqueue(
        SPAWN,
        _ok(
            {
                "status": "STATUS_CODE_SUCCESS",
                "custom_message": "actual-runtime-99",
            }
        ),
    )
    fake.enqueue(GET_STATE, _transport_failure())
    controller = _controller(fake)

    with pytest.raises(module.SpawnVerificationError) as captured:
        controller.spawn_npc(
            request_id="npc-request-99",
            model_name="30200003",
            label="npc",
            xyz=(1.0, 2.0, 3.0),
            rpy_deg=(0.0, 0.0, 0.0),
            velocity=10.0,
        )

    pending = captured.value.spawned_actor
    assert pending.actor.actor_id == "actual-runtime-99"
    assert pending.verification_state == "pending"
    assert pending.to_manifest()["reconciliation_required"] is True
    assert controller.created_manifest == (pending,)
    assert [method for method, _request, _timeout in fake.calls].count(SPAWN) == 1


def test_spawn_rejected_result_status_is_not_treated_as_created():
    fake = RecordingClient()
    fake.enqueue(GET_ALL, _ok({"states": []}))
    fake.enqueue(
        SPAWN,
        _ok(
            {
                "status": "STATUS_CODE_UNKNOWN_FAILURE",
                "description": "model unavailable",
                "custom_message": "must-not-be-used",
            }
        ),
    )
    controller = _controller(fake)

    with pytest.raises(RuntimeError, match="model unavailable"):
        controller.spawn_npc(
            request_id="npc-request-1",
            model_name="missing-model",
            label="npc",
            xyz=(1.0, 2.0, 3.0),
            rpy_deg=(0.0, 0.0, 0.0),
            velocity=10.0,
        )

    assert [method for method, _request, _timeout in fake.calls] == [GET_ALL, SPAWN]
    assert controller.created_manifest == ()


def test_mutations_preserve_exact_identity_status_and_canonical_route():
    fake = RecordingClient()
    created = _spawned()
    controller = _controller(fake)
    fake.enqueue(GET_STATE, _ok(_state("runtime-42", client_key="heven-task4")))
    controller.register_created(created)
    for method in (SET_PAUSE, GET_STATE, SET_PAUSE, GET_STATE, SET_ROUTE, GET_STATE):
        if method == GET_STATE:
            fake.enqueue(method, _ok(_state("runtime-42", client_key="heven-task4")))
        else:
            fake.enqueue(method, _ok({"status": "STATUS_CODE_SUCCESS"}))

    controller.pause_actor(created.actor)
    controller.resume_actor(created.actor)
    controller.route_actor(
        created.actor,
        decision_range=15.0,
        links=(
            ("A2256W000133", 3),
            ("A2256W000134", 0),
        ),
    )

    identity = _info("runtime-42", client_key="heven-task4")
    mutation_calls = [call for call in fake.calls if call[0] != GET_STATE]
    assert mutation_calls == [
        (SET_PAUSE, {"actor_info": identity, "enable": True}, 2.5),
        (SET_PAUSE, {"actor_info": identity, "enable": False}, 2.5),
        (
            SET_ROUTE,
            {
                "actor_info": identity,
                "decision_range": 15.0,
                "links": [
                    {"id": {"value": "A2256W000133"}, "waypoint_idx": 3},
                    {"id": {"value": "A2256W000134"}, "waypoint_idx": 0},
                ],
            },
            2.5,
        ),
    ]


@pytest.mark.parametrize("malformed_status", [True, 1.0, "1", None])
def test_mutation_rejects_noncanonical_status_types(malformed_status):
    fake = RecordingClient()
    created = _spawned()
    controller = _controller(fake)
    fake.enqueue(GET_STATE, _ok(_state("runtime-42", client_key="heven-task4")))
    controller.register_created(created)
    fake.enqueue(SET_PAUSE, _ok({"status": malformed_status}))
    fake.enqueue(GET_STATE, _ok(_state("runtime-42", client_key="heven-task4")))

    with pytest.raises(RuntimeError, match="rejected"):
        controller.pause_actor(created.actor)

    assert [method for method, _request, _timeout in fake.calls].count(SET_PAUSE) == 1
    assert [method for method, _request, _timeout in fake.calls].count(GET_STATE) == 1


@pytest.mark.parametrize(
    "links,decision_range",
    [([], 10.0), ([('', 0)], 10.0), ([('L1', -1)], 10.0), ([('L1', 0)], float('nan'))],
)
def test_invalid_route_is_rejected_before_any_call(links, decision_range):
    fake = RecordingClient()
    controller = _controller(fake)
    created = _spawned()
    fake.enqueue(GET_STATE, _ok(_state("runtime-42", client_key="heven-task4")))
    controller.register_created(created)
    calls_before = list(fake.calls)

    with pytest.raises(ValueError):
        controller.route_actor(created.actor, decision_range=decision_range, links=links)

    assert fake.calls == calls_before


def test_hold_ego_uses_exact_discovered_identity_zero_velocity_then_actor_pause():
    fake = RecordingClient()
    controller = _controller(fake)
    fake.enqueue(GET_ALL, _ok({"states": [_state("Ego", client_key="world")]}))
    for method in (SET_VELOCITY, GET_STATE, SET_PAUSE, GET_STATE):
        fake.enqueue(
            method,
            _ok(_state("Ego", client_key="world"))
            if method == GET_STATE
            else _ok({"status": "STATUS_CODE_SUCCESS"}),
        )

    ego = controller.discover_ego()
    controller.hold_ego(ego)

    assert [call for call in fake.calls if call[0] in (SET_VELOCITY, SET_PAUSE)] == [
        (SET_VELOCITY, {"actor_info": _info("Ego"), "velocity": 0.0}, 2.5),
        (SET_PAUSE, {"actor_info": _info("Ego"), "enable": True}, 2.5),
    ]


def test_destroy_created_uses_one_exact_destroy_and_verifies_absence():
    fake = RecordingClient()
    controller = _controller(fake)
    created = _spawned()
    fake.enqueue(GET_ALL, _ok({"states": [_state("runtime-42", client_key="heven-task4")]}))
    fake.enqueue(DESTROY, _ok({"status": "STATUS_CODE_SUCCESS"}))
    fake.enqueue(GET_ALL, _ok({"states": []}))

    assert controller.destroy_created(created) is True

    assert fake.calls == [
        (GET_ALL, {"vehicle": True, "pedestrian": True, "obstacle": True}, 2.5),
        (DESTROY, _info("runtime-42", client_key="heven-task4"), 2.5),
        (GET_ALL, {"vehicle": True, "pedestrian": True, "obstacle": True}, 2.5),
    ]
    assert all("DestroyAllActors" not in method for method, _request, _timeout in fake.calls)


def test_destroy_skips_identity_mismatch_before_destructive_call():
    fake = RecordingClient()
    controller = _controller(fake)
    created = _spawned()
    fake.enqueue(GET_ALL, _ok({"states": [_state("runtime-42", client_key="other-owner")]}))

    with pytest.raises(RuntimeError, match="identity mismatch"):
        controller.destroy_created(created)

    assert [method for method, _request, _timeout in fake.calls] == [GET_ALL]


def test_destroy_transport_timeout_is_not_retried():
    fake = RecordingClient()
    controller = _controller(fake)
    created = _spawned()
    fake.enqueue(GET_ALL, _ok({"states": [_state("runtime-42", client_key="heven-task4")]}))
    fake.enqueue(DESTROY, _transport_failure())

    with pytest.raises(RuntimeError, match="deadline"):
        controller.destroy_created(created)

    assert [method for method, _request, _timeout in fake.calls].count(DESTROY) == 1


@pytest.mark.parametrize(
    "forbidden",
    [
        "morai_sim_api.scenario.Scenario/LoadMoraiScenario",
        "morai_sim_api.map.Map/Load",
        "morai_sim_api.simulation.Simulation/Start",
        "morai_sim_api.simulation.Simulation/Stop",
        "morai_sim_api.simulation.Simulation/Pause",
        "morai_sim_api.simulation.Simulation/Resume",
        "morai_sim_api.actor.Actor/DestroyAllActors",
    ],
)
def test_forbidden_rpc_is_rejected_before_client_call(forbidden):
    fake = RecordingClient()
    controller = _controller(fake)

    with pytest.raises(RuntimeError, match="prohibited MORAI actor RPC"):
        controller._rpc(forbidden, {})

    assert fake.calls == []
