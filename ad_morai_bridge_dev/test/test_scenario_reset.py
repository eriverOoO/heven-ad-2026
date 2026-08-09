import json

import pytest

from ad_morai_bridge_dev.simulator_grpc.client import GrpcCallResult


def write_scenario(path):
    path.write_text(
        json.dumps(
            {
                "egoVehicle": {
                    "initPosition": {
                        "pos": {
                            "x": -131.486,
                            "y": -427.961,
                            "z": 28.883,
                        },
                        "rot": {
                            "roll": "-359.540",
                            "pitch": "0.121",
                            "yaw": "62.515",
                        },
                    }
                },
                "vehicleList": [
                    {
                        "UNIQUEID": 13,
                        "initPosition": {
                            "pos": {"x": 65.378, "y": -31.143, "z": 28.646},
                            "rot": {
                                "roll": "-359.674",
                                "pitch": "0.450",
                                "yaw": "-91.588",
                            },
                        },
                        "currentVelocity": 49.9945,
                    }
                ],
                "pedestrianList": [
                    {
                        "UNIQUEID": 3,
                        "pos": {
                            "x": -56.744,
                            "y": -11.614,
                            "z": 28.561,
                        },
                        "rot": {
                            "roll": "0",
                            "pitch": "0",
                            "yaw": "-83.262",
                        },
                        "speed": 10.0,
                    }
                ],
                "objectList": [
                    {
                        "UNIQUEID": 2,
                        "pos": {
                            "x": -60.610,
                            "y": -142.178,
                            "z": 28.374,
                        },
                        "rot": {
                            "roll": "-0.199",
                            "pitch": "0.016",
                            "yaw": "-83.219",
                        },
                    }
                ],
                "waypointDataList": [
                    {
                        "pathName": "PedPath - 3",
                        "waypointData": [
                            {
                                "pos": {
                                    "x": 56.744,
                                    "y": 28.561,
                                    "z": 11.614,
                                }
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_load_reset_plan_uses_world_pose_and_unique_ids(tmp_path):
    from ad_morai_bridge_dev.scenarios.reset import load_reset_plan

    path = tmp_path / "scene.json"
    write_scenario(path)

    plan = load_reset_plan(path)

    assert plan.ego.actor_id == "Ego"
    assert plan.ego.pose.location == (-131.486, -427.961, 28.883)
    assert [(actor.actor_id, actor.object_type) for actor in plan.actors] == [
        ("13", "OBJECT_TYPE_VEHICLE"),
        ("3", "OBJECT_TYPE_PEDESTRIAN"),
        ("2", "OBJECT_TYPE_OBSTACLE"),
    ]
    assert plan.actors[0].velocity == pytest.approx(49.9945)
    assert plan.actors[1].pose.location == (-56.744, -11.614, 0.0)


def test_load_reset_plan_reports_missing_required_field(tmp_path):
    from ad_morai_bridge_dev.scenarios.reset import load_reset_plan

    path = tmp_path / "scene.json"
    path.write_text(
        '{"pedestrianList": [], "objectList": []}', encoding="utf-8"
    )

    with pytest.raises(ValueError, match="egoVehicle"):
        load_reset_plan(path)


class FakeClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def call_json(self, method, request_json, timeout):
        self.calls.append((method, json.loads(request_json), timeout))
        return next(self.responses)


def ok_result(payload=None):
    payload = payload or {"status": "STATUS_CODE_SUCCESS"}
    return GrpcCallResult(True, 0, "OK", json.dumps(payload))


def actor_states(*actor_ids):
    return {
        "states": [
            {"actor_info": {"id": {"value": actor_id}}}
            for actor_id in actor_ids
        ]
    }


def test_reset_uses_loaded_pedestrian_path_and_always_resumes(tmp_path):
    from ad_morai_bridge_dev.scenarios.reset import (
        load_reset_plan,
        reset_scenario,
    )

    path = tmp_path / "scene.json"
    write_scenario(path)
    client = FakeClient(
        [ok_result(), ok_result(actor_states("Ego", "13", "3", "2"))]
        + [ok_result()] * 9
    )

    reset_scenario(client, load_reset_plan(path), 5.0)

    methods = [call[0] for call in client.calls]
    assert methods[0].endswith("Simulation/Pause")
    assert methods[1].endswith("Actor/GetAllActorsState")
    assert methods[-1].endswith("Simulation/Resume")
    assert not any(
        method.endswith("Actor/SetPedestrianWaypoint") for method in methods
    )
    vehicle_velocity_calls = [
        payload
        for method, payload, _ in client.calls
        if method.endswith("Actor/SetVelocity")
        and payload["actor_info"]["id"]["value"] == "13"
    ]
    assert vehicle_velocity_calls == [
        {
            "actor_info": {
                "id": {"value": "13"},
                "object_type": "OBJECT_TYPE_VEHICLE",
            },
            "velocity": pytest.approx(49.9945),
        }
    ]


def test_reset_resumes_after_mid_reset_rejection(tmp_path):
    from ad_morai_bridge_dev.scenarios.reset import (
        load_reset_plan,
        reset_scenario,
    )

    path = tmp_path / "scene.json"
    write_scenario(path)
    rejected = ok_result(
        {
            "status": "STATUS_CODE_UNKNOWN_ERROR",
            "description": "rejected",
        }
    )
    client = FakeClient(
        [
            ok_result(),
            ok_result(actor_states("Ego", "13", "3", "2")),
            rejected,
            ok_result(),
        ]
    )

    with pytest.raises(RuntimeError, match="rejected"):
        reset_scenario(client, load_reset_plan(path), 5.0)

    assert client.calls[-1][0].endswith("Simulation/Resume")
