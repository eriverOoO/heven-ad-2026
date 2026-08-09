import json

import pytest

from ad_morai_bridge_dev.simulator_grpc.client import GrpcCallResult


class FakeClient:
    def __init__(self, results):
        self.results = iter(results)
        self.calls = []

    def call_json(self, method, request_json, timeout):
        self.calls.append((method, json.loads(request_json), timeout))
        return next(self.results)


def success_result(status="STATUS_CODE_SUCCESS", description=""):
    return GrpcCallResult(
        True,
        0,
        "OK",
        response_json=json.dumps(
            {"status": status, "description": description}
        ),
    )


def test_scenario_is_loaded_before_simulation_resumes():
    from ad_morai_bridge_dev.scenarios.setup import load_scenario_and_resume

    client = FakeClient([success_result(), success_result()])

    load_scenario_and_resume(client, "/sim/2026_molit_comp_sample_scene.json", 8.0)

    assert client.calls == [
        (
            "morai_sim_api.scenario.Scenario/LoadMoraiScenario",
            {"value": "/sim/2026_molit_comp_sample_scene.json"},
            8.0,
        ),
        ("morai_sim_api.simulation.Simulation/Resume", {}, 8.0),
    ]


def test_transport_failure_stops_before_resume():
    from ad_morai_bridge_dev.scenarios.setup import load_scenario_and_resume

    client = FakeClient(
        [GrpcCallResult(False, 14, "UNAVAILABLE", error="simulator offline")]
    )

    with pytest.raises(RuntimeError, match="simulator offline"):
        load_scenario_and_resume(client, "scene.json", 5.0)

    assert len(client.calls) == 1


def test_morai_rejection_stops_before_resume():
    from ad_morai_bridge_dev.scenarios.setup import load_scenario_and_resume

    client = FakeClient(
        [success_result("STATUS_CODE_UNKNOWN_ERROR", "scenario not found")]
    )

    with pytest.raises(RuntimeError, match="scenario not found"):
        load_scenario_and_resume(client, "missing.json", 5.0)

    assert len(client.calls) == 1


def test_empty_default_result_is_failure_not_success():
    from ad_morai_bridge_dev.scenarios.setup import load_scenario_and_resume

    client = FakeClient([GrpcCallResult(True, 0, "OK", response_json="{}")])

    with pytest.raises(RuntimeError, match="MORAI rejected scenario load"):
        load_scenario_and_resume(client, "scene.json", 5.0)

    assert len(client.calls) == 1
