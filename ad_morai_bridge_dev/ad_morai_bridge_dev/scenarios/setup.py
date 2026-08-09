import json


LOAD_SCENARIO = "morai_sim_api.scenario.Scenario/LoadMoraiScenario"
RESUME_SIMULATION = "morai_sim_api.simulation.Simulation/Resume"


def _require_morai_success(action, result):
    if not result.success:
        raise RuntimeError(result.error or f"{result.grpc_status} ({result.grpc_code})")

    response = json.loads(result.response_json or "{}")
    if response.get("status") not in ("STATUS_CODE_SUCCESS", 1):
        raise RuntimeError(response.get("description") or f"MORAI rejected {action}")


def load_scenario_and_resume(client, scenario_file, timeout_sec):
    load_result = client.call_json(
        LOAD_SCENARIO,
        json.dumps({"value": scenario_file}),
        timeout_sec,
    )
    _require_morai_success("scenario load", load_result)

    resume_result = client.call_json(RESUME_SIMULATION, "{}", timeout_sec)
    _require_morai_success("simulation resume", resume_result)
