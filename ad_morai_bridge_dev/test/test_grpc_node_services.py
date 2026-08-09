import json

from ad_morai_interfaces_dev.srv import MoraiGrpcCall, MoraiStep, MoraiSyncMode

from ad_morai_bridge_dev.simulator_grpc.client import GrpcCallResult
from ad_morai_bridge_dev.bridge.node import AdMoraiDevBridge


class RecordingClient:
    def __init__(self, *results):
        self.results = list(results)
        self.calls = []
        self.closed = False

    def call_json(self, method, request_json, timeout=None):
        self.calls.append((method, request_json, timeout))
        return self.results.pop(0)

    def close(self):
        self.closed = True


def _bridge(client):
    bridge = object.__new__(AdMoraiDevBridge)
    bridge._grpc_client = client
    bridge._grpc_timeout = 5.0
    return bridge


def test_generic_service_forwards_method_json_and_timeout():
    client = RecordingClient(
        GrpcCallResult(True, 0, "OK", '{"values":["KCity"]}')
    )
    bridge = _bridge(client)
    request = MoraiGrpcCall.Request()
    request.method = "morai_sim_api.simualtor.Simulator/GetAvailableMaps"
    request.request_json = "{}"
    request.timeout_sec = 2.0

    response = bridge._on_grpc_call(request, MoraiGrpcCall.Response())

    assert response.success
    assert response.grpc_code == 0
    assert response.grpc_status == "OK"
    assert response.response_json == '{"values":["KCity"]}'
    assert response.error_message == ""
    assert client.calls == [(request.method, "{}", 2.0)]


def test_sync_query_reports_active_mode_and_tick_period():
    client = RecordingClient(
        GrpcCallResult(
            True,
            0,
            "OK",
            '{"type":"SYNC_MODE_TYPE_SYNCHRONOUS","tick_period":20}',
        )
    )
    bridge = _bridge(client)
    request = MoraiSyncMode.Request()
    request.set_mode = False

    response = bridge._on_sync_mode(request, MoraiSyncMode.Response())

    assert response.success
    assert response.active_mode == MoraiSyncMode.Request.MODE_SYNCHRONOUS
    assert response.active_tick_period_ms == 20
    assert client.calls == [
        ("morai_sim_api.simulation.Simulation/GetSynchronousMode", "{}", 5.0)
    ]


def test_sync_set_uses_official_fields_then_queries_active_mode():
    client = RecordingClient(
        GrpcCallResult(True, 0, "OK", '{"status":"STATUS_CODE_SUCCESS"}'),
        GrpcCallResult(
            True,
            0,
            "OK",
            '{"type":"SYNC_MODE_TYPE_TIME_MANAGEMENT","tick_period":10}',
        ),
    )
    bridge = _bridge(client)
    request = MoraiSyncMode.Request()
    request.set_mode = True
    request.mode = MoraiSyncMode.Request.MODE_TIME_MANAGEMENT
    request.tick_period_ms = 10

    response = bridge._on_sync_mode(request, MoraiSyncMode.Response())

    assert response.success
    assert response.active_mode == MoraiSyncMode.Request.MODE_TIME_MANAGEMENT
    assert response.active_tick_period_ms == 10
    assert client.calls[0][0].endswith("/SetSynchronousMode")
    assert json.loads(client.calls[0][1]) == {"type": 1, "tick_period": 10}
    assert client.calls[1][0].endswith("/GetSynchronousMode")


def test_step_returns_sync_timestamp_and_rejects_non_positive_count():
    client = RecordingClient(
        GrpcCallResult(
            True,
            0,
            "OK",
            '{"frame_count":"7","elapsed_time":"20"}',
        )
    )
    bridge = _bridge(client)
    request = MoraiStep.Request()
    request.frame_count = 1

    response = bridge._on_step(request, MoraiStep.Response())

    assert response.success
    assert response.current_frame == 7
    assert response.elapsed_time == 20
    assert json.loads(client.calls[0][1]) == {"value": 1}

    invalid = MoraiStep.Request()
    invalid.frame_count = 0
    rejected = bridge._on_step(invalid, MoraiStep.Response())
    assert not rejected.success
    assert rejected.grpc_code == 3
    assert rejected.grpc_status == "INVALID_ARGUMENT"
    assert len(client.calls) == 1


def test_close_transports_closes_grpc_client_without_calling_simulator():
    client = RecordingClient()
    bridge = _bridge(client)
    bridge._closed = False
    bridge._receivers = {}
    bridge._senders = {}

    bridge._close_transports()

    assert client.closed
    assert client.calls == []
