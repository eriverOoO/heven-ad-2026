import json
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
DESCRIPTOR_SET = (
    REPOSITORY / "ad_morai_bridge_dev" / "data" / "morai_api.desc"
)


class FakeCode:
    def __init__(self, number, name):
        self.value = (number, name.lower())
        self.name = name


class FakeRpcError(Exception):
    def __init__(self, code, details):
        super().__init__(details)
        self._code = code
        self._details = details

    def code(self):
        return self._code

    def details(self):
        return self._details


class FakeGrpc:
    __version__ = "1.82.1"
    RpcError = FakeRpcError

    @staticmethod
    def insecure_channel(target):
        raise AssertionError(f"unexpected channel creation for {target}")


class FakeChannel:
    def __init__(self, api, *, error=None):
        self.api = api
        self.error = error
        self.calls = []
        self.closed = False

    def unary_unary(self, path, request_serializer, response_deserializer):
        method = self.api.resolve_method(path)

        def invoke(request, timeout):
            self.calls.append((path, request_serializer(request), timeout))
            if self.error is not None:
                raise self.error
            response_type = _message_type(self.api, method.output_type)
            response = response_type()
            if method.name == "Tick":
                response.frame_count = 2
                response.elapsed_time = 40
            elif method.name == "GetActorState":
                response.actor_info.id.value = "zero-actor"
                response.actor_info.object_type = 1
                response.actor_info.client_key = "key"
                response.transform.location.SetInParent()
                response.transform.rotation.SetInParent()
                response.velocity.SetInParent()
                response.global_velocity.SetInParent()
            return response_deserializer(response.SerializeToString())

        return invoke

    def close(self):
        self.closed = True


def _message_type(api, descriptor):
    from google.protobuf import message_factory

    if hasattr(message_factory, "GetMessageClass"):
        return message_factory.GetMessageClass(descriptor)
    return message_factory.MessageFactory(api.pool).GetPrototype(descriptor)


def _client(*, error=None, default_timeout=5.0):
    from ad_morai_bridge_dev.simulator_grpc.client import MoraiGrpcClient
    from ad_morai_bridge_dev.simulator_grpc.descriptors import MoraiApi

    api = MoraiApi.load(DESCRIPTOR_SET)
    channel = FakeChannel(api, error=error)
    return MoraiGrpcClient(
        api,
        channel,
        default_timeout=default_timeout,
        grpc_module=FakeGrpc,
    ), channel


def test_call_json_uses_dynamic_request_and_response_types():
    client, channel = _client()

    result = client.call_json(
        "morai_sim_api.simulation.Simulation/Tick",
        '{"value": 2}',
        timeout=1.5,
    )

    assert result.success
    assert result.code == 0
    assert result.status == "OK"
    assert json.loads(result.response_json) == {
        "frame_count": "2",
        "elapsed_time": "40",
    }
    assert channel.calls[0][0] == "/morai_sim_api.simulation.Simulation/Tick"
    assert channel.calls[0][2] == 1.5


def test_non_positive_timeout_uses_configured_default():
    client, channel = _client(default_timeout=3.25)

    result = client.call_json(
        "morai_sim_api.simualtor.Simulator/GetAvailableMaps",
        "{}",
        timeout=0.0,
    )

    assert result.success
    assert channel.calls[0][2] == 3.25


def test_invalid_json_and_unknown_method_are_rejected_without_network_call():
    client, channel = _client()

    invalid_json = client.call_json(
        "morai_sim_api.simulation.Simulation/Tick", "not-json"
    )
    unknown_method = client.call_json("unknown.Service/Method", "{}")

    assert not invalid_json.success
    assert invalid_json.code == 3
    assert invalid_json.status == "INVALID_ARGUMENT"
    assert not unknown_method.success
    assert unknown_method.code == 3
    assert channel.calls == []


def test_grpc_error_preserves_status_code_and_details():
    error = FakeRpcError(FakeCode(14, "UNAVAILABLE"), "server is offline")
    client, _channel = _client(error=error)

    result = client.call_json(
        "morai_sim_api.simualtor.Simulator/GetTimestamp", "{}"
    )

    assert not result.success
    assert result.code == 14
    assert result.status == "UNAVAILABLE"
    assert result.error == "server is offline"


def test_close_closes_the_single_channel():
    client, channel = _client()

    client.close()

    assert channel.closed


def test_zero_vector_responses_include_every_canonical_axis():
    client, _channel = _client()

    result = client.call_json(
        "morai_sim_api.actor.Actor/GetActorState",
        json.dumps(_actor_info("zero-actor")),
    )

    response = json.loads(result.response_json)
    for vector in (
        response["transform"]["location"],
        response["transform"]["rotation"],
        response["velocity"],
        response["global_velocity"],
    ):
        assert vector == {"x": 0.0, "y": 0.0, "z": 0.0}


def _actor_info(actor_id):
    return {
        "id": {"value": actor_id},
        "object_type": "OBJECT_TYPE_VEHICLE",
        "client_key": "key",
    }
