"""Smoke test: the built development interfaces import and expose the fields
the dev bridge relies on."""

from ad_morai_interfaces_dev.msg import (
    EgoVehicleStatus,
    ObjectStatusArray,
    ScenarioLoad,
)
from ad_morai_interfaces_dev.srv import MoraiGrpcCall


def test_dev_messages_construct():
    assert EgoVehicleStatus() is not None
    assert ObjectStatusArray() is not None
    assert ScenarioLoad() is not None


def test_grpc_call_service_keeps_the_json_bridge_fields():
    request = MoraiGrpcCall.Request()
    response = MoraiGrpcCall.Response()
    assert hasattr(request, "method")
    assert hasattr(request, "request_json")
    assert hasattr(request, "timeout_sec")
    assert hasattr(response, "success")
    assert hasattr(response, "grpc_code")
    assert hasattr(response, "response_json")
