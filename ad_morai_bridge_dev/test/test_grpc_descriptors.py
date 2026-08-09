import hashlib
from pathlib import Path

import pytest
from google.protobuf.descriptor_pb2 import FileDescriptorSet

from ad_morai_bridge_dev.simulator_grpc.descriptors import MoraiApi


REPOSITORY = Path(__file__).resolve().parents[2]
DESCRIPTOR_SET = REPOSITORY / "ad_morai_bridge_dev" / "data" / "morai_api.desc"


def test_load_resolves_real_methods_from_the_committed_descriptor_set():
    api = MoraiApi.load()

    assert api.methods
    tick = api.resolve_method("morai_sim_api.simulation.Simulation/Tick")
    assert tick.name == "Tick"
    assert tick.input_type is not None
    # A leading slash (gRPC path form) resolves to the same method.
    assert (
        api.resolve_method("/morai_sim_api.simulation.Simulation/Tick") is tick
    )


def test_unknown_method_is_rejected_locally():
    api = MoraiApi.load()
    with pytest.raises(ValueError, match="unknown MORAI gRPC method"):
        api.resolve_method("morai_sim_api.simulation.Simulation/DoesNotExist")


def test_actor_descriptor_inventory_and_provenance_are_pinned():
    api = MoraiApi.load(DESCRIPTOR_SET)
    actor_methods = {
        name: method
        for name, method in api.methods.items()
        if name.startswith("morai_sim_api.actor.Actor/")
    }

    assert len(api.methods) == 93
    assert len(actor_methods) == 38
    descriptor_set = FileDescriptorSet.FromString(DESCRIPTOR_SET.read_bytes())
    actor_service_protos = [
        service
        for file_proto in descriptor_set.file
        for service in file_proto.service
        if f"{file_proto.package}.{service.name}" == "morai_sim_api.actor.Actor"
    ]
    assert len(actor_service_protos) == 1
    assert all(
        not method.client_streaming and not method.server_streaming
        for method in actor_service_protos[0].method
    )
    assert hashlib.sha256(DESCRIPTOR_SET.read_bytes()).hexdigest() == (
        "764afbae20d60ec4614d6b9abe47218c72925671ffa20773a60f192277b9ce5a"
    )
    contracts = {
        name.rsplit("/", 1)[-1]: (
            method.input_type.full_name,
            method.output_type.full_name,
        )
        for name, method in actor_methods.items()
    }
    assert contracts["SpawnVehicle"] == (
        "morai_sim_api.actor.VehicleSpawnParam",
        "morai_sim_api.Result",
    )
    assert contracts["DestroyActor"] == (
        "morai_sim_api.ObjectInfo",
        "morai_sim_api.Result",
    )
    assert contracts["SetVehicleRoute"] == (
        "morai_sim_api.actor.VehicleRoute",
        "morai_sim_api.Result",
    )
