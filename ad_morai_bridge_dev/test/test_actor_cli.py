import importlib
import json
from pathlib import Path
from collections import defaultdict, deque

import pytest

from ad_morai_bridge_dev.simulator_grpc.client import GrpcCallResult


def _module():
    return importlib.import_module("ad_morai_bridge_dev.actors.cli")


class RejectConnection:
    def __call__(self, *args, **kwargs):
        raise AssertionError("dry-run must not connect")


class CliRecordingClient:
    def __init__(self):
        self.results = defaultdict(deque)
        self.calls = []
        self.closed = False

    def enqueue(self, method, *results):
        self.results[method].extend(results)

    def call_json(self, method, request_json, timeout):
        self.calls.append((method, json.loads(request_json), timeout))
        if not self.results[method]:
            raise AssertionError(f"no result queued for {method}")
        return self.results[method].popleft()

    def close(self):
        self.closed = True


def _ok(payload):
    return GrpcCallResult(True, 0, "OK", response_json=json.dumps(payload))


def _deadline():
    return GrpcCallResult(False, 4, "DEADLINE_EXCEEDED", error="deadline")


def _spawn_state(actor_id):
    return {
        "actor_info": {
            "id": {"value": actor_id},
            "object_type": "OBJECT_TYPE_VEHICLE",
            "client_key": "heven-task4",
        },
        "transform": {
            "location": {"x": 1.0, "y": 2.0, "z": 3.0},
            "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
        },
        "velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
        "global_velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
    }


def _spawn_args(manifest_out=None):
    args = [
        "--client-key",
        "heven-task4",
        "spawn-npc",
        "--request-id",
        "npc-1",
        "--model-name",
        "30200003",
        "--label",
        "npc",
        "--xyz",
        "1",
        "2",
        "3",
        "--rpy-deg",
        "0",
        "0",
        "0",
        "--velocity",
        "10",
    ]
    if manifest_out is not None:
        args.extend(("--manifest-out", str(manifest_out)))
    return args


def test_all_exact_commands_are_registered():
    parser = _module().build_parser()
    subparsers = next(
        action for action in parser._actions if action.__class__.__name__ == "_SubParsersAction"
    )

    assert set(subparsers.choices) == {
        "models",
        "list",
        "hold-ego",
        "teleport-ego",
        "spawn-npc",
        "route-npc",
        "pause-actor",
        "resume-actor",
        "state",
        "destroy-created",
    }


def test_ego_mutation_commands_do_not_accept_an_arbitrary_actor_id():
    parser = _module().build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["hold-ego", "--actor-id", "foreign-npc"])
    configured = parser.parse_args(
        [
            "--ego-id",
            "Ego",
            "--ego-client-key",
            "world",
            "hold-ego",
        ]
    )
    assert configured.ego_id == "Ego"
    assert configured.ego_client_key == "world"


def test_spawn_dry_run_prints_canonical_json_without_connecting(capsys):
    result = _module().main(
        [
            "--dry-run",
            "--client-key",
            "heven-task4",
            "spawn-npc",
            "--request-id",
            "npc-1",
            "--model-name",
            "30200003",
            "--label",
            "npc",
            "--xyz",
            "1",
            "2",
            "3",
            "--rpy-deg",
            "0",
            "0",
            "90",
            "--velocity",
            "20",
        ],
        connect=RejectConnection(),
    )

    assert result == 0
    output = json.loads(capsys.readouterr().out)
    assert output == {
        "dry_run": True,
        "operations": [
            {
                "method": "morai_sim_api.actor.Actor/SpawnVehicle",
                "request": {
                    "spawn_info": {
                        "actor_info": {
                            "id": {"value": "npc-1"},
                            "object_type": "OBJECT_TYPE_VEHICLE",
                            "client_key": "heven-task4",
                        },
                        "transform": {
                            "location": {"x": 1.0, "y": 2.0, "z": 3.0},
                            "rotation": {"x": 0.0, "y": 0.0, "z": 90.0},
                        },
                        "model_name": "30200003",
                        "label": "npc",
                        "is_multi_object_one_mode": False,
                    },
                    "velocity": 20.0,
                    "pause": False,
                    "multi_ego": False,
                },
            }
        ],
    }


def test_destroy_dry_run_requires_exact_spawn_manifest_and_never_connects(tmp_path, capsys):
    manifest = tmp_path / "created.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "actor": {
                    "actor_id": "runtime-42",
                    "object_type": "OBJECT_TYPE_VEHICLE",
                    "client_key": "heven-task4",
                },
                "request_id": "npc-1",
                "model_name": "30200003",
                "label": "npc",
            }
        ),
        encoding="utf-8",
    )

    result = _module().main(
        [
            "--dry-run",
            "--client-key",
            "heven-task4",
            "destroy-created",
            "--manifest",
            str(manifest),
        ],
        connect=RejectConnection(),
    )

    assert result == 0
    output = json.loads(capsys.readouterr().out)
    assert output["operations"] == [
        {
            "method": "morai_sim_api.actor.Actor/DestroyActor",
            "request": {
                "id": {"value": "runtime-42"},
                "object_type": "OBJECT_TYPE_VEHICLE",
                "client_key": "heven-task4",
            },
        }
    ]


def test_read_only_dry_run_does_not_load_descriptors_or_connect(capsys):
    assert _module().main(["--dry-run", "models"], connect=RejectConnection()) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["operations"][0]["method"].endswith("GetAvailableObject")


def test_state_dry_run_keeps_selector_outside_canonical_request(capsys):
    assert (
        _module().main(
            ["--dry-run", "state", "--actor-id", "Ego"],
            connect=RejectConnection(),
        )
        == 0
    )
    operation = json.loads(capsys.readouterr().out)["operations"][0]

    assert operation["request"] == {
        "vehicle": True,
        "pedestrian": True,
        "obstacle": True,
    }
    assert operation["select_exact_vehicle_id"] == "Ego"


@pytest.mark.parametrize("timeout", ["0", "-1", "nan", "inf"])
def test_invalid_real_call_timeout_is_rejected_before_connection(timeout):
    with pytest.raises(ValueError, match="timeout-sec"):
        _module().main(
            ["--timeout-sec", timeout, "models"],
            connect=RejectConnection(),
        )


def test_spawn_verification_failure_emits_pending_actual_id_and_returns_nonzero(capsys):
    module = _module()
    fake = CliRecordingClient()
    fake.enqueue(module.GET_ALL_ACTORS_STATE, _ok({"states": []}))
    fake.enqueue(
        module.SPAWN_VEHICLE,
        _ok(
            {
                "status": "STATUS_CODE_SUCCESS",
                "custom_message": "actual-runtime-99",
            }
        ),
    )
    fake.enqueue("morai_sim_api.actor.Actor/GetActorState", _deadline())

    code = module.main(
        _spawn_args(),
        connect=lambda _target, _timeout: fake,
    )

    assert code != 0
    recovery = json.loads(capsys.readouterr().err)
    assert recovery["error"] == "spawn_verification_failed"
    assert recovery["manifest"]["actor"]["actor_id"] == "actual-runtime-99"
    assert recovery["manifest"]["verification_state"] == "pending"
    assert recovery["manifest"]["reconciliation_required"] is True
    assert [call[0] for call in fake.calls].count(module.SPAWN_VEHICLE) == 1


def test_manifest_destination_is_preflighted_before_spawn_call(tmp_path):
    module = _module()
    destination = tmp_path / "created.json"
    destination.write_text("do not replace", encoding="utf-8")
    fake = CliRecordingClient()

    with pytest.raises(ValueError, match="already exists"):
        module.main(
            _spawn_args(destination),
            connect=lambda _target, _timeout: fake,
        )

    assert fake.calls == []
    assert destination.read_text(encoding="utf-8") == "do not replace"


def test_post_spawn_manifest_write_failure_emits_verified_recovery_identity(
    tmp_path, monkeypatch, capsys
):
    module = _module()
    destination = tmp_path / "created.json"
    fake = CliRecordingClient()
    fake.enqueue(module.GET_ALL_ACTORS_STATE, _ok({"states": []}))
    fake.enqueue(
        module.SPAWN_VEHICLE,
        _ok(
            {
                "status": "STATUS_CODE_SUCCESS",
                "custom_message": "actual-runtime-42",
            }
        ),
    )
    fake.enqueue(
        "morai_sim_api.actor.Actor/GetActorState",
        _ok(_spawn_state("actual-runtime-42")),
    )

    def fail_write(_self, _manifest):
        raise OSError("disk full")

    monkeypatch.setattr(module._ManifestDestination, "write", fail_write)
    code = module.main(
        _spawn_args(destination),
        connect=lambda _target, _timeout: fake,
    )

    assert code != 0
    recovery = json.loads(capsys.readouterr().err)
    assert recovery["error"] == "manifest_write_failed"
    assert recovery["manifest"]["actor"]["actor_id"] == "actual-runtime-42"
    assert recovery["manifest"]["verification_state"] == "verified"
    assert recovery["manifest"]["reconciliation_required"] is False
    assert [call[0] for call in fake.calls].count(module.SPAWN_VEHICLE) == 1


def test_setup_declares_console_entrypoint_and_installs_preset():
    repository = Path(__file__).resolve().parents[2]
    setup_text = (repository / "ad_morai_bridge_dev" / "setup.py").read_text(
        encoding="utf-8"
    )

    assert "ad_morai_actor = ad_morai_bridge_dev.actors.cli:main" in setup_text
    assert 'glob("config/*.yaml")' in setup_text
