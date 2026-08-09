from ad_vehicle_profiling.loop_guard_node import (
    ActorState,
    GuardLimits,
    reset_reason,
    reposition_ego,
)


class Result:
    def __init__(self, response_json='{"status":"STATUS_CODE_SUCCESS"}'):
        self.success = True
        self.response_json = response_json
        self.error = ""
        self.status = "OK"


class RecordingClient:
    def __init__(self):
        self.calls = []

    def call_json(self, method, payload, timeout):
        self.calls.append((method, payload, timeout))
        return Result()


def test_loop_boundary_only_repositions_between_measurements():
    limits = GuardLimits()
    outside = ActorState(0.0, -1001.0, 0.36)

    assert reset_reason(outside, "REACH_TARGET_SPEED", limits) == (
        "longitudinal_loop_boundary"
    )
    assert reset_reason(outside, "APPLY_TEST_COMMAND", limits) is None


def test_fall_triggers_emergency_in_every_phase():
    state = ActorState(0.0, -1001.0, -1.0)

    assert reset_reason(
        state, "APPLY_TEST_COMMAND", GuardLimits()
    ) == "emergency_fall"


def test_reposition_pauses_moves_restores_speed_and_resumes():
    client = RecordingClient()

    reposition_ego(
        client,
        speed_mps=12.5,
        target_location=(0.0, 0.0, 0.36),
        target_rotation_deg=(0.0, 0.0, -88.62),
        timeout_sec=3.0,
    )

    methods = [method for method, _, _ in client.calls]
    assert methods[0].endswith("Simulation/Pause")
    assert methods[1].endswith("Actor/ControlVehicle")
    assert methods[2].endswith("Actor/SetTransform")
    assert methods[3].endswith("Actor/SetVelocity")
    assert methods[4].endswith("Simulation/Resume")
    assert '"velocity": 12.5' in client.calls[3][1]
