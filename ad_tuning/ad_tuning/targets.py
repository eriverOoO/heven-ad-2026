from __future__ import annotations

from dataclasses import dataclass

from ad_tuning.parameters import FloatParameter, IntParameter, SuggestingTrial


Parameter = FloatParameter | IntParameter


@dataclass(frozen=True)
class TuningTarget:
    domain: str
    algorithm: str
    parameters: tuple[Parameter, ...]

    def __post_init__(self) -> None:
        if not self.domain or "/" in self.domain:
            raise ValueError("target domain must be one path component")
        if not self.algorithm or "/" in self.algorithm:
            raise ValueError("target algorithm must be one path component")
        names = tuple(parameter.name for parameter in self.parameters)
        if len(names) != len(set(names)):
            raise ValueError("target parameter names must not be duplicate")
        prefix = self.algorithm + "."
        if any(not name.startswith(prefix) for name in names):
            raise ValueError("target parameters must use the algorithm namespace")

    @property
    def identifier(self) -> str:
        return self.domain + "/" + self.algorithm

    def suggest(self, trial: SuggestingTrial) -> dict[str, float | int]:
        return {parameter.name: parameter.suggest(trial) for parameter in self.parameters}


def stanley_parameters(prefix: str) -> tuple[Parameter, ...]:
    return (
        FloatParameter(f"{prefix}.cross_track_gain", 0.5, 1.5),
        FloatParameter(f"{prefix}.speed_softening_mps", 1.0, 4.0),
        FloatParameter(f"{prefix}.heading_error_gain", 0.7, 1.3),
        FloatParameter(f"{prefix}.lookahead_time_s", 0.08, 0.30),
        FloatParameter(f"{prefix}.curvature_lookahead_m", 0.75, 3.0),
        FloatParameter(f"{prefix}.speed_pid.kp", 0.05, 1.2, log=True),
        FloatParameter(f"{prefix}.speed_pid.kd", 0.0, 0.1),
        FloatParameter(f"{prefix}.brake_pid.kp", 0.02, 0.6, log=True),
        FloatParameter(f"{prefix}.brake_pid.kd", 0.0, 0.05),
    )


STANLEY = TuningTarget(
    domain="control",
    algorithm="stanley",
    parameters=stanley_parameters("stanley"),
)

PROFILE_STANLEY = TuningTarget(
    domain="control",
    algorithm="profile_stanley",
    parameters=stanley_parameters("profile_stanley"),
)

DWA = TuningTarget(
    domain="planning",
    algorithm="dwa",
    parameters=(
        FloatParameter("dwa.goal_weight", 0.1, 2.0, log=True),
        FloatParameter("dwa.heading_weight", 0.1, 4.0, log=True),
        FloatParameter("dwa.clearance_weight", 0.5, 8.0, log=True),
        FloatParameter("dwa.smoothness_weight", 0.01, 0.5, log=True),
        FloatParameter("dwa.path_distance_weight", 0.25, 8.0, log=True),
        FloatParameter("dwa.speed_weight", 0.05, 3.0, log=True),
        FloatParameter("dwa.speed_pid.kp", 0.05, 1.2, log=True),
        FloatParameter("dwa.speed_pid.kd", 0.0, 0.1),
        FloatParameter("dwa.brake_pid.kp", 0.25, 0.8, log=True),
        FloatParameter("dwa.brake_pid.kd", 0.0, 0.05),
    ),
)

TARGETS = {
    target.identifier: target
    for target in (STANLEY, PROFILE_STANLEY, DWA)
}


def get_target(identifier: str) -> TuningTarget:
    if not isinstance(identifier, str) or identifier.count("/") != 1:
        raise ValueError("tuning target must use exact domain/algorithm form")
    try:
        return TARGETS[identifier]
    except KeyError as error:
        raise ValueError(f"unknown tuning target: {identifier}") from error


def list_targets() -> tuple[str, ...]:
    return tuple(sorted(TARGETS))
