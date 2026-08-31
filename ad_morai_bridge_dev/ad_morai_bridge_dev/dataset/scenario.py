"""Deterministic scenario catalog for the dataset factory.

A scenario is data: a catalog entry (purpose, intended range regime, MORAI
scenario file) plus the standard MORAI scenario JSON that
``ad_morai_bridge_dev.scenarios.reset.load_reset_plan`` already parses. A
``scenario_seed`` deterministically derives small per-run placement /
velocity perturbations so repeated captures of one ``(scenario, seed)`` are
reproducible on the factory side. Whether the simulator itself guarantees
identical physics for a seed is recorded separately and defaults to false.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ad_morai_bridge_dev.scenarios.reset import ResetPlan, load_reset_plan

RANGE_REGIMES = ("near", "mid", "far", "mixed")

# Bounds for seed-derived perturbations. Deliberately small: they vary
# placement without changing a scenario's qualitative purpose.
MAX_LONGITUDINAL_JITTER_M = 4.0
MAX_LATERAL_JITTER_M = 1.0
MAX_SPEED_JITTER_MPS = 1.5


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    purpose: str
    range_regime: str
    scenario_file: Path
    scenario_sha256: str
    notes: str = ""

    def base_plan(self) -> ResetPlan:
        return load_reset_plan(self.scenario_file)


@dataclass(frozen=True)
class RunParameters:
    scenario_id: str
    requested_seed: int
    simulator_determinism_guaranteed: bool
    actor_jitter: dict[str, dict[str, float]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "requested_seed": self.requested_seed,
            "simulator_determinism_guaranteed": self.simulator_determinism_guaranteed,
            "actor_jitter": self.actor_jitter,
        }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ScenarioCatalog:
    def __init__(self, scenarios: dict[str, Scenario]) -> None:
        self._scenarios = dict(scenarios)

    def __iter__(self):
        return iter(sorted(self._scenarios.values(), key=lambda s: s.scenario_id))

    def __len__(self) -> int:
        return len(self._scenarios)

    def ids(self) -> list[str]:
        return sorted(self._scenarios)

    def get(self, scenario_id: str) -> Scenario:
        if scenario_id not in self._scenarios:
            raise KeyError(
                f"unknown scenario {scenario_id!r}; known: {self.ids()}"
            )
        return self._scenarios[scenario_id]

    @classmethod
    def load(cls, catalog_path: Path) -> "ScenarioCatalog":
        catalog_path = Path(catalog_path)
        data = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
        base_dir = catalog_path.parent
        scenarios: dict[str, Scenario] = {}
        for entry in data.get("scenarios", []):
            scenario_id = str(entry["scenario_id"])
            if scenario_id in scenarios:
                raise ValueError(f"duplicate scenario_id {scenario_id!r}")
            regime = str(entry["range_regime"])
            if regime not in RANGE_REGIMES:
                raise ValueError(
                    f"scenario {scenario_id!r} range_regime {regime!r} not in "
                    f"{RANGE_REGIMES}"
                )
            scenario_file = Path(entry["scenario_file"])
            if not scenario_file.is_absolute():
                scenario_file = base_dir / scenario_file
            if not scenario_file.is_file():
                raise FileNotFoundError(
                    f"scenario {scenario_id!r} file missing: {scenario_file}"
                )
            # Fail fast if the JSON is not a valid reset plan.
            load_reset_plan(scenario_file)
            scenarios[scenario_id] = Scenario(
                scenario_id=scenario_id,
                purpose=str(entry.get("purpose", "")),
                range_regime=regime,
                scenario_file=scenario_file,
                scenario_sha256=_sha256(scenario_file),
                notes=str(entry.get("notes", "")),
            )
        if not scenarios:
            raise ValueError("scenario catalog is empty")
        return cls(scenarios)


def derive_run_parameters(
    scenario: Scenario,
    requested_seed: int,
    simulator_determinism_guaranteed: bool = False,
) -> RunParameters:
    """Reproducible per-run perturbation for one ``(scenario, seed)``."""

    rng = random.Random(f"{scenario.scenario_id}:{scenario.scenario_sha256}:{requested_seed}")
    plan = scenario.base_plan()
    jitter: dict[str, dict[str, float]] = {}
    for actor in plan.actors:
        jitter[actor.actor_id] = {
            "longitudinal_m": round(
                rng.uniform(-MAX_LONGITUDINAL_JITTER_M, MAX_LONGITUDINAL_JITTER_M), 4
            ),
            "lateral_m": round(
                rng.uniform(-MAX_LATERAL_JITTER_M, MAX_LATERAL_JITTER_M), 4
            ),
            "speed_mps": round(
                rng.uniform(-MAX_SPEED_JITTER_MPS, MAX_SPEED_JITTER_MPS), 4
            ),
        }
    return RunParameters(
        scenario_id=scenario.scenario_id,
        requested_seed=int(requested_seed),
        simulator_determinism_guaranteed=bool(simulator_determinism_guaranteed),
        actor_jitter=jitter,
    )


def build_reset_plan(scenario: Scenario, parameters: RunParameters) -> ResetPlan:
    """Apply the seed-derived jitter to the base plan.

    Ego is never jittered - it is always reset to its scenario pose at zero
    velocity. Actor jitter is applied along the actor's own heading
    (longitudinal) and perpendicular (lateral).
    """

    import math

    from ad_morai_bridge_dev.scenarios.reset import Pose, ResetActor

    base = scenario.base_plan()
    new_actors = []
    for actor in base.actors:
        j = parameters.actor_jitter.get(actor.actor_id)
        if j is None:
            new_actors.append(actor)
            continue
        yaw = actor.pose.rotation[2]
        dx = j["longitudinal_m"] * math.cos(yaw) - j["lateral_m"] * math.sin(yaw)
        dy = j["longitudinal_m"] * math.sin(yaw) + j["lateral_m"] * math.cos(yaw)
        loc = (
            actor.pose.location[0] + dx,
            actor.pose.location[1] + dy,
            actor.pose.location[2],
        )
        velocity = actor.velocity
        if velocity is not None:
            velocity = max(0.0, velocity + j["speed_mps"])
        new_actors.append(
            ResetActor(
                actor.actor_id,
                actor.object_type,
                Pose(loc, actor.pose.rotation),
                velocity,
            )
        )
    return ResetPlan(base.ego, tuple(new_actors))


def default_catalog_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "config"
        / "dataset_factory"
        / "scenario_catalog.yaml"
    )
