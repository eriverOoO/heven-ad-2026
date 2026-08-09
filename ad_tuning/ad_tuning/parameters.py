from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol


class SuggestingTrial(Protocol):
    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        step: float | None = None,
        log: bool = False,
    ) -> float: ...

    def suggest_int(
        self,
        name: str,
        low: int,
        high: int,
        *,
        step: int = 1,
        log: bool = False,
    ) -> int: ...


def _validate_name(name: str) -> None:
    if not isinstance(name, str) or not name:
        raise ValueError("parameter name must be non-empty")


@dataclass(frozen=True)
class FloatParameter:
    name: str
    low: float
    high: float
    step: float | None = None
    log: bool = False

    def __post_init__(self) -> None:
        _validate_name(self.name)
        if not all(math.isfinite(value) for value in (self.low, self.high)):
            raise ValueError("float parameter bounds must be finite")
        if self.low >= self.high:
            raise ValueError("float parameter low must be less than high")
        if self.step is not None and (
            not math.isfinite(self.step) or self.step <= 0.0
        ):
            raise ValueError("float parameter step must be finite and positive")
        if self.log and self.low <= 0.0:
            raise ValueError("log float parameter low must be positive")
        if self.log and self.step is not None:
            raise ValueError("log float parameters cannot use a step")

    def suggest(self, trial: SuggestingTrial) -> float:
        return trial.suggest_float(
            self.name, self.low, self.high, step=self.step, log=self.log
        )


@dataclass(frozen=True)
class IntParameter:
    name: str
    low: int
    high: int
    step: int = 1
    log: bool = False

    def __post_init__(self) -> None:
        _validate_name(self.name)
        if any(type(value) is not int for value in (self.low, self.high, self.step)):
            raise ValueError("integer parameter bounds and step must be exact integers")
        if self.low >= self.high:
            raise ValueError("integer parameter low must be less than high")
        if self.step <= 0:
            raise ValueError("integer parameter step must be positive")
        if self.log and self.low <= 0:
            raise ValueError("log integer parameter low must be positive")
        if self.log and self.step != 1:
            raise ValueError("log integer parameters cannot use a step")

    def suggest(self, trial: SuggestingTrial) -> int:
        return trial.suggest_int(
            self.name, self.low, self.high, step=self.step, log=self.log
        )
