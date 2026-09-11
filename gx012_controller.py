"""GX-012 Dual Cone Variator controller.

Provider-neutral reference implementation for GX-012-A controller simulation.
The controller allocates separate input and output apertures and records every
transition so the policy can be evaluated independently of an LLM provider.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Optional


def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))


@dataclass(frozen=True)
class GX012Signals:
    """Observed signals used to calculate target apertures."""

    relevance: float = 0.5
    cohesion: float = 0.5
    lineage_depth: float = 0.0
    uncertainty: float = 0.5
    task_complexity: float = 0.5
    contradiction: float = 0.0
    resource_pressure: float = 0.0

    def normalized(self) -> "GX012Signals":
        return GX012Signals(
            relevance=_clip(self.relevance),
            cohesion=_clip(self.cohesion),
            lineage_depth=_clip(self.lineage_depth),
            uncertainty=_clip(self.uncertainty),
            task_complexity=_clip(self.task_complexity),
            contradiction=_clip(self.contradiction),
            resource_pressure=_clip(self.resource_pressure),
        )


@dataclass(frozen=True)
class GX012Allocation:
    input_aperture: float
    output_aperture: float
    raw_fraction: float
    lineage_fraction: float
    output_budget: int


@dataclass(frozen=True)
class GX012Step:
    step: int
    signals: Dict[str, float]
    target_input: float
    target_output: float
    input_aperture: float
    output_aperture: float
    raw_fraction: float
    lineage_fraction: float
    output_budget: int


class DualConeVariator:
    """Continuous, independently controlled input/output aperture policy.

    This class intentionally does not call a model or claim compute savings.
    It is the controller under test for GX-012-A.
    """

    def __init__(
        self,
        *,
        mu: float = 0.25,
        initial_input: float = 0.5,
        initial_output: float = 0.5,
        output_min: int = 32,
        output_max: int = 512,
    ) -> None:
        if not 0.0 < mu <= 1.0:
            raise ValueError("mu must be in (0, 1]")
        if output_min < 1 or output_max < output_min:
            raise ValueError("invalid output budget bounds")
        self.mu = float(mu)
        self.input_aperture = _clip(initial_input)
        self.output_aperture = _clip(initial_output)
        self.output_min = int(output_min)
        self.output_max = int(output_max)
        self.step_count = 0
        self.history = []

    def target(self, signals: GX012Signals) -> tuple[float, float]:
        """Return experimental target apertures from normalized signals.

        Higher relevance/cohesion/lineage and uncertainty increase input
        allocation. Complexity/uncertainty increase output allocation.
        Resource pressure reduces both. The weights are explicit so they can
        be frozen, ablated, or replaced in later experiments.
        """
        s = signals.normalized()
        input_target = (
            0.20
            + 0.30 * s.relevance
            + 0.20 * s.cohesion
            + 0.15 * s.lineage_depth
            + 0.15 * s.uncertainty
            + 0.10 * s.contradiction
            - 0.20 * s.resource_pressure
        )
        output_target = (
            0.15
            + 0.40 * s.task_complexity
            + 0.25 * s.uncertainty
            + 0.10 * s.contradiction
            + 0.10 * s.relevance
            - 0.25 * s.resource_pressure
        )
        return _clip(input_target), _clip(output_target)

    def allocation(self, input_budget: int) -> GX012Allocation:
        """Translate current apertures into observable allocation values."""
        if input_budget < 1:
            raise ValueError("input_budget must be positive")
        raw_fraction = 1.0 - self.input_aperture
        lineage_fraction = self.input_aperture
        output_budget = round(
            self.output_min
            + self.output_aperture * (self.output_max - self.output_min)
        )
        return GX012Allocation(
            input_aperture=self.input_aperture,
            output_aperture=self.output_aperture,
            raw_fraction=raw_fraction,
            lineage_fraction=lineage_fraction,
            output_budget=output_budget,
        )

    def update(self, signals: GX012Signals, input_budget: int) -> GX012Allocation:
        """Apply one closed-loop controller update and record the transition."""
        target_input, target_output = self.target(signals)
        self.input_aperture = _clip(
            self.input_aperture + self.mu * (target_input - self.input_aperture)
        )
        self.output_aperture = _clip(
            self.output_aperture + self.mu * (target_output - self.output_aperture)
        )
        self.step_count += 1
        allocation = self.allocation(input_budget)
        normalized = signals.normalized()
        self.history.append(
            GX012Step(
                step=self.step_count,
                signals=asdict(normalized),
                target_input=target_input,
                target_output=target_output,
                input_aperture=allocation.input_aperture,
                output_aperture=allocation.output_aperture,
                raw_fraction=allocation.raw_fraction,
                lineage_fraction=allocation.lineage_fraction,
                output_budget=allocation.output_budget,
            )
        )
        return allocation

    def reset(self) -> None:
        self.input_aperture = 0.5
        self.output_aperture = 0.5
        self.step_count = 0
        self.history.clear()

    def export_history(self):
        return [asdict(item) for item in self.history]
