"""GX-012-A deterministic controller simulation.

Runs the Dual Cone Variator against synthetic task profiles. This is a
controller-behavior experiment only; it does not measure LLM quality, compute,
or energy savings.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List

from gx012_controller import DualConeVariator, GX012Signals


@dataclass(frozen=True)
class GX012TaskProfile:
    name: str
    signals: GX012Signals
    input_budget: int


@dataclass(frozen=True)
class GX012SimulationResult:
    task: str
    steps: int
    final_input_aperture: float
    final_output_aperture: float
    final_raw_fraction: float
    final_lineage_fraction: float
    final_output_budget: int


def default_profiles() -> List[GX012TaskProfile]:
    return [
        GX012TaskProfile(
            "low_complexity_low_pressure",
            GX012Signals(relevance=.30, cohesion=.40, lineage_depth=.20,
                         uncertainty=.20, task_complexity=.20,
                         contradiction=.00, resource_pressure=.10),
            4096,
        ),
        GX012TaskProfile(
            "high_memory_relevance",
            GX012Signals(relevance=.95, cohesion=.90, lineage_depth=.90,
                         uncertainty=.35, task_complexity=.60,
                         contradiction=.10, resource_pressure=.10),
            4096,
        ),
        GX012TaskProfile(
            "high_uncertainty",
            GX012Signals(relevance=.70, cohesion=.55, lineage_depth=.60,
                         uncertainty=.95, task_complexity=.80,
                         contradiction=.25, resource_pressure=.15),
            4096,
        ),
        GX012TaskProfile(
            "resource_constrained",
            GX012Signals(relevance=.70, cohesion=.70, lineage_depth=.70,
                         uncertainty=.50, task_complexity=.70,
                         contradiction=.10, resource_pressure=.95),
            4096,
        ),
    ]


def run_profile(profile: GX012TaskProfile, steps: int = 20, mu: float = .25) -> GX012SimulationResult:
    controller = DualConeVariator(mu=mu)
    allocation = controller.allocation(profile.input_budget)
    for _ in range(steps):
        allocation = controller.update(profile.signals, profile.input_budget)
    return GX012SimulationResult(
        task=profile.name,
        steps=steps,
        final_input_aperture=allocation.input_aperture,
        final_output_aperture=allocation.output_aperture,
        final_raw_fraction=allocation.raw_fraction,
        final_lineage_fraction=allocation.lineage_fraction,
        final_output_budget=allocation.output_budget,
    )


def run_suite(steps: int = 20, mu: float = .25) -> List[Dict[str, object]]:
    return [asdict(run_profile(profile, steps=steps, mu=mu)) for profile in default_profiles()]


if __name__ == "__main__":
    for row in run_suite():
        print(row)
