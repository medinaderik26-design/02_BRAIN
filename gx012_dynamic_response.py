"""GX-012-A.1 dynamic response simulation.

Provider-neutral controller dynamics only. No LLM or compute claims.
"""

from dataclasses import dataclass
from typing import Dict, List

from gx012_controller import DualConeVariator, GX012Signals


@dataclass(frozen=True)
class SignalFrame:
    name: str
    signals: GX012Signals


@dataclass
class DynamicTrace:
    sequence: str
    rows: List[Dict[str, float]]


def default_sequence() -> List[SignalFrame]:
    base = GX012Signals(
        relevance=0.60, cohesion=0.60, lineage_depth=0.50,
        uncertainty=0.20, task_complexity=0.40,
        contradiction=0.00, resource_pressure=0.10,
    )
    pressure = GX012Signals(
        relevance=0.60, cohesion=0.60, lineage_depth=0.50,
        uncertainty=0.20, task_complexity=0.40,
        contradiction=0.00, resource_pressure=0.90,
    )
    recovery = GX012Signals(
        relevance=0.60, cohesion=0.60, lineage_depth=0.50,
        uncertainty=0.20, task_complexity=0.40,
        contradiction=0.00, resource_pressure=0.10,
    )
    uncertainty = GX012Signals(
        relevance=0.60, cohesion=0.60, lineage_depth=0.50,
        uncertainty=0.90, task_complexity=0.40,
        contradiction=0.00, resource_pressure=0.10,
    )
    return (
        [SignalFrame("baseline", base)] * 5
        + [SignalFrame("pressure_spike", pressure)] * 5
        + [SignalFrame("recovery", recovery)] * 5
        + [SignalFrame("uncertainty_spike", uncertainty)] * 5
        + [SignalFrame("recovery_2", recovery)] * 5
    )


def run_dynamic_sequence(frames=None, mu: float = 0.25) -> DynamicTrace:
    controller = DualConeVariator(mu=mu)
    rows: List[Dict[str, float]] = []
    for step, frame in enumerate(frames or default_sequence()):
        before = controller.input_aperture
        allocation = controller.update(frame.signals, input_budget=4096)
        current = controller.history[-1]
        rows.append({
            "step": float(step),
            "input_aperture_before": before,
            "input_aperture": allocation.input_aperture,
            "output_aperture": allocation.output_aperture,
            "input_target": current.target_input,
            "output_target": current.target_output,
            "resource_pressure": frame.signals.resource_pressure,
            "uncertainty": frame.signals.uncertainty,
        })
    return DynamicTrace(sequence="default_dynamic_response", rows=rows)


if __name__ == "__main__":
    trace = run_dynamic_sequence()
    for row in trace.rows:
        print(row)
