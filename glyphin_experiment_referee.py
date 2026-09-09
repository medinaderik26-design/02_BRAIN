"""Independent audit of recorded Glyphin experiment runs.

The recorder captures an operation trace and final memory. This module creates
a fresh runtime, replays the trace, and compares the resulting GlyphinMemory
to the recorded final state with the canonical GlyphState referee.

A successful result means the run record is internally reproducible. It does
not establish scientific validity, semantic truth, or model consciousness.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from glyphin_experiment import ExperimentRun
from glyphin_research_core import GlyphinMemory
from glyphin_runtime import GlyphinRuntime
from glyphin_state_referee import StateRefereeResult, referee_memory


@dataclass(frozen=True)
class ExperimentRefereeResult:
    replay_succeeded: bool
    replay_error: str | None
    memory: StateRefereeResult | None

    @property
    def exact(self) -> bool:
        return self.replay_succeeded and self.memory is not None and self.memory.exact

    def to_dict(self) -> dict[str, Any]:
        memory = self.memory
        return {
            "replay_succeeded": self.replay_succeeded,
            "replay_error": self.replay_error,
            "exact": self.exact,
            "memory": None
            if memory is None
            else {
                "exact": memory.exact,
                "source_count": memory.source_count,
                "candidate_count": memory.candidate_count,
                "missing_states": list(memory.missing_states),
                "extra_states": list(memory.extra_states),
                "field_mismatches": list(memory.field_mismatches),
                "parameter_mismatches": list(memory.parameter_mismatches),
            },
        }


def referee_run(run: ExperimentRun) -> ExperimentRefereeResult:
    """Replay a recorded run from scratch and independently referee its state."""
    runtime = GlyphinRuntime()
    try:
        runtime.run(run.operations)
    except Exception as exc:  # noqa: BLE001 - audit must record failure type/message
        return ExperimentRefereeResult(
            replay_succeeded=False,
            replay_error=f"{type(exc).__name__}: {exc}",
            memory=None,
        )

    recorded = GlyphinMemory.from_dict(run.final_memory)
    result = referee_memory(recorded, runtime.memory)
    return ExperimentRefereeResult(
        replay_succeeded=True,
        replay_error=None,
        memory=result,
    )


__all__ = ["ExperimentRefereeResult", "referee_run"]
