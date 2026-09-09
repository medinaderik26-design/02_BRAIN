"""Independent referee for GlyphinMemory state fidelity.

The topology referee answers a narrower question: do two directed graphs have
identical nodes and edges? This referee answers a different question: after a
memory serialization/reconstruction, are the full persisted GlyphState values
and model parameters identical?

No semantic inference is performed. Equality is structural and field-level.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from glyphin_research_core import GlyphinMemory


_STATE_FIELDS = (
    "name",
    "level",
    "cohesion",
    "parent",
    "children",
    "frequency",
    "resonance",
    "sigma",
    "created_at",
)
_PARAMETER_FIELDS = ("decay_lambda", "alpha", "beta")


@dataclass(frozen=True)
class MemoryRefereeResult:
    exact: bool
    source_state_count: int
    candidate_state_count: int
    missing_states: tuple[str, ...]
    extra_states: tuple[str, ...]
    state_mismatches: tuple[dict[str, Any], ...]
    parameter_mismatches: tuple[dict[str, Any], ...]


def _state_dict(memory: GlyphinMemory, name: str) -> dict[str, Any]:
    state = memory.states[name]
    return {field: getattr(state, field) for field in _STATE_FIELDS}


def referee(source: GlyphinMemory, candidate: GlyphinMemory) -> MemoryRefereeResult:
    """Compare two memories without inferring semantic equivalence."""
    source_names = set(source.states)
    candidate_names = set(candidate.states)
    missing = tuple(sorted(source_names - candidate_names))
    extra = tuple(sorted(candidate_names - source_names))

    mismatches: list[dict[str, Any]] = []
    for name in sorted(source_names & candidate_names):
        left = _state_dict(source, name)
        right = _state_dict(candidate, name)
        for field in _STATE_FIELDS:
            if left[field] != right[field]:
                mismatches.append(
                    {
                        "state": name,
                        "field": field,
                        "source": left[field],
                        "candidate": right[field],
                    }
                )

    parameter_mismatches: list[dict[str, Any]] = []
    for field in _PARAMETER_FIELDS:
        left = getattr(source, field)
        right = getattr(candidate, field)
        if left != right:
            parameter_mismatches.append(
                {"field": field, "source": left, "candidate": right}
            )

    exact = not missing and not extra and not mismatches and not parameter_mismatches
    return MemoryRefereeResult(
        exact=exact,
        source_state_count=len(source.states),
        candidate_state_count=len(candidate.states),
        missing_states=missing,
        extra_states=extra,
        state_mismatches=tuple(mismatches),
        parameter_mismatches=tuple(parameter_mismatches),
    )


__all__ = ["MemoryRefereeResult", "referee"]
