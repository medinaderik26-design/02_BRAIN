"""Independent referee for GlyphState-level fidelity.

The topology referee answers a structural question: did the same graph come
back? This referee answers the separate state question: did the named
GlyphState attributes, lineage relationships, and model parameters come back?
It intentionally uses direct field comparison rather than trusting a
producer's success flag.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from glyphin_research_core import GlyphinMemory


STATE_FIELDS = (
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
PARAMETER_FIELDS = ("decay_lambda", "alpha", "beta")


@dataclass(frozen=True)
class StateRefereeResult:
    exact: bool
    source_count: int
    candidate_count: int
    missing_states: tuple[str, ...]
    extra_states: tuple[str, ...]
    field_mismatches: tuple[dict[str, Any], ...]
    parameter_mismatches: tuple[dict[str, Any], ...]


def _state_dict(memory: GlyphinMemory, name: str) -> dict[str, Any]:
    state = memory.states[name]
    return {field: getattr(state, field) for field in STATE_FIELDS}


def referee_memory(
    source: GlyphinMemory,
    candidate: GlyphinMemory,
    *,
    float_tolerance: float = 0.0,
) -> StateRefereeResult:
    """Compare memories field-by-field without relying on serialization hashes."""
    if float_tolerance < 0:
        raise ValueError("float_tolerance must be non-negative")

    source_names = set(source.states)
    candidate_names = set(candidate.states)
    missing = tuple(sorted(source_names - candidate_names))
    extra = tuple(sorted(candidate_names - source_names))
    mismatches: list[dict[str, Any]] = []

    for name in sorted(source_names & candidate_names):
        expected = _state_dict(source, name)
        actual = _state_dict(candidate, name)
        for field in STATE_FIELDS:
            left = expected[field]
            right = actual[field]
            equal = left == right
            if field in {"cohesion", "resonance"} and isinstance(left, float) and isinstance(right, float):
                equal = abs(left - right) <= float_tolerance
            if not equal:
                mismatches.append({
                    "state": name,
                    "field": field,
                    "expected": left,
                    "actual": right,
                })

    parameter_mismatches: list[dict[str, Any]] = []
    for field in PARAMETER_FIELDS:
        expected = getattr(source, field)
        actual = getattr(candidate, field)
        equal = expected == actual
        if isinstance(expected, float) and isinstance(actual, float):
            equal = abs(expected - actual) <= float_tolerance
        if not equal:
            parameter_mismatches.append({
                "field": field,
                "expected": expected,
                "actual": actual,
            })

    return StateRefereeResult(
        exact=not (missing or extra or mismatches or parameter_mismatches),
        source_count=len(source_names),
        candidate_count=len(candidate_names),
        missing_states=missing,
        extra_states=extra,
        field_mismatches=tuple(mismatches),
        parameter_mismatches=tuple(parameter_mismatches),
    )


__all__ = ["STATE_FIELDS", "PARAMETER_FIELDS", "StateRefereeResult", "referee_memory"]
