"""Tests for independent GlyphState-level verification."""
from __future__ import annotations

from glyphin_research_core import GlyphinMemory
from glyphin_state_referee import referee_memory


def _memory() -> GlyphinMemory:
    memory = GlyphinMemory(decay_lambda=0.1, alpha=0.25, beta=0.25)
    memory.add_state("root", cohesion=0.5)
    memory.add_state("child", parent="root", level=1)
    memory.reinforce("child", kappa=0.5)
    return memory


def test_identical_memories_are_exact() -> None:
    source = _memory()
    candidate = GlyphinMemory.from_json(source.to_json())
    result = referee_memory(source, candidate)
    assert result.exact
    assert result.field_mismatches == ()


def test_state_change_is_detected_even_when_topology_is_unchanged() -> None:
    source = _memory()
    candidate = GlyphinMemory.from_json(source.to_json())
    candidate.states["child"].cohesion += 0.01
    result = referee_memory(source, candidate)
    assert not result.exact
    assert {item["field"] for item in result.field_mismatches} == {"cohesion"}


def test_lineage_change_is_detected() -> None:
    source = _memory()
    candidate = GlyphinMemory.from_json(source.to_json())
    candidate.states["child"].parent = None
    candidate.states["root"].children = []
    result = referee_memory(source, candidate)
    assert not result.exact
    fields = {(item["state"], item["field"]) for item in result.field_mismatches}
    assert ("child", "parent") in fields
    assert ("root", "children") in fields
