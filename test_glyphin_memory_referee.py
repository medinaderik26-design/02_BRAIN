"""Tests for full persisted GlyphState fidelity."""
from glyphin_memory_referee import referee
from glyphin_research_core import GlyphinMemory


def build_memory() -> GlyphinMemory:
    memory = GlyphinMemory(decay_lambda=0.02, alpha=0.3, beta=0.4)
    memory.add_state("root", level=0, cohesion=0.5)
    memory.add_state("child", level=1, parent="root")
    memory.add_state("leaf", level=2, parent="child")
    memory.reinforce("child", kappa=0.75)
    memory.decay(1.5)
    return memory


def test_serialization_is_full_state_exact() -> None:
    source = build_memory()
    candidate = GlyphinMemory.from_json(source.to_json())
    result = referee(source, candidate)
    assert result.exact
    assert result.missing_states == ()
    assert result.extra_states == ()
    assert result.state_mismatches == ()
    assert result.parameter_mismatches == ()


def test_state_field_change_is_detected() -> None:
    source = build_memory()
    candidate = GlyphinMemory.from_json(source.to_json())
    candidate.states["child"].cohesion += 0.01
    result = referee(source, candidate)
    assert not result.exact
    assert any(
        item["state"] == "child" and item["field"] == "cohesion"
        for item in result.state_mismatches
    )


def test_lineage_change_is_detected_even_when_nodes_remain() -> None:
    source = build_memory()
    candidate = GlyphinMemory.from_json(source.to_json())
    candidate.states["leaf"].parent = "root"
    candidate._rebuild_children()
    result = referee(source, candidate)
    assert not result.exact
    assert any(
        item["state"] == "leaf" and item["field"] == "parent"
        for item in result.state_mismatches
    )
    assert any(
        item["state"] == "root" and item["field"] == "children"
        for item in result.state_mismatches
    )


def test_parameter_change_is_detected() -> None:
    source = build_memory()
    candidate = GlyphinMemory.from_json(source.to_json())
    candidate.beta = 0.41
    result = referee(source, candidate)
    assert not result.exact
    assert result.parameter_mismatches == (
        {"field": "beta", "source": 0.4, "candidate": 0.41},
    )


if __name__ == "__main__":
    tests = [
        test_serialization_is_full_state_exact,
        test_state_field_change_is_detected,
        test_lineage_change_is_detected_even_when_nodes_remain,
        test_parameter_change_is_detected,
    ]
    for test in tests:
        test()
    print(f"PASS: {len(tests)} memory referee tests")
