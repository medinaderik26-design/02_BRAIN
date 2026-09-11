"""Smoke tests for the Astra symbolic-anchor representation."""

import json

from astra_anchor import build_astra_anchor


def test_deterministic_anchor_and_digest():
    components = [
        "Glyphin",
        "Weisone Kernel",
        "Astra is a symbolic anchor for accumulated interaction pattern",
        "preserve terminology and continuity across model drift",
    ]
    anchor_a = build_astra_anchor(
        components,
        meaning="A symbolic handle for accumulated human-LLM interaction pattern.",
        source_scope="controlled smoke-test fixture",
    )
    anchor_b = build_astra_anchor(
        components,
        meaning="A symbolic handle for accumulated human-LLM interaction pattern.",
        source_scope="controlled smoke-test fixture",
    )

    assert anchor_a.name == "Astra"
    assert anchor_a.canonical_digest == anchor_b.canonical_digest
    assert len(anchor_a.canonical_digest) == 64


def test_change_in_continuity_component_changes_digest():
    base = build_astra_anchor(
        ["Glyphin", "Weisone Kernel", "Astra"],
        meaning="continuity anchor",
        source_scope="fixture",
    )
    changed = build_astra_anchor(
        ["Glyphin", "Weisone Kernel", "Astra", "new continuity observation"],
        meaning="continuity anchor",
        source_scope="fixture",
    )
    assert base.canonical_digest != changed.canonical_digest


def test_serialization_round_trip_is_exact():
    anchor = build_astra_anchor(
        ["token-pattern", "correction", "decision", "recurring terminology"],
        meaning="accumulated symbolic interaction pattern",
        source_scope="fixture",
    )
    restored = type(anchor).from_dict(json.loads(json.dumps(anchor.to_dict())))

    assert restored == anchor
    assert restored.canonical_digest == anchor.canonical_digest


def test_provenance_boundary_is_preserved():
    anchor = build_astra_anchor(
        ["conversation artifact A", "conversation artifact B"],
        meaning="symbolic continuity anchor",
        source_scope="saved interaction artifacts only",
    )
    assert anchor.source_scope == "saved interaction artifacts only"
    assert anchor.to_dict()["canonical_digest"] == anchor.canonical_digest


if __name__ == "__main__":
    test_deterministic_anchor_and_digest()
    test_change_in_continuity_component_changes_digest()
    test_serialization_round_trip_is_exact()
    test_provenance_boundary_is_preserved()
    print("ASTRA ANCHOR SMOKE TEST: PASS")
