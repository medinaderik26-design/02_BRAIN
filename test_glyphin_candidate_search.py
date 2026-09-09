"""Tests for bounded Glyphin candidate generation and independent validation."""
from __future__ import annotations

from glyphin_candidate_search import generate_candidates, search
from glyphin_encoder import encode_explicit_edges
from glyphin_topology import DirectedTopology


def test_search_is_deterministic() -> None:
    topology = DirectedTopology.from_edges(
        [("root", "a"), ("a", "b"), ("root", "c")]
    )
    first = search(topology)
    second = search(topology)
    assert first == second


def test_explicit_baseline_is_always_a_candidate() -> None:
    topology = DirectedTopology.from_edges(
        [("root", "a"), ("a", "b"), ("root", "c")]
    )
    assert encode_explicit_edges(topology) in generate_candidates(topology)


def test_exact_candidates_are_refereed() -> None:
    topology = DirectedTopology.from_edges(
        [("root", "a"), ("a", "b"), ("root", "c")]
    )
    result = search(topology)
    assert result.exact_candidates
    assert result.shortest_exact is not None
    assert result.shortest_exact.exact


def test_search_does_not_claim_global_optimality() -> None:
    topology = DirectedTopology.from_edges(
        [("a", "b"), ("b", "c"), ("c", "d")]
    )
    result = search(topology)
    assert result.shortest_exact is not None
    assert result.candidates_generated >= result.candidates_evaluated
    assert result.candidates_generated == result.candidates_evaluated


def test_search_finds_exact_representation_with_isolated_node() -> None:
    topology = DirectedTopology.from_edges(
        [("root", "a"), ("a", "b"), ("root", "c")], nodes=["isolated"]
    )
    result = search(topology)
    assert result.shortest_exact is not None
    assert result.shortest_exact.exact
    assert "isolated" in result.shortest_exact.encoding


def test_candidate_set_contains_chain_candidate() -> None:
    topology = DirectedTopology.from_edges([("a", "b"), ("b", "c")])
    candidates = generate_candidates(topology)
    assert "a->b;b->c" in candidates
    assert "a->b->c" in candidates
