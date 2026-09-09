"""Tests for the transparent topology encoder."""
from __future__ import annotations

from glyphin_encoder import encode, encode_chains_and_edges, encode_explicit_edges
from glyphin_referee import referee
from glyphin_reconstruction import reconstruct
from glyphin_topology import DirectedTopology


def test_explicit_encoder_preserves_isolated_nodes() -> None:
    topology = DirectedTopology.from_edges(
        [("root", "a"), ("a", "b"), ("root", "c")], nodes=["isolated"]
    )
    encoded = encode_explicit_edges(topology)
    assert referee(topology, encoded).exact_match
    assert encoded.endswith(";isolated")
    assert "isolated" in reconstruct(encoded).nodes


def test_chain_encoder_preserves_isolated_nodes() -> None:
    topology = DirectedTopology.from_edges(
        [("root", "a"), ("a", "b")], nodes=["isolated"]
    )
    encoded = encode_chains_and_edges(topology)
    result = referee(topology, encoded)
    assert result.exact_match
    assert "isolated" in reconstruct(encoded).nodes


def test_chain_encoder_is_exact_on_lineage() -> None:
    topology = DirectedTopology.from_edges(
        [("root", "a"), ("a", "b"), ("b", "c")]
    )
    encoded = encode_chains_and_edges(topology)
    assert encoded == "root->a->b->c"
    assert referee(topology, encoded).exact_match


def test_chain_encoder_preserves_branching() -> None:
    topology = DirectedTopology.from_edges(
        [("root", "a"), ("root", "b"), ("a", "c"), ("b", "d")]
    )
    encoded = encode(topology)
    result = referee(topology, encoded)
    assert result.exact_match


def test_cycle_falls_back_without_dropping_edges() -> None:
    topology = DirectedTopology.from_edges(
        [("a", "b"), ("b", "c"), ("c", "a")]
    )
    encoded = encode(topology)
    assert referee(topology, encoded).exact_match


def test_empty_topology_encodes_to_empty_text() -> None:
    topology = DirectedTopology()
    encoded = encode(topology)
    assert encoded == ""
    assert referee(topology, encoded).exact_match


def test_unknown_strategy_is_rejected() -> None:
    topology = DirectedTopology.from_edges([("a", "b")])
    try:
        encode(topology, strategy="not-a-strategy")
    except ValueError as exc:
        assert "unknown encoding strategy" in str(exc)
    else:
        raise AssertionError("unknown strategy must fail")
