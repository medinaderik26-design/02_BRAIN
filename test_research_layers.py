"""Cross-layer smoke tests for the research implementation."""
from glyphin_topology import DirectedTopology
from glyphin_reconstruction import reconstruct
from glyphin_referee import referee
from glyphin_compression import measure


def build_graph() -> DirectedTopology:
    g = DirectedTopology()
    for node in ["A", "B", "C", "D"]:
        g.add_node(node)
    g.add_edge("A", "B")
    g.add_edge("A", "C")
    g.add_edge("B", "D")
    g.add_edge("C", "D")
    return g


def test_round_trip_and_referee() -> None:
    source = build_graph()
    encoded = "A->{B->D,C->D}"
    candidate = reconstruct(encoded)
    assert source.compare(candidate)["exact_match"]
    result = referee(source, encoded)
    assert result.parse_ok and result.exact_match
    assert result.source_fingerprint == result.candidate_fingerprint


def test_mismatch_is_not_exact() -> None:
    result = referee(build_graph(), "A->{B,C}")
    assert result.parse_ok
    assert not result.exact_match
    assert result.missing_edges


def test_compression_never_invents_tokens() -> None:
    metrics = measure("A -> B -> C", "A->B->C")
    assert metrics.char_reduction_pct > 0
    assert metrics.token_reduction_pct is None
