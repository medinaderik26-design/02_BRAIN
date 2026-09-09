"""Deterministic tests for the general directed topology layer."""

from glyphin_topology import DirectedTopology


def test_nodes_and_edges_are_canonicalized() -> None:
    graph = DirectedTopology.from_edges(
        [("B", "C"), ("A", "B"), ("A", "B")], nodes=["Z"]
    )
    assert graph.nodes == {"A", "B", "C", "Z"}
    assert graph.edges == {("A", "B"), ("B", "C")}
    assert graph.canonical() == {
        "nodes": ["A", "B", "C", "Z"],
        "edges": [["A", "B"], ["B", "C"]],
    }


def test_fanout_and_convergence() -> None:
    graph = DirectedTopology.from_edges(
        [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")]
    )
    assert graph.successors("A") == ["B", "C"]
    assert graph.predecessors("D") == ["B", "C"]
    assert graph.out_degree("A") == 2
    assert graph.in_degree("D") == 2


def test_cycle_detection() -> None:
    dag = DirectedTopology.from_edges([("A", "B"), ("B", "C")])
    cyclic = DirectedTopology.from_edges([("A", "B"), ("B", "A")])
    assert not dag.has_cycle()
    assert cyclic.has_cycle()


def test_compare_reports_exact_differences() -> None:
    source = DirectedTopology.from_edges(
        [("A", "B"), ("B", "C")], nodes=["D"]
    )
    candidate = DirectedTopology.from_edges(
        [("A", "B"), ("C", "D")], nodes=["E"]
    )
    result = source.compare(candidate)
    assert result["exact"] is False
    assert result["missing_nodes"] == []
    assert result["extra_nodes"] == ["E"]
    assert result["missing_edges"] == [["B", "C"]]
    assert result["extra_edges"] == [["C", "D"]]


def test_empty_and_isolated_nodes_survive() -> None:
    graph = DirectedTopology(nodes={"isolated"})
    assert graph.roots() == ["isolated"]
    assert graph.leaves() == ["isolated"]
    assert not graph.has_cycle()


if __name__ == "__main__":
    tests = [
        test_nodes_and_edges_are_canonicalized,
        test_fanout_and_convergence,
        test_cycle_detection,
        test_compare_reports_exact_differences,
        test_empty_and_isolated_nodes_survive,
    ]
    for test in tests:
        test()
    print(f"PASS: {len(tests)} topology tests")
