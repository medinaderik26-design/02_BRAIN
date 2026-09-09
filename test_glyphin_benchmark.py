"""Tests for the Glyphin generalization benchmark harness."""
from __future__ import annotations

from glyphin_benchmark import adversarial_cases, measure_case, random_graph, run


def test_random_graph_generation_is_deterministic() -> None:
    assert random_graph(17, cyclic=False).canonical() == random_graph(17, cyclic=False).canonical()
    assert random_graph(17, cyclic=True).canonical() == random_graph(17, cyclic=True).canonical()


def test_random_cyclic_graph_contains_cycle() -> None:
    topology = random_graph(17, cyclic=True)
    assert topology.has_cycle()


def test_adversarial_cases_are_exactly_measurable() -> None:
    cases = list(adversarial_cases())
    assert len(cases) >= 6
    measured = [measure_case(family, topology, seed) for family, topology, seed in cases]
    assert all(case.exact for case in measured)


def test_small_benchmark_is_reproducible() -> None:
    first = run(dag_count=3, cyclic_count=3)
    second = run(dag_count=3, cyclic_count=3)
    assert first == second
    assert first["summary"]["cases"] == 13
