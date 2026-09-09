"""Generalization benchmark for the bounded Glyphin encoder/search.

This runner is deliberately measurement-first. It generates graphs without
looking up a known solution, reconstructs candidate encodings, and records
exact referee outcomes plus character/word compression. Token counts remain
unmeasured unless a tokenizer is explicitly supplied.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Iterable

from glyphin_candidate_search import search
from glyphin_compression import measure
from glyphin_encoder import encode_explicit_edges
from glyphin_topology import DirectedTopology


@dataclass(frozen=True)
class CaseResult:
    family: str
    seed: int
    nodes: int
    edges: int
    candidates: int
    parse_failures: int
    inexact_candidates: int
    exact: bool
    baseline_chars: int
    shortest_chars: int | None
    char_reduction_pct: float | None
    baseline_words: int
    shortest_words: int | None
    word_reduction_pct: float | None


def random_graph(seed: int, *, cyclic: bool, node_count: int | None = None) -> DirectedTopology:
    """Generate a deterministic random graph without graph-specific lookup."""
    rng = random.Random(seed)
    count = node_count if node_count is not None else rng.randint(2, 12)
    nodes = [f"N{i:02d}" for i in range(count)]
    edges: set[tuple[str, str]] = set()

    # DAG: only forward edges. Cyclic: add a guaranteed small cycle after
    # generating the same forward-edge family.
    for i in range(count):
        for j in range(i + 1, count):
            if rng.random() < 0.20:
                edges.add((nodes[i], nodes[j]))

    if cyclic and count >= 3:
        edges.update({(nodes[0], nodes[1]), (nodes[1], nodes[2]), (nodes[2], nodes[0])})

    return DirectedTopology.from_edges(sorted(edges), nodes=nodes)


def adversarial_cases() -> Iterable[tuple[str, DirectedTopology, int]]:
    """Return deterministic structures that stress representation boundaries."""
    yield "empty", DirectedTopology(), 0
    yield "isolated", DirectedTopology.from_edges([], nodes=["A", "B", "C", "D"]), 1
    yield "chain", DirectedTopology.from_edges([(f"N{i}", f"N{i+1}") for i in range(10)]), 2
    yield "star", DirectedTopology.from_edges([("R", f"N{i}") for i in range(10)]), 3
    yield "fan_in", DirectedTopology.from_edges([(f"N{i}", "Z") for i in range(10)]), 4
    yield "cycle", DirectedTopology.from_edges([("A", "B"), ("B", "C"), ("C", "A")]), 5
    yield "mixed", DirectedTopology.from_edges(
        [("R", "A"), ("R", "B"), ("A", "C"), ("B", "C"), ("C", "D"), ("D", "C")],
        nodes=["isolated"],
    ), 6


def measure_case(family: str, topology: DirectedTopology, seed: int) -> CaseResult:
    result = search(topology)
    baseline = encode_explicit_edges(topology)
    shortest = result.shortest_exact
    metrics = measure(source=baseline, encoded=shortest.encoding) if shortest else None
    return CaseResult(
        family=family,
        seed=seed,
        nodes=len(topology.nodes),
        edges=len(topology.edges),
        candidates=result.candidates_generated,
        parse_failures=result.parse_failures,
        inexact_candidates=result.inexact_candidates,
        exact=shortest is not None,
        baseline_chars=len(baseline),
        shortest_chars=shortest.chars if shortest else None,
        char_reduction_pct=metrics.char_reduction_pct if metrics else None,
        baseline_words=len(baseline.split()),
        shortest_words=shortest.words if shortest else None,
        word_reduction_pct=metrics.word_reduction_pct if metrics else None,
    )


def run(*, dag_count: int, cyclic_count: int) -> dict[str, object]:
    cases: list[CaseResult] = []
    for seed in range(dag_count):
        cases.append(measure_case("random_dag", random_graph(seed, cyclic=False), seed))
    for seed in range(cyclic_count):
        cases.append(measure_case("random_cyclic", random_graph(seed, cyclic=True), seed))
    for family, topology, seed in adversarial_cases():
        cases.append(measure_case(family, topology, seed))

    exact = sum(case.exact for case in cases)
    reductions = [case.char_reduction_pct for case in cases if case.char_reduction_pct is not None]
    return {
        "schema": "glyphin-benchmark-1",
        "configuration": {
            "dag_count": dag_count,
            "cyclic_count": cyclic_count,
            "random_node_range": [2, 12],
            "edge_probability": 0.20,
            "search": "bounded candidate search; not exhaustive or globally optimal",
            "tokens": "NOT_MEASURED",
        },
        "summary": {
            "cases": len(cases),
            "exact_cases": exact,
            "exact_rate_pct": (exact / len(cases) * 100.0) if cases else 0.0,
            "mean_char_reduction_pct": (sum(reductions) / len(reductions)) if reductions else None,
            "min_char_reduction_pct": min(reductions) if reductions else None,
            "max_char_reduction_pct": max(reductions) if reductions else None,
        },
        "cases": [asdict(case) for case in cases],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Glyphin generalization benchmarks")
    parser.add_argument("--dag-count", type=int, default=1000)
    parser.add_argument("--cyclic-count", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=Path("glyphin_benchmark.json"))
    args = parser.parse_args(argv)
    if args.dag_count < 0 or args.cyclic_count < 0:
        parser.error("graph counts must be non-negative")

    evidence = run(dag_count=args.dag_count, cyclic_count=args.cyclic_count)
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(evidence["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
