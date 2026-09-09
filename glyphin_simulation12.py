"""Simulation 12: generalization benchmark for the frozen Glyphin encoder/search.

The benchmark deliberately keeps the encoder/search unchanged. It generates
random graph families plus adversarial topologies, reconstructs each candidate
through the independent referee, and records exactness/compression statistics.

This is a bounded benchmark, not a proof of global optimality. Token counts are
reported only when a real tokenizer is explicitly supplied by the caller.
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Callable, Iterable

from glyphin_candidate_search import search
from glyphin_compression import measure
from glyphin_encoder import encode_explicit_edges
from glyphin_topology import DirectedTopology


BENCHMARK_VERSION = "12.0"
DEFAULT_SEED = 12012026


@dataclass(frozen=True)
class CaseResult:
    family: str
    case_id: str
    seed: int | None
    nodes: int
    edges: int
    has_cycle: bool
    baseline_chars: int
    shortest_chars: int | None
    char_reduction_pct: float | None
    shortest_words: int | None
    candidates_generated: int
    candidates_evaluated: int
    parse_failures: int
    inexact_candidates: int
    exact_candidates: int
    exact: bool
    missing_edges: int
    extra_edges: int
    elapsed_ms: float


@dataclass(frozen=True)
class BenchmarkResult:
    benchmark_version: str
    seed: int
    python: str
    platform: str
    config: dict[str, int]
    graph_families: dict[str, int]
    cases: list[CaseResult]
    summary: dict[str, object]
    source_sha256: str


def _random_dag(rng: random.Random, node_count: int, probability: float) -> DirectedTopology:
    nodes = [f"N{i:02d}" for i in range(node_count)]
    edges = set()
    for i, source in enumerate(nodes):
        for target in nodes[i + 1 :]:
            if rng.random() < probability:
                edges.add((source, target))
    return DirectedTopology.from_edges(edges, nodes)


def _random_cyclic(rng: random.Random, node_count: int, probability: float) -> DirectedTopology:
    topology = _random_dag(rng, node_count, probability)
    nodes = sorted(topology.nodes)
    # Force one directed cycle without using graph-specific labels.
    if node_count >= 2:
        source = nodes[-1]
        target = nodes[0]
        topology.add_edge(source, target)
    return topology


def _adversarial_cases() -> list[tuple[str, DirectedTopology]]:
    cases: list[tuple[str, DirectedTopology]] = []

    def add(name: str, nodes: Iterable[str], edges: Iterable[tuple[str, str]]) -> None:
        cases.append((name, DirectedTopology.from_edges(edges, nodes)))

    add("isolated", ["A", "B", "C", "D", "E"], [])
    add("single_chain", ["A", "B", "C", "D", "E", "F"], [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E"), ("E", "F")])
    add("fan_out", ["R", "A", "B", "C", "D", "E"], [("R", x) for x in ["A", "B", "C", "D", "E"]])
    add("fan_in", ["A", "B", "C", "D", "E", "J"], [(x, "J") for x in ["A", "B", "C", "D", "E"]])
    add("cycle", ["A", "B", "C", "D"], [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")])
    add("two_cycles", ["A", "B", "C", "D", "E", "F"], [("A", "B"), ("B", "C"), ("C", "A"), ("D", "E"), ("E", "F"), ("F", "D")])
    add("mixed_disconnected", ["A", "B", "C", "D", "E", "F", "G", "H"], [("A", "B"), ("B", "C"), ("D", "E"), ("E", "D"), ("F", "G"), ("G", "H"), ("F", "H")])
    add("diamond", ["A", "B", "C", "D"], [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")])
    add("dense_dag", [f"N{i}" for i in range(8)], [(f"N{i}", f"N{j}") for i in range(8) for j in range(i + 1, 8)])
    add("cycle_with_chains", ["A", "B", "C", "D", "E", "F", "G", "H"], [("A", "B"), ("B", "C"), ("C", "A"), ("C", "D"), ("D", "E"), ("F", "G"), ("G", "H"), ("H", "F"), ("E", "F")])
    return cases


def _run_case(family: str, case_id: str, seed: int | None, topology: DirectedTopology) -> CaseResult:
    baseline = encode_explicit_edges(topology)
    baseline_metrics = measure(baseline, baseline)
    started = time.perf_counter()
    result = search(topology)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    shortest = result.shortest_exact
    reduction = None
    if shortest is not None and baseline_metrics.encoded_chars:
        reduction = (1.0 - shortest.chars / baseline_metrics.encoded_chars) * 100.0
    return CaseResult(
        family=family,
        case_id=case_id,
        seed=seed,
        nodes=len(topology.nodes),
        edges=len(topology.edges),
        has_cycle=topology.has_cycle(),
        baseline_chars=baseline_metrics.encoded_chars,
        shortest_chars=shortest.chars if shortest else None,
        char_reduction_pct=reduction,
        shortest_words=shortest.words if shortest else None,
        candidates_generated=result.candidates_generated,
        candidates_evaluated=result.candidates_evaluated,
        parse_failures=result.parse_failures,
        inexact_candidates=result.inexact_candidates,
        exact_candidates=len(result.exact_candidates),
        exact=shortest is not None,
        missing_edges=len(shortest.missing_edges) if shortest else -1,
        extra_edges=len(shortest.extra_edges) if shortest else -1,
        elapsed_ms=elapsed_ms,
    )


def _run_random_family(
    family: str,
    count: int,
    seed: int,
    cyclic: bool,
    min_nodes: int,
    max_nodes: int,
    probability: float,
) -> list[CaseResult]:
    rng = random.Random(seed)
    cases: list[CaseResult] = []
    for index in range(count):
        node_count = rng.randint(min_nodes, max_nodes)
        graph_seed = rng.getrandbits(64)
        graph_rng = random.Random(graph_seed)
        topology = _random_cyclic(graph_rng, node_count, probability) if cyclic else _random_dag(graph_rng, node_count, probability)
        cases.append(_run_case(family, f"{family}-{index:04d}", graph_seed, topology))
    return cases


def _summarize(cases: list[CaseResult]) -> dict[str, object]:
    exact_cases = [case for case in cases if case.exact]
    reductions = [case.char_reduction_pct for case in exact_cases if case.char_reduction_pct is not None]
    return {
        "total_cases": len(cases),
        "exact_cases": len(exact_cases),
        "exact_rate_pct": (len(exact_cases) / len(cases) * 100.0) if cases else 0.0,
        "mean_char_reduction_pct_exact": (sum(reductions) / len(reductions)) if reductions else None,
        "median_char_reduction_pct_exact": _median(reductions),
        "min_shortest_chars": min((case.shortest_chars for case in exact_cases if case.shortest_chars is not None), default=None),
        "total_parse_failures": sum(case.parse_failures for case in cases),
        "total_inexact_candidates": sum(case.inexact_candidates for case in cases),
        "total_candidates_generated": sum(case.candidates_generated for case in cases),
        "max_elapsed_ms": max((case.elapsed_ms for case in cases), default=0.0),
    }


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def run_benchmark(
    *,
    seed: int = DEFAULT_SEED,
    dag_count: int = 1000,
    cyclic_count: int = 1000,
    adversarial: bool = True,
    min_nodes: int = 4,
    max_nodes: int = 12,
    probability: float = 0.20,
) -> BenchmarkResult:
    if min_nodes < 2 or max_nodes < min_nodes:
        raise ValueError("node bounds must satisfy 2 <= min_nodes <= max_nodes")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("edge probability must be between 0 and 1")

    cases = _run_random_family("random_dag", dag_count, seed, False, min_nodes, max_nodes, probability)
    cases += _run_random_family("random_cyclic", cyclic_count, seed ^ 0xC0FFEE, True, min_nodes, max_nodes, probability)
    if adversarial:
        for index, (name, topology) in enumerate(_adversarial_cases()):
            cases.append(_run_case("adversarial", f"adversarial-{index:02d}-{name}", None, topology))

    config = {
        "dag_count": dag_count,
        "cyclic_count": cyclic_count,
        "min_nodes": min_nodes,
        "max_nodes": max_nodes,
        "edge_probability_pct": int(probability * 100),
        "adversarial_cases": 10 if adversarial else 0,
    }
    families = {
        "random_dag": dag_count,
        "random_cyclic": cyclic_count,
        "adversarial": 10 if adversarial else 0,
    }
    payload = json.dumps([asdict(case) for case in cases], sort_keys=True, separators=(",", ":")).encode()
    return BenchmarkResult(
        benchmark_version=BENCHMARK_VERSION,
        seed=seed,
        python=sys.version.split()[0],
        platform=platform.platform(),
        config=config,
        graph_families=families,
        cases=cases,
        summary=_summarize(cases),
        source_sha256=sha256(payload).hexdigest(),
    )


def _main() -> int:
    parser = argparse.ArgumentParser(description="Run Glyphin Simulation 12 benchmark")
    parser.add_argument("--dag-count", type=int, default=1000)
    parser.add_argument("--cyclic-count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--min-nodes", type=int, default=4)
    parser.add_argument("--max-nodes", type=int, default=12)
    parser.add_argument("--edge-probability", type=float, default=0.20)
    parser.add_argument("--no-adversarial", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("glyphin_simulation12_result.json"))
    args = parser.parse_args()

    result = run_benchmark(
        seed=args.seed,
        dag_count=args.dag_count,
        cyclic_count=args.cyclic_count,
        adversarial=not args.no_adversarial,
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
        probability=args.edge_probability,
    )
    args.output.write_text(json.dumps(asdict(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary": result.summary}, indent=2, sort_keys=True))
    return 0 if result.summary["exact_cases"] == result.summary["total_cases"] else 2


if __name__ == "__main__":
    raise SystemExit(_main())
