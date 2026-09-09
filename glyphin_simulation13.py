"""Simulation 13: scaling benchmark for the frozen Glyphin search stack.

This benchmark is deliberately downstream of Simulation 12. It does not alter
or optimize the encoder/search. It measures how representation size, exactness,
and bounded-search cost change as topology size and density increase.

Simulation 13 is a measurement harness, not a claim of global optimality.
Run it only after the frozen Simulation 12 result has been archived/reviewed.
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

from glyphin_candidate_search import search
from glyphin_encoder import encode_explicit_edges
from glyphin_topology import DirectedTopology


BENCHMARK_VERSION = "13.0"
DEFAULT_SEED = 13012026


@dataclass(frozen=True)
class ScalingCase:
    family: str
    size: int
    density: float
    replicate: int
    seed: int
    nodes: int
    edges: int
    has_cycle: bool
    baseline_chars: int
    shortest_chars: int | None
    char_reduction_pct: float | None
    candidates_generated: int
    candidates_evaluated: int
    parse_failures: int
    inexact_candidates: int
    exact: bool
    elapsed_ms: float


@dataclass(frozen=True)
class ScalingResult:
    benchmark_version: str
    seed: int
    python: str
    platform: str
    config: dict[str, object]
    cases: list[ScalingCase]
    summary: dict[str, object]
    source_sha256: str
    data_sha256: str


def _random_dag(rng: random.Random, n: int, probability: float) -> DirectedTopology:
    nodes = [f"N{i:03d}" for i in range(n)]
    edges = {
        (nodes[i], nodes[j])
        for i in range(n)
        for j in range(i + 1, n)
        if rng.random() < probability
    }
    return DirectedTopology.from_edges(edges, nodes)


def _random_cyclic(rng: random.Random, n: int, probability: float) -> DirectedTopology:
    topology = _random_dag(rng, n, probability)
    nodes = sorted(topology.nodes)
    if n >= 2:
        topology.add_edge(nodes[-1], nodes[0])
    return topology


def _chain(n: int) -> DirectedTopology:
    nodes = [f"N{i:03d}" for i in range(n)]
    return DirectedTopology.from_edges(
        [(nodes[i], nodes[i + 1]) for i in range(n - 1)], nodes
    )


def _cycle(n: int) -> DirectedTopology:
    topology = _chain(n)
    nodes = sorted(topology.nodes)
    if n >= 2:
        topology.add_edge(nodes[-1], nodes[0])
    return topology


def _star(n: int) -> DirectedTopology:
    nodes = [f"N{i:03d}" for i in range(n)]
    return DirectedTopology.from_edges(
        [(nodes[0], node) for node in nodes[1:]], nodes
    )


def _run_case(family: str, size: int, density: float, replicate: int, seed: int,
              topology: DirectedTopology) -> ScalingCase:
    baseline = encode_explicit_edges(topology)
    started = time.perf_counter()
    result = search(topology)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    shortest = result.shortest_exact
    reduction = None
    if shortest is not None and baseline:
        reduction = (1.0 - shortest.chars / len(baseline)) * 100.0
    return ScalingCase(
        family=family,
        size=size,
        density=density,
        replicate=replicate,
        seed=seed,
        nodes=len(topology.nodes),
        edges=len(topology.edges),
        has_cycle=topology.has_cycle(),
        baseline_chars=len(baseline),
        shortest_chars=shortest.chars if shortest else None,
        char_reduction_pct=reduction,
        candidates_generated=result.candidates_generated,
        candidates_evaluated=result.candidates_evaluated,
        parse_failures=result.parse_failures,
        inexact_candidates=result.inexact_candidates,
        exact=shortest is not None,
        elapsed_ms=elapsed_ms,
    )


def run_benchmark(
    *,
    seed: int = DEFAULT_SEED,
    sizes: tuple[int, ...] = (4, 8, 12, 16, 20, 24, 28),
    replicates: int = 10,
    densities: tuple[float, ...] = (0.10, 0.20, 0.35),
    include_structured: bool = True,
) -> ScalingResult:
    if not sizes or any(size < 2 for size in sizes):
        raise ValueError("all sizes must be >= 2")
    if replicates < 1:
        raise ValueError("replicates must be >= 1")
    if not densities or any(not 0.0 <= d <= 1.0 for d in densities):
        raise ValueError("densities must be between 0 and 1")

    cases: list[ScalingCase] = []
    master = random.Random(seed)

    for size in sizes:
        for density in densities:
            for replicate in range(replicates):
                graph_seed = master.getrandbits(64)
                rng = random.Random(graph_seed)
                topology = _random_dag(rng, size, density)
                cases.append(_run_case("random_dag", size, density, replicate, graph_seed, topology))

                graph_seed = master.getrandbits(64)
                rng = random.Random(graph_seed)
                topology = _random_cyclic(rng, size, density)
                cases.append(_run_case("random_cyclic", size, density, replicate, graph_seed, topology))

        if include_structured:
            for family, topology in (
                ("chain", _chain(size)),
                ("cycle", _cycle(size)),
                ("star", _star(size)),
            ):
                cases.append(_run_case(family, size, 0.0, 0, master.getrandbits(64), topology))

    exact_cases = [case for case in cases if case.exact]
    reductions = [case.char_reduction_pct for case in exact_cases if case.char_reduction_pct is not None]
    summary = {
        "total_cases": len(cases),
        "exact_cases": len(exact_cases),
        "exact_rate_pct": (len(exact_cases) / len(cases) * 100.0) if cases else 0.0,
        "mean_char_reduction_pct_exact": (sum(reductions) / len(reductions)) if reductions else None,
        "median_char_reduction_pct_exact": (
            sorted(reductions)[len(reductions) // 2] if reductions else None
        ),
        "max_candidates_generated": max((case.candidates_generated for case in cases), default=0),
        "max_elapsed_ms": max((case.elapsed_ms for case in cases), default=0.0),
        "total_parse_failures": sum(case.parse_failures for case in cases),
        "total_inexact_candidates": sum(case.inexact_candidates for case in cases),
    }
    data_payload = json.dumps([asdict(case) for case in cases], sort_keys=True, separators=(",", ":")).encode()
    source_hash = sha256(Path(__file__).read_bytes()).hexdigest()
    return ScalingResult(
        benchmark_version=BENCHMARK_VERSION,
        seed=seed,
        python=sys.version.split()[0],
        platform=platform.platform(),
        config={
            "sizes": list(sizes),
            "replicates": replicates,
            "densities": list(densities),
            "include_structured": include_structured,
        },
        cases=cases,
        summary=summary,
        source_sha256=source_hash,
        data_sha256=sha256(data_payload).hexdigest(),
    )


def _main() -> int:
    parser = argparse.ArgumentParser(description="Run Glyphin Simulation 13 scaling benchmark")
    parser.add_argument("--sizes", nargs="+", type=int, default=[4, 8, 12, 16, 20, 24, 28])
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--densities", nargs="+", type=float, default=[0.10, 0.20, 0.35])
    parser.add_argument("--no-structured", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("glyphin_simulation13_result.json"))
    args = parser.parse_args()

    result = run_benchmark(
        seed=args.seed,
        sizes=tuple(args.sizes),
        replicates=args.replicates,
        densities=tuple(args.densities),
        include_structured=not args.no_structured,
    )
    args.output.write_text(json.dumps(asdict(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary": result.summary}, indent=2, sort_keys=True))
    return 0 if result.summary["exact_cases"] == result.summary["total_cases"] else 2


if __name__ == "__main__":
    raise SystemExit(_main())
