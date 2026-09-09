from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from glyphin_candidate_search import bounded_candidate_search
from glyphin_reconstruction import reconstruct
from glyphin_topology import DirectedTopology

BENCHMARK_VERSION = "13.1"
DEFAULT_SEED = 13012026


@dataclass(frozen=True)
class ScalingCase:
    family: str
    n_nodes: int
    density: float
    replicate: int
    nodes: int
    edges: int
    baseline_chars: int
    shortest_chars: int | None
    char_reduction_pct: float | None
    candidates_generated: int
    parse_failures: int
    inexact_candidates: int
    elapsed_ms: float
    exact: bool


@dataclass(frozen=True)
class ScalingResult:
    benchmark_version: str
    seed: int
    python: str
    platform: str
    config: dict
    cases: list[ScalingCase]
    summary: dict
    source_sha256: str
    data_sha256: str


def _random_dag(rng: random.Random, n: int, density: float) -> DirectedTopology:
    g = DirectedTopology()
    for i in range(n):
        g.add_node(f"N{i}")
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < density:
                g.add_edge(f"N{i}", f"N{j}")
    return g


def _random_cyclic(rng: random.Random, n: int, density: float) -> DirectedTopology:
    g = _random_dag(rng, n, density)
    if n > 1:
        g.add_edge(f"N{n - 1}", "N0")
    return g


def _structured(kind: str, n: int) -> DirectedTopology:
    g = DirectedTopology()
    for i in range(n):
        g.add_node(f"N{i}")
    if kind == "chain":
        for i in range(n - 1):
            g.add_edge(f"N{i}", f"N{i + 1}")
    elif kind == "cycle":
        for i in range(n):
            g.add_edge(f"N{i}", f"N{(i + 1) % n}")
    elif kind == "star":
        for i in range(1, n):
            g.add_edge("N0", f"N{i}")
    else:
        raise ValueError(kind)
    return g


def _run_case(family: str, g: DirectedTopology, density: float, replicate: int) -> ScalingCase:
    start = time.perf_counter()
    baseline = ";".join(f"{a}->{b}" for a, b in sorted(g.edges))
    search = bounded_candidate_search(g)
    shortest = search.shortest_exact
    reconstructed_exact = False
    if shortest is not None:
        try:
            reconstructed_exact = reconstruct(shortest).compare(g)["exact"]
        except Exception:
            reconstructed_exact = False
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    reduction = None if shortest is None else (1.0 - len(shortest) / len(baseline)) * 100.0 if baseline else 0.0
    return ScalingCase(
        family=family,
        n_nodes=len(g.nodes),
        density=density,
        replicate=replicate,
        nodes=len(g.nodes),
        edges=len(g.edges),
        baseline_chars=len(baseline),
        shortest_chars=None if shortest is None else len(shortest),
        char_reduction_pct=reduction,
        candidates_generated=search.candidates_generated,
        parse_failures=search.parse_failures,
        inexact_candidates=search.inexact_candidates,
        elapsed_ms=elapsed_ms,
        exact=reconstructed_exact,
    )


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def run_benchmark(
    seed: int = DEFAULT_SEED,
    sizes: tuple[int, ...] = (4, 8, 12, 16, 20, 24, 28),
    replicates: int = 10,
    densities: tuple[float, ...] = (0.10, 0.20, 0.35),
    include_structured: bool = True,
) -> ScalingResult:
    rng = random.Random(seed)
    cases: list[ScalingCase] = []
    for n in sizes:
        for density in densities:
            for replicate in range(replicates):
                cases.append(_run_case("random_dag", _random_dag(rng, n, density), density, replicate))
                cases.append(_run_case("random_cyclic", _random_cyclic(rng, n, density), density, replicate))
        if include_structured:
            for kind in ("chain", "cycle", "star"):
                cases.append(_run_case(f"structured_{kind}", _structured(kind, n), 0.0, 0))

    exact_cases = [case for case in cases if case.exact]
    reductions = [case.char_reduction_pct for case in exact_cases if case.char_reduction_pct is not None]
    summary = {
        "total_cases": len(cases),
        "exact_cases": len(exact_cases),
        "exact_rate_pct": (len(exact_cases) / len(cases) * 100.0) if cases else 0.0,
        "mean_char_reduction_pct_exact": (sum(reductions) / len(reductions)) if reductions else None,
        "median_char_reduction_pct_exact": _median(reductions),
        "max_candidates_generated": max((case.candidates_generated for case in cases), default=0),
        "max_elapsed_ms": max((case.elapsed_ms for case in cases), default=0.0),
        "total_parse_failures": sum(case.parse_failures for case in cases),
        "total_inexact_candidates": sum(case.inexact_candidates for case in cases),
    }
    data_payload = json.dumps([asdict(case) for case in cases], sort_keys=True, separators=(",", ":")).encode()
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
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
        data_sha256=hashlib.sha256(data_payload).hexdigest(),
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
    result = run_benchmark(seed=args.seed, sizes=tuple(args.sizes), replicates=args.replicates, densities=tuple(args.densities), include_structured=not args.no_structured)
    args.output.write_text(json.dumps(asdict(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary": result.summary}, indent=2, sort_keys=True))
    return 0 if result.summary["exact_cases"] == result.summary["total_cases"] else 2


if __name__ == "__main__":
    raise SystemExit(_main())
