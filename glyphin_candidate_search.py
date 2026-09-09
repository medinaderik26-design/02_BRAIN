"""Bounded, transparent candidate search for Glyphin symbolic encodings.

This module is intentionally not called an exhaustive or global optimizer.
It generates a finite set of representations from topology-local structure,
then uses the independent reconstruction/referee layer to decide fidelity.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from glyphin_compression import measure_compression
from glyphin_encoder import encode_explicit_edges
from glyphin_referee import referee
from glyphin_topology import DirectedTopology


@dataclass(frozen=True)
class CandidateResult:
    encoding: str
    exact: bool
    parse_ok: bool
    chars: int
    words: int
    token_count: int | None
    missing_edges: tuple[tuple[str, str], ...]
    extra_edges: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class SearchResult:
    candidates_generated: int
    candidates_evaluated: int
    exact_candidates: tuple[CandidateResult, ...]
    shortest_exact: CandidateResult | None


def _edge_statements(topology: DirectedTopology) -> Iterable[str]:
    for source, target in sorted(topology.edges):
        yield f"{source}->{target}"


def _chain_candidates(topology: DirectedTopology) -> Iterable[str]:
    """Generate maximal chains beginning at structural boundary nodes."""
    for source in sorted(topology.nodes):
        if topology.out_degree(source) == 0:
            continue
        if topology.in_degree(source) == 1:
            continue
        for target in topology.successors(source):
            chain = [source, target]
            current = target
            seen = {(source, target)}
            while topology.in_degree(current) == 1 and topology.out_degree(current) == 1:
                nxt = topology.successors(current)[0]
                edge = (current, nxt)
                if edge in seen:
                    break
                seen.add(edge)
                chain.append(nxt)
                current = nxt
            yield "->".join(chain)


def generate_candidates(topology: DirectedTopology) -> list[str]:
    """Generate deterministic bounded candidates without graph-specific lookup."""
    candidates: set[str] = set()
    edges = list(_edge_statements(topology))
    candidates.add(encode_explicit_edges(topology))
    candidates.update(edges)

    chains = sorted(set(_chain_candidates(topology)))
    for chain in chains:
        remaining = [edge for edge in edges if edge not in _edges_from_chain(chain)]
        candidates.add(";".join([chain, *remaining]))

    return sorted(candidate for candidate in candidates if candidate)


def _edges_from_chain(chain: str) -> set[str]:
    nodes = chain.split("->")
    return {f"{nodes[i]}->{nodes[i + 1]}" for i in range(len(nodes) - 1)}


def evaluate_candidates(
    topology: DirectedTopology, candidates: Iterable[str]
) -> list[CandidateResult]:
    results: list[CandidateResult] = []
    for encoded in candidates:
        result = referee(topology, encoded)
        metrics = measure_compression(
            source=encode_explicit_edges(topology), encoded=encoded
        )
        comparison = result.comparison
        results.append(
            CandidateResult(
                encoding=encoded,
                exact=result.exact_match,
                parse_ok=result.parse_ok,
                chars=metrics.encoded_chars,
                words=metrics.encoded_words,
                token_count=metrics.encoded_tokens,
                missing_edges=tuple(tuple(edge) for edge in comparison.get("missing_edges", [])),
                extra_edges=tuple(tuple(edge) for edge in comparison.get("extra_edges", [])),
            )
        )
    return results


def search(topology: DirectedTopology) -> SearchResult:
    candidates = generate_candidates(topology)
    evaluated = evaluate_candidates(topology, candidates)
    exact = tuple(sorted((item for item in evaluated if item.exact), key=lambda item: (item.chars, item.words, item.encoding)))
    return SearchResult(
        candidates_generated=len(candidates),
        candidates_evaluated=len(evaluated),
        exact_candidates=exact,
        shortest_exact=exact[0] if exact else None,
    )


__all__ = ["CandidateResult", "SearchResult", "generate_candidates", "evaluate_candidates", "search"]
