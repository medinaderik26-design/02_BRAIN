"""Deterministic topology-to-symbolic encoder for Glyphin research.

This is deliberately a transparent baseline encoder, not an optimizer.
It emits the bounded grammar consumed by ``glyphin_reconstruction`` and
never embeds knowledge of a particular graph. Exact fidelity must be checked
by the independent referee after reconstruction.
"""
from __future__ import annotations

from glyphin_topology import DirectedTopology


def _isolated_nodes(topology: DirectedTopology) -> list[str]:
    """Return nodes with no incident edges in canonical order."""
    return sorted(
        node
        for node in topology.nodes
        if topology.in_degree(node) == 0 and topology.out_degree(node) == 0
    )


def encode_explicit_edges(topology: DirectedTopology) -> str:
    """Encode every edge plus every isolated node.

    A bare identifier is a valid grammar statement and is required to preserve
    isolated nodes. This makes the explicit representation an exact topology
    baseline rather than an edge-only approximation.
    """
    statements = [
        f"{source}->{target}" for source, target in sorted(topology.edges)
    ]
    statements.extend(_isolated_nodes(topology))
    return ";".join(statements)


def encode_chains_and_edges(topology: DirectedTopology) -> str:
    """Encode maximal directed chains, falling back to edge statements.

    A chain is extended only through a node with exactly one predecessor and
    one successor. Branching, merging, and cyclic structures are emitted as
    individual edges so this baseline never silently changes topology.
    Isolated nodes are emitted as bare identifiers.
    """
    visited: set[tuple[str, str]] = set()
    statements: list[str] = []

    def extend(source: str, target: str) -> list[str]:
        nodes = [source, target]
        visited.add((source, target))
        current = target
        while topology.in_degree(current) == 1 and topology.out_degree(current) == 1:
            nxt = topology.successors(current)[0]
            edge = (current, nxt)
            if edge in visited:
                break
            nodes.append(nxt)
            visited.add(edge)
            current = nxt
        return nodes

    for source, target in sorted(topology.edges):
        if (source, target) in visited:
            continue
        if topology.in_degree(source) != 1:
            statements.append("->".join(extend(source, target)))

    for source, target in sorted(topology.edges):
        if (source, target) not in visited:
            statements.append("->".join(extend(source, target)))

    statements.extend(_isolated_nodes(topology))
    return ";".join(statements)


def encode(topology: DirectedTopology, strategy: str = "chains") -> str:
    """Encode topology using a named deterministic baseline strategy."""
    if strategy == "edges":
        return encode_explicit_edges(topology)
    if strategy == "chains":
        return encode_chains_and_edges(topology)
    raise ValueError(f"unknown encoding strategy: {strategy}")


__all__ = ["encode", "encode_explicit_edges", "encode_chains_and_edges"]
