"""Glyphin research topology primitives.

This module deliberately separates a general directed research graph from the
single-parent lineage field in GlyphState. That distinction matters because
experimental graphs may contain convergence, fan-out, and cycles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Set, Tuple

Edge = Tuple[str, str]


@dataclass
class DirectedTopology:
    """Canonical directed topology used by reconstruction/referee experiments."""

    nodes: Set[str] = field(default_factory=set)
    edges: Set[Edge] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.nodes = set(self.nodes)
        self.edges = {tuple(edge) for edge in self.edges}
        for edge in self.edges:
            if len(edge) != 2:
                raise ValueError("edges must contain exactly two node names")
            self.nodes.update(edge)
        self.validate()

    def validate(self) -> None:
        if any(not isinstance(node, str) or not node for node in self.nodes):
            raise ValueError("node names must be non-empty strings")
        for source, target in self.edges:
            if source not in self.nodes or target not in self.nodes:
                raise ValueError("every edge endpoint must be a node")

    @classmethod
    def from_edges(
        cls, edges: Iterable[Edge], nodes: Iterable[str] = ()
    ) -> "DirectedTopology":
        return cls(set(nodes), set(edges))

    def add_node(self, name: str) -> None:
        if not name:
            raise ValueError("node name must not be empty")
        self.nodes.add(name)

    def add_edge(self, source: str, target: str) -> None:
        self.nodes.update((source, target))
        self.edges.add((source, target))

    def successors(self, node: str) -> List[str]:
        return sorted(target for source, target in self.edges if source == node)

    def predecessors(self, node: str) -> List[str]:
        return sorted(source for source, target in self.edges if target == node)

    def roots(self) -> List[str]:
        return sorted(node for node in self.nodes if not self.predecessors(node))

    def leaves(self) -> List[str]:
        return sorted(node for node in self.nodes if not self.successors(node))

    def out_degree(self, node: str) -> int:
        return len(self.successors(node))

    def in_degree(self, node: str) -> int:
        return len(self.predecessors(node))

    def has_cycle(self) -> bool:
        visiting: Set[str] = set()
        visited: Set[str] = set()

        def visit(node: str) -> bool:
            if node in visiting:
                return True
            if node in visited:
                return False
            visiting.add(node)
            for child in self.successors(node):
                if visit(child):
                    return True
            visiting.remove(node)
            visited.add(node)
            return False

        return any(visit(node) for node in sorted(self.nodes))

    def canonical(self) -> Dict[str, object]:
        return {
            "nodes": sorted(self.nodes),
            "edges": [list(edge) for edge in sorted(self.edges)],
        }

    def fingerprint(self) -> Tuple[Tuple[str, ...], Tuple[Edge, ...]]:
        return tuple(sorted(self.nodes)), tuple(sorted(self.edges))

    def compare(self, other: "DirectedTopology") -> Dict[str, object]:
        missing_nodes = sorted(self.nodes - other.nodes)
        extra_nodes = sorted(other.nodes - self.nodes)
        missing_edges = sorted(self.edges - other.edges)
        extra_edges = sorted(other.edges - self.edges)
        return {
            "exact": not (missing_nodes or extra_nodes or missing_edges or extra_edges),
            "node_fidelity": (
                1.0 if self.nodes == other.nodes else len(self.nodes & other.nodes) / max(1, len(self.nodes))
            ),
            "edge_fidelity": (
                1.0 if self.edges == other.edges else len(self.edges & other.edges) / max(1, len(self.edges))
            ),
            "missing_nodes": missing_nodes,
            "extra_nodes": extra_nodes,
            "missing_edges": [list(edge) for edge in missing_edges],
            "extra_edges": [list(edge) for edge in extra_edges],
        }


__all__ = ["DirectedTopology", "Edge"]
