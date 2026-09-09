"""Explicit adapters between GlyphinMemory and DirectedTopology.

This module does not claim the two representations are equivalent.
GlyphinMemory is single-parent and state-rich; DirectedTopology is a general
edge graph. The adapter reports information that cannot survive conversion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from glyphin_research_core import GlyphinMemory
from glyphin_topology import DirectedTopology


@dataclass(frozen=True)
class AdaptationReport:
    source_nodes: int
    source_edges: int
    target_nodes: int
    target_edges: int
    unsupported_edges: List[tuple[str, str]] = field(default_factory=list)
    lost_state_fields: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def lossless(self) -> bool:
        return not self.unsupported_edges and not self.lost_state_fields


def memory_to_topology(memory: GlyphinMemory) -> tuple[DirectedTopology, AdaptationReport]:
    """Convert parent/child lineage to a directed topology.

    Every parent link becomes one edge. State attributes remain in the source
    memory and are intentionally not smuggled into topology node identities.
    """
    edges: list[tuple[str, str]] = []
    for name in sorted(memory.states):
        state = memory.states[name]
        if state.parent is not None:
            edges.append((state.parent, name))
    topology = DirectedTopology.from_edges(edges)
    for name in memory.states:
        topology.add_node(name)
    report = AdaptationReport(
        source_nodes=len(memory.states),
        source_edges=len(edges),
        target_nodes=len(topology.nodes),
        target_edges=len(topology.edges),
        lost_state_fields=["level", "cohesion", "frequency", "resonance", "sigma", "created_at"],
        warnings=["topology preserves structure, not GlyphState dynamics"],
    )
    return topology, report


def _cycle_edges(topology: DirectedTopology) -> set[tuple[str, str]]:
    """Return edges whose endpoints are mutually reachable through a cycle."""
    result: set[tuple[str, str]] = set()
    for parent, child in topology.edges:
        if parent == child:
            result.add((parent, child))
            continue
        stack = [child]
        seen = {child}
        while stack:
            current = stack.pop()
            for nxt in topology.successors(current):
                if nxt == parent:
                    result.add((parent, child))
                    stack = []
                    break
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
    return result


def topology_to_memory(
    topology: DirectedTopology, *, strict: bool = True
) -> tuple[GlyphinMemory, AdaptationReport]:
    """Convert a topology while making representation loss explicit.

    ``GlyphinMemory`` permits at most one parent per node and no cycles.
    In strict mode, any fan-in, self-loop, or cycle blocks *all* relationship
    mutation: the returned memory contains nodes only. In non-strict mode,
    representable acyclic single-parent edges are retained and unsupported
    edges are reported. Neither mode invents a parent relationship.
    """
    unsupported: set[tuple[str, str]] = set()
    parent_for: dict[str, str] = {}

    for parent, child in sorted(topology.edges):
        if parent == child:
            unsupported.add((parent, child))
            continue
        existing = parent_for.get(child)
        if existing is not None and existing != parent:
            unsupported.add((parent, child))
            unsupported.add((existing, child))
            continue
        parent_for[child] = parent

    unsupported.update(_cycle_edges(topology))

    if strict and unsupported:
        parent_for = {}

    memory = GlyphinMemory()
    for name in sorted(topology.nodes):
        memory.add_state(name)

    for child, parent in sorted(parent_for.items()):
        if (parent, child) in unsupported:
            continue
        memory.states[child].parent = parent
        memory.states[parent].children.append(child)

    warnings: list[str] = []
    if unsupported:
        warnings.append("GlyphState cannot faithfully encode general fan-in, self-loops, or cycles")
    if strict and unsupported:
        warnings.append("strict conversion blocked relationship mutation because unsupported structure was detected")

    report = AdaptationReport(
        source_nodes=len(topology.nodes),
        source_edges=len(topology.edges),
        target_nodes=len(memory.states),
        target_edges=sum(1 for s in memory.states.values() if s.parent is not None),
        unsupported_edges=sorted(unsupported),
        warnings=warnings,
    )
    return memory, report


__all__ = ["AdaptationReport", "memory_to_topology", "topology_to_memory"]
