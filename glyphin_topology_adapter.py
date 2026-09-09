"""Explicit adapters between GlyphinMemory and DirectedTopology.

This module does not claim the two representations are equivalent.
GlyphinMemory is single-parent and state-rich; DirectedTopology is a general
edge graph. The adapter reports information that cannot survive conversion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

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


def topology_to_memory(topology: DirectedTopology) -> tuple[GlyphinMemory, AdaptationReport]:
    """Convert a topology only when its edges admit a single-parent forest.

    Fan-in and cycles cannot be represented faithfully by GlyphinMemory's
    single-parent lineage model, so those edges are reported as unsupported
    instead of being silently discarded or overwritten.
    """
    memory = GlyphinMemory()
    for name in sorted(topology.nodes):
        memory.add_state(name)

    unsupported: list[tuple[str, str]] = []
    parent_for: Dict[str, str] = {}
    for parent, child in sorted(topology.edges):
        if child in parent_for and parent_for[child] != parent:
            unsupported.append((parent, child))
            continue
        if child == parent:
            unsupported.append((parent, child))
            continue
        parent_for[child] = parent

    # Only apply a parent mapping after validating it; this keeps the report
    # honest when a graph contains fan-in or cycles.
    for child, parent in sorted(parent_for.items()):
        if (parent, child) in unsupported:
            continue
        if memory.states[child].parent is None:
            memory.states[child].parent = parent
            memory.states[parent].children.append(child)

    if topology.has_cycle():
        for edge in sorted(topology.edges):
            if edge not in unsupported:
                child_path = memory.get_path(edge[1]) if memory.states[edge[1]].parent else [edge[1]]
                if edge[0] in child_path:
                    unsupported.append(edge)

    report = AdaptationReport(
        source_nodes=len(topology.nodes),
        source_edges=len(topology.edges),
        target_nodes=len(memory.states),
        target_edges=sum(1 for s in memory.states.values() if s.parent is not None),
        unsupported_edges=sorted(set(unsupported)),
        lost_state_fields=[],
        warnings=["GlyphState cannot faithfully encode general fan-in or cycles"],
    )
    return memory, report


__all__ = ["AdaptationReport", "memory_to_topology", "topology_to_memory"]
