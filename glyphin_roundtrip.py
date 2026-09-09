"""Round-trip referee for the GlyphinMemory/topology representation boundary.

This module verifies structural preservation only. It deliberately does not
claim that topology round-tripping preserves GlyphState dynamics such as
cohesion, resonance, frequency, sigma, or timestamps.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from glyphin_research_core import GlyphinMemory
from glyphin_topology import DirectedTopology
from glyphin_topology_adapter import AdaptationReport, memory_to_topology, topology_to_memory


@dataclass(frozen=True)
class RoundTripResult:
    structural_exact: bool
    original_nodes: int
    original_edges: int
    round_trip_nodes: int
    round_trip_edges: int
    first_adaptation: AdaptationReport
    second_adaptation: AdaptationReport
    missing_nodes: list[str]
    extra_nodes: list[str]
    missing_edges: list[tuple[str, str]]
    extra_edges: list[tuple[str, str]]
    dynamics_verified: bool = False

    def to_dict(self) -> dict[str, Any]:
        def report_dict(report: AdaptationReport) -> dict[str, Any]:
            return {
                "source_nodes": report.source_nodes,
                "source_edges": report.source_edges,
                "target_nodes": report.target_nodes,
                "target_edges": report.target_edges,
                "unsupported_edges": [list(e) for e in report.unsupported_edges],
                "lost_state_fields": report.lost_state_fields,
                "warnings": report.warnings,
                "lossless": report.lossless,
            }

        return {
            "structural_exact": self.structural_exact,
            "original_nodes": self.original_nodes,
            "original_edges": self.original_edges,
            "round_trip_nodes": self.round_trip_nodes,
            "round_trip_edges": self.round_trip_edges,
            "first_adaptation": report_dict(self.first_adaptation),
            "second_adaptation": report_dict(self.second_adaptation),
            "missing_nodes": self.missing_nodes,
            "extra_nodes": self.extra_nodes,
            "missing_edges": [list(e) for e in self.missing_edges],
            "extra_edges": [list(e) for e in self.extra_edges],
            "dynamics_verified": self.dynamics_verified,
        }


def round_trip_memory(memory: GlyphinMemory, *, strict: bool = True) -> RoundTripResult:
    """Convert memory to topology and back, then independently compare topology."""
    original, first = memory_to_topology(memory)
    reconstructed_memory, second = topology_to_memory(original, strict=strict)
    round_trip, _ = memory_to_topology(reconstructed_memory)
    diff = original.compare(round_trip)
    return RoundTripResult(
        structural_exact=bool(diff["exact"]),
        original_nodes=len(original.nodes),
        original_edges=len(original.edges),
        round_trip_nodes=len(round_trip.nodes),
        round_trip_edges=len(round_trip.edges),
        first_adaptation=first,
        second_adaptation=second,
        missing_nodes=diff["missing_nodes"],
        extra_nodes=diff["extra_nodes"],
        missing_edges=[tuple(e) for e in diff["missing_edges"]],
        extra_edges=[tuple(e) for e in diff["extra_edges"]],
    )


def round_trip_topology(topology: DirectedTopology, *, strict: bool = True) -> RoundTripResult:
    """Convert topology to GlyphinMemory and back, then compare structure."""
    memory, first = topology_to_memory(topology, strict=strict)
    reconstructed, second = memory_to_topology(memory)
    diff = topology.compare(reconstructed)
    return RoundTripResult(
        structural_exact=bool(diff["exact"]),
        original_nodes=len(topology.nodes),
        original_edges=len(topology.edges),
        round_trip_nodes=len(reconstructed.nodes),
        round_trip_edges=len(reconstructed.edges),
        first_adaptation=first,
        second_adaptation=second,
        missing_nodes=diff["missing_nodes"],
        extra_nodes=diff["extra_nodes"],
        missing_edges=[tuple(e) for e in diff["missing_edges"]],
        extra_edges=[tuple(e) for e in diff["extra_edges"]],
    )


__all__ = ["RoundTripResult", "round_trip_memory", "round_trip_topology"]
