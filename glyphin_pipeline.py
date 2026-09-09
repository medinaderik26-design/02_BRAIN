"""End-to-end deterministic Glyphin research pipeline.

The pipeline joins the state model, persistence referee, topology adapter,
symbolic encoder, independent topology referee, compression measurement, and
bounded candidate search. It deliberately reports each boundary separately so
loss at one representation layer cannot be mistaken for full-state fidelity.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from glyphin_candidate_search import SearchResult, search
from glyphin_compression import CompressionMetrics, measure
from glyphin_encoder import encode, encode_explicit_edges
from glyphin_reconstruction import reconstruct
from glyphin_referee import RefereeResult, referee as referee_topology
from glyphin_research_core import GlyphinMemory
from glyphin_state_referee import StateRefereeResult, referee_memory
from glyphin_topology_adapter import AdaptationReport, memory_to_topology


@dataclass(frozen=True)
class GlyphinPipelineReport:
    memory_state_count: int
    persistence_referee: StateRefereeResult
    adaptation: AdaptationReport
    encoded: str
    topology_referee: RefereeResult
    compression: CompressionMetrics
    candidate_search: SearchResult

    def to_dict(self) -> dict[str, Any]:
        state = self.persistence_referee
        adaptation = self.adaptation
        topology = self.topology_referee
        compression = self.compression
        candidates = self.candidate_search
        return {
            "memory_state_count": self.memory_state_count,
            "persistence_referee": {
                "exact": state.exact,
                "source_count": state.source_count,
                "candidate_count": state.candidate_count,
                "missing_states": list(state.missing_states),
                "extra_states": list(state.extra_states),
                "field_mismatches": list(state.field_mismatches),
                "parameter_mismatches": list(state.parameter_mismatches),
            },
            "adaptation": {
                "source_nodes": adaptation.source_nodes,
                "source_edges": adaptation.source_edges,
                "target_nodes": adaptation.target_nodes,
                "target_edges": adaptation.target_edges,
                "unsupported_edges": [list(edge) for edge in adaptation.unsupported_edges],
                "lost_state_fields": list(adaptation.lost_state_fields),
                "warnings": list(adaptation.warnings),
                "lossless": adaptation.lossless,
            },
            "encoded": self.encoded,
            "topology_referee": {
                "exact_match": topology.exact_match,
                "parse_ok": topology.parse_ok,
                "source_fingerprint": topology.source_fingerprint,
                "candidate_fingerprint": topology.candidate_fingerprint,
                "missing_nodes": list(topology.missing_nodes),
                "extra_nodes": list(topology.extra_nodes),
                "missing_edges": [list(edge) for edge in topology.missing_edges],
                "extra_edges": [list(edge) for edge in topology.extra_edges],
            },
            "compression": {
                "source_chars": compression.source_chars,
                "encoded_chars": compression.encoded_chars,
                "source_words": compression.source_words,
                "encoded_words": compression.encoded_words,
                "source_tokens": compression.source_tokens,
                "encoded_tokens": compression.encoded_tokens,
                "char_reduction_pct": compression.char_reduction_pct,
                "word_reduction_pct": compression.word_reduction_pct,
                "token_reduction_pct": compression.token_reduction_pct,
            },
            "candidate_search": {
                "candidates_generated": candidates.candidates_generated,
                "candidates_evaluated": candidates.candidates_evaluated,
                "parse_failures": candidates.parse_failures,
                "inexact_candidates": candidates.inexact_candidates,
                "exact_candidate_count": len(candidates.exact_candidates),
                "shortest_exact": None
                if candidates.shortest_exact is None
                else {
                    "encoding": candidates.shortest_exact.encoding,
                    "chars": candidates.shortest_exact.chars,
                    "words": candidates.shortest_exact.words,
                    "token_count": candidates.shortest_exact.token_count,
                },
            },
        }


def analyze(memory: GlyphinMemory) -> GlyphinPipelineReport:
    """Measure one memory through every currently implemented research layer."""
    persisted = GlyphinMemory.from_json(memory.to_json())
    persistence = referee_memory(memory, persisted)

    topology, adaptation = memory_to_topology(memory)
    encoded = encode(topology, strategy="chains")
    topology_result = referee_topology(topology, encoded)
    reconstructed = reconstruct(encoded)
    baseline = encode_explicit_edges(topology)
    compression = measure(baseline, encoded)
    candidates = search(topology)

    # Exercise the actual parser output while leaving exactness authority with
    # the independent referee. These are intentionally separate checks.
    if reconstructed.canonical() != topology.canonical():
        raise AssertionError("reconstruction changed topology outside referee report")
    if not topology_result.parse_ok:
        raise AssertionError("pipeline encoder produced an unparsable representation")

    return GlyphinPipelineReport(
        memory_state_count=len(memory.states),
        persistence_referee=persistence,
        adaptation=adaptation,
        encoded=encoded,
        topology_referee=topology_result,
        compression=compression,
        candidate_search=candidates,
    )


__all__ = ["GlyphinPipelineReport", "analyze"]
