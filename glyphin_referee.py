"""Independent referee for Glyphin topology experiments."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from glyphin_reconstruction import ReconstructionError, reconstruct
from glyphin_topology import DirectedTopology


@dataclass(frozen=True)
class RefereeResult:
    exact_match: bool
    parse_ok: bool
    source_nodes: int
    candidate_nodes: int
    source_edges: int
    candidate_edges: int
    missing_nodes: list[str]
    extra_nodes: list[str]
    missing_edges: list[tuple[str, str]]
    extra_edges: list[tuple[str, str]]
    source_fingerprint: str
    candidate_fingerprint: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "exact_match": self.exact_match,
            "parse_ok": self.parse_ok,
            "source_nodes": self.source_nodes,
            "candidate_nodes": self.candidate_nodes,
            "source_edges": self.source_edges,
            "candidate_edges": self.candidate_edges,
            "missing_nodes": self.missing_nodes,
            "extra_nodes": self.extra_nodes,
            "missing_edges": [list(e) for e in self.missing_edges],
            "extra_edges": [list(e) for e in self.extra_edges],
            "source_fingerprint": self.source_fingerprint,
            "candidate_fingerprint": self.candidate_fingerprint,
            "error": self.error,
        }


def _fingerprint(graph: DirectedTopology) -> str:
    payload = graph.to_canonical_dict()
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def referee(source: DirectedTopology, encoded: str) -> RefereeResult:
    try:
        candidate = reconstruct(encoded)
    except (ReconstructionError, ValueError) as exc:
        return RefereeResult(
            exact_match=False,
            parse_ok=False,
            source_nodes=len(source.nodes), candidate_nodes=0,
            source_edges=len(source.edges), candidate_edges=0,
            missing_nodes=sorted(source.nodes), extra_nodes=[],
            missing_edges=sorted(source.edges), extra_edges=[],
            source_fingerprint=_fingerprint(source), candidate_fingerprint="",
            error=str(exc),
        )

    diff = source.compare(candidate)
    return RefereeResult(
        exact_match=bool(diff["exact_match"]),
        parse_ok=True,
        source_nodes=len(source.nodes), candidate_nodes=len(candidate.nodes),
        source_edges=len(source.edges), candidate_edges=len(candidate.edges),
        missing_nodes=diff["missing_nodes"], extra_nodes=diff["extra_nodes"],
        missing_edges=diff["missing_edges"], extra_edges=diff["extra_edges"],
        source_fingerprint=_fingerprint(source),
        candidate_fingerprint=_fingerprint(candidate),
    )


__all__ = ["RefereeResult", "referee"]
