"""Run a small, reproducible Glyphin research trajectory."""
from __future__ import annotations

import json
from pathlib import Path

from glyphin_candidate_search import search
from glyphin_compression import measure
from glyphin_encoder import encode_explicit_edges
from glyphin_engine import GlyphinExecutionEngine
from glyphin_topology_adapter import memory_to_topology


OPERATIONS = [
    {"operation": "create", "arguments": {"name": "root", "cohesion": 0.5}},
    {"operation": "create", "arguments": {"name": "child", "level": 1}},
    {"operation": "create", "arguments": {"name": "grandchild", "level": 2}},
    {"operation": "link", "arguments": {"parent": "root", "child": "child"}},
    {"operation": "link", "arguments": {"parent": "child", "child": "grandchild"}},
    {"operation": "reinforce", "arguments": {"name": "child", "kappa": 1.0}},
    {"operation": "decay", "arguments": {"delta": 2.0}},
]


def build_evidence() -> dict[str, object]:
    engine = GlyphinExecutionEngine()
    engine.execute(OPERATIONS)
    recalled = engine.recall("grandchild")
    report = engine.verify_reload(["grandchild"])
    topology, adaptation = memory_to_topology(engine.memory)

    result = search(topology)
    shortest = result.shortest_exact
    baseline_encoding = encode_explicit_edges(topology)
    compression = measure(source=baseline_encoding, encoded=shortest.encoding) if shortest else None

    return {
        "schema": "glyphin-research-run-5",
        "operations": [
            {"operation": event.operation, "arguments": event.arguments}
            for event in report.events
        ],
        "execution": report.to_dict(),
        "recall": recalled,
        "topology": topology.canonical(),
        "adaptation": {
            "lossless": adaptation.lossless,
            "unsupported_edges": adaptation.unsupported_edges,
            "lost_state_fields": adaptation.lost_state_fields,
        },
        "candidate_search": {
            "candidates_generated": result.candidates_generated,
            "candidates_evaluated": result.candidates_evaluated,
            "exact_candidates": [candidate.encoding for candidate in result.exact_candidates],
            "shortest_exact": shortest.encoding if shortest else None,
            "shortest_chars": shortest.chars if shortest else None,
            "baseline_encoding": baseline_encoding,
            "baseline_chars": len(baseline_encoding),
            "compression_vs_explicit_baseline": compression.__dict__ if compression else None,
            "note": "bounded candidate search; not exhaustive or globally optimal",
        },
    }


def main() -> None:
    output = Path("glyphin_research_run.json")
    evidence = build_evidence()
    output.write_text(json.dumps(evidence, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "state_count": evidence["execution"]["state_count"],
        "relationship_count": evidence["execution"]["relationship_count"],
        "reload_exact": evidence["execution"]["reload_exact"],
        "adaptation_lossless": evidence["adaptation"]["lossless"],
        "candidates_generated": evidence["candidate_search"]["candidates_generated"],
        "exact_candidates": len(evidence["candidate_search"]["exact_candidates"]),
        "shortest_chars": evidence["candidate_search"]["shortest_chars"],
        "token_reduction_pct": None,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
