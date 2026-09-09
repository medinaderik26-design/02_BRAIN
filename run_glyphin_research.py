"""Run a small, reproducible Glyphin research trajectory.

The harness emits evidence rather than interpretation: operation trace,
state/edge counts, canonical hash, reload fidelity, lineage, an explicit
symbolic candidate, independent referee result, and compression metrics.
"""

from __future__ import annotations

import json
from pathlib import Path

from glyphin_compression import measure_compression
from glyphin_engine import GlyphinExecutionEngine
from glyphin_referee import referee
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

SYMBOLIC = "root->child->grandchild"


def build_evidence() -> dict[str, object]:
    engine = GlyphinExecutionEngine()
    engine.execute(OPERATIONS)
    recalled = engine.recall("grandchild")
    report = engine.verify_reload(["grandchild"])
    topology, adaptation = memory_to_topology(engine.memory)
    referee_result = referee(topology, SYMBOLIC).to_dict()
    compression = measure_compression(
        source=engine.memory.to_json(), encoded=SYMBOLIC
    ).to_dict()

    return {
        "schema": "glyphin-research-run-2",
        "operations": [
            {"operation": event.operation, "arguments": event.arguments}
            for event in report.events
        ],
        "execution": report.to_dict(),
        "recall": recalled,
        "topology": topology.canonical(),
        "encoded_topology": SYMBOLIC,
        "adaptation": {
            "lossless": adaptation.lossless,
            "unsupported_edges": adaptation.unsupported_edges,
            "lost_state_fields": adaptation.lost_state_fields,
        },
        "symbolic_referee": referee_result,
        "compression": compression,
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
        "symbolic_exact": evidence["symbolic_referee"]["exact_match"],
        "compression": evidence["compression"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
