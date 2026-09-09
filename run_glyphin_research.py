"""Run a small, reproducible Glyphin research trajectory.

The harness intentionally emits evidence rather than interpretation: operation
trace, state/edge counts, canonical hash, reload fidelity, lineage, and an
independent topology referee result.
"""

from __future__ import annotations

import json
from pathlib import Path

from glyphin_engine import GlyphinExecutionEngine
from glyphin_topology_adapter import memory_to_topology
from glyphin_referee import referee


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

    # Referee requires a symbolic encoding, so construct one directly from
    # the topology for this smoke run. This validates the referee pipeline,
    # not compression optimality.
    encoded = ";".join(
        f"{parent}->{child}" for parent, child in sorted(topology.edges)
    )
    referee_result = referee(topology, encoded)

    return {
        "schema": "glyphin-research-run-1",
        "operations": [
            {"operation": event.operation, "arguments": event.arguments}
            for event in report.events
        ],
        "execution": report.to_dict(),
        "recall": recalled,
        "topology": {
            "nodes": sorted(topology.nodes),
            "edges": sorted(topology.edges),
        },
        "encoded_topology": encoded,
        "adaptation": {
            "lossless": adaptation.lossless,
            "unsupported_edges": adaptation.unsupported_edges,
            "lost_state_fields": adaptation.lost_state_fields,
        },
        "referee": referee_result.to_dict(),
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
        "referee_exact": evidence["referee"]["exact_match"],
        "referee_parse_ok": evidence["referee"]["parse_ok"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
