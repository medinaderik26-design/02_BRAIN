"""High-level execution layer for repeatable Glyphin research runs.

This layer is intentionally small: it orchestrates GlyphinMemory operations,
records the operation trace, and exposes a deterministic verification report.
It does not replace the state model or infer semantics from text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Iterable

from glyphin_research_core import GlyphinMemory


@dataclass(frozen=True)
class ExecutionEvent:
    """One explicit operation applied to research memory."""

    operation: str
    arguments: dict[str, Any]


@dataclass
class ExecutionReport:
    """Machine-readable result of an execution and reload check."""

    events: list[ExecutionEvent] = field(default_factory=list)
    state_count: int = 0
    relationship_count: int = 0
    canonical_sha256: str = ""
    reload_exact: bool = False
    lineage_checks: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "events": [
                {"operation": e.operation, "arguments": e.arguments}
                for e in self.events
            ],
            "state_count": self.state_count,
            "relationship_count": self.relationship_count,
            "canonical_sha256": self.canonical_sha256,
            "reload_exact": self.reload_exact,
            "lineage_checks": self.lineage_checks,
        }


class GlyphinExecutionEngine:
    """Deterministic orchestration around :class:`GlyphinMemory`."""

    def __init__(self, memory: GlyphinMemory | None = None) -> None:
        self.memory = memory or GlyphinMemory()
        self.events: list[ExecutionEvent] = []

    def _record(self, operation: str, **arguments: Any) -> None:
        self.events.append(ExecutionEvent(operation, dict(arguments)))

    def create(self, name: str, **kwargs: Any) -> None:
        self.memory.add_state(name, **kwargs)
        self._record("create", name=name, **kwargs)

    def link(self, parent: str, child: str) -> None:
        self.memory.link_state(parent, child)
        self._record("link", parent=parent, child=child)

    def reinforce(self, name: str, kappa: float = 1.0) -> None:
        self.memory.reinforce(name, kappa=kappa)
        self._record("reinforce", name=name, kappa=kappa)

    def decay(self, delta: float) -> None:
        self.memory.decay(delta)
        self._record("decay", delta=delta)

    def recall(self, name: str) -> dict[str, Any]:
        result = self.memory.recall(name)
        self._record("recall", name=name)
        return result

    def execute(self, operations: Iterable[dict[str, Any]]) -> None:
        """Apply a JSON-like operation sequence in order.

        Supported operations are ``create``, ``link``, ``reinforce`` and
        ``decay``. Recall is intentionally query-only and is not required in
        a mutation script.
        """
        for item in operations:
            operation = item.get("operation")
            arguments = dict(item.get("arguments", {}))
            if operation == "create":
                self.create(**arguments)
            elif operation == "link":
                self.link(**arguments)
            elif operation == "reinforce":
                self.reinforce(**arguments)
            elif operation == "decay":
                self.decay(**arguments)
            else:
                raise ValueError(f"unsupported execution operation: {operation!r}")

    def verify_reload(self, lineage_targets: Iterable[str] = ()) -> ExecutionReport:
        encoded = self.memory.to_json()
        digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        restored = GlyphinMemory.from_json(encoded)
        exact = restored.to_json() == encoded
        lineage = {name: restored.get_path(name) for name in lineage_targets}
        relationships = sum(
            1 for state in self.memory.states.values() if state.parent is not None
        )
        return ExecutionReport(
            events=list(self.events),
            state_count=len(self.memory.states),
            relationship_count=relationships,
            canonical_sha256=digest,
            reload_exact=exact,
            lineage_checks=lineage,
        )


__all__ = ["ExecutionEvent", "ExecutionReport", "GlyphinExecutionEngine"]
