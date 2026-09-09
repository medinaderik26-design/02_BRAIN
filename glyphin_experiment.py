"""Reproducible experiment-run wrapper for the Glyphin research runtime.

A GlyphinMemory snapshot describes state. This module additionally records the
ordered operations used to reach that state, making a run auditable without
requiring a model provider or external database.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from glyphin_runtime import GlyphinRuntime


@dataclass(frozen=True)
class ExperimentRun:
    """Canonical, serializable record of one Glyphin runtime execution."""

    version: str
    operations: tuple[dict[str, Any], ...]
    results: tuple[dict[str, Any], ...]
    final_memory: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "operations": list(self.operations),
            "results": list(self.results),
            "final_memory": self.final_memory,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def fingerprint(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json() + "\n", encoding="utf-8")
        return target


class GlyphinExperiment:
    """Execute and capture a complete Glyphin experiment run."""

    VERSION = "glyphin-experiment-0.1"

    def __init__(self, runtime: GlyphinRuntime | None = None) -> None:
        self.runtime = runtime or GlyphinRuntime()

    def run(self, operations: Iterable[dict[str, Any]]) -> ExperimentRun:
        frozen_ops = tuple(dict(op) for op in operations)
        results = tuple(self.runtime.run(frozen_ops))
        return ExperimentRun(
            version=self.VERSION,
            operations=frozen_ops,
            results=results,
            final_memory=self.runtime.snapshot(),
        )

    @classmethod
    def load_memory(cls, path: str | Path) -> "GlyphinExperiment":
        return cls(GlyphinRuntime.load(path))


__all__ = ["ExperimentRun", "GlyphinExperiment"]
