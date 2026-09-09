"""Minimal executable entry point for the Glyphin research core.

This module is intentionally small: it exposes one runtime object around
``GlyphinMemory`` so experiments do not need to import internal methods or use
the legacy ``glyphin.py`` REPL. It performs no model/API calls.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

from glyphin_research_core import GlyphinMemory


class GlyphinRuntime:
    """Research-facing execution layer backed exclusively by GlyphinMemory."""

    def __init__(self, memory: Optional[GlyphinMemory] = None) -> None:
        self.memory = memory or GlyphinMemory()

    def create(self, name: str, *, level: int = 0, cohesion: float = 0.0) -> dict:
        state = self.memory.add_state(name, level=level, cohesion=cohesion)
        return {"operation": "create", "state": state.name}

    def link(self, parent: str, child: str) -> dict:
        self.memory.link_state(parent, child)
        return {"operation": "link", "parent": parent, "child": child}

    def reinforce(self, name: str, kappa: float = 1.0) -> dict:
        state = self.memory.reinforce(name, kappa=kappa)
        return {
            "operation": "reinforce",
            "state": name,
            "cohesion": state.cohesion,
            "resonance": state.resonance,
            "frequency": state.frequency,
            "sigma": state.sigma,
        }

    def decay(self, delta: float) -> dict:
        self.memory.decay(delta)
        return {"operation": "decay", "delta": delta}

    def recall(self, name: str) -> dict:
        return {"operation": "recall", **self.memory.recall(name)}

    def snapshot(self) -> dict:
        return self.memory.to_dict()

    def save(self, path: str | Path) -> Path:
        return self.memory.save(path)

    @classmethod
    def load(cls, path: str | Path) -> "GlyphinRuntime":
        return cls(GlyphinMemory.load(path))

    def run(self, operations: Iterable[dict]) -> list[dict]:
        """Execute a deterministic operation sequence and return its results."""
        results: list[dict] = []
        for operation in operations:
            name = operation.get("operation")
            if name == "create":
                results.append(self.create(operation["name"], level=operation.get("level", 0),
                                            cohesion=operation.get("cohesion", 0.0)))
            elif name == "link":
                results.append(self.link(operation["parent"], operation["child"]))
            elif name == "reinforce":
                results.append(self.reinforce(operation["name"], operation.get("kappa", 1.0)))
            elif name == "decay":
                results.append(self.decay(operation["delta"]))
            elif name == "recall":
                results.append(self.recall(operation["name"]))
            else:
                raise ValueError(f"unknown Glyphin operation: {name!r}")
        return results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Glyphin research-core runtime")
    parser.add_argument("--load", type=Path, help="load an existing Glyphin memory JSON")
    parser.add_argument("--save", type=Path, help="save memory JSON after operations")
    parser.add_argument("--ops", type=Path, help="JSON file containing an operation list")
    parser.add_argument("--recall", metavar="NAME", help="recall one state after operations")
    parser.add_argument("--pretty", action="store_true", help="pretty-print JSON output")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    runtime = GlyphinRuntime.load(args.load) if args.load else GlyphinRuntime()
    results: list[dict] = []
    if args.ops:
        operations = json.loads(args.ops.read_text(encoding="utf-8"))
        if not isinstance(operations, list):
            raise ValueError("--ops JSON must contain a list")
        results.extend(runtime.run(operations))
    if args.recall:
        results.append(runtime.recall(args.recall))
    if args.save:
        runtime.save(args.save)
    output = {"results": results, "memory": runtime.snapshot()}
    print(json.dumps(output, indent=2 if args.pretty else None, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["GlyphinRuntime", "main"]
