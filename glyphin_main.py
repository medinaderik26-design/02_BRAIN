"""Executable entry point for the Glyphin research memory core."""

from __future__ import annotations

import argparse
from pathlib import Path

from glyphin_research_core import GlyphinMemory


def build_demo_memory() -> GlyphinMemory:
    memory = GlyphinMemory()
    memory.add_state("root", level=0, cohesion=0.5)
    memory.add_state("research", level=1, parent="root")
    memory.add_state("glyphin", level=2, parent="research")
    memory.reinforce("research")
    memory.decay(1.0)
    return memory


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Glyphin research memory core.")
    parser.add_argument("--memory", type=Path, default=Path("glyphin_memory.json"))
    parser.add_argument("--target", default="glyphin")
    parser.add_argument("--demo", action="store_true", help="create a deterministic demo memory")
    args = parser.parse_args()

    if args.demo or not args.memory.exists():
        memory = build_demo_memory()
        memory.save(args.memory)
        print(f"created: {args.memory}")
    else:
        memory = GlyphinMemory.load(args.memory)
        print(f"loaded: {args.memory}")

    if args.target not in memory.states:
        parser.error(f"unknown target: {args.target}")

    state = memory.states[args.target]
    recall = memory.recall(args.target)
    print(f"target={args.target}")
    print(f"cohesion={state.cohesion:.6f}")
    print(f"resonance={state.resonance:.6f}")
    print(f"frequency={state.frequency}")
    print(f"sigma={state.sigma}")
    print("lineage=" + " -> ".join(recall["lineage"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
