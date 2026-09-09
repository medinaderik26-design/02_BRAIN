# Glyphin Research Runtime

This repository contains two distinct Glyphin code paths:

- `glyphin_research_core.py` — **research source of truth** for the formal `GlyphState` / `GlyphinMemory` model.
- `glyphin.py` — legacy Fleet OS / REPL code retained for historical compatibility; it is **not** the research engine.

## Research execution path

`glyphin_runtime.py` is the minimal executable entry point around `GlyphinMemory`.
It supports deterministic experiment operations:

`create -> link -> reinforce -> decay -> recall -> save/load`

The runtime makes no model-provider calls and does not claim token savings.

## Representation boundary

`glyphin_topology.py` and `glyphin_topology_adapter.py` provide a separate graph representation. `GlyphinMemory` is single-parent and state-rich; `DirectedTopology` is a general directed graph. The adapter reports unsupported fan-in, self-loops, and cycles rather than silently converting them.

`glyphin_roundtrip.py` and `glyphin_referee.py` are structural verification layers. They can establish graph/topology fidelity; they do **not** establish full GlyphState dynamic fidelity unless the relevant state variables are explicitly tested.

## Evidence rule

A claim is not considered verified merely because an implementation prints a successful result. Reproduce the run, compare the actual artifacts, and preserve mismatches. Token metrics remain unmeasured unless an explicit tokenizer is supplied.

## Tests

The research modules use Python's standard `unittest` runner. From this directory:

```text
python -m unittest discover -p 'test_*.py'
```

No external package is required for the research core/runtime.
