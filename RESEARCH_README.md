# Glyphin Research Runtime

This repository contains two distinct Glyphin code paths:

- `glyphin_research_core.py` — **research source of truth** for the formal `GlyphState` / `GlyphinMemory` model.
- `glyphin.py` — legacy Fleet OS / REPL code retained for historical compatibility; it is **not** the research engine.

## Research execution path

`glyphin_runtime.py` is the minimal executable entry point around `GlyphinMemory`.
`glyphin_engine.py` adds operation tracing and reload verification for reproducible runs.
The supported research sequence is:

`create -> link -> reinforce -> decay -> recall -> save/load`

The runtime makes no model-provider calls and does not claim token savings.

## Representation boundary

`glyphin_topology.py` and `glyphin_topology_adapter.py` provide a separate graph representation. `GlyphinMemory` is single-parent and state-rich; `DirectedTopology` is a general directed graph. The adapter reports unsupported fan-in, self-loops, and cycles rather than silently converting them.

`glyphin_roundtrip.py` and `glyphin_referee.py` are structural verification layers. They can establish graph/topology fidelity. `glyphin_state_referee.py` separately checks GlyphState fields and lineage, so a topology match is not incorrectly treated as full state fidelity.

## Encoding and reconstruction

`glyphin_encoder.py` provides transparent deterministic encoders. Exact topology encoding includes isolated nodes as bare identifiers; this is required for true node-set fidelity.

`glyphin_candidate_search.py` evaluates a finite, topology-derived candidate set using the independent reconstruction/referee layer. It is explicitly **bounded**, not exhaustive, global, or optimal, and now records parse-failure and inexact-candidate counts.

`glyphin_benchmark.py` generates deterministic random DAGs, cyclic graphs, and adversarial structures for generalization testing. It records exact-referee outcomes and character/word compression. Token counts remain `NOT_MEASURED` unless a tokenizer is supplied.

## Evidence rule

A claim is not considered verified merely because an implementation prints a successful result. Reproduce the run, compare the actual artifacts, and preserve mismatches. Token metrics remain unmeasured unless an explicit tokenizer is supplied.

## Tests

`run_glyphin_tests.py` is the canonical dependency-free test runner. It collects both `unittest.TestCase` classes and zero-argument `test_*` functions so the repository has one test command:

```text
python run_glyphin_tests.py
```

The GitHub Actions workflow runs this suite and the research smoke test on pushes and pull requests to `main`.
