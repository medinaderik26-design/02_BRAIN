# Glyphin Evidence Matrix

| Research question | Evidence | Status | Boundary |
|---|---|---|---|
| Can Glyphin represent named structured states? | `glyphin_research_core.py`, `GlyphState` / `GlyphinMemory` | Demonstrated | Implementation evidence |
| Can lineage be represented and checked? | Parent/child lineage + state/topology referees | Demonstrated | Tested structures |
| Can state survive persistence? | Canonical serialization, save/load, SHA verification | Demonstrated | Deterministic persistence |
| Can structural representations reduce token requirements? | Simulations 16, 18, 19, 20, 21, 24, 26 | Demonstrated in tested benchmarks | Not a universal compression law |
| Can compressed state be exactly reconstructed? | Simulations 15, 18, 20, 21, 24, 26, 27 | Demonstrated in tested benchmarks | Exact referee scope |
| Does compression preserve deterministic downstream queries? | Simulation 27: 450/450 exact state cases and 450/450 exact answer cases | Demonstrated in tested benchmark | Representation-level utility |
| Does efficiency persist as memory grows? | Simulation 24: 256–8192 states | Observed | Benchmark-specific scaling |
| Does tokenizer choice change the result? | Simulation 26: cl100k_base, o200k_base, o200k_harmony | Measured | Tested tokenizers only |
| Does Glyphin improve actual LLM long-term memory? | GX-010 | Open | Requires controlled LLM experiment |
| Does Glyphin reduce LLM memory drift? | GX-010 | Open | Not yet demonstrated |
| Does Glyphin preserve identity across sessions? | GX-010 | Open | Not yet demonstrated |
| Does Glyphin outperform RAG/vector memory? | GX-010 | Open | Must use matched reader/model conditions |

## Current strongest defensible result

In the tested high-entropy benchmark family, the state-columnar representation achieved roughly 53.86% mean input-token reduction while preserving exact deterministic reconstruction and exact downstream query answers across 450 tested cases.

This is representation-level evidence. It is not yet evidence of semantic equivalence for arbitrary LLM workloads, universal generalization, or global optimality.

## Scientific next step

GX-010 should test the representation with a fixed local LLM and controlled baselines: full history, conventional retrieval memory, and Glyphin. Where possible, deterministic answer keys and exact state checks should be used instead of relying exclusively on an LLM judge.
