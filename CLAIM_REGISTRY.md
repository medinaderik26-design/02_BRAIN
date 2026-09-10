# Glyphin Claim Registry

Claims are tracked separately from code so experimental evidence is not confused with hypotheses.

| Claim | Statement | Evidence | Status |
|---|---|---|---|
| CLAIM-GX-001 | Glyphin can represent structured memory in a reconstructable representation. | Simulations 15, 18, 20, 21, 24, 26, 27; research core/referees | Supported within tested benchmarks |
| CLAIM-GX-002 | State-columnar representation reduces input-token requirements relative to the tested compact baseline. | Simulations 20, 21, 24, 26 | Supported within tested benchmark families |
| CLAIM-GX-003 | Compressed/reconstructed Glyphin state preserves deterministic downstream query answers. | Simulation 27: 450/450 exact state cases and 450/450 exact answer cases | Supported within tested benchmark |
| CLAIM-GX-004 | Glyphin improves actual long-term LLM memory performance. | GX-010 not yet run | OPEN |
| CLAIM-GX-005 | Glyphin reduces LLM memory drift across sessions. | No controlled experiment yet | OPEN |
| CLAIM-GX-006 | Glyphin is superior to RAG/vector memory on real conversations. | No controlled experiment yet | OPEN |
| CLAIM-GX-007 | Glyphin provides universal or globally optimal compression. | No evidence; explicitly excluded by current benchmark scope | NOT CLAIMED |

## Evidence rule

A benchmark result supports only the scope actually tested. Exact reconstruction of a representation is not equivalent to semantic equivalence of arbitrary LLM behavior.

## Current priority

GX-010 should test CLAIM-GX-004 using controlled comparisons against full-history and conventional memory baselines, with deterministic evaluation where possible.
