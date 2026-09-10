# Glyphin Experiment Index

| Experiment | Main question | Primary evidence |
|---|---|---|
| GX-001 / Sim15 | Can structured states round-trip exactly? | Exact reconstruction across 4 cases |
| GX-002 / Sim16 | What happens to token economics as memory grows? | cl100k_base measurements through 256 states |
| GX-003 / Sim18 | Does structural lineage improve compression? | Exact reconstruction and token measurements |
| GX-004 / Sim19 | Does a structural dictionary add meaningful savings? | Exact reconstruction; limited incremental token gain |
| GX-005 / Sim20 | Does state-columnar representation improve compression? | Exact reconstruction; improved benchmark token reduction |
| GX-006 / Sim21 | Does the result survive high-entropy/adversarial structures? | Exact reconstruction and measured token reduction |
| GX-007 / Sim24 | Does the compression advantage persist with larger memories? | 256–8192 state scaling benchmark |
| GX-008 / Sim26 | Is the result tokenizer-specific? | cl100k_base, o200k_base, o200k_harmony |
| GX-009 / Sim27 | Does compression preserve downstream deterministic utility? | 450/450 exact state cases; 450/450 exact answer cases |
| GX-010 | Does Glyphin improve actual LLM memory behavior? | OPEN |

## Next experiment design

GX-010 should compare, under the same model and task set:

1. Full history / replay baseline
2. Conventional retrieval memory baseline
3. Glyphin memory

Primary measures: factual recall, relationship recall, multi-hop recall, temporal reasoning, contradiction handling, abstention, cross-session continuity, memory drift, supplied input tokens, generated output tokens, latency, retrieval count, and failure rate.

Ablation candidates: remove lineage, cohesion, resonance, recurrence/reinforcement, compression, and state-aware retrieval independently.
