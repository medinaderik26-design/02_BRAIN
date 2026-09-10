# GX-010 — LLM Long-Term Memory Evaluation Specification

## Purpose

GX-010 is the next experimental layer for Glyphin. It tests whether the Glyphin representation improves long-term memory behavior in an actual language model while reducing the amount of contextual material supplied to that model.

This experiment is intentionally designed so that Glyphin can lose. The goal is measurement, not confirmation.

## Research question

Does a Glyphin-based memory layer improve long-term recall and continuity relative to full-history and conventional retrieval memory, while supplying less contextual material to the same language model?

## Core comparison

All conditions should use the same model, same task set, same questions, and equivalent access to the underlying information.

### A — Full History

The model receives the available conversation/history directly, subject only to the defined context budget.

### B — Conventional Retrieval Memory

A conventional retrieval baseline stores history and supplies retrieved passages to the model.

### C — Glyphin

Glyphin converts eligible information into named relational states, maintains lineage/state properties, retrieves the relevant state representation, and supplies the resulting evidence to the same model.

## Required controls

- Same LLM/model version across conditions.
- Same temperature and generation settings.
- Same question order or randomized order with a recorded seed.
- Same underlying memory corpus.
- Same evaluation questions.
- Pinned software versions and repository commit SHA.
- Recorded tokenizer and token-counting method.
- Separate development and evaluation fixtures where practical.
- No benchmark-specific lookup tables or hidden answer injection.

## Primary measurements

1. Factual recall accuracy.
2. Relationship recall accuracy.
3. Temporal ordering accuracy.
4. Multi-hop recall accuracy.
5. Knowledge-update accuracy after contradictory/new information.
6. Contradiction rate.
7. Appropriate abstention when evidence is absent.
8. Cross-session recall.
9. Identity/continuity consistency where the task explicitly tests persistent identity information.
10. Memory drift across repeated sessions.
11. Input tokens supplied to the model.
12. Output tokens generated.
13. Memory-store size.
14. Retrieval latency.
15. End-to-end latency.
16. Number of retrieval operations.
17. Reconstruction failures or state corruption.

## Secondary analysis

Measure performance as history length increases. The central hypothesis is not simply that Glyphin compresses memory; it is that a structured representation may preserve useful relationships while reducing the contextual burden placed on the model.

Results should therefore be reported at multiple memory sizes rather than only as one aggregate score.

## Ablation conditions

If implementation time permits, run the following Glyphin ablations:

- G0 — Full Glyphin.
- G1 — Glyphin without lineage.
- G2 — Glyphin without cohesion.
- G3 — Glyphin without resonance.
- G4 — Glyphin without recurrence/reinforcement.
- G5 — Glyphin without compression.
- G6 — Compression + reconstruction without state-aware retrieval.

The purpose is to determine which mechanisms contribute to observed performance rather than attributing every result to Glyphin as a whole.

## Evaluation layers

### Layer 1 — Deterministic state integrity

Verify canonical serialization, save/load equality, state referee equality, lineage integrity, and reconstruction integrity.

### Layer 2 — Deterministic question utility

Where a question has an exact answer key, evaluate exact correctness without an LLM judge.

### Layer 3 — Model-mediated answers

For questions requiring natural-language reasoning, evaluate with a pinned reader model and record model/version/prompt. If an LLM judge is used, retain raw answers and judge outputs so the result can be independently audited.

### Layer 4 — Long-term memory benchmarks

Where practical, evaluate against established long-term-memory benchmark protocols, including LongMemEval and LongMemEval-V2. LongMemEval-V2 currently provides 451 manually curated questions across five memory abilities and evaluates both answer accuracy and query latency.

## LongMemEval-V2 mapping

Map Glyphin capabilities to:

- Static state recall.
- Dynamic state tracking.
- Workflow knowledge.
- Environment gotchas.
- Premise awareness.

Do not claim that a local GX-010 task suite is equivalent to LongMemEval-V2. A benchmark reproduction must use the benchmark's released data and evaluation protocol.

## Failure preservation

Every failure must be retained. Record:

- question ID
- condition
- expected answer/state
- observed answer/state
- retrieved evidence
- token counts
- latency
- reconstruction status
- software commit
- relevant configuration
- error category

Negative results are part of the research record and must not be removed merely because they weaken the hypothesis.

## Interpretation rules

A successful implementation is not proof of improved LLM memory.

A compression result is not proof of semantic equivalence.

A benchmark win is not proof of universal superiority.

A result on one tokenizer is not a tokenizer-independent claim.

An LLM-judge score is not equivalent to deterministic exact correctness.

Claims must be scoped to the tested model, data, benchmark, tokenizer, configuration, and evaluation protocol.

## Proposed result record

Each run should produce a machine-readable record containing:

```text
experiment_id
run_id
git_sha
model_name
model_version
prompt_version
tokenizer_name
tokenizer_version
condition
ablation
fixture_seed
memory_size
question_count
correct_count
accuracy
input_tokens
output_tokens
memory_store_size
retrieval_latency_ms
end_to_end_latency_ms
reconstruction_exact
state_referee_exact
failure_count
artifact_sha256
```

## Decision gate

GX-010 advances toward publication evidence only if the experiment can be reproduced from the recorded configuration and raw artifacts.

If Glyphin does not outperform a baseline, the result remains valuable: it identifies the boundary conditions under which symbolic state compression does or does not translate into better model behavior.
