# GX-010 Controlled Runbook

## Purpose

GX-010 tests whether the Glyphin representation improves long-term memory behavior in an actual language model while reducing the contextual material supplied to that model.

The first executable stage is provider-neutral. Conditions A, B, and C must use the same reader, question set, generation settings, and memory fixture.

## Conditions

- `A_FULL_HISTORY` — complete history supplied to the reader.
- `B_RETRIEVAL_MEMORY` — conventional deterministic retrieval baseline.
- `C_GLYPHIN` — existing `GlyphinMemory` research core through `gx010_glyphin_adapter.py`.
- `D_GLYPHIN_WEISONE_KERNEL` — reserved for the separate Weisone Kernel integration experiment.

D must not replace C. This preserves attribution of any measured effect.

## Stage 0: harness validation

Run the deterministic smoke fixture first. Expected result:

- all three conditions execute through the same `gx010_runner.py` path;
- no model provider is contacted;
- exact deterministic answers are measured;
- C additionally reports exact canonical reconstruction of the Glyphin state representation.

A smoke-test success is infrastructure evidence only. It is not evidence that Glyphin improves an LLM.

## Stage 1: local LLM reader

Connect the reader to the same local model for A, B, and C. Pin:

- model name and exact model version/hash when available;
- inference backend;
- generation settings;
- prompt version;
- tokenizer name/version;
- memory fixture version and SHA-256;
- experiment runner Git SHA.

Do not change the model between conditions.

## Primary measurements

For every question record:

- answer correctness;
- information extraction accuracy;
- relationship recall;
- temporal ordering;
- multi-hop recall;
- knowledge-update behavior;
- contradiction handling;
- abstention correctness;
- cross-session recall;
- identity continuity;
- memory drift;
- input tokens;
- output tokens;
- memory-store size;
- retrieval latency;
- end-to-end latency;
- retrieval operations;
- reconstruction exactness;
- state-referee exactness;
- failure category and artifact reference.

## Interpretation rules

Do not infer semantic equivalence from compression alone.

Do not treat exact representation reconstruction as proof of correct LLM reasoning.

Do not claim universal superiority from a benchmark fixture.

Do not claim tokenizer-independent compression unless multiple tokenizers have been measured.

Do not use an LLM judge as a substitute for deterministic ground truth where deterministic evaluation is available.

Do not merge Weisone Kernel integration effects into the Glyphin condition.

## External benchmark path

After the local controlled experiment is stable, map the harness to established long-term-memory benchmarks such as LongMemEval and LongMemEval-V2. LongMemEval evaluates information extraction, multi-session reasoning, temporal reasoning, knowledge updates, and abstention. LongMemEval-V2 uses a context-gathering formulation in which memory systems consume long histories and return compact evidence for downstream answering.

External benchmark results must be reported separately from the local GX-010 fixture results.
