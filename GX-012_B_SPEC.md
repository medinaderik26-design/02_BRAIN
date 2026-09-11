# GX-012-B — Controlled Local LLM Integration

## Purpose

GX-012-B connects the Dual Cone Variator to a real local LLM through the existing provider-neutral GX-010 reader boundary.

The goal is to test whether adaptive input/output allocation can reduce measured token usage while preserving task performance relative to a fixed-allocation baseline.

This stage does **not** claim compute or energy savings. Those require later measurement.

## Architecture

```text
Glyphin / memory evidence
          |
          v
   Dual Cone Variator
       |        |
       v        v
 input cone   output cone
       |        |
       +--- local LLM ---+
                |
                v
          measured result
                |
                +----> controller feedback
```

## Conditions

1. **B0 — Fixed allocation**: fixed context budget and fixed output budget.
2. **B1 — Adaptive input**: controller changes contextual allocation; output budget remains fixed.
3. **B2 — Adaptive output**: controller changes output budget; input allocation remains fixed.
4. **B3 — Fully adaptive**: controller controls both input and output allocation.

The same local model, model file/version, sampling settings, questions, memory evidence, and evaluation order must be used across conditions.

## Local model boundary

The existing `gx010_reader.py` provides an OpenAI-compatible HTTP interface and can target a local server by supplying its base URL and model name. The reader remains provider-neutral and can use an injected transport for tests.

Examples of compatible local-server classes include Ollama, llama.cpp server, and other OpenAI-compatible runtimes. The experiment must record the exact runtime, model identifier, quantization, context configuration, and commit/version where available.

## Controlled measurements

Record at minimum:

- condition;
- model/runtime identifier;
- input evidence characters and tokens, when available;
- allocated input budget;
- allocated output budget;
- actual input tokens;
- actual output tokens;
- total tokens;
- answer correctness/task score;
- retrieval/memory evidence identifiers;
- controller input/output aperture;
- controller targets;
- end-to-end latency;
- reader/model latency when available;
- controller overhead;
- failure category.

## Primary hypothesis

Under a predefined quality tolerance, B3 will reduce unnecessary contextual and/or generation tokens relative to B0 without unacceptable task-performance degradation.

The hypothesis is falsified if adaptive allocation provides no meaningful reduction after overhead, causes quality loss beyond the predefined tolerance, or fails to reproduce across held-out tasks.

## Quality rule

A resource reduction is not considered successful by itself. A run is only favorable when:

```text
quality_adaptive >= quality_fixed - epsilon
AND
resource_adaptive < resource_fixed
```

where `epsilon` is frozen before evaluation.

## Required controls

Freeze before the first comparative run:

- model and model revision;
- quantization;
- runtime version;
- prompt template;
- tokenizer;
- temperature/top-p and other sampling settings;
- benchmark questions;
- memory corpus;
- epsilon;
- controller weights;
- `mu`;
- minimum/maximum output budget;
- hardware and power mode where possible.

## Ablations

After the primary comparison:

- fixed aperture;
- input-only adaptation;
- output-only adaptation;
- no uncertainty signal;
- no resource-pressure signal;
- no cohesion/lineage signal;
- several `mu` values;
- controller overhead included/excluded.

## Scientific boundary

Token reduction is not automatically compute reduction, latency reduction, or energy reduction. GX-012-B establishes real-LLM behavior and measured token/resource allocation. Direct hardware utilization, power, or energy claims belong to GX-012-D.
