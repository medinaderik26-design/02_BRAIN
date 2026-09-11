# GX-012-B Local LLM Setup

## Purpose

This document defines the first controlled procedure for connecting GX-012-B to a real local OpenAI-compatible LLM server.

The provider-free boundary is already tested with an injected transport. The local stage replaces that fake transport with a real server while keeping the experiment contract unchanged.

## Required recording

Before the first comparative run, record:

- local runtime name and exact version;
- model identifier and model revision/file hash where available;
- quantization;
- context-window configuration;
- tokenizer name and version;
- hardware and runtime power/performance mode;
- prompt version;
- controller version/SHA;
- controller `mu`;
- input budget;
- minimum/maximum output budget;
- temperature and other sampling parameters;
- benchmark fixture version;
- epsilon quality tolerance.

## Local endpoint

`gx010_reader.py` uses an OpenAI-compatible `/v1/chat/completions` endpoint and records prompt/completion token counts when the server reports them. The implementation remains provider-neutral. fileciteturn148file0

Set the server base URL and model identifier in the experiment configuration. Do not place credentials in GitHub, source files, fixtures, or committed logs.

## First local run

Run the provider-free boundary tests first:

```bash
python test_gx012_b_runner.py
```

Then point `GX012BAdaptiveReader` at the local server and execute a small smoke fixture before comparative evaluation.

## Comparative conditions

Run the same questions and evidence under:

- B0 fixed allocation;
- B1 adaptive input;
- B2 adaptive output;
- B3 fully adaptive.

Keep model, runtime, prompt, tokenizer, sampling, hardware, corpus, and evaluation order fixed.

## Measurements

At minimum capture actual prompt tokens, completion tokens, total tokens, correctness, latency, controller overhead, selected context, input/output apertures, targets, and failure category.

Token reduction alone is not evidence of compute or energy reduction. Those claims belong to GX-012-D.

## Safety of the experiment

The initial local run should use a non-sensitive fixture. Never commit API keys or private model paths containing sensitive information. Raw outputs should be archived separately from public research artifacts when necessary.

## Boundary

A successful local connection proves only that the controller can alter requests sent to a real local LLM. It does not prove that adaptive allocation improves quality or reduces total computational cost. Those are empirical questions for the comparative runs.
