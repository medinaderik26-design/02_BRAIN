# GX-012-B Runbook — Controlled Local LLM Integration

## Purpose

GX-012-B is the first stage where the Dual Cone Variator crosses from controller simulation into an actual LLM interface.

The experiment uses the existing provider-neutral GX-010 reader boundary and can target a local OpenAI-compatible server. The provider-free CI test uses an injected transport and therefore does not contact a model server.

## Conditions

- B0: fixed input/output allocation.
- B1: adaptive input allocation only.
- B2: adaptive output allocation only.
- B3: fully adaptive input/output allocation.

## Current implementation

- `GX-012_B_SPEC.md` — experiment specification.
- `gx012_b_runner.py` — adaptive reader boundary.
- `test_gx012_b_runner.py` — provider-free boundary tests.
- `.github/workflows/gx012-b.yml` — CI validation.
- `gx010_reader.py` — existing OpenAI-compatible local reader.
- `gx012_controller.py` — Dual Cone Variator.

## Important boundary

The default context selector currently uses a deterministic character budget because exact model tokenization is runtime/model-specific. It must not be used to make token-efficiency claims.

Before a real comparison, replace it with the exact tokenizer used by the local model and record tokenizer name/version.

## Local execution path

A local OpenAI-compatible server is supplied through `base_url` and `model`. No cloud service is required by the architecture.

The first real run should use a tiny deterministic fixture before moving to the full GX-010 memory benchmark.

## Required measurements

Record actual prompt/input tokens, output tokens, total tokens, correctness, allocation values, latency, model/runtime version, hardware, and controller overhead.

## Scientific boundary

GX-012-B does not establish energy savings. It establishes whether the adaptive controller can influence a real model request and whether quality/resource measurements can be collected under controlled conditions.

Energy and direct compute measurements belong to later GX-012-D work.
