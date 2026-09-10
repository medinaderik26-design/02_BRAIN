# GX-011 — Weisone Kernel / Glyphin Integration Audit

## Purpose

This document establishes the experimental boundary between the **Weisone Kernel** and **Glyphin** without modifying either project's historical implementation.

## Canonical identity

- Company: **Weisone Systems**
- Core kernel: **Weisone Kernel**
- Memory architecture/research system: **Glyphin**
- Research repository: `02_BRAIN`
- Kernel repository: `weisone-kernel`

## Current repository findings

The `weisone-kernel` repository contains a Python runtime defining `WeisoneKernel` v0.4.0_Directors_Cut. The runtime currently initializes `marrow_vault` and exposes `nexus_forge`, `chorus_weave`, and `fracture_stitch_v2`.

The current kernel repository does **not** expose a model-provider interface, LLM request loop, Glyphin adapter, or GX-010 measurement interface. Therefore the kernel should be treated as an operational prototype/runtime boundary, not yet as a completed LLM-memory experiment harness.

`02_BRAIN` contains the independently measured Glyphin research implementation and GX-010 harness. The two repositories are presently separate.

## Integration hypothesis

The experimental question is not whether the kernel and Glyphin are conceptually similar. It is whether a controlled integration of the two changes measurable LLM memory behavior.

### Conditions

- A — `A_FULL_HISTORY`
- B — `B_RETRIEVAL_MEMORY`
- C — `C_GLYPHIN`
- D — `D_GLYPHIN_WEISONE_KERNEL`

A/B/C establish the Glyphin effect. D measures the incremental effect of the Weisone Kernel integration.

## Required boundary

The integration should use an adapter rather than rewriting `weisone_runtime.py` or the Glyphin research core.

```text
Weisone Systems
      |
      v
Weisone Kernel
      |
      v
GX-011 Adapter
      |
      +------> GlyphinMemory
      |
      v
Memory Evidence
      |
      v
Same Local LLM
```

## Measurements

The D condition must use the same model, model version, prompt, generation settings, question set, tokenizer, dataset, seeds, and evaluation procedure as A/B/C.

Record:

- answer accuracy
- relationship accuracy
- temporal accuracy
- multi-hop accuracy
- update accuracy
- contradiction rate
- abstention accuracy
- cross-session continuity
- identity continuity
- memory drift
- input/output tokens
- memory-store size
- retrieval operations
- retrieval latency
- end-to-end latency
- reconstruction exactness
- state-referee exactness
- failure category/detail

## Existing kernel token claim

The kernel repository contains a case study reporting a reduction from 1,200,000 to 155,000 tokens and describing this as 87.08% reduction with 100% semantic accuracy. This is retained as a **historical/project claim**, not as independently verified GX evidence.

The claim must be reproduced under a pinned tokenizer, explicit workload, baseline, model conditions, and deterministic evaluation before being merged with Glyphin's measured benchmark results.

## Scientific boundary

- Token reduction is not semantic preservation by itself.
- Exact Glyphin reconstruction is not proof of LLM reasoning equivalence.
- Kernel integration is not independent validation of Glyphin.
- Glyphin benchmark results are not independent validation of the whole Weisone Kernel.
- A case-study claim is not equivalent to a controlled benchmark result.

## Next implementation step

Create `gx011_weisone_adapter.py` in `02_BRAIN` as a thin adapter around the existing kernel runtime and Glyphin memory. The adapter should expose only the operations required by GX-010 and should preserve the existing kernel implementation unchanged.

The first integration test should be deterministic and provider-free. A real local LLM should be attached only after the adapter passes structural and reconstruction checks.
