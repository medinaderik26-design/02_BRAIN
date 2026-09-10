# Glyphin Research Registry

Canonical research registry for the 02_BRAIN repository.

## Source-of-truth rule

`02_BRAIN` is the technical source of truth for executable Glyphin research, simulations, protocols, and reproducible artifacts. ChatGPT conversations, Consensus, and Scite are supporting research layers, not replacements for repository evidence.

## Project identity

- Project: **Glyphin**
- Company: **Weisone Systems**
- Core kernel: **Weisone Kernel**
- Kernel/integration layer for Glyphin: **Weisone Kernel**
- Canonical repository: `medinaderik26-design/02_BRAIN`
- Research implementation: `glyphin_research_core.py`
- Legacy implementation: `glyphin.py`

**Naming rule:** The kernel and company name are authoritative project terminology. Use **Weisone Kernel** and **Weisone Systems** in future research records. Do not revert to the previous Wison spelling.

## Experiment registry

| ID | Experiment | Purpose | Status |
|---|---|---|---|
| GX-001 | Simulation 15 | Semantic roundtrip / exact reconstruction | Complete |
| GX-002 | Simulation 16 | Token economics across memory sizes | Complete |
| GX-003 | Simulation 18 | Structural lineage compression | Complete |
| GX-004 | Simulation 19 | Structural dictionary compression | Complete |
| GX-005 | Simulation 20 | State-columnar representation | Complete |
| GX-006 | Simulation 21 | High-entropy adversarial stress test | Complete |
| GX-007 | Simulation 24 | Scaling behavior through 8192 states | Complete |
| GX-008 | Simulation 26 | Multi-tokenizer comparison | Complete |
| GX-009 | Simulation 27 | Deterministic downstream query utility | Complete |
| GX-010 | LLM Memory Evaluation | Compare full history, conventional memory, and Glyphin with a controlled LLM evaluation | Open |

## Evidence ladder

Representation -> Compression -> Reconstruction -> Functional utility -> Actual LLM memory behavior

The first four layers have benchmark-scoped evidence in the repository. Actual LLM memory behavior remains an open experimental claim.

## Research rules

1. Preserve the spelling **Glyphin**.
2. Preserve **Weisone Kernel** and **Weisone Systems** exactly.
3. Preserve historical experiments, failures, negative results, and terminology.
4. Do not convert benchmark-specific results into universal claims.
5. Token metrics are only claims when an explicit tokenizer was used.
6. Exact reconstruction must be independently checked.
7. Deterministic utility tests should be preferred over LLM self-grading where possible.
8. Weisone Kernel integration results must be labeled separately from Glyphin representation results.

## External research layers

- **Consensus:** literature discovery, paper library, saved searches, and manuscript/document analysis when available.
- **Scite:** citation-context and prior-art verification; Smart Citations and literature screening.
- **ChatGPT:** research continuity, synthesis, and orchestration.

## Current open question

Does the Glyphin representation improve long-term LLM memory performance while reducing the contextual material supplied to the model, relative to full-history and conventional memory baselines?
