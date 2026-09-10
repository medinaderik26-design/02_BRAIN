# Simulation 35 — Confidence/Coverage Policy Boundary

## Purpose
Measure how a downstream Glyphin structural pipeline behaves when an upstream resolver supplies confidence-scored outcomes. Confidence and resolver outcomes are synthetic injections; this benchmark does not evaluate semantic resolution.

## Matrix
- Seeds: 5
- Memory sizes: 256, 1024, 2048
- Targets per memory: first 64 state IDs
- Resolver outcomes: correct, wrong_unique, ambiguous, miss
- Thresholds: 0.00, 0.25, 0.50, 0.75, 0.90, 0.99
- Encodings: compact, structural-lineage, state-columnar
- Total cases: 51,840

## Synthetic confidence policy
- correct: 0.99
- wrong_unique: 0.90
- ambiguous: 0.50
- miss: 0.00

Acceptance requires a resolved outcome whose confidence meets the threshold. Accepted anchors undergo structural parent-closure, induced-memory reconstruction, deterministic encoding/decoding, and exact answer checks.

## Interpretation boundary
This measures threshold policy and resolver-error propagation only. It does not measure calibration of a real semantic resolver, natural-language understanding, learned retrieval, LLM answer equivalence, latency, universal generalization, consciousness, or optimality.
