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
- Total cases: 69,120
- Cases per variant/threshold: 3,840
- Cases per outcome within each variant/threshold: 960

## Synthetic confidence policy
- correct: 0.99
- wrong_unique: 0.90
- ambiguous: 0.50
- miss: 0.00

Acceptance requires a resolved outcome whose confidence meets the threshold. Ambiguous and miss outcomes are therefore rejected at every threshold because they do not have status `resolved`. Accepted anchors undergo structural parent-closure, induced-memory reconstruction, deterministic encoding/decoding, and exact answer checks.

Expected coverage boundary:
- thresholds 0.00, 0.25, 0.50, 0.75, 0.90: 50% (correct + wrong_unique)
- threshold 0.99: 25% (correct only)

## Interpretation boundary
This measures threshold policy and resolver-error propagation only. Confidence values and resolver outcomes are synthetic; this is not calibration of a real semantic resolver. It does not measure natural-language understanding, learned retrieval, LLM answer equivalence, latency, universal generalization, consciousness, or optimality.
