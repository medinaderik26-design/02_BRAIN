# Glyphin Research Continuation Manifest

## Purpose
This file is the handoff point for continuing the Glyphin research program in a new chat if the current conversation becomes unavailable or limited.

## Repository
- GitHub: https://github.com/medinaderik26-design/02_BRAIN
- Active research branch: `sim31-ci-verify`
- Keep the work evidence-first: never convert deterministic representation results into claims about consciousness, semantic equivalence, universal generalization, latency, optimality, or learned retrieval unless separately measured.

## Core thesis
> Don’t move the entire history. Move the structure that makes the history meaningful.

Research chain:
`interaction → recurrence → association → topology → compact state → transfer → reconstruction`

Structural continuity means relational/contextual topology, not model weights and not consciousness.

## Accepted simulation lineage
- Sim12: 2010/2010 exact; mean character reduction 6.6915%.
- Sim16: actual `cl100k_base`; 7/7 exact; mean token reduction 40.6377%.
- Sim18: structural-lineage; 14/14 exact; 256-state reduction 55.1632%.
- Sim20: state-columnar; 21/21 exact; mean reduction 58.9652%.
- Sim24: scaling 256..8192; 18/18 exact; columnar 8192-state reduction 53.5932%.
- Sim25: cross-seed; 120/120 exact; columnar mean reduction 53.9134%.
- Sim26: seed×tokenizer; 135/135 exact; columnar mean 53.8647% on cl100k and 54.8513% on o200k.
- Sim27: 450/450 exact state and query answers; deterministic downstream query utility; columnar mean reduction 53.8593%.
- Sim28.4: deterministic query-conditioned retrieval; 450 cases/variant; 100% closure recall; selected-state reduction about 99.0%; explicit anchors and dependency closure supplied/deterministic.
- Sim29: structural retrieval index discovers closure; 150/150 exact per variant; selected-state reduction about 99.1%; index amortization measured. Archive should be rechecked if needed.
- Sim30.2: lexical resolution from query text + structural index; 150/150 state/anchor/index/answer exact per variant; selected reduction ~99.18% compact, ~99.12% structural/columnar. Archive commit `d94bcb662725ab2fc330f165a2a256cde7b54d7f`.
- Sim31.2: opaque alias resolution + collision rejection; 450 primary + 405 collision cases; 100% state/index/answer exact; ambiguous aliases rejected; primary selected reduction ~99.20% compact, ~99.13% structural/columnar. Final CI run `34421251485`; artifact digest `sha256:a20f6179bbc5f87d2b5703f03833bc4cdc10ac0db39836c4fc951fbaa641010e`.
- Sim32: controlled descriptor resolution; 1350 cases; 100% state/anchor/index/answer exact; unique resolution 405/450 per variant because anchor-free cases are intentionally retained. Mean selected reduction ~89.14–89.20% because unsupported/anchor-free cases remain in aggregate.
- Sim33.2: surface-form boundary. 4800 cases. canonical/punctuation/reordered/synonymized = 960/960 exact each; free paraphrase = 0/960 and marked external-required. Final run `34476867976`; artifact digest `sha256:62e40e7e83388fe6e3d07a8731cf8f3082e3428c01020d3dddc67e27fbe98504`; result hash `a782358ac67ff850d3de3c9f9745f701f71163d392f22215b3786d121356ec60`.
- Sim34.3: resolver error propagation. 11,520 cases; exact/miss/ambiguous/wrong_unique outcomes. Exact: 960 accepted and fully exact per variant. Miss and ambiguous: rejected. Wrong_unique: accepted but closure exactness 0; transport exact 960; transport-answer exact 8/960 due structural coincidences. Exact-path mean selected reductions: compact 99.2334%, structural 99.4213%, columnar 99.4350%. Final run `34484029755`; artifact digest `sha256:708dff17fad84a6b0e25a2b5eacc68a420487cd9a3a80c13215003d7ccc26854`; result hash `794f67dc94c1e91e823a394db8f4d9500504f6b2c62e664c00cd52fa6b369bcc`.

## Sim35 current state
Goal: confidence/coverage policy boundary. Resolver outcomes and confidence scores are synthetic/injected upstream; the benchmark is **not** semantic resolver evaluation.

Current source:
- `glyphin_simulation35.py`
- Current version: `35.1`
- Confidence outcomes:
  - correct = 0.99
  - wrong_unique = 0.90
  - ambiguous = 0.50, but status is `ambiguous` and therefore never accepted
  - miss = 0.00
- Thresholds: `0.0, 0.25, 0.5, 0.75, 0.9, 0.99`
- Matrix: `5 seeds × 3 sizes × 64 targets × 4 outcomes × 6 thresholds × 3 variants = 69,120 cases`.
- Therefore each variant/threshold slice has `3,840 cases`.
- Each outcome contributes `960 cases` per variant/threshold.

### Critical corrections still required before accepting Sim35
1. Update `glyphin_simulation35.py` from version `35.1` to `35.2`.
2. Fix summary denominators from `240` to `960` for correct/wrong/ambiguous acceptance percentages.
3. Fetch and update `.github/workflows/glyphin-simulation35.yml`.
4. Workflow assertions must use:
   - version `35.2`
   - total cases `69,120`
   - cases per variant/threshold `3,840`
   - threshold 0.0 coverage `50.0%`, NOT 75%, because ambiguous is rejected by status.
   - threshold 0.9 coverage `50.0%`
   - threshold 0.99 coverage `25.0%`
   - threshold 0.99 correct acceptance `100.0%`
   - threshold 0.99 wrong false acceptance `0.0%`
   - threshold 0.99 ambiguous false acceptance `0.0%`
5. Trigger fresh CI.
6. If CI fails, inspect logs and correct the actual defect; do not accept numbers from a failing validator.
7. If CI succeeds, download the artifact and independently verify:
   - artifact SHA-256 against GitHub digest
   - `sha256(json.dumps(cases, sort_keys=True).encode())` against embedded `result_data_sha256`
   - exact case count and threshold summaries.
8. Only then call Sim35 accepted evidence.

### Expected Sim35 policy behavior
Because only `status == resolved` is eligible:
- threshold 0.00 → correct + wrong_unique = 50% coverage
- threshold 0.25 → 50%
- threshold 0.50 → 50%
- threshold 0.75 → 50%
- threshold 0.90 → 50% (wrong confidence is exactly 0.90 and acceptance uses >=)
- threshold 0.99 → 25% (correct only)

This benchmark therefore tests the boundary between accepting a resolved low-confidence candidate and requiring the high-confidence correct candidate; it does not calibrate a real semantic resolver.

## Important implementation history
- Sim20 decoder was corrected to preserve zero UTC offsets and zero deltas, and to preserve an empty parent field for a singleton root so vector lengths remain aligned.
- Sim31 source still contains the original alias-index orientation defect; CI used `glyphin_simulation31_runner.py` as an explicit corrected runner. Do not silently describe the source as fixed unless it is actually fixed.
- Sim33 uses an external runner because the scaffold intentionally remains a boundary artifact.
- Maintain this transparency: corrected runners are acceptable only when explicitly documented and independently verified.

## Standard evidence protocol
For every accepted simulation:
1. Source commit SHA and relevant file content.
2. Workflow run ID and job ID.
3. All CI validation steps successful.
4. Artifact ID and GitHub artifact digest.
5. Independent artifact download and SHA-256 verification.
6. Independent recomputation of embedded result hash.
7. Inspect actual result JSON, not just CI's exit status.
8. State exact scope and exclusions.

## What not to claim
Do not claim:
- global minimum or optimal compression
- universal/general human memory transfer
- semantic equivalence to an LLM
- natural-language understanding from deterministic lexical/controlled resolvers
- learned retrieval recall
- end-to-end latency improvement
- consciousness or identity transfer
- that topology alone preserves all dynamic memory semantics

## Likely next research sequence after Sim35
- Sim36: resolver calibration model / score distributions, still synthetic unless a real resolver is introduced.
- Sim37: noisy resolver with controlled false-positive/false-negative rates and cost-sensitive acceptance.
- Sim38: index-update and mutation consistency under additions/deletions/reparenting.
- Sim39: concurrent query batches and index amortization under changing memory state.
- Sim40: independent implementation/specification conformance test to reduce dependence on the same code path for encoder/decoder validation.

## Fast continuation prompt
Paste this into a new chat:

> Continue the Glyphin research from `medinaderik26-design/02_BRAIN`, branch `sim31-ci-verify`, using `CONTINUATION_MANIFEST.md` as the source of truth. Do not ask me to restate prior work. First inspect the current Sim35 source and workflow. Fix the known Sim35 issues (matrix is 69,120 total / 3,840 per variant-threshold; source acceptance-rate denominators must be 960; threshold 0.0 coverage is 50%, not 75%; bump implementation to 35.2), run CI, inspect failures, and independently verify the final artifact and canonical result hash before accepting evidence. Then continue to the next scientifically useful simulation. Preserve strict evidence discipline and do not make unsupported claims.

## If the conversation is limited
The repository is the durable handoff. Commit all meaningful source, protocol, workflow, result, and status changes to `sim31-ci-verify`. Keep `CONTINUATION_MANIFEST.md` current whenever a simulation reaches a new accepted milestone or a major bug is discovered.
