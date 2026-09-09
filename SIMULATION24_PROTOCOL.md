# Simulation 24 — Scaling Law / Break-Even

## Question
Does the token-efficiency advantage of structural state factorization persist, increase, flatten, or reverse as Glyphin memory size grows?

## Fixture
Simulation 21 high-entropy memory generator, extended beyond the previous 256-state ceiling.

## Sizes
256, 512, 1024, 2048, 4096, 8192 states.

## Variants
- `sim17-compact`
- `structural-lineage`
- `state-columnar`

## Controls
- `cl100k_base` tokenizer, version 0.11.0
- same semantic state model and independent `glyphin_state_referee.py`
- exact reconstruction required
- deterministic seed derived from Sim21 seed + state count
- no fixture-specific lookup tables or fingerprints in the encoders

## Measurements
For every case:
- baseline and encoded character counts
- baseline and encoded token counts
- character reduction percentage
- token reduction percentage
- encode/decode wall time
- exact state-referee result and mismatch details

## Scaling analysis
Token-reduction slope is measured against `log2(state_count)` for each variant. State-columnar advantage over Sim17 compact is reported in percentage points at each tested size. The first tested size at which the advantage is positive is reported as the observed break-even size.

## Interpretation rule
Results describe only the tested high-entropy benchmark family. They do not establish a universal scaling law, global optimum, or guaranteed 80% compression target.
