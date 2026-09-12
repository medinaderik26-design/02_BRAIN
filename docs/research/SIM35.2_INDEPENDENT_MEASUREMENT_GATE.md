# Sim35.2 — Independent Measurement Gate

**Status:** PROPOSED — NOT ACCEPTED

**Lineage:** Sim35.0 → Sim35.1 → Sim35.2

**Frozen source baseline:** `b77465778a6e6345c0d24a02087ef09ff9562eff`

## Purpose

Sim35.2 is an independent measurement gate for the Sim35 benchmark family. It is intended to test whether a pre-specified, independently computed measurement can identify a reproducible relationship in the benchmark without defining the result into the metric itself.

This specification does **not** validate the underlying Glyphin hypothesis. A passing benchmark can establish only the stated experimental result under the frozen protocol.

## Preservation rule

The existing Sim35.0 and Sim35.1 artifacts are experimental evidence and must not be rewritten as part of Sim35.2. Corrections to the validator are preserved as lineage; this gate evaluates a new measurement layer.

## Frozen benchmark dimensions

- Seeds: `21092026`, `31092026`, `41092026`, `51092026`, `61092026`
- State sizes: `256`, `512`, `1024`, `2048`
- Confidence levels: `0.5`, `0.6`, `0.7`, `0.8`, `0.9`, `0.95`, `0.99`
- Variants:
  - `sim17-compact`
  - `structural-lineage`
  - `state-columnar`
- Tokenizer: `cl100k_base`

These dimensions are inherited from the frozen Sim35 baseline. They are not to be tuned after inspecting Sim35.2 outcomes.

## Measurement independence requirement

The Sim35.2 measurement must not derive its target score by reusing the Sim35 confidence construction:

`min(confidence_level, (state.resonance + state.frequency) / 200.0)`

In particular, the measurement must not define success by the same threshold it is later claimed to predict.

The producer and validator should share as little critical scoring logic as practical. The validator must recompute the measured quantity from recorded inputs and independently verify the expected invariants.

## Required controls

### A. Label-shuffle control

Randomly permute the outcome/measurement association using deterministic control seeds that are distinct from the experimental seeds. The measured relationship should collapse under the permutation.

### B. Parameter-perturbation control

Apply pre-specified perturbations to relevant state parameters or inputs. The independent measurement must respond according to the frozen directional expectation rather than remaining artificially invariant.

### C. Null/control configuration

Run a configuration for which the proposed relationship is not expected to hold. The measurement must distinguish the experimental condition from the null condition under the same analysis procedure.

### D. Independent-seed replication

Use held-out deterministic seeds for replication. No threshold, transformation, or acceptance criterion may be tuned on those held-out results.

## Acceptance gate

Sim35.2 is accepted only if all of the following are satisfied:

1. The frozen benchmark executes successfully.
2. The independent validator reconstructs the recorded cases without unexplained discrepancies.
3. The measurement is computed independently of the original Sim35 confidence formula.
4. The experimental condition shows the pre-specified effect with a recorded effect-size measure.
5. The label-shuffle control removes the effect to the pre-specified degree.
6. The parameter-perturbation control changes the measurement in the pre-specified direction.
7. The null/control configuration does not reproduce the experimental effect at the acceptance threshold.
8. The result replicates on held-out seeds.
9. Raw outputs, parameters, seeds, environment information, and result hashes are preserved.
10. No acceptance criterion was changed after inspecting the Sim35.2 result.

A single failed gate blocks acceptance and becomes a preserved experimental result requiring diagnosis.

## Rejection and preservation rule

Failure is not to be silently repaired by changing the protocol. The failing run, inputs, outputs, validator result, and commit SHA must be preserved. A subsequent correction receives a new version and new acceptance decision.

## Evidence requirements

The final Sim35.2 evidence package should include:

- exact implementation commit SHA;
- exact validator commit SHA;
- benchmark configuration;
- experimental and control seeds;
- raw case-level measurements;
- aggregate effect-size statistics;
- invariant/validator report;
- environment/runtime information;
- deterministic result hash;
- explicit pass/fail status for every acceptance criterion.

## Scope boundary

Sim35.2 is a reproducibility and measurement-quality gate. It must not be described as proof of a new physical law, proof of the Glyphin hypothesis, or proof that an observed computational relationship generalizes beyond the tested protocol.

## Change control

This specification is the initial frozen proposal on branch `sim35-2-independent-gate`. Implementation changes should occur in subsequent commits and should not modify the frozen baseline experiments.
