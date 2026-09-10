# GX-010 — Experimental Data Schema

## Purpose

This schema defines the machine-readable research record for GX-010. It separates experimental configuration, per-question observations, aggregate results, failures, and artifact integrity so that the Weisone Kernel can be integrated later without changing the scientific measurement framework.

## Design principles

1. **Immutable run identity** — every result identifies the exact code, model, tokenizer, prompt, fixture, and configuration used.
2. **Condition isolation** — Full History, Conventional Retrieval, Glyphin, and future Weisone Kernel integrations are explicit condition IDs.
3. **Raw evidence first** — preserve raw answers/evidence references before aggregation.
4. **Deterministic checks where possible** — exact state and answer checks take precedence over model judging when an answer key exists.
5. **Failure preservation** — failures are records, not exceptions to hide.
6. **No scientific leakage** — the schema does not encode a preferred outcome.

## Run-level record

```text
experiment_id
run_id
run_timestamp_utc
git_sha
repository
branch
model_name
model_version
provider
inference_backend
prompt_version
harness_version
temperature
top_p
max_output_tokens
tokenizer_name
tokenizer_version
dataset_name
dataset_version
dataset_sha256
fixture_seed
memory_generator_version
condition
ablation
memory_size
question_count
config_sha256
artifact_sha256
```

## Per-question record

```text
experiment_id
run_id
question_id
session_id
memory_size
condition
ablation
question_type
expected_answer
observed_answer
answer_exact
answer_score
retrieved_evidence_ids
retrieved_state_ids
retrieval_count
input_tokens
output_tokens
memory_store_bytes
retrieval_latency_ms
end_to_end_latency_ms
reconstruction_exact
state_referee_exact
contradiction_detected
abstention_expected
abstention_correct
identity_continuity_expected
identity_continuity_correct
failure_category
failure_detail
raw_artifact_ref
```

## Aggregate record

```text
experiment_id
run_id
condition
ablation
memory_size
question_count
correct_count
accuracy
relationship_accuracy
temporal_accuracy
multi_hop_accuracy
update_accuracy
contradiction_rate
abstention_accuracy
cross_session_accuracy
identity_continuity_accuracy
memory_drift_rate
mean_input_tokens
median_input_tokens
mean_output_tokens
mean_memory_store_bytes
mean_retrieval_latency_ms
mean_end_to_end_latency_ms
mean_retrieval_count
reconstruction_exact_rate
state_referee_exact_rate
failure_count
artifact_sha256
```

## Failure taxonomy

Use one primary category per failed observation and preserve the detailed evidence.

- `WRITE_LOSS` — required information was not represented in memory.
- `RETRIEVAL_MISS` — relevant stored information existed but was not retrieved.
- `RECONSTRUCTION_FAILURE` — compressed representation could not be reconstructed exactly.
- `STATE_CORRUPTION` — state or lineage integrity changed unexpectedly.
- `REASONING_FAILURE` — evidence was supplied but the model produced an incorrect answer.
- `UPDATE_FAILURE` — newer information did not correctly replace or modify prior knowledge.
- `CONTRADICTION_FAILURE` — conflicting information was handled incorrectly.
- `ABSTENTION_FAILURE` — the system answered when evidence was absent or failed to answer when evidence was sufficient.
- `IDENTITY_DRIFT` — persistent identity information changed incorrectly across sessions.
- `TEMPORAL_FAILURE` — ordering or time-dependent information was incorrect.
- `MULTI_HOP_FAILURE` — required relational traversal failed.
- `INFRASTRUCTURE_FAILURE` — execution, provider, timeout, or harness failure.
- `OTHER` — only when no defined category applies.

## Condition identifiers

### Baseline conditions

- `A_FULL_HISTORY`
- `B_RETRIEVAL_MEMORY`
- `C_GLYPHIN`

### Integration condition

- `D_GLYPHIN_WEISONE_KERNEL`

`D_GLYPHIN_WEISONE_KERNEL` must not be treated as evidence for Glyphin alone. It measures the integrated system and must remain scientifically separable from the Glyphin representation condition.

## Glyphin ablations

- `G0_FULL`
- `G1_NO_LINEAGE`
- `G2_NO_COHESION`
- `G3_NO_RESONANCE`
- `G4_NO_REINFORCEMENT`
- `G5_NO_COMPRESSION`
- `G6_COMPRESSION_NO_STATE_AWARE_RETRIEVAL`

## Evaluation hierarchy

```text
Raw run
  |
  +-- configuration
  +-- per-question observations
  |     +-- deterministic checks
  |     +-- model-mediated answer
  |     +-- failures
  |
  +-- aggregate metrics
  |
  +-- artifact hashes
  |
  v
Claim Registry
```

## Reproducibility requirement

A published GX-010 result should be reproducible from the repository commit, pinned configuration, dataset/version/hash, tokenizer/version, model/version, prompt/harness version, fixture seed, and preserved raw artifacts.

## Scientific boundary

This schema deliberately does not contain a field such as `hypothesis_confirmed`. The experiment records observations; interpretation belongs in the analysis and claim registry.

In particular:

- token reduction does not imply semantic preservation;
- exact state reconstruction does not imply correct LLM reasoning;
- benchmark accuracy does not establish universal superiority;
- Weisone Kernel integration does not independently validate the Glyphin hypothesis.

## Relationship to existing research core

GX-010 consumes the existing Glyphin research implementation and verification layers. It does not replace them. The existing deterministic state integrity tests remain the foundation beneath model-mediated evaluation.
