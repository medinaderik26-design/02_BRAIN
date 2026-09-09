# Context Gateway Policy v0.1

## Purpose

The gateway sits between an application/user and a downstream model API. Its
job is to reduce repeated context transmission while preserving an auditable
record of what was selected and omitted.

## Invariants

1. The gateway does not claim semantic equivalence unless a referee verifies it.
2. Token savings are `NOT_MEASURED` unless an explicit tokenizer is supplied.
3. Full source material remains outside the model request when the application
   chooses to store it locally.
4. Every packet has source and packet hashes for auditability.
5. Selection is deterministic for identical inputs/configuration except for the
   timestamp field.
6. Exclusion is not deletion: omitted context remains in the local source set.
7. Provider privacy is not guaranteed by this module; users must evaluate the
   network, API, logging, retention, and telemetry policies of their provider.

## Intended pipeline

`user/app -> local context store -> gateway -> compact context packet -> model API`

The gateway is an optimization and evidence layer. It is not a security proxy,
privacy guarantee, or replacement for provider controls.

## Future research

- lineage-aware selection using Glyphin topology
- exact reconstruction/referee before semantic compression claims
- cache/reuse of stable context packets
- provider-neutral adapters
- measured cost/latency/token benchmarks
- configurable redaction and data-minimization policy
- local-only execution mode
