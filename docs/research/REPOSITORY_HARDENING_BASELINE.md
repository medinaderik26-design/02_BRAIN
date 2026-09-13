# Repository Hardening Baseline

**Status:** Proposed — not yet accepted

## Purpose

Establish a repository-wide integrity gate for `02_BRAIN` without replacing or weakening the specialized Glyphin simulation workflows.

## Frozen requirements

The baseline gate must run for every pull request targeting `main` and every push to `main`. It must not use path filters that allow ordinary repository changes to bypass the baseline.

The gate currently checks:

1. Python source compilation across the checkout.
2. Presence of required repository documents: `README.md`, `.gitignore`, and `LICENSE`.
3. Absence of Python bytecode/cache artifacts in the checkout.
4. Static detection of the external `ollama` dependency boundary in `ollama_bridge.py` without requiring a live Ollama service.

## Known portability issue

`ollama_bridge.py` currently resolves `../04_TOOLS/tools_core.py` relative to its own location. A standalone `02_BRAIN` checkout therefore does not necessarily contain the expected sibling monorepo path. This remains an explicit follow-up defect and must not be silently masked by the baseline gate.

## Dependency declaration

The repository should establish one authoritative Python dependency declaration before claiming clean-checkout reproducibility. The absence of `requirements.txt` is not itself a failure if another authoritative mechanism is adopted.

## Scope boundary

This gate is repository integrity infrastructure. It does not validate the Glyphin research hypothesis, Sim35 results, GX results, or any scientific claim. Specialized simulation and research workflows remain independent evidence-producing layers.

## Acceptance criteria

Accept this hardening PR only after:

- baseline CI passes on a clean checkout;
- the Ollama bridge portability issue is either fixed or explicitly documented with a supported monorepo installation path;
- dependency installation is reproducible from a declared source;
- the baseline remains independent of specialized simulation path filters;
- no scientific artifact is modified as part of repository hardening.
