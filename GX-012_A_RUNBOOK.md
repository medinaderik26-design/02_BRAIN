# GX-012-A Runbook — Dual Cone Variator Controller Simulation

## Purpose

GX-012-A is the controller-only stage of GX-012. It tests whether the Dual Cone Variator produces bounded, deterministic, interpretable aperture behavior before any local LLM is introduced.

This stage does **not** measure LLM quality, compute savings, energy savings, or semantic preservation.

## Frozen reference configuration

- Controller: `gx012_controller.py`
- Simulation: `gx012_simulation.py`
- Tests: `test_gx012_simulation.py`
- Adaptation rate: `mu = 0.25`
- Initial input aperture: `0.5`
- Initial output aperture: `0.5`
- Output budget: `32..512`
- Input budget in synthetic profiles: `4096`
- Default duration: `20` controller updates

The target-weight policy currently implemented in `DualConeVariator.target()` is an **initial experimental policy**, not a learned or optimized policy. Its weights must be treated as frozen for a baseline run and changed only as a documented ablation or later experiment.

## Local execution

Run the deterministic test harness:

```bash
python test_gx012_simulation.py
```

Run the profile suite:

```bash
python gx012_simulation.py
```

## CI execution

GitHub Actions runs the GX-012-A test harness when the controller, simulation, test, specification, or workflow changes. The workflow also exports the complete controller history for the four default profiles as a build artifact.

The CI artifact is evidence of reproducible controller execution only. It must not be interpreted as evidence of LLM-level resource savings.

## Current synthetic profiles

1. `low_complexity_low_pressure`
2. `high_memory_relevance`
3. `high_uncertainty`
4. `resource_constrained`

The profiles test directional behavior and convergence under controlled signals. They are not a substitute for a representative task distribution.

## Required checks

The baseline GX-012-A run should establish:

- aperture bounds remain within `[0,1]`;
- output budgets remain within configured bounds;
- raw and lineage fractions remain valid;
- resource pressure moves both apertures downward relative to the high-memory-relevance profile;
- repeated runs are deterministic;
- the controller moves toward its calculated target rather than remaining at its initial state.

## Next experimental extension

The next useful controller-only experiment is a **dynamic signal-response suite** rather than more repeated steady-state profiles. It should include:

- sudden resource-pressure increase/decrease;
- sudden relevance increase/decrease;
- alternating high/low pressure;
- uncertainty spikes;
- step-response settling time;
- overshoot/oscillation measurement;
- comparison across several `mu` values;
- controller overhead accounting.

Only after that should GX-012-B connect the controller to a controlled local LLM.

## Scientific boundary

A successful GX-012-A run demonstrates that the proposed controller can execute reproducibly under synthetic signals. It does not establish that the controller makes better allocation decisions for a real LLM. That question belongs to GX-012-B and later stages.
