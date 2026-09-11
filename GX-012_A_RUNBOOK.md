# GX-012-A Runbook — Dual Cone Variator Controller Simulation

## Purpose

GX-012-A is the controller-only stage of GX-012. It tests whether the Dual Cone Variator produces bounded, deterministic, interpretable aperture behavior before any local LLM is introduced.

This stage does **not** measure LLM quality, compute savings, energy savings, or semantic preservation.

## Frozen reference configuration

- Controller: `gx012_controller.py`
- Static simulation: `gx012_simulation.py`
- Dynamic response simulation: `gx012_dynamic_response.py`
- Static tests: `test_gx012_simulation.py`
- Dynamic tests: `test_gx012_dynamic_response.py`
- Adaptation rate: `mu = 0.25`
- Initial input aperture: `0.5`
- Initial output aperture: `0.5`
- Output budget: `32..512`
- Input budget in synthetic profiles: `4096`
- Static duration: `20` controller updates
- Dynamic response duration: `25` controller updates

The target-weight policy currently implemented in `DualConeVariator.target()` is an **initial experimental policy**, not a learned or optimized policy. Its weights must be treated as frozen for a baseline run and changed only as a documented ablation or later experiment.

## Local execution

Run the static deterministic test harness:

```bash
python test_gx012_simulation.py
```

Run the dynamic response tests:

```bash
python test_gx012_dynamic_response.py
```

Run the static profile suite:

```bash
python gx012_simulation.py
```

Run the dynamic response trace:

```bash
python gx012_dynamic_response.py
```

## GX-012-A.1 dynamic response

The dynamic stage replaces repeated steady-state signals with a controlled sequence:

```text
baseline
   |
   v
resource-pressure spike
   |
   v
recovery
   |
   v
uncertainty spike
   |
   v
recovery
```

The sequence contains five steps in each phase. It records input/output apertures, targets, resource pressure, and uncertainty.

Required checks:

- deterministic repeated traces;
- aperture and target bounds remain within `[0,1]`;
- resource pressure lowers input aperture;
- aperture begins recovering after pressure is removed;
- uncertainty increases the output target.

These checks establish controller response behavior only; they do not establish downstream LLM performance or resource savings.

## CI execution

GitHub Actions runs the GX-012-A test harness when the controller, static/dynamic simulations, tests, specification, runbook, or workflow changes. The workflow exports both the steady-state controller history and the GX-012-A.1 dynamic response trace as build artifacts.

The CI artifacts are evidence of reproducible controller execution only. They must not be interpreted as evidence of LLM-level resource savings.

## Current synthetic profiles

1. `low_complexity_low_pressure`
2. `high_memory_relevance`
3. `high_uncertainty`
4. `resource_constrained`

The profiles test directional behavior and convergence under controlled signals. They are not a substitute for a representative task distribution.

## Scientific boundary

A successful GX-012-A.1 run demonstrates that the proposed controller can respond reproducibly to changing synthetic signals. It does not establish that the controller makes better allocation decisions for a real LLM, saves tokens, reduces compute, reduces energy, or lowers cost.

The next stages remain:

```text
GX-012-A.1  -> dynamic controller behavior
GX-012-B    -> controlled local LLM
GX-012-C    -> Glyphin-coupled signals
GX-012-D    -> direct resource/energy evaluation
```

Adaptive resource allocation is an active research direction, including constrained test-time compute allocation and budget-conditioned dynamic inference. Those works establish relevant scientific context, not validation of the Dual Cone Variator or Weisone architecture.
