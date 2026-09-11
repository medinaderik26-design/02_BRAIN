# GX-012 — Dual Cone Variator Specification

## Purpose

GX-012 defines the Dual Cone Variator as a measurable adaptive resource-allocation layer between structured memory and language-model execution.

The objective is not compression for its own sake. The objective is to determine whether dynamic allocation can reduce unnecessary contextual and generation resources while preserving task performance within a predefined tolerance.

## Architecture

```text
Glyphin state
     |
     v
relevance / cohesion / lineage / uncertainty
     |
     v
Dual Cone Variator
  |             |
  v             v
Input cone    Output cone
  |             |
  +------> LLM <+
             |
             v
          feedback
             |
             +------> next aperture
```

## Input aperture

Let:

`A_in(t) in [0, 1]`

be the input aperture at time `t`, and let `B_in` be the available input/context budget.

Initial allocation:

`B_raw = (1 - A_in) * B_in`

`B_lineage = A_in * B_in`

The interpretation is experimental: the controller determines how much of the available contextual budget is allocated to raw/current evidence versus structured lineage-aware memory evidence.

The implementation must record the actual token counts for each allocation rather than infer them from the aperture value.

## Output aperture

Let:

`A_out(t) in [0, 1]`

and define an initial output budget as:

`B_out = B_out_min + A_out * (B_out_max - B_out_min)`

The output controller must remain independently measurable from input allocation.

## Continuous update

The initial controller family is:

`A(t+1) = clip(A(t) + mu * (T(t) - A(t)), 0, 1)`

where:

- `A(t)` is the current aperture;
- `T(t)` is the target aperture derived from current experimental signals;
- `mu` is the adaptation rate;
- `clip` constrains the aperture to `[0,1]`.

No LOW/MID/HIGH state buckets are required. Continuous values are the default representation.

## Candidate control signals

The controller may consume:

- Glyphin retrieval relevance;
- cohesion;
- lineage depth;
- number of relevant states;
- unresolved references;
- task complexity;
- answer confidence or evaluator confidence;
- contradiction indicators;
- resource budget;
- measured input tokens;
- measured output tokens;
- retrieval latency;
- end-to-end latency.

Signals must be logged so that later analysis can determine which signals actually contributed to useful allocation decisions.

## Feedback loop

The controller must treat allocation as a closed-loop experiment:

`state -> allocation -> model execution -> measured outcome -> controller update`

A larger aperture is not inherently better. A smaller aperture is not inherently better. The desired behavior is sufficient allocation for the task under the available resource constraint.

## Experimental hypothesis

> Adaptive aperture allocation can reduce contextual and/or generation resource consumption while maintaining task performance within a predefined tolerance of an equivalent fixed-budget baseline.

## Primary comparisons

At minimum compare:

1. Fixed full context / fixed output budget.
2. Fixed compressed context / fixed output budget.
3. Adaptive input aperture / fixed output budget.
4. Fixed input allocation / adaptive output aperture.
5. Fully adaptive input + output apertures.

All conditions must use the same model, task set, prompt version, generation settings, tokenizer, and evaluation procedure.

## Ablations

The following ablations should be available:

- no cohesion signal;
- no lineage signal;
- no uncertainty signal;
- no resource-budget signal;
- fixed aperture;
- input-only adaptation;
- output-only adaptation;
- no feedback update;
- controller overhead included versus excluded from reported resource totals.

## Primary measurements

Record per task:

- task correctness;
- exact-match or task-appropriate quality metric;
- input tokens;
- output tokens;
- total tokens;
- retrieval operations;
- retrieval latency;
- end-to-end latency;
- controller computation/time;
- memory-store size;
- selected Glyphin states;
- aperture values over time;
- allocation target values;
- failure category.

Where available, record hardware/runtime compute or energy measurements separately from token metrics.

## Success criterion

Before running the experiment, define a quality tolerance `epsilon`.

A configuration is resource-efficient only if:

`quality_adaptive >= quality_baseline - epsilon`

and it produces a statistically and practically meaningful reduction in the selected resource metric after controller overhead is included.

The exact epsilon and minimum resource reduction must be frozen in the experiment configuration before evaluating results.

## Falsification conditions

GX-012 does not support the hypothesis if, under controlled evaluation:

- adaptive allocation saves no meaningful resources;
- controller overhead consumes the measured savings;
- quality falls beyond the predefined tolerance;
- benefits disappear across reasonable task distributions;
- fixed allocation consistently dominates adaptive allocation at comparable quality;
- results cannot be reproduced from the recorded configuration and artifacts.

## Scientific boundaries

This specification does not claim that the Dual Cone Variator currently saves energy or compute.

Token reduction is not equivalent to energy reduction.

Context compression is not equivalent to semantic preservation.

A deterministic controller test is not evidence of improved LLM intelligence.

Energy or compute claims require direct measurements or validated runtime proxies.

## Planned implementation stages

### GX-012-A — Controller simulation

Test the continuous aperture controller against synthetic task complexity and budget signals without an LLM.

### GX-012-B — Controlled local LLM

Run fixed versus adaptive conditions against the same local model and task set.

### GX-012-C — Glyphin-coupled controller

Use actual Glyphin state/relevance signals to influence the input aperture.

### GX-012-D — Resource evaluation

Measure token, latency, compute-proxy, and—where available—direct energy effects including controller overhead.

## Relationship to GX-010 and GX-011

GX-010 measures long-term memory behavior in an LLM.

GX-011 establishes the provider-neutral Weisone Kernel + Glyphin integration boundary.

GX-012 tests whether that structured state can drive adaptive resource allocation.

The experiments remain separate so that improvements can be attributed rather than bundled into one claim.
