# Dual Cone Variator — Resource Conservation

## Status

**Preserve as a planned core mechanism. Do not remove from the architecture.**

The Dual Cone Variator is intended to regulate runtime context and token allocation so the system can conserve computational resources rather than treating available context or output budget as something that should always be maximized.

## Architectural role

The current conceptual pipeline is:

`Glyphin -> Dual Cone Variator -> Weisone Kernel -> LLM -> feedback -> Dual Cone Variator`

- **Glyphin** represents memory as named, relational states with lineage and measurable state properties.
- **Dual Cone Variator** determines how much contextual information and generation budget should be active for the current task.
- **Weisone Kernel** coordinates the memory/runtime integration layer.
- **LLM** performs inference and generation.
- **Feedback** allows the controller to adjust allocation rather than using a fixed compression ratio.

## Resource-conservation principle

The objective is not simply token compression. The intended objective is:

> **Conserve computational resources by dynamically using only the context and output capacity required to preserve task performance.**

This includes eventual measurement of:

- input/context tokens
- retrieved memory tokens
- lineage-derived context
- output tokens
- retrieval operations
- latency
- memory-store size
- task accuracy and memory behavior
- energy/compute proxies where reliable measurements are available

## Dual allocation concept

The mechanism should eventually support coupled runtime control of at least:

1. **Input/context allocation** — how much raw context versus lineage-derived/contextual memory is allowed through.
2. **Output allocation** — how much generation budget is appropriate for the task.

The existing thesis seed defines an aperture `A in [0,1]` for continuous allocation between raw context and lineage-derived context, with a continuous update of the form:

`A(t+1) = A(t) + mu(T - A(t))`

This is a starting hypothesis, not a validated final controller.

## Scientific boundary

The system must not assume that fewer tokens automatically means less energy or equal reasoning quality. The controller must be evaluated experimentally against fixed-budget and non-adaptive baselines.

The key future question is:

> **Can adaptive symbolic memory and context allocation reduce unnecessary inference workload while maintaining task performance?**

## Planned experiment sequence

- **GX-010** — actual LLM memory behavior.
- **GX-011** — Glyphin + Weisone integration.
- **GX-012** — Dual Cone Variator adaptive context allocation.
- **GX-013** — resource-conservation evaluation: adaptive allocation versus fixed allocation, measuring token use, latency, memory behavior, and task performance.

The Variator is therefore preserved as an architectural extension point while the underlying Glyphin and Weisone interfaces are stabilized.
