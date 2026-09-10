# Glyphin Research Core

This repository contains two distinct generations of Glyphin code.

## Research source of truth

The executable research implementation is `glyphin_research_core.py` (`GlyphinMemory`). It implements named states, parent/child lineage, recurrence, cohesion, resonance, decay, recall, canonical serialization, and persistence.

`glyphin_main.py` is the executable entry point for the research core.

Research topology, reconstruction, referee, and compression modules are separate experimental layers. They must not be treated as proof of the full Glyphin hypothesis.

## Planned adaptive resource-conservation layer

`DUAL_CONE_VARIATOR_RESOURCE_CONSERVATION.md` preserves the **Dual Cone Variator** as a planned core mechanism. Its purpose is to dynamically regulate contextual and output-token allocation so the system can conserve computational resources while preserving task performance. It is intended to sit between Glyphin memory representation and Weisone Kernel/LLM execution, with runtime feedback for adaptive allocation.

The Variator is **not yet validated** and should not be treated as a demonstrated energy-saving mechanism. Its future evaluation is planned through GX-012/GX-013 against fixed-allocation baselines.

## Legacy code

`glyphin.py` is the legacy Glyphin v1.7 REPL/runtime. It is preserved for historical reference and is **not** the research engine.

## Evidence rule

A working implementation demonstrates that an architecture can be executed. It does not by itself validate the hypothesis. Experimental claims require reproducible measurements and independent referee checks.
