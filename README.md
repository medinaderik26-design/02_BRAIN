# Glyphin Research Core

This repository contains two distinct generations of Glyphin code.

> **PROPRIETARY WORK — ALL RIGHTS RESERVED.** See [`LICENSE`](./LICENSE) and [`COPYRIGHT.md`](./COPYRIGHT.md) before using, copying, modifying, redistributing, commercializing, or incorporating this work elsewhere.

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

## License and usage

This is **not an open-source release**. The repository is publicly visible for research transparency and controlled evaluation. The `LICENSE` grants only the limited rights expressly stated there. In particular, commercial exploitation, redistribution, sublicensing, derivative works, removal of provenance, and unauthorized AI/ML training or competing-system use are not permitted.

Public GitHub visibility does not itself grant those additional rights. GitHub's platform terms separately govern viewing and forking of public repositories.

For permission beyond the license, obtain written authorization from the copyright owner.
