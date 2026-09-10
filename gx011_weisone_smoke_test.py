"""Provider-free GX-011 smoke test.

This validates the Weisone Kernel memory boundary and the combined
Weisone/Glyphin adapter before a local LLM is introduced.

Set ``WEISONE_KERNEL_REPO`` to the local checkout of the separate
``weisone-kernel`` repository. A sibling ``../weisone-kernel`` checkout is
used by default for developer convenience.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from gx010_baselines import make_fixture
from gx010_glyphin_adapter import DeterministicContextReader
from gx010_runner import run_condition
from gx011_weisone_adapter import WeisoneGlyphinAdapter


def _load_kernel_memory_module():
    configured = os.environ.get("WEISONE_KERNEL_REPO")
    repo_path = Path(configured) if configured else Path(__file__).resolve().parent.parent / "weisone-kernel"
    module_path = repo_path / "memory_interface.py"
    if not module_path.is_file():
        raise FileNotFoundError(
            "Could not find Weisone Kernel memory_interface.py. "
            f"Set WEISONE_KERNEL_REPO to the kernel checkout; looked at {module_path}."
        )

    spec = importlib.util.spec_from_file_location("weisone_kernel_memory_interface", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load kernel memory interface from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    kernel_memory_module = _load_kernel_memory_module()
    memory = WeisoneGlyphinAdapter(
        kernel_memory_module.InMemoryKernelMemory(),
        record_factory=kernel_memory_module.MemoryRecord,
    )

    items, questions = make_fixture()
    run = run_condition(
        questions=questions,
        memory_items=items,
        memory=memory,
        reader=DeterministicContextReader(),
        experiment_id="GX-011",
        run_id="GX-011-WEISONE-GLYPHIN-SMOKE-001",
        git_sha="WORKTREE",
        model_name="deterministic-smoke-reader",
        model_version="fixture-1",
        prompt_version="gx011-smoke-1",
        tokenizer_name="NOT_MEASURED",
        tokenizer_version="NOT_MEASURED",
        condition="D_GLYPHIN_WEISONE_KERNEL",
        fixture_seed=1001,
    )

    print("condition:", run.condition)
    print("questions:", run.question_count)
    print("correct:", run.correct_count)
    print("accuracy:", run.accuracy)
    print("reconstruction_exact:", run.reconstruction_exact)
    print("failures:", run.failure_count)

    if run.accuracy != 1.0 or run.reconstruction_exact is not True or run.failure_count != 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
