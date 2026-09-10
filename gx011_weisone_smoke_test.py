"""Provider-free GX-011 smoke test.

This validates the Weisone Kernel memory boundary and the combined
Weisone/Glyphin adapter before a local LLM is introduced.
"""

from gx010_baselines import make_fixture
from gx010_runner import run_condition
from gx011_weisone_adapter import WeisoneGlyphinAdapter


def main() -> int:
    from memory_interface import InMemoryKernelMemory

    items, questions = make_fixture()
    memory = WeisoneGlyphinAdapter(InMemoryKernelMemory())

    run = run_condition(
        questions=questions,
        memory_items=items,
        memory=memory,
        reader=__import__("gx010_glyphin_adapter", fromlist=["DeterministicContextReader"]).DeterministicContextReader(),
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
