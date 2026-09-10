"""Deterministic GX-010 smoke test for the real Glyphin research core."""

from gx010_glyphin_adapter import DeterministicContextReader, GlyphinMemoryAdapter
from gx010_baselines import make_fixture
from gx010_runner import run_condition


def main() -> int:
    items, questions = make_fixture()
    run = run_condition(
        questions=questions,
        memory_items=items,
        memory=GlyphinMemoryAdapter(),
        reader=DeterministicContextReader(),
        experiment_id="GX-010",
        run_id="GX-010-GLYPHIN-SMOKE-001",
        git_sha="WORKTREE",
        model_name="deterministic-smoke-reader",
        model_version="fixture-1",
        prompt_version="gx010-smoke-1",
        tokenizer_name="NOT_MEASURED",
        tokenizer_version="NOT_MEASURED",
        condition="C_GLYPHIN",
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
