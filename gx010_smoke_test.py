"""GX-010 adapter smoke test.

This test validates the experiment plumbing without calling an LLM. It is not a
memory-quality result; it only verifies that conditions can be executed through
the common runner and that exact fixture answers are handled consistently.
"""

from __future__ import annotations

from gx010_baselines import FullHistoryMemory, LexicalRetrievalMemory, make_fixture
from gx010_runner import ReaderResult, run_condition


class FixtureReader:
    """Deterministic reader used only for harness plumbing validation."""

    ANSWERS = {
        "q1": "Glyphin",
        "q2": "Weisone Systems",
        "q3": "Weisone Kernel",
    }

    def answer(self, question, evidence):
        return ReaderResult(self.ANSWERS[question.question_id])


def main() -> None:
    items, questions = make_fixture()
    reader = FixtureReader()

    full = run_condition(
        questions=questions,
        memory_items=items,
        memory=FullHistoryMemory(),
        reader=reader,
        experiment_id="GX-010",
        run_id="SMOKE-A",
        git_sha="LOCAL",
        model_name="fixture-reader",
        model_version="fixture-1",
        prompt_version="smoke-1",
        tokenizer_name="none",
        tokenizer_version="none",
        condition="A_FULL_HISTORY",
        fixture_seed=1001,
    )

    retrieval = run_condition(
        questions=questions,
        memory_items=items,
        memory=LexicalRetrievalMemory(top_k=2),
        reader=reader,
        experiment_id="GX-010",
        run_id="SMOKE-B",
        git_sha="LOCAL",
        model_name="fixture-reader",
        model_version="fixture-1",
        prompt_version="smoke-1",
        tokenizer_name="none",
        tokenizer_version="none",
        condition="B_RETRIEVAL_MEMORY",
        fixture_seed=1001,
    )

    assert full.accuracy == 1.0
    assert retrieval.accuracy == 1.0
    assert full.question_count == 3
    assert retrieval.question_count == 3
    print("GX-010 smoke test: PASS")
    print(f"A_FULL_HISTORY accuracy={full.accuracy:.3f}")
    print(f"B_RETRIEVAL_MEMORY accuracy={retrieval.accuracy:.3f}")


if __name__ == "__main__":
    main()
