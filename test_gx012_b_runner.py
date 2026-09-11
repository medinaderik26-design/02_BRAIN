"""Provider-free tests for the GX-012-B local LLM boundary."""

from gx010_runner import MemoryEvidence, Question
from gx012_b_runner import GX012BAdaptiveReader
from gx012_controller import DualConeVariator, GX012Signals


class FakeTransport:
    def __init__(self):
        self.payloads = []

    def __call__(self, payload):
        self.payloads.append(payload)
        return {
            "choices": [{"message": {"content": "Glyphin"}}],
            "usage": {"prompt_tokens": 17, "completion_tokens": 3},
        }


def make_fixture():
    question = Question(
        question_id="b-test-001",
        prompt="What is the preferred project name?",
        expected="Glyphin",
        category="identity",
    )
    evidence = MemoryEvidence(
        context="Glyphin is the preferred project name. " * 100,
        metadata={"source": "gx012-b-test"},
    )
    return question, evidence


def test_adaptive_runner_changes_request_allocation():
    transport = FakeTransport()
    reader = GX012BAdaptiveReader(
        base_url="http://local.test",
        model="test-local-model",
        controller=DualConeVariator(mu=1.0),
        input_budget=1000,
        transport=transport,
    )
    question, evidence = make_fixture()

    low_pressure = GX012Signals(
        relevance=0.9, cohesion=0.8, lineage_depth=0.8,
        uncertainty=0.4, task_complexity=0.4,
        contradiction=0.0, resource_pressure=0.0,
    )
    high_pressure = GX012Signals(
        relevance=0.9, cohesion=0.8, lineage_depth=0.8,
        uncertainty=0.4, task_complexity=0.4,
        contradiction=0.0, resource_pressure=0.9,
    )

    first = reader.answer(question, evidence, low_pressure)
    second = reader.answer(question, evidence, high_pressure)

    assert first.result.answer == "Glyphin"
    assert second.result.answer == "Glyphin"
    assert second.requested_input_budget < first.requested_input_budget
    assert transport.payloads[1]["max_tokens"] < transport.payloads[0]["max_tokens"]
    assert len(second.selected_context) < len(first.selected_context)


def test_runner_is_provider_free_with_injected_transport():
    transport = FakeTransport()
    reader = GX012BAdaptiveReader(
        base_url="http://local.test",
        model="test-local-model",
        input_budget=500,
        transport=transport,
    )
    question, evidence = make_fixture()
    result = reader.answer(question, evidence, GX012Signals())
    assert result.result.output_tokens == 3
    assert result.result.metadata["input_tokens"] == 17
    assert len(transport.payloads) == 1


if __name__ == "__main__":
    test_adaptive_runner_changes_request_allocation()
    test_runner_is_provider_free_with_injected_transport()
    print("GX-012-B runner tests passed")
