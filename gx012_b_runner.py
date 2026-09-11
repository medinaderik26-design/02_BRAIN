"""GX-012-B adaptive local-LLM runner boundary.

The runner connects the existing GX-010 reader to Dual Cone Variator without
requiring a specific local model provider. Network calls occur only when the
caller supplies a real OpenAI-compatible reader transport/server.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from gx010_reader import OpenAICompatibleReader, PromptConfig
from gx010_runner import MemoryEvidence, Question, ReaderResult
from gx012_controller import DualConeVariator, GX012Signals, GX012Allocation


@dataclass(frozen=True)
class GX012BResult:
    result: ReaderResult
    allocation: GX012Allocation
    selected_context: str
    requested_input_budget: int
    requested_output_budget: int


ContextSelector = Callable[[str, int], str]


def character_budget_selector(context: str, budget: int) -> str:
    """Deterministic context selector used until a real tokenizer is supplied.

    The budget is treated as a character budget, not a token count. GX-012-B
    must replace this selector with the exact model tokenizer before making
    token-efficiency claims.
    """
    if budget <= 0:
        return ""
    return context[:budget]


class GX012BAdaptiveReader:
    """Apply Dual Cone allocation before invoking a local OpenAI-compatible LLM."""

    def __init__(
        self,
        base_url: str,
        model: str,
        controller: Optional[DualConeVariator] = None,
        api_key: Optional[str] = None,
        prompt_config: Optional[PromptConfig] = None,
        input_budget: int = 4096,
        context_selector: ContextSelector = character_budget_selector,
        transport=None,
    ) -> None:
        self.controller = controller or DualConeVariator(mu=0.25)
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.prompt_config = prompt_config or PromptConfig()
        self.input_budget = input_budget
        self.context_selector = context_selector
        self.transport = transport

    def answer(self, question: Question, evidence: MemoryEvidence, signals: GX012Signals) -> GX012BResult:
        allocation = self.controller.update(signals, self.input_budget)
        selected_budget = max(1, int(self.input_budget * allocation.input_aperture))
        selected_context = self.context_selector(evidence.context, selected_budget)
        selected_evidence = MemoryEvidence(
            context=selected_context,
            memory_store_size=evidence.memory_store_size,
            retrieval_operations=evidence.retrieval_operations,
            reconstruction_exact=evidence.reconstruction_exact,
            state_referee_exact=evidence.state_referee_exact,
            metadata=dict(evidence.metadata),
        )

        reader = OpenAICompatibleReader(
            base_url=self.base_url,
            model=self.model,
            api_key=self.api_key,
            temperature=0.0,
            max_output_tokens=allocation.output_budget,
            prompt_config=self.prompt_config,
            transport=self.transport,
        )
        result = reader.answer(question, selected_evidence)
        return GX012BResult(
            result=result,
            allocation=allocation,
            selected_context=selected_context,
            requested_input_budget=selected_budget,
            requested_output_budget=allocation.output_budget,
        )


__all__ = ["GX012BAdaptiveReader", "GX012BResult", "character_budget_selector"]
