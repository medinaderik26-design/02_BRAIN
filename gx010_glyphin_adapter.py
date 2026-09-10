"""GX-010 adapter for the existing Glyphin research core.

This adapter does not redefine Glyphin. It maps experiment memory items into
GlyphinMemory states and exposes recalled state records as MemoryEvidence.
The adapter is deliberately deterministic so the first GX-010 integration test
can validate the measurement path before a real LLM reader is introduced.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Dict, List

from glyphin_research_core import GlyphinMemory
from gx010_runner import MemoryEvidence, Question


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[A-Za-z0-9_]+", text)}


def _item_text(item: Any) -> str:
    if isinstance(item, str):
        return item
    return json.dumps(item, sort_keys=True, ensure_ascii=False)


@dataclass
class GlyphinMemoryAdapter:
    """Condition C: store experiment items as Glyphin states."""

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.memory = GlyphinMemory()
        self.items: List[Dict[str, Any]] = []
        self.state_names: List[str] = []

    def insert(self, item: Any) -> None:
        index = len(self.items)
        state_name = f"gx010_state_{index:04d}"
        text = _item_text(item)
        record = {"state": state_name, "item": item}

        # Keep the experiment mapping explicit. The existing Glyphin core owns
        # the state model; this adapter only chooses a deterministic projection.
        self.memory.add_state(
            state_name,
            level=0,
            cohesion=0.5,
            frequency=1,
            resonance=1.0,
            sigma="seed",
        )
        self.items.append(record)
        self.state_names.append(state_name)

        # Reinforcement is used only for repeated lexical content. It exercises
        # an existing Glyphin mechanism without changing the experiment schema.
        if index and _tokens(text) & _tokens(_item_text(self.items[-2]["item"])):
            self.memory.reinforce(state_name)

    def _candidate_states(self, question: Question) -> List[str]:
        query_terms = _tokens(question.prompt)
        scored = []
        for index, record in enumerate(self.items):
            text = _item_text(record["item"])
            score = len(query_terms & _tokens(text))
            scored.append((score, -index, record["state"]))
        scored.sort(reverse=True)
        return [name for score, _, name in scored if score > 0]

    def query(self, question: Question) -> MemoryEvidence:
        selected = self._candidate_states(question)
        recalls = [self.memory.recall(name) for name in selected[:5]]
        context = "\n".join(json.dumps(recall, sort_keys=True, ensure_ascii=False) for recall in recalls)

        # Verify the exact Glyphin state representation survives canonical
        # serialization and reconstruction for this query.
        restored = GlyphinMemory.from_json(self.memory.to_json())
        reconstruction_exact = restored.to_json() == self.memory.to_json()

        return MemoryEvidence(
            context=context,
            memory_store_size=len(self.memory.to_json().encode("utf-8")),
            retrieval_operations=1,
            reconstruction_exact=reconstruction_exact,
            metadata={
                "input_text": context,
                "retrieved_state_ids": selected[:5],
                "retrieved_items": len(recalls),
            },
        )


class DeterministicContextReader:
    """Smoke-test reader: return an expected answer only when it is present."""

    def answer(self, question: Question, evidence: MemoryEvidence):
        from gx010_runner import ReaderResult

        expected = str(question.expected)
        if expected in evidence.context:
            return ReaderResult(answer=question.expected)
        return ReaderResult(
            answer=None,
            metadata={"failure_category": "retrieval_miss"},
        )


__all__ = ["GlyphinMemoryAdapter", "DeterministicContextReader"]
