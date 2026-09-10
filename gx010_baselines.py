"""Deterministic GX-010 baseline memory adapters.

These adapters are intentionally small and provider-neutral. They establish the
experimental interfaces before a local LLM reader is connected.

A_FULL_HISTORY returns the complete inserted history.
B_RETRIEVAL_MEMORY performs deterministic lexical retrieval over inserted items.
C_GLYPHIN is reserved for the existing Glyphin research core and is implemented
in a separate adapter module so that this baseline file does not redefine Glyphin.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Iterable, List

from gx010_runner import MemoryEvidence, Question


def _text(item: Any) -> str:
    if isinstance(item, str):
        return item
    return json.dumps(item, sort_keys=True, ensure_ascii=False)


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[A-Za-z0-9_]+", text)}


@dataclass
class FullHistoryMemory:
    """Condition A: supply the complete memory history."""

    items: List[Any] = None

    def __post_init__(self) -> None:
        if self.items is None:
            self.items = []

    def reset(self) -> None:
        self.items.clear()

    def insert(self, item: Any) -> None:
        self.items.append(item)

    def query(self, question: Question) -> MemoryEvidence:
        context = "\n".join(_text(item) for item in self.items)
        return MemoryEvidence(
            context=context,
            memory_store_size=len(context.encode("utf-8")),
            retrieval_operations=1,
            metadata={"input_text": context},
        )


@dataclass
class LexicalRetrievalMemory:
    """Condition B: deterministic lexical retrieval baseline."""

    top_k: int = 5
    items: List[Any] = None

    def __post_init__(self) -> None:
        if self.items is None:
            self.items = []

    def reset(self) -> None:
        self.items.clear()

    def insert(self, item: Any) -> None:
        self.items.append(item)

    def query(self, question: Question) -> MemoryEvidence:
        query_terms = _tokens(question.prompt)
        scored = []
        for index, item in enumerate(self.items):
            text = _text(item)
            score = len(query_terms & _tokens(text))
            scored.append((score, -index, text))
        scored.sort(reverse=True)
        selected = [text for score, _, text in scored[: self.top_k] if score > 0]
        context = "\n".join(selected)
        return MemoryEvidence(
            context=context,
            memory_store_size=sum(len(_text(item).encode("utf-8")) for item in self.items),
            retrieval_operations=1,
            metadata={"input_text": context, "retrieved_items": len(selected)},
        )


def make_fixture() -> tuple[list[dict[str, Any]], list[Question]]:
    """Small deterministic smoke fixture for adapter and harness validation."""
    items = [
        {"session": "s1", "fact": "The user's preferred project name is Glyphin."},
        {"session": "s1", "fact": "Glyphin uses named states and lineage."},
        {"session": "s2", "fact": "The company is Weisone Systems."},
        {"session": "s2", "fact": "The core integration layer is the Weisone Kernel."},
    ]
    questions = [
        Question("q1", "What is the preferred project name?", "Glyphin", "information_extraction", "s1"),
        Question("q2", "What is the company name?", "Weisone Systems", "information_extraction", "s2"),
        Question("q3", "What is the core integration layer?", "Weisone Kernel", "information_extraction", "s2"),
    ]
    return items, questions
