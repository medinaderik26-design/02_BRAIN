"""GX-011 adapter for Weisone Kernel + Glyphin.

The adapter composes, rather than rewrites, the existing Weisone Kernel and
Glyphin research core. The kernel memory boundary is injected so this module
can be tested without requiring a particular installation path or LLM server.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Callable, Dict, List, Optional

from glyphin_research_core import GlyphinMemory
from gx010_runner import MemoryEvidence, Question


@dataclass(frozen=True)
class _KernelRecord:
    """Structural fallback record for provider-free kernel implementations."""

    record_id: str
    payload: Any
    metadata: Dict[str, Any]


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[A-Za-z0-9_]+", text)}


def _item_text(item: Any) -> str:
    if isinstance(item, str):
        return item
    return json.dumps(item, sort_keys=True, ensure_ascii=False)


@dataclass
class WeisoneGlyphinAdapter:
    """Condition D: Weisone Kernel boundary with Glyphin-backed memory.

    ``kernel`` is expected to expose the provider-neutral memory interface from
    the Weisone Kernel repository. The adapter does not import or mutate the
    historical ``weisone_runtime.py`` module.

    ``record_factory`` is injectable so the adapter never depends on a Python
    import path from the separate kernel repository. The provider-free smoke
    test supplies the kernel's actual ``MemoryRecord`` class; other compatible
    implementations may use the structural fallback.
    """

    kernel: Any
    top_k: int = 5
    record_factory: Optional[Callable[[str, Any, Dict[str, Any]], Any]] = None

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")
        if self.record_factory is None:
            self.record_factory = (
                lambda record_id, payload, metadata: _KernelRecord(
                    record_id=record_id,
                    payload=payload,
                    metadata=metadata,
                )
            )
        self.reset()

    def reset(self) -> None:
        self.memory = GlyphinMemory()
        self.items: List[Any] = []
        self.state_names: List[str] = []
        self.kernel_memory = self.kernel
        if hasattr(self.kernel_memory, "restore"):
            snapshot_type = type(self.kernel_memory.snapshot())
            self.kernel_memory.restore(snapshot_type(records=[]))

    def insert(self, item: Any) -> None:
        index = len(self.items)
        state_name = f"gx011_state_{index:04d}"
        text = _item_text(item)

        self.memory.add_state(
            state_name,
            level=0,
            cohesion=0.5,
            frequency=1,
            resonance=1.0,
            sigma="seed",
        )
        self.items.append(item)
        self.state_names.append(state_name)

        # The kernel stores the same experiment item through its explicit
        # interface. Glyphin remains the state representation being measured.
        record = self.record_factory(
            state_name,
            text,
            {"glyphin_state": state_name},
        )
        self.kernel_memory.write(record)

    def _candidate_states(self, question: Question) -> List[str]:
        query_terms = _tokens(question.prompt)
        scored = []
        for index, item in enumerate(self.items):
            text = _item_text(item)
            score = len(query_terms & _tokens(text))
            scored.append((score, -index, self.state_names[index]))
        scored.sort(reverse=True)
        return [name for score, _, name in scored if score > 0][: self.top_k]

    def query(self, question: Question) -> MemoryEvidence:
        selected = self._candidate_states(question)
        recalls = [self.memory.recall(name) for name in selected]

        # Query the kernel boundary as part of the D condition. The returned
        # records are included as provenance, while the answer context remains
        # the canonical Glyphin reconstruction used by the experiment.
        kernel_matches = self.kernel_memory.retrieve(question.prompt, limit=self.top_k)
        context_records = [
            {"glyphin": recall, "kernel_record_id": record.record_id}
            for recall, record in zip(recalls, kernel_matches)
        ]
        context = "\n".join(
            json.dumps(record, sort_keys=True, ensure_ascii=False)
            for record in context_records
        )

        canonical = self.memory.to_json()
        restored = GlyphinMemory.from_json(canonical)
        reconstruction_exact = restored.to_json() == canonical

        return MemoryEvidence(
            context=context,
            memory_store_size=len(canonical.encode("utf-8")),
            retrieval_operations=1,
            reconstruction_exact=reconstruction_exact,
            metadata={
                "input_text": context,
                "retrieved_state_ids": selected,
                "kernel_retrieved_record_ids": [record.record_id for record in kernel_matches],
                "retrieved_items": len(recalls),
                "kernel_version": getattr(self.kernel, "version", "UNKNOWN"),
            },
        )


__all__ = ["WeisoneGlyphinAdapter"]
