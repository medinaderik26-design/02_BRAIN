"""User-to-model context gateway for token/resource efficiency.

Design goal: keep the user's full working context outside the model request when
possible. Build a deterministic, auditable context packet containing only the
material needed for the next model call.

This is a transport/context optimization layer, not a claim of privacy against
a model provider. Network, provider, and application logging policies remain
outside this module.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Optional

from glyphin_compression import CompressionMetrics, measure


@dataclass(frozen=True)
class ContextItem:
    key: str
    text: str
    priority: int = 0
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.key.strip():
            raise ValueError("context item key must not be empty")
        if self.priority < 0:
            raise ValueError("priority must be non-negative")


@dataclass(frozen=True)
class ContextPacket:
    request_id: str
    created_at: str
    query: str
    selected_keys: tuple[str, ...]
    context: str
    source_chars: int
    packet_chars: int
    source_words: int
    packet_words: int
    source_sha256: str
    packet_sha256: str
    excluded_keys: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "version": "context-gateway-0.1",
            "request_id": self.request_id,
            "created_at": self.created_at,
            "query": self.query,
            "selected_keys": list(self.selected_keys),
            "context": self.context,
            "source_chars": self.source_chars,
            "packet_chars": self.packet_chars,
            "source_words": self.source_words,
            "packet_words": self.packet_words,
            "source_sha256": self.source_sha256,
            "packet_sha256": self.packet_sha256,
            "excluded_keys": list(self.excluded_keys),
        }


class ContextGateway:
    """Deterministic context selector with an explicit character budget."""

    def __init__(self, max_chars: int = 12000):
        if max_chars < 1:
            raise ValueError("max_chars must be positive")
        self.max_chars = max_chars
        self._items: dict[str, ContextItem] = {}

    def add(self, item: ContextItem) -> None:
        if item.key in self._items:
            raise ValueError(f"duplicate context key: {item.key}")
        self._items[item.key] = item

    def add_many(self, items: Iterable[ContextItem]) -> None:
        for item in items:
            self.add(item)

    @staticmethod
    def _query_terms(query: str) -> set[str]:
        return {x.lower() for x in re.findall(r"[A-Za-z0-9_]+", query) if len(x) > 2}

    def _score(self, item: ContextItem, query_terms: set[str]) -> tuple[int, int, str]:
        haystack = (item.key + " " + item.text + " " + " ".join(item.tags)).lower()
        overlap = sum(1 for term in query_terms if term in haystack)
        # Stable tie-breakers make packets reproducible across runs.
        return (item.priority + overlap, overlap, item.key)

    def build(self, query: str, request_id: Optional[str] = None) -> ContextPacket:
        if not query.strip():
            raise ValueError("query must not be empty")
        terms = self._query_terms(query)
        ranked = sorted(
            self._items.values(),
            key=lambda item: self._score(item, terms),
            reverse=True,
        )

        selected: list[ContextItem] = []
        used = 0
        for item in ranked:
            block = f"[{item.key}]\n{item.text.strip()}"
            extra = len(block) + (2 if selected else 0)
            if used + extra <= self.max_chars:
                selected.append(item)
                used += extra

        context = "\n\n".join(f"[{x.key}]\n{x.text.strip()}" for x in selected)
        source = "\n\n".join(f"[{x.key}]\n{x.text.strip()}" for x in self._items.values())
        request_id = request_id or hashlib.sha256(query.encode()).hexdigest()[:16]
        return ContextPacket(
            request_id=request_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            query=query,
            selected_keys=tuple(x.key for x in selected),
            context=context,
            source_chars=len(source),
            packet_chars=len(context),
            source_words=len(source.split()),
            packet_words=len(context.split()),
            source_sha256=hashlib.sha256(source.encode()).hexdigest(),
            packet_sha256=hashlib.sha256(context.encode()).hexdigest(),
            excluded_keys=tuple(x.key for x in self._items.values() if x not in selected),
        )

    def metrics(self, packet: ContextPacket) -> CompressionMetrics:
        source = "\n\n".join(
            f"[{key}]\n{self._items[key].text.strip()}" for key in self._items
        )
        return measure(source, packet.context)


__all__ = ["ContextItem", "ContextPacket", "ContextGateway"]
