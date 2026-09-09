"""Deterministic in-process cache for context packets."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CacheStats:
    requests: int
    hits: int
    misses: int
    hit_rate: float
    entries: int


class ContextCache:
    """Cache packets by their deterministic context fingerprint.

    This cache optimizes reuse; it does not provide persistence, encryption, or
    a privacy guarantee. The implementation is intentionally dependency-free.
    """

    def __init__(self) -> None:
        self._entries: dict[str, Any] = {}
        self._requests = 0
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Any | None:
        self._requests += 1
        value = self._entries.get(key)
        if value is None:
            self._misses += 1
        else:
            self._hits += 1
        return value

    def put(self, key: str, value: Any) -> None:
        self._entries[key] = value

    def contains(self, key: str) -> bool:
        return key in self._entries

    def clear(self) -> None:
        self._entries.clear()

    def stats(self) -> CacheStats:
        rate = self._hits / self._requests if self._requests else 0.0
        return CacheStats(
            requests=self._requests,
            hits=self._hits,
            misses=self._misses,
            hit_rate=rate,
            entries=len(self._entries),
        )
