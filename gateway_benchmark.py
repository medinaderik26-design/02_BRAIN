"""Benchmark harness for measuring context-boundary savings honestly."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from context_gateway import ContextGateway, ContextItem
from gateway_runtime import GatewayRuntime
from provider_adapter import DryRunAdapter
from context_cache import ContextCache


@dataclass(frozen=True)
class BenchmarkResult:
    queries: int
    cache_requests: int
    cache_hits: int
    cache_misses: int
    cache_hit_rate_pct: float
    source_chars: int
    packet_chars: int
    source_words: int
    packet_words: int
    char_reduction_pct: float | None
    word_reduction_pct: float | None
    selected: tuple[str, ...]
    excluded: tuple[str, ...]


def _reduction(source: int, encoded: int) -> float | None:
    return None if source == 0 else (1.0 - encoded / source) * 100.0


def run(
    queries: str | Iterable[str],
    items: list[ContextItem],
    max_chars: int = 12000,
) -> BenchmarkResult:
    """Run repeated queries through the full gateway with a dry-run provider.

    Character/word reduction measures context selected for the model boundary.
    Token savings are intentionally not reported here because tokenizer choice
    is provider/model specific and no tokenizer is assumed by the core.
    """
    query_list = [queries] if isinstance(queries, str) else list(queries)
    gateway = ContextGateway(max_chars=max_chars)
    gateway.add_many(items)
    runtime = GatewayRuntime(gateway, ContextCache(), DryRunAdapter())

    source_chars = packet_chars = source_words = packet_words = 0
    selected: tuple[str, ...] = ()
    excluded: tuple[str, ...] = ()
    for query in query_list:
        runtime.run(query)
        packet = gateway.build(query)
        source_chars += packet.source_chars
        packet_chars += packet.packet_chars
        source_words += packet.source_words
        packet_words += packet.packet_words
        selected = packet.selected_keys
        excluded = packet.excluded_keys

    stats = runtime.cache.stats()
    return BenchmarkResult(
        queries=len(query_list),
        cache_requests=stats.requests,
        cache_hits=stats.hits,
        cache_misses=stats.misses,
        cache_hit_rate_pct=stats.hit_rate * 100.0,
        source_chars=source_chars,
        packet_chars=packet_chars,
        source_words=source_words,
        packet_words=packet_words,
        char_reduction_pct=_reduction(source_chars, packet_chars),
        word_reduction_pct=_reduction(source_words, packet_words),
        selected=selected,
        excluded=excluded,
    )

__all__ = ["BenchmarkResult", "run"]
