"""Benchmark harness for measuring context-boundary savings honestly."""
from __future__ import annotations
from dataclasses import dataclass
from context_gateway import ContextGateway, ContextItem

@dataclass(frozen=True)
class BenchmarkResult:
    source_chars: int
    packet_chars: int
    source_words: int
    packet_words: int
    char_reduction_pct: float
    word_reduction_pct: float
    selected: tuple[str, ...]
    excluded: tuple[str, ...]


def run(query: str, items: list[ContextItem], max_chars: int = 12000) -> BenchmarkResult:
    gateway = ContextGateway(max_chars=max_chars)
    gateway.add_many(items)
    packet = gateway.build(query, request_id="benchmark")
    metrics = gateway.metrics(packet)
    return BenchmarkResult(
        metrics.source_chars, metrics.encoded_chars,
        metrics.source_words, metrics.encoded_words,
        metrics.char_reduction_pct, metrics.word_reduction_pct,
        packet.selected_keys, packet.excluded_keys,
    )

__all__ = ["BenchmarkResult", "run"]
