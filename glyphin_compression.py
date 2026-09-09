"""Reproducible compression measurements for Glyphin artifacts.

Character and whitespace-delimited word counts are deterministic. Token counts
are explicitly NOT measured unless a tokenizer is supplied by the caller.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class CompressionMetrics:
    source_chars: int
    encoded_chars: int
    source_words: int
    encoded_words: int
    source_tokens: Optional[int]
    encoded_tokens: Optional[int]
    char_reduction_pct: float
    word_reduction_pct: float
    token_reduction_pct: Optional[float]


def _reduction(source: int, encoded: int) -> float:
    if source == 0:
        return 0.0
    return (source - encoded) / source * 100.0


def measure(source: str, encoded: str, tokenizer: Optional[Callable[[str], int]] = None) -> CompressionMetrics:
    source_chars = len(source)
    encoded_chars = len(encoded)
    source_words = len(source.split())
    encoded_words = len(encoded.split())

    if tokenizer is None:
        source_tokens = encoded_tokens = token_reduction = None
    else:
        source_tokens = int(tokenizer(source))
        encoded_tokens = int(tokenizer(encoded))
        token_reduction = _reduction(source_tokens, encoded_tokens)

    return CompressionMetrics(
        source_chars=source_chars,
        encoded_chars=encoded_chars,
        source_words=source_words,
        encoded_words=encoded_words,
        source_tokens=source_tokens,
        encoded_tokens=encoded_tokens,
        char_reduction_pct=_reduction(source_chars, encoded_chars),
        word_reduction_pct=_reduction(source_words, encoded_words),
        token_reduction_pct=token_reduction,
    )


__all__ = ["CompressionMetrics", "measure"]
