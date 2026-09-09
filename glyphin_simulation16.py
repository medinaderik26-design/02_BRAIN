"""Glyphin Simulation 16 — semantic compression scaling with real tokens."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path

import tiktoken

from glyphin_compression import measure
from glyphin_research_core import GlyphinMemory
from glyphin_state_referee import referee_memory

VERSION = "16.1"
SEED = 16092026
SIZES = (4, 8, 16, 32, 64, 128, 256)
NULL = ""
ESCAPE = "\\"
TOKENIZER_NAME = "cl100k_base"


def esc(value: str) -> str:
    out = []
    for char in str(value):
        if char in (ESCAPE, "|", ";", ","):
            out.append(ESCAPE)
        out.append(char)
    return "".join(out)


def split_escaped(text: str, delimiter: str, preserve_escapes: bool = False) -> list[str]:
    parts, current, escaped = [], [], False
    for char in text:
        if escaped:
            if preserve_escapes:
                current.extend((ESCAPE, char))
            else:
                current.append(char)
            escaped = False
        elif char == ESCAPE:
            escaped = True
        elif char == delimiter:
            parts.append("".join(current)); current = []
        else:
            current.append(char)
    if escaped:
        raise ValueError("dangling escape")
    parts.append("".join(current))
    return parts


def encode_semantic(memory: GlyphinMemory) -> str:
    params = f"P{memory.decay_lambda!r},{memory.alpha!r},{memory.beta!r}"
    records = [params]
    for name in sorted(memory.states):
        state = memory.states[name]
        fields = [state.name, str(state.level), repr(state.cohesion),
                  NULL if state.parent is None else state.parent,
                  str(state.frequency), repr(state.resonance), state.sigma,
                  state.created_at]
        records.append("S" + "|".join(esc(field) for field in fields))
    return ";".join(records)


def reconstruct_semantic(encoded: str) -> GlyphinMemory:
    records = split_escaped(encoded, ";", preserve_escapes=True)
    if not records or not records[0].startswith("P"):
        raise ValueError("missing parameter record")
    params = records[0][1:].split(",")
    if len(params) != 3:
        raise ValueError("parameter record must contain three values")
    memory = GlyphinMemory(decay_lambda=float(params[0]), alpha=float(params[1]), beta=float(params[2]))
    rows = []
    for record in records[1:]:
        if not record.startswith("S"):
            raise ValueError("unknown record")
        fields = split_escaped(record[1:], "|", preserve_escapes=False)
        if len(fields) != 8:
            raise ValueError("state record must contain eight fields")
        rows.append((fields[0], int(fields[1]), float(fields[2]),
                     None if fields[3] == NULL else fields[3], int(fields[4]),
                     float(fields[5]), fields[6], fields[7]))
    remaining = {row[0]: row for row in rows}
    while remaining:
        progressed = False
        for name in sorted(tuple(remaining)):
            row = remaining[name]
            if row[3] is None or row[3] in memory.states:
                memory.add_state(row[0], level=row[1], cohesion=row[2], parent=row[3],
                                 frequency=row[4], resonance=row[5], sigma=row[6], created_at=row[7])
                del remaining[name]
                progressed = True
        if not progressed:
            raise ValueError("unresolvable parent relationships")
    return memory


def build_memory(n: int) -> GlyphinMemory:
    memory = GlyphinMemory(decay_lambda=0.02, alpha=0.30, beta=0.20)
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    memory.add_state("root", level=0, cohesion=.90, parent=None, frequency=4,
                     resonance=.80, sigma="seed", created_at=base.isoformat())
    for i in range(1, n):
        parent = "root" if i < 3 else f"s{i // 2}"
        # Periodic adversarial values force the frozen escaping grammar to be exercised.
        if i % 17 == 0:
            name, sigma = f"s{i}|x", f"sig,{i};\\x"
        else:
            name, sigma = f"s{i}", f"sigma-{i % 5}"
        memory.add_state(name, level=i, cohesion=round(.9 / (1 + i * .01), 8),
                         parent=parent, frequency=(i % 7) + 1,
                         resonance=round((i % 11) / 10, 8), sigma=sigma,
                         created_at=(base + timedelta(minutes=i)).isoformat())
    return memory


def canonical(memory: GlyphinMemory) -> str:
    return memory.to_json()


def run_case(n: int, tokenizer) -> dict:
    source = build_memory(n)
    encoded = encode_semantic(source)
    reconstructed = reconstruct_semantic(encoded)
    verdict = referee_memory(source, reconstructed)
    source_text = canonical(source)
    metrics = measure(source_text, encoded)
    source_tokens = len(tokenizer.encode(source_text, disallowed_special=()))
    encoded_tokens = len(tokenizer.encode(encoded, disallowed_special=()))
    token_reduction = (source_tokens - encoded_tokens) / source_tokens * 100 if source_tokens else 0.0
    return {
        "states": n,
        "source_chars": metrics.source_chars,
        "encoded_chars": metrics.encoded_chars,
        "char_reduction_pct": metrics.char_reduction_pct,
        "source_words": metrics.source_words,
        "encoded_words": metrics.encoded_words,
        "word_reduction_pct": metrics.word_reduction_pct,
        "source_tokens": source_tokens,
        "encoded_tokens": encoded_tokens,
        "token_reduction_pct": token_reduction,
        "exact": verdict.exact,
        "field_mismatches": verdict.field_mismatches,
        "parameter_mismatches": verdict.parameter_mismatches,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=list(SIZES))
    parser.add_argument("--output", default="glyphin_simulation16_result.json")
    args = parser.parse_args()
    tokenizer = tiktoken.get_encoding(TOKENIZER_NAME)
    cases = [run_case(n, tokenizer) for n in args.sizes]
    reductions = [c["token_reduction_pct"] for c in cases]
    char_reductions = [c["char_reduction_pct"] for c in cases]
    result = {
        "benchmark_version": VERSION,
        "seed": SEED,
        "tokenizer": TOKENIZER_NAME,
        "sizes": args.sizes,
        "total_cases": len(cases),
        "exact_cases": sum(c["exact"] for c in cases),
        "exact_rate_pct": 100.0 * sum(c["exact"] for c in cases) / len(cases) if cases else 0.0,
        "mean_char_reduction_pct": statistics.mean(char_reductions) if cases else 0.0,
        "median_char_reduction_pct": statistics.median(char_reductions) if cases else 0.0,
        "mean_token_reduction_pct": statistics.mean(reductions) if cases else 0.0,
        "median_token_reduction_pct": statistics.median(reductions) if cases else 0.0,
        "cases": cases,
    }
    raw = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["data_sha256"] = hashlib.sha256(raw).hexdigest()
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["exact_cases"] == result["total_cases"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
