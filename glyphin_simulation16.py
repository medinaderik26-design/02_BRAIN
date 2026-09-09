"""Glyphin Simulation 16 — semantic compression scaling benchmark.

Purpose:
    Measure whether the verified Simulation 15 semantic representation scales
    with Glyphin state size while preserving exact state semantics.

This benchmark is intentionally a baseline, not an optimizer. It measures
characters, whitespace-delimited words, and optional tokenizer tokens.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path

from glyphin_compression import measure
from glyphin_research_core import GlyphState, GlyphinMemory
from glyphin_state_referee import referee_memory

BENCHMARK_VERSION = "16.0"
SEED = 16092026
SIZES = (4, 8, 16, 32, 64, 128, 256)


def build_memory(n: int) -> GlyphinMemory:
    memory = GlyphinMemory(decay_lambda=0.02, alpha=0.30, beta=0.20)
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    memory.add_state(GlyphState("root", 0, 1.0, None, 1, 1.0, "seed", (base).isoformat()))
    for i in range(1, n):
        parent = "root" if i < 3 else f"s{i // 2}"
        state = GlyphState(
            name=f"s{i}",
            level=i,
            cohesion=round(1.0 / (1 + i * 0.01), 8),
            parent=parent,
            frequency=(i % 7) + 1,
            resonance=round((i % 11) / 10, 8),
            sigma=f"sigma-{i % 5}",
            created_at=(base + timedelta(minutes=i)).isoformat(),
        )
        memory.add_state(state)
    return memory


def esc(value: str) -> str:
    return value.replace("\\", "\\\\").replace("|", "\\|").replace(";", "\\;").replace(",", "\\,")


def encode(memory: GlyphinMemory) -> str:
    p = f"P{memory.decay_lambda!r},{memory.alpha!r},{memory.beta!r}"
    records = []
    for name in sorted(memory.states):
        s = memory.states[name]
        parent = "" if s.parent is None else esc(s.parent)
        records.append("S" + "|".join([
            esc(s.name), str(s.level), repr(s.cohesion), parent,
            str(s.frequency), repr(s.resonance), esc(s.sigma), esc(s.created_at),
        ]))
    return p + ";" + ";".join(records)


def split_escaped(text: str, delimiter: str) -> list[str]:
    out, current, escaped = [], [], False
    for ch in text:
        if escaped:
            current.append("\\" + ch)
            escaped = False
        elif ch == "\\":
            escaped = True
        elif ch == delimiter:
            out.append("".join(current))
            current = []
        else:
            current.append(ch)
    if escaped:
        current.append("\\")
    out.append("".join(current))
    return out


def unesc(value: str) -> str:
    out, escaped = [], False
    for ch in value:
        if escaped:
            out.append(ch); escaped = False
        elif ch == "\\":
            escaped = True
        else:
            out.append(ch)
    if escaped:
        out.append("\\")
    return "".join(out)


def decode(text: str) -> GlyphinMemory:
    records = split_escaped(text, ";")
    if not records or not records[0].startswith("P"):
        raise ValueError("invalid parameter record")
    params = records[0][1:].split(",")
    if len(params) != 3:
        raise ValueError("invalid parameters")
    memory = GlyphinMemory(float(params[0]), float(params[1]), float(params[2]))
    pending = []
    for record in records[1:]:
        if not record:
            continue
        fields = split_escaped(record[1:], "|") if record.startswith("S") else []
        if len(fields) != 8:
            raise ValueError("state record must contain eight fields")
        pending.append(fields)
    unresolved = {unesc(f[0]): f for f in pending}
    while unresolved:
        progressed = False
        for name, f in list(unresolved.items()):
            parent = unesc(f[3]) or None
            if parent is not None and parent not in memory.states:
                continue
            state = GlyphState(
                name=name, level=int(f[1]), cohesion=float(f[2]), parent=parent,
                frequency=int(f[4]), resonance=float(f[5]), sigma=unesc(f[6]),
                created_at=unesc(f[7]),
            )
            memory.add_state(state)
            del unresolved[name]
            progressed = True
        if not progressed:
            raise ValueError("unresolved parent dependency")
    return memory


def canonical(memory: GlyphinMemory) -> str:
    return json.dumps(memory.to_dict(), sort_keys=True, separators=(",", ":"))


def run_case(n: int) -> dict:
    source = build_memory(n)
    encoded = encode(source)
    reconstructed = decode(encoded)
    check = referee_memory(source, reconstructed)
    metrics = measure(canonical(source), encoded)
    return {
        "states": n,
        "source_chars": metrics.source_chars,
        "encoded_chars": metrics.encoded_chars,
        "char_reduction_pct": metrics.char_reduction_pct,
        "source_words": metrics.source_words,
        "encoded_words": metrics.encoded_words,
        "word_reduction_pct": metrics.word_reduction_pct,
        "token_metrics": None,
        "exact": check.exact,
        "field_mismatches": check.field_mismatches,
        "parameter_mismatches": check.parameter_mismatches,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=list(SIZES))
    parser.add_argument("--output", default="glyphin_simulation16_result.json")
    args = parser.parse_args()
    cases = [run_case(n) for n in args.sizes]
    result = {
        "benchmark_version": BENCHMARK_VERSION,
        "seed": SEED,
        "sizes": args.sizes,
        "total_cases": len(cases),
        "exact_cases": sum(c["exact"] for c in cases),
        "exact_rate_pct": 100.0 * sum(c["exact"] for c in cases) / len(cases) if cases else 0.0,
        "mean_char_reduction_pct": statistics.mean(c["char_reduction_pct"] for c in cases) if cases else 0.0,
        "median_char_reduction_pct": statistics.median(c["char_reduction_pct"] for c in cases) if cases else 0.0,
        "cases": cases,
    }
    payload = json.dumps(result, sort_keys=True, separators=(",", ":"))
    result["data_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["exact_cases"] == result["total_cases"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
