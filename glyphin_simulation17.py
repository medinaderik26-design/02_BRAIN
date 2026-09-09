"""Glyphin Simulation 17.0 — tokenizer-aware semantic encoding A/B benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from datetime import datetime, timedelta, timezone

import tiktoken

from glyphin_compression import measure
from glyphin_research_core import GlyphinMemory
from glyphin_state_referee import referee_memory

VERSION = "17.0"
SEED = 17092026
SIZES = (4, 8, 16, 32, 64, 128, 256)
TOKENIZER_NAME = "cl100k_base"
ESC = "\\"


def esc(value: str, reserved: set[str]) -> str:
    out = []
    for ch in str(value):
        if ch == ESC or ch in reserved:
            out.append(ESC)
        out.append(ch)
    return "".join(out)


def split_escaped(text: str, delimiter: str, preserve: bool = False) -> list[str]:
    parts, cur, escaped = [], [], False
    for ch in text:
        if escaped:
            if preserve:
                cur.extend((ESC, ch))
            else:
                cur.append(ch)
            escaped = False
        elif ch == ESC:
            escaped = True
        elif ch == delimiter:
            parts.append("".join(cur)); cur = []
        else:
            cur.append(ch)
    if escaped:
        raise ValueError("dangling escape")
    parts.append("".join(cur))
    return parts


def build_memory(n: int) -> GlyphinMemory:
    memory = GlyphinMemory(decay_lambda=0.02, alpha=0.30, beta=0.20)
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    memory.add_state("root", level=0, cohesion=.90, parent=None, frequency=4,
                     resonance=.80, sigma="seed", created_at=base.isoformat())
    names = {0: "root"}
    for i in range(1, n):
        name = f"s{i}|x" if i % 17 == 0 else f"s{i}"
        sigma = f"sig,{i};\\x" if i % 17 == 0 else f"sigma-{i % 5}"
        names[i] = name
        parent = names[0] if i < 3 else names[(i - 1) // 2]
        memory.add_state(name, level=i, cohesion=round(.9 / (1 + i * .01), 8),
                         parent=parent, frequency=(i % 7) + 1,
                         resonance=round((i % 11) / 10, 8), sigma=sigma,
                         created_at=(base + timedelta(minutes=i)).isoformat())
    return memory


def encode_15(memory: GlyphinMemory) -> str:
    """Frozen Sim 15.3 reference grammar."""
    records = [f"P{memory.decay_lambda!r},{memory.alpha!r},{memory.beta!r}"]
    for name in sorted(memory.states):
        s = memory.states[name]
        fields = [s.name, str(s.level), repr(s.cohesion), "" if s.parent is None else s.parent,
                  str(s.frequency), repr(s.resonance), s.sigma, s.created_at]
        records.append("S" + "|".join(esc(v, {"|", ";", ","}) for v in fields))
    return ";".join(records)


def decode_15(text: str) -> GlyphinMemory:
    records = split_escaped(text, ";", True)
    p = records[0][1:].split(",")
    memory = GlyphinMemory(decay_lambda=float(p[0]), alpha=float(p[1]), beta=float(p[2]))
    rows = []
    for r in records[1:]:
        f = split_escaped(r[1:], "|")
        rows.append((f[0], int(f[1]), float(f[2]), None if f[3] == "" else f[3], int(f[4]),
                     float(f[5]), f[6], f[7]))
    remaining = {r[0]: r for r in rows}
    while remaining:
        progress = False
        for name in sorted(tuple(remaining)):
            r = remaining[name]
            if r[3] is None or r[3] in memory.states:
                memory.add_state(r[0], level=r[1], cohesion=r[2], parent=r[3], frequency=r[4],
                                 resonance=r[5], sigma=r[6], created_at=r[7])
                del remaining[name]; progress = True
        if not progress:
            raise ValueError("unresolvable parent relationships")
    return memory


def encode_compact(memory: GlyphinMemory) -> str:
    """Same information as 15.3, removing non-semantic P/S record markers."""
    records = [f"{memory.decay_lambda!r},{memory.alpha!r},{memory.beta!r}"]
    for name in sorted(memory.states):
        s = memory.states[name]
        fields = [s.name, str(s.level), repr(s.cohesion), "" if s.parent is None else s.parent,
                  str(s.frequency), repr(s.resonance), s.sigma, s.created_at]
        records.append("|".join(esc(v, {"|", ";", ","}) for v in fields))
    return ";".join(records)


def decode_compact(text: str) -> GlyphinMemory:
    records = split_escaped(text, ";", True)
    p = records[0].split(",")
    memory = GlyphinMemory(decay_lambda=float(p[0]), alpha=float(p[1]), beta=float(p[2]))
    rows = []
    for r in records[1:]:
        f = split_escaped(r, "|")
        if len(f) != 8:
            raise ValueError("state record must contain eight fields")
        rows.append((f[0], int(f[1]), float(f[2]), None if f[3] == "" else f[3], int(f[4]),
                     float(f[5]), f[6], f[7]))
    remaining = {r[0]: r for r in rows}
    while remaining:
        progress = False
        for name in sorted(tuple(remaining)):
            r = remaining[name]
            if r[3] is None or r[3] in memory.states:
                memory.add_state(r[0], level=r[1], cohesion=r[2], parent=r[3], frequency=r[4],
                                 resonance=r[5], sigma=r[6], created_at=r[7])
                del remaining[name]; progress = True
        if not progress:
            raise ValueError("unresolvable parent relationships")
    return memory


def encode_tilde(memory: GlyphinMemory) -> str:
    """Alternative grammar using tilde for records and colon for fields."""
    records = [f"{memory.decay_lambda!r},{memory.alpha!r},{memory.beta!r}"]
    for name in sorted(memory.states):
        s = memory.states[name]
        fields = [s.name, str(s.level), repr(s.cohesion), "" if s.parent is None else s.parent,
                  str(s.frequency), repr(s.resonance), s.sigma, s.created_at]
        records.append(":".join(esc(v, {":", "~", ","}) for v in fields))
    return "~".join(records)


def decode_tilde(text: str) -> GlyphinMemory:
    records = split_escaped(text, "~", True)
    p = records[0].split(",")
    memory = GlyphinMemory(decay_lambda=float(p[0]), alpha=float(p[1]), beta=float(p[2]))
    rows = []
    for r in records[1:]:
        f = split_escaped(r, ":")
        if len(f) != 8:
            raise ValueError("state record must contain eight fields")
        rows.append((f[0], int(f[1]), float(f[2]), None if f[3] == "" else f[3], int(f[4]),
                     float(f[5]), f[6], f[7]))
    remaining = {r[0]: r for r in rows}
    while remaining:
        progress = False
        for name in sorted(tuple(remaining)):
            r = remaining[name]
            if r[3] is None or r[3] in memory.states:
                memory.add_state(r[0], level=r[1], cohesion=r[2], parent=r[3], frequency=r[4],
                                 resonance=r[5], sigma=r[6], created_at=r[7])
                del remaining[name]; progress = True
        if not progress:
            raise ValueError("unresolvable parent relationships")
    return memory

VARIANTS = {
    "15.3-reference": (encode_15, decode_15),
    "compact-markers-removed": (encode_compact, decode_compact),
    "tilde-colon": (encode_tilde, decode_tilde),
}


def evaluate(name: str, encode, decode, memory: GlyphinMemory, tokenizer) -> dict:
    encoded = encode(memory)
    reconstructed = decode(encoded)
    verdict = referee_memory(memory, reconstructed)
    source = memory.to_json()
    metrics = measure(source, encoded)
    st = len(tokenizer.encode(source, disallowed_special=()))
    et = len(tokenizer.encode(encoded, disallowed_special=()))
    return {
        "variant": name, "states": len(memory.states), "source_chars": metrics.source_chars,
        "encoded_chars": metrics.encoded_chars, "char_reduction_pct": metrics.char_reduction_pct,
        "source_tokens": st, "encoded_tokens": et,
        "token_reduction_pct": (st-et)/st*100 if st else 0.0,
        "exact": verdict.exact, "field_mismatches": verdict.field_mismatches,
        "parameter_mismatches": verdict.parameter_mismatches,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", type=int, default=list(SIZES))
    ap.add_argument("--output", default="glyphin_simulation17_result.json")
    args = ap.parse_args()
    tokenizer = tiktoken.get_encoding(TOKENIZER_NAME)
    cases = []
    for n in args.sizes:
        memory = build_memory(n)
        for name, (enc, dec) in VARIANTS.items():
            cases.append(evaluate(name, enc, dec, memory, tokenizer))
    exact = [c for c in cases if c["exact"]]
    token_reductions = [c["token_reduction_pct"] for c in exact]
    result = {
        "benchmark_version": VERSION, "seed": SEED, "tokenizer": TOKENIZER_NAME,
        "sizes": args.sizes, "variants": list(VARIANTS), "total_cases": len(cases),
        "exact_cases": len(exact), "exact_rate_pct": 100.0*len(exact)/len(cases) if cases else 0.0,
        "exact_token_mean_pct": statistics.mean(token_reductions) if token_reductions else 0.0,
        "exact_token_median_pct": statistics.median(token_reductions) if token_reductions else 0.0,
        "cases": cases,
    }
    raw = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["data_sha256"] = hashlib.sha256(raw).hexdigest()
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, sort_keys=True); fh.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if len(exact) == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
