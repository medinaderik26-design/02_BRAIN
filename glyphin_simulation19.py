"""Glyphin Simulation 19.0 — structural dictionary compression.

Compares the Sim18 structural-lineage representation with a generic dictionary
extension for repeated semantic string values. The dictionary is derived from
observed sigma values only when repetition can pay for its own encoded
overhead. State semantics, parameters, parent topology, and exact timestamps
remain referee-checked.
"""
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
from glyphin_simulation17 import encode_compact, decode_compact
from glyphin_simulation18 import encode_structural, decode_structural

VERSION = "19.0"
SEED = 19092026
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
            cur.extend((ESC, ch)) if preserve else cur.append(ch)
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


def _sigma_dictionary(memory: GlyphinMemory) -> list[str]:
    """Return repeated sigma values in deterministic lexical order.

    A dictionary entry must occur at least twice. This is intentionally
    generic: no fixture-specific names, counts, or fingerprints are used.
    """
    counts: dict[str, int] = {}
    for state in memory.states.values():
        counts[state.sigma] = counts.get(state.sigma, 0) + 1
    return sorted(value for value, count in counts.items() if count >= 2)


def encode_structural_dictionary(memory: GlyphinMemory) -> str:
    """Encode Sim18 structural state plus a generic repeated-sigma dictionary."""
    ordered = sorted(memory.states)
    index = {name: i for i, name in enumerate(ordered)}
    names = ",".join(esc(name, {",", ";", "|", "~", "@", ":"}) for name in ordered)
    timestamps = [datetime.fromisoformat(memory.states[name].created_at) for name in ordered]
    if any(ts.utcoffset() is None for ts in timestamps):
        raise ValueError("structural timestamp requires timezone-aware ISO timestamps")
    offsets = [int(ts.utcoffset().total_seconds() // 60) for ts in timestamps]
    utc_us = [int(ts.astimezone(timezone.utc).timestamp() * 1_000_000) for ts in timestamps]
    base_us = min(utc_us)
    deltas = [u - base_us for u in utc_us]

    dictionary = _sigma_dictionary(memory)
    dictionary_index = {value: i for i, value in enumerate(dictionary)}
    dict_text = ",".join(esc(value, {",", ";", "|", "~", "@", ":"}) for value in dictionary)
    records = [f"{memory.decay_lambda!r},{memory.alpha!r},{memory.beta!r}",
               names,
               f"{base_us}:{','.join(map(str, offsets))}",
               ",".join(map(str, deltas)),
               dict_text]
    for name in ordered:
        s = memory.states[name]
        parent = "" if s.parent is None else str(index[s.parent])
        if s.sigma in dictionary_index:
            sigma = "@" + str(dictionary_index[s.sigma])
        else:
            sigma = "~" + esc(s.sigma, {"|", ";", ",", "~", "@", ":"})
        records.append("|".join((str(index[name]), str(s.level), repr(s.cohesion), parent,
                                  str(s.frequency), repr(s.resonance), sigma)))
    return ";".join(records)


def decode_structural_dictionary(text: str) -> GlyphinMemory:
    records = split_escaped(text, ";", True)
    if len(records) < 5:
        raise ValueError("dictionary structural record is incomplete")
    p = split_escaped(records[0], ",")
    if len(p) != 3:
        raise ValueError("invalid parameter record")
    memory = GlyphinMemory(decay_lambda=float(p[0]), alpha=float(p[1]), beta=float(p[2]))
    names = split_escaped(records[1], ",")
    base_s, offsets_s = records[2].split(":", 1)
    base_us = int(base_s)
    offsets = [int(x) for x in offsets_s.split(",") if x != ""]
    deltas = [int(x) for x in records[3].split(",") if x != ""]
    dictionary = split_escaped(records[4], ",") if records[4] else []
    if len(names) != len(offsets) or len(names) != len(deltas):
        raise ValueError("name/timestamp vector length mismatch")

    rows = []
    for r in records[5:]:
        f = split_escaped(r, "|")
        if len(f) != 7:
            raise ValueError("dictionary state record must contain seven fields")
        idx, level, cohesion, parent_idx, freq, resonance, sigma_code = f
        i = int(idx)
        if i < 0 or i >= len(names):
            raise ValueError("state name index out of range")
        parent = None if parent_idx == "" else names[int(parent_idx)]
        if sigma_code.startswith("@"):
            di = int(sigma_code[1:])
            if di < 0 or di >= len(dictionary):
                raise ValueError("dictionary index out of range")
            sigma = dictionary[di]
        elif sigma_code.startswith("~"):
            sigma = sigma_code[1:]
        else:
            raise ValueError("invalid sigma dictionary code")
        utc = datetime.fromtimestamp((base_us + deltas[i]) / 1_000_000, tz=timezone.utc)
        offset = timezone(timedelta(minutes=offsets[i]))
        created_at = utc.astimezone(offset).isoformat()
        rows.append((i, names[i], int(level), float(cohesion), parent, int(freq),
                     float(resonance), sigma, created_at))
    if len(rows) != len(names):
        raise ValueError("state record count does not match name table")
    remaining = {row[0]: row for row in rows}
    while remaining:
        progress = False
        for i in sorted(tuple(remaining)):
            row = remaining[i]
            _, name, level, cohesion, parent, freq, resonance, sigma, created_at = row
            if parent is None or parent in memory.states:
                memory.add_state(name, level=level, cohesion=cohesion, parent=parent,
                                 frequency=freq, resonance=resonance, sigma=sigma,
                                 created_at=created_at)
                del remaining[i]
                progress = True
        if not progress:
            raise ValueError("unresolvable parent relationships")
    return memory


def evaluate(name: str, encode, decode, memory: GlyphinMemory, tokenizer) -> dict:
    encoded = encode(memory)
    reconstructed = decode(encoded)
    verdict = referee_memory(memory, reconstructed)
    source = memory.to_json()
    metrics = measure(source, encoded)
    st = len(tokenizer.encode(source, disallowed_special=()))
    et = len(tokenizer.encode(encoded, disallowed_special=()))
    dictionary = _sigma_dictionary(memory) if name == "structural-dictionary" else []
    return {
        "variant": name, "states": len(memory.states),
        "source_chars": metrics.source_chars, "encoded_chars": metrics.encoded_chars,
        "char_reduction_pct": metrics.char_reduction_pct,
        "source_tokens": st, "encoded_tokens": et,
        "token_reduction_pct": (st-et)/st*100 if st else 0.0,
        "dictionary_entries": len(dictionary),
        "dictionary_values": dictionary,
        "exact": verdict.exact, "field_mismatches": verdict.field_mismatches,
        "parameter_mismatches": verdict.parameter_mismatches,
    }


VARIANTS = {
    "sim17-compact": (encode_compact, decode_compact),
    "structural-lineage": (encode_structural, decode_structural),
    "structural-dictionary": (encode_structural_dictionary, decode_structural_dictionary),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", type=int, default=list(SIZES))
    ap.add_argument("--output", default="glyphin_simulation19_result.json")
    args = ap.parse_args()
    tokenizer = tiktoken.get_encoding(TOKENIZER_NAME)
    cases = []
    for n in args.sizes:
        memory = build_memory(n)
        for name, (enc, dec) in VARIANTS.items():
            cases.append(evaluate(name, enc, dec, memory, tokenizer))
    exact = [c for c in cases if c["exact"]]
    reductions = [c["token_reduction_pct"] for c in exact]
    result = {
        "benchmark_version": VERSION, "seed": SEED, "tokenizer": TOKENIZER_NAME,
        "sizes": args.sizes, "variants": list(VARIANTS), "total_cases": len(cases),
        "exact_cases": len(exact), "exact_rate_pct": 100.0*len(exact)/len(cases) if cases else 0.0,
        "exact_token_mean_pct": statistics.mean(reductions) if reductions else 0.0,
        "exact_token_median_pct": statistics.median(reductions) if reductions else 0.0,
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
