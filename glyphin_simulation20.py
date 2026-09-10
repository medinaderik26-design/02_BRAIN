"""Glyphin Simulation 20.0 — state-field factorization.

Tests whether columnar factorization of dynamic state fields adds compression
beyond Sim18 structural lineage. Every state field remains represented and is
checked by the independent state referee. This is a benchmark, not a claim of
optimality.
"""
from __future__ import annotations

import argparse, hashlib, json, statistics
from datetime import datetime, timedelta, timezone
import tiktoken

from glyphin_compression import measure
from glyphin_research_core import GlyphinMemory
from glyphin_state_referee import referee_memory
from glyphin_simulation17 import encode_compact, decode_compact
from glyphin_simulation18 import encode_structural, decode_structural
from glyphin_simulation19 import build_memory, split_escaped, esc

VERSION = "20.0"
SEED = 20092026
SIZES = (4, 8, 16, 32, 64, 128, 256)
TOKENIZER_NAME = "cl100k_base"


def encode_columnar(memory: GlyphinMemory) -> str:
    ordered = sorted(memory.states)
    index = {name: i for i, name in enumerate(ordered)}
    names = ",".join(esc(n, {",", ";", "|", ":", "~", "@"}) for n in ordered)
    parents = ",".join("" if memory.states[n].parent is None else str(index[memory.states[n].parent]) for n in ordered)
    levels = ",".join(str(memory.states[n].level) for n in ordered)
    cohesions = ",".join(repr(memory.states[n].cohesion) for n in ordered)
    freqs = ",".join(str(memory.states[n].frequency) for n in ordered)
    resonances = ",".join(repr(memory.states[n].resonance) for n in ordered)
    sigmas = ",".join(esc(memory.states[n].sigma, {",", ";", "|", ":", "~", "@"}) for n in ordered)
    timestamps = [datetime.fromisoformat(memory.states[n].created_at) for n in ordered]
    if any(t.utcoffset() is None for t in timestamps):
        raise ValueError("timestamps must be timezone-aware")
    offsets = [int(t.utcoffset().total_seconds() // 60) for t in timestamps]
    utc_us = [int(t.astimezone(timezone.utc).timestamp()) * 1_000_000 + t.astimezone(timezone.utc).microsecond for t in timestamps]
    base_us = min(utc_us)
    deltas = [u - base_us for u in utc_us]
    return ";".join([
        f"{memory.decay_lambda!r},{memory.alpha!r},{memory.beta!r}", names, parents,
        levels, cohesions, freqs, resonances, sigmas,
        f"{base_us}:{','.join(map(str, offsets))}", ",".join(map(str, deltas))
    ])


def decode_columnar(text: str) -> GlyphinMemory:
    r = split_escaped(text, ";", True)
    if len(r) != 10:
        raise ValueError("columnar record must contain ten sections")
    p = split_escaped(r[0], ",")
    if len(p) != 3: raise ValueError("invalid parameters")
    memory = GlyphinMemory(decay_lambda=float(p[0]), alpha=float(p[1]), beta=float(p[2]))
    names = split_escaped(r[1], ",") if r[1] else []
    parents = r[2].split(",") if r[2] else []
    levels = [int(x) for x in r[3].split(",")] if r[3] else []
    cohesions = [float(x) for x in r[4].split(",")] if r[4] else []
    freqs = [int(x) for x in r[5].split(",")] if r[5] else []
    resonances = [float(x) for x in r[6].split(",")] if r[6] else []
    sigmas = split_escaped(r[7], ",") if r[7] else []
    base_s, offsets_s = r[8].split(":", 1)
    base_us = int(base_s); offsets = [int(x) for x in offsets_s.split(",") if x]
    # Preserve zero deltas: the minimum timestamp is legitimately encoded as 0.
    deltas = [int(x) for x in r[9].split(",")] if r[9] else []
    vectors = (parents, levels, cohesions, freqs, resonances, sigmas, offsets, deltas)
    if any(len(v) != len(names) for v in vectors): raise ValueError("column lengths differ")
    pending = set(range(len(names)))
    while pending:
        progress = False
        for i in sorted(pending):
            parent_idx = parents[i]
            if parent_idx and (int(parent_idx) in pending):
                continue
            parent = None if parent_idx == "" else names[int(parent_idx)]
            utc = datetime.fromtimestamp(base_us // 1_000_000, tz=timezone.utc) + timedelta(microseconds=(base_us % 1_000_000) + deltas[i])
            created_at = utc.astimezone(timezone(timedelta(minutes=offsets[i]))).isoformat()
            memory.add_state(names[i], level=levels[i], cohesion=cohesions[i], parent=parent,
                             frequency=freqs[i], resonance=resonances[i], sigma=sigmas[i], created_at=created_at)
            pending.remove(i); progress = True
        if not progress: raise ValueError("unresolvable parent graph")
    return memory

VARIANTS = {
    "sim17-compact": (encode_compact, decode_compact),
    "structural-lineage": (encode_structural, decode_structural),
    "state-columnar": (encode_columnar, decode_columnar),
}


def evaluate(name, enc, dec, memory, tokenizer):
    encoded = enc(memory); reconstructed = dec(encoded)
    verdict = referee_memory(memory, reconstructed)
    source = memory.to_json(); m = measure(source, encoded)
    st = len(tokenizer.encode(source, disallowed_special=()))
    et = len(tokenizer.encode(encoded, disallowed_special=()))
    return {"variant": name, "states": len(memory.states), "source_chars": m.source_chars,
            "encoded_chars": m.encoded_chars, "char_reduction_pct": m.char_reduction_pct,
            "source_tokens": st, "encoded_tokens": et,
            "token_reduction_pct": (1 - et / st) * 100.0,
            "exact": verdict.exact, "missing_states": verdict.missing_states,
            "extra_states": verdict.extra_states, "field_mismatches": verdict.field_mismatches,
            "parameter_mismatches": verdict.parameter_mismatches}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--output", default="glyphin_simulation20_result.json"); args = ap.parse_args()
    tokenizer = tiktoken.get_encoding(TOKENIZER_NAME); cases=[]
    for size in SIZES:
        mem = build_memory(size, SEED + size)
        for name,(enc,dec) in VARIANTS.items(): cases.append(evaluate(name,enc,dec,mem,tokenizer))
    summary={}
    for name in VARIANTS:
        rows=[r for r in cases if r["variant"]==name]
        reds=[r["token_reduction_pct"] for r in rows]
        summary[name]={"mean_token_reduction_pct":sum(reds)/len(reds),"median_token_reduction_pct":statistics.median(reds),"exact_cases":sum(r["exact"] for r in rows),"total_cases":len(rows)}
    out={"simulation":20,"benchmark_version":VERSION,"seed":SEED,"sizes":SIZES,"tokenizer":TOKENIZER_NAME,"variants":list(VARIANTS),"cases":cases,"summary":summary,"result_data_sha256":hashlib.sha256(json.dumps(cases,sort_keys=True).encode()).hexdigest(),"scope":"Benchmark-specific evidence only; not a global optimum or universal claim."}
    Path(args.output).write_text(json.dumps(out,indent=2,sort_keys=True),encoding="utf-8"); print(json.dumps(summary,indent=2,sort_keys=True))

if __name__ == "__main__": main()
