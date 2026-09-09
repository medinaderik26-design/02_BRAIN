"""Glyphin Simulation 26.0 — seed × tokenizer interaction benchmark.

Tests whether the Sim25 cross-seed compression pattern survives simultaneous
changes in fixture seed and tokenizer. Exact state reconstruction is required.
This is benchmark-specific evidence, not a universal optimality claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time

import tiktoken

from glyphin_simulation21 import build_memory
from glyphin_simulation17 import encode_compact, decode_compact
from glyphin_simulation18 import encode_structural, decode_structural
from glyphin_simulation20 import encode_columnar, decode_columnar
from glyphin_state_referee import referee_memory

VERSION = "26.0"
SEEDS = (21092026, 31092026, 41092026, 51092026, 61092026)
SIZES = (256, 1024, 2048)
TOKENIZERS = ("cl100k_base", "o200k_base", "o200k_harmony")
VARIANTS = {
    "sim17-compact": (encode_compact, decode_compact),
    "structural-lineage": (encode_structural, decode_structural),
    "state-columnar": (encode_columnar, decode_columnar),
}


def evaluate(seed, size, tokenizer_name, variant, encoder, decoder, memory, tokenizer, baseline_tokens):
    t0 = time.perf_counter()
    encoded = encoder(memory)
    encode_ms = (time.perf_counter() - t0) * 1000.0
    t1 = time.perf_counter()
    rebuilt = decoder(encoded)
    decode_ms = (time.perf_counter() - t1) * 1000.0
    verdict = referee_memory(memory, rebuilt)
    encoded_tokens = len(tokenizer.encode(encoded, disallowed_special=()))
    return {
        "seed": seed, "size": size, "tokenizer": tokenizer_name, "variant": variant,
        "baseline_tokens": baseline_tokens, "encoded_tokens": encoded_tokens,
        "token_reduction_pct": (1.0 - encoded_tokens / baseline_tokens) * 100.0,
        "baseline_chars": len(memory.to_json()), "encoded_chars": len(encoded),
        "char_reduction_pct": (1.0 - len(encoded) / len(memory.to_json())) * 100.0,
        "encode_ms": encode_ms, "decode_ms": decode_ms, "exact": verdict.exact,
        "missing_states": verdict.missing_states, "extra_states": verdict.extra_states,
        "field_mismatches": verdict.field_mismatches,
        "parameter_mismatches": verdict.parameter_mismatches,
    }


def stats(rows):
    vals = [r["token_reduction_pct"] for r in rows if r["exact"]]
    return {"exact_cases": sum(r["exact"] for r in rows), "total_cases": len(rows),
            "exact_rate_pct": 100.0 * sum(r["exact"] for r in rows) / len(rows),
            "mean_token_reduction_pct": statistics.mean(vals) if vals else None,
            "median_token_reduction_pct": statistics.median(vals) if vals else None,
            "stdev_token_reduction_pct": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "min_token_reduction_pct": min(vals) if vals else None,
            "max_token_reduction_pct": max(vals) if vals else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="glyphin_simulation26_result.json")
    args = ap.parse_args()
    tokenizers = {name: tiktoken.get_encoding(name) for name in TOKENIZERS}
    cases = []
    for seed in SEEDS:
        for size in SIZES:
            memory = build_memory(size, seed)
            source = memory.to_json()
            for tokenizer_name, tokenizer in tokenizers.items():
                baseline_tokens = len(tokenizer.encode(source, disallowed_special=()))
                for variant, (encoder, decoder) in VARIANTS.items():
                    cases.append(evaluate(seed, size, tokenizer_name, variant, encoder, decoder, memory, tokenizer, baseline_tokens))

    summary = {v: {t: stats([r for r in cases if r["variant"] == v and r["tokenizer"] == t]) for t in TOKENIZERS} for v in VARIANTS}
    by_size = {str(s): {v: {t: stats([r for r in cases if r["size"] == s and r["variant"] == v and r["tokenizer"] == t]) for t in TOKENIZERS} for v in VARIANTS} for s in SIZES}
    compact = {(r["seed"], r["size"], r["tokenizer"]): r for r in cases if r["variant"] == "sim17-compact" and r["exact"]}
    columnar = [r for r in cases if r["variant"] == "state-columnar" and r["exact"]]
    paired = [r["token_reduction_pct"] - compact[(r["seed"], r["size"], r["tokenizer"])] ["token_reduction_pct"] for r in columnar]
    advantage_by_tokenizer = {t: [r for r in paired] for t in TOKENIZERS}
    # Rebuild the tokenizer-specific paired vectors deterministically from the case rows.
    advantage_by_tokenizer = {}
    for t in TOKENIZERS:
        vals = []
        for r in columnar:
            if r["tokenizer"] == t:
                vals.append(r["token_reduction_pct"] - compact[(r["seed"], r["size"], t)]["token_reduction_pct"])
        advantage_by_tokenizer[t] = {"mean_pp": statistics.mean(vals), "median_pp": statistics.median(vals), "stdev_pp": statistics.stdev(vals) if len(vals) > 1 else 0.0, "min_pp": min(vals), "max_pp": max(vals)}

    out = {"simulation": 26, "benchmark_version": VERSION,
           "purpose": "Seed × tokenizer interaction for exact structural state compression.",
           "fixture_family": "Sim21 high-entropy memory", "seeds": list(SEEDS), "sizes": list(SIZES),
           "tokenizers": list(TOKENIZERS), "variants": list(VARIANTS), "cases": cases,
           "summary": summary, "by_size": by_size,
           "paired_state_columnar_advantage_over_compact_pp": {
               "mean_pp": statistics.mean(paired), "median_pp": statistics.median(paired),
               "stdev_pp": statistics.stdev(paired), "min_pp": min(paired), "max_pp": max(paired)},
           "advantage_by_tokenizer": advantage_by_tokenizer,
           "scope": "Benchmark-specific evidence only; no universal generalization, scaling, or optimality claim."}
    raw = json.dumps(out, sort_keys=True, separators=(",", ":")).encode()
    out["result_data_sha256"] = hashlib.sha256(raw).hexdigest()
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, sort_keys=True); fh.write("\n")
    print(json.dumps(out["summary"], indent=2, sort_keys=True))
    print(json.dumps(out["paired_state_columnar_advantage_over_compact_pp"], indent=2, sort_keys=True))
    return 0 if all(r["exact"] for r in cases) else 1

if __name__ == "__main__":
    raise SystemExit(main())
