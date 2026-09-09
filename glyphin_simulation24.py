"""Glyphin Simulation 24.0 — scaling law / break-even benchmark.

Extends the Sim21 high-entropy fixture family from 256 states to 8192 states.
The benchmark measures whether structural factorization retains its token
advantage as memory size grows. Exact state reconstruction is required before
compression results are considered valid. This is benchmark-specific evidence,
not a global optimum claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time

import tiktoken

from glyphin_simulation21 import build_memory, SEED
from glyphin_simulation17 import encode_compact, decode_compact
from glyphin_simulation18 import encode_structural, decode_structural
from glyphin_simulation20 import encode_columnar, decode_columnar
from glyphin_state_referee import referee_memory

VERSION = "24.0"
TOKENIZER = "cl100k_base"
SIZES = (256, 512, 1024, 2048, 4096, 8192)
VARIANTS = {
    "sim17-compact": (encode_compact, decode_compact),
    "structural-lineage": (encode_structural, decode_structural),
    "state-columnar": (encode_columnar, decode_columnar),
}


def evaluate(size: int, name: str, encoder, decoder, memory, tokenizer, baseline_tokens: int) -> dict:
    t0 = time.perf_counter()
    encoded = encoder(memory)
    encode_ms = (time.perf_counter() - t0) * 1000.0
    t1 = time.perf_counter()
    rebuilt = decoder(encoded)
    decode_ms = (time.perf_counter() - t1) * 1000.0
    verdict = referee_memory(memory, rebuilt)
    encoded_tokens = len(tokenizer.encode(encoded, disallowed_special=()))
    source_json = memory.to_json()
    return {
        "size": size,
        "variant": name,
        "baseline_tokens": baseline_tokens,
        "encoded_tokens": encoded_tokens,
        "token_reduction_pct": (1.0 - encoded_tokens / baseline_tokens) * 100.0,
        "baseline_chars": len(source_json),
        "encoded_chars": len(encoded),
        "char_reduction_pct": (1.0 - len(encoded) / len(source_json)) * 100.0,
        "encode_ms": encode_ms,
        "decode_ms": decode_ms,
        "exact": verdict.exact,
        "missing_states": verdict.missing_states,
        "extra_states": verdict.extra_states,
        "field_mismatches": verdict.field_mismatches,
        "parameter_mismatches": verdict.parameter_mismatches,
    }


def regression_slope(points: list[tuple[float, float]]) -> float | None:
    if len(points) < 2:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    xm = statistics.mean(xs)
    ym = statistics.mean(ys)
    den = sum((x - xm) ** 2 for x in xs)
    return sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / den if den else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", type=int, default=list(SIZES))
    ap.add_argument("--output", default="glyphin_simulation24_result.json")
    args = ap.parse_args()
    tokenizer = tiktoken.get_encoding(TOKENIZER)
    cases = []

    for size in args.sizes:
        memory = build_memory(size, SEED + size)
        baseline = memory.to_json()
        baseline_tokens = len(tokenizer.encode(baseline, disallowed_special=()))
        for name, (encoder, decoder) in VARIANTS.items():
            cases.append(evaluate(size, name, encoder, decoder, memory, tokenizer, baseline_tokens))

    summary = {}
    for name in VARIANTS:
        rows = [r for r in cases if r["variant"] == name]
        reductions = [r["token_reduction_pct"] for r in rows if r["exact"]]
        summary[name] = {
            "exact_cases": sum(r["exact"] for r in rows),
            "total_cases": len(rows),
            "exact_rate_pct": 100.0 * sum(r["exact"] for r in rows) / len(rows),
            "mean_token_reduction_pct": statistics.mean(reductions) if reductions else None,
            "median_token_reduction_pct": statistics.median(reductions) if reductions else None,
            "token_reduction_slope_per_log2_state": regression_slope([
                (__import__("math").log2(r["size"]), r["token_reduction_pct"])
                for r in rows if r["exact"]
            ]),
        }

    compact = {r["size"]: r for r in cases if r["variant"] == "sim17-compact" and r["exact"]}
    structural = {r["size"]: r for r in cases if r["variant"] == "state-columnar" and r["exact"]}
    deltas = []
    for size in args.sizes:
        if size in compact and size in structural:
            deltas.append({
                "size": size,
                "state_columnar_advantage_pp": structural[size]["token_reduction_pct"] - compact[size]["token_reduction_pct"],
            })
    break_even = next((d["size"] for d in deltas if d["state_columnar_advantage_pp"] > 0), None)

    out = {
        "simulation": 24,
        "benchmark_version": VERSION,
        "purpose": "Scaling law and break-even analysis for exact structural state compression.",
        "fixture_family": "Sim21 high-entropy memory",
        "seed": SEED,
        "sizes": args.sizes,
        "tokenizer": TOKENIZER,
        "variants": list(VARIANTS),
        "cases": cases,
        "summary": summary,
        "state_columnar_vs_compact": deltas,
        "first_observed_break_even_size": break_even,
        "scope": "Benchmark-specific evidence only; no universal scaling or optimality claim.",
    }
    raw = json.dumps(out, sort_keys=True, separators=(",", ":")).encode()
    out["result_data_sha256"] = hashlib.sha256(raw).hexdigest()
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps(out["summary"], indent=2, sort_keys=True))
    print(json.dumps(out["state_columnar_vs_compact"], indent=2, sort_keys=True))
    return 0 if all(r["exact"] for r in cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
