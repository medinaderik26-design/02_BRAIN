"""Glyphin Simulation 22 — Cross-Tokenizer Robustness.

Tests whether the compression advantage observed in Simulation 21 survives
across multiple tokenization regimes. The underlying high-entropy memories
are deterministic and identical across every tokenizer and representation.
No tokenizer-specific encoder logic is used.

Evidence scope: benchmark-specific only. This does not establish a global
optimum, universal compression ratio, or cross-model semantic equivalence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import tiktoken

from glyphin_simulation17 import encode_compact, decode_compact
from glyphin_simulation18 import encode_structural, decode_structural
from glyphin_simulation20 import encode_columnar, decode_columnar
from glyphin_simulation21 import build_memory, SEED as SIM21_SEED
from glyphin_state_referee import referee_memory

VERSION = "22.0"
SIZES = (4, 8, 16, 32, 64, 128, 256)
TOKENIZER_CANDIDATES = ("cl100k_base", "o200k_base", "o200k_harmony")
VARIANTS = {
    "sim17-compact": (encode_compact, decode_compact),
    "structural-lineage": (encode_structural, decode_structural),
    "state-columnar": (encode_columnar, decode_columnar),
}


def load_tokenizers() -> dict[str, object]:
    """Load all supported OpenAI tiktoken regimes available in the pinned package."""
    loaded = {}
    unavailable = {}
    for name in TOKENIZER_CANDIDATES:
        try:
            loaded[name] = tiktoken.get_encoding(name)
        except Exception as exc:  # package/version availability is an experimental result
            unavailable[name] = type(exc).__name__
    if not loaded:
        raise RuntimeError("No requested tiktoken encoding is available")
    return loaded


def count_tokens(encoding, text: str) -> int:
    return len(encoding.encode(text, disallowed_special=()))


def count_bytes(text: str) -> int:
    return len(text.encode("utf-8"))


def evaluate_case(size: int, memory, variant: str, encoder, decoder, regimes: dict[str, object]) -> dict:
    encoded = encoder(memory)
    rebuilt = decoder(encoded)
    verdict = referee_memory(memory, rebuilt)
    row = {
        "size": size,
        "variant": variant,
        "exact": verdict.exact,
        "missing_states": verdict.missing_states,
        "extra_states": verdict.extra_states,
        "field_mismatches": verdict.field_mismatches,
        "parameter_mismatches": verdict.parameter_mismatches,
        "chars": len(encoded),
        "bytes": count_bytes(encoded),
        "regimes": {},
    }
    baseline = memory.to_json()
    for name, tokenizer in regimes.items():
        source_n = count_tokens(tokenizer, baseline)
        encoded_n = count_tokens(tokenizer, encoded)
        row["regimes"][name] = {
            "baseline_tokens": source_n,
            "encoded_tokens": encoded_n,
            "token_reduction_pct": (1.0 - encoded_n / source_n) * 100.0 if source_n else 0.0,
        }
    source_bytes = count_bytes(baseline)
    row["regimes"]["utf8-bytes"] = {
        "baseline_tokens": source_bytes,
        "encoded_tokens": row["bytes"],
        "token_reduction_pct": (1.0 - row["bytes"] / source_bytes) * 100.0 if source_bytes else 0.0,
    }
    return row


def summarize(cases: list[dict], regime_names: list[str]) -> dict:
    out = {}
    for variant in VARIANTS:
        rows = [r for r in cases if r["variant"] == variant]
        per_regime = {}
        for regime in regime_names + ["utf8-bytes"]:
            vals = [r["regimes"][regime]["token_reduction_pct"] for r in rows if r["exact"]]
            per_regime[regime] = {
                "mean_token_reduction_pct": statistics.mean(vals) if vals else 0.0,
                "median_token_reduction_pct": statistics.median(vals) if vals else 0.0,
                "min_token_reduction_pct": min(vals) if vals else 0.0,
                "max_token_reduction_pct": max(vals) if vals else 0.0,
                "exact_cases": sum(r["exact"] for r in rows),
                "total_cases": len(rows),
            }
        out[variant] = per_regime
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="glyphin_simulation22_result.json")
    args = ap.parse_args()

    regimes = load_tokenizers()
    cases = []
    for size in SIZES:
        memory = build_memory(size, SIM21_SEED + size)
        for variant, (encoder, decoder) in VARIANTS.items():
            cases.append(evaluate_case(size, memory, variant, encoder, decoder, regimes))

    regime_names = list(regimes)
    summary = summarize(cases, regime_names)
    exact_cases = sum(r["exact"] for r in cases)
    out = {
        "simulation": 22,
        "benchmark_version": VERSION,
        "purpose": "Cross-tokenizer robustness of exact compression on the deterministic Sim21 high-entropy fixture family.",
        "fixture_source": "glyphin_simulation21.build_memory with SEED + size",
        "fixture_seed": SIM21_SEED,
        "sizes": list(SIZES),
        "tokenizer_regimes": regime_names,
        "byte_baseline": "utf8-bytes",
        "variants": list(VARIANTS),
        "total_cases": len(cases),
        "exact_cases": exact_cases,
        "exact_rate_pct": 100.0 * exact_cases / len(cases) if cases else 0.0,
        "summary": summary,
        "cases": cases,
        "scope": "Benchmark-specific evidence only; not a global optimum, universal claim, or proof of cross-model semantic equivalence.",
    }
    raw = json.dumps(out, sort_keys=True, separators=(",", ":")).encode()
    out["result_data_sha256"] = hashlib.sha256(raw).hexdigest()
    Path(args.output).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(out["summary"], indent=2, sort_keys=True))
    return 0 if exact_cases == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
