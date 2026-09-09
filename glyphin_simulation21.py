"""Simulation 21 — Adversarial State Entropy.

Stress-tests exact Glyphin memory encodings when redundancy is intentionally
reduced. The benchmark compares Sim17 compact, structural-lineage, and the
Sim20 state-columnar encoder using high-entropy state fields and irregular
parent structures. Results are independently refereed and tokenized with
cl100k_base.
"""
from __future__ import annotations
import argparse, hashlib, json, random, string
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tiktoken

from glyphin_research_core import GlyphinMemory
from glyphin_state_referee import referee_memory
from glyphin_simulation20 import encode_sim17_compact, encode_structural_lineage, encode_state_columnar

SEED = 21092026
SIZES = [4, 8, 16, 32, 64, 128, 256]
ENCODING = "cl100k_base"


def entropy_string(rng: random.Random, n: int = 17) -> str:
    alphabet = string.ascii_letters + string.digits + "_:/|;,@#%[]{}"
    return "".join(rng.choice(alphabet) for _ in range(n))


def build_memory(n: int, seed: int) -> GlyphinMemory:
    rng = random.Random(seed)
    mem = GlyphinMemory(decay_lambda=0.0137, alpha=0.271, beta=0.193)
    names = [f"{entropy_string(rng, 13)}-{i:04x}" for i in range(n)]
    base = datetime(2024, 1, 3, 9, 17, 11, 123457, tzinfo=timezone(timedelta(hours=-5)))
    for i, name in enumerate(names):
        parent = None
        if i:
            # Irregular but acyclic parent selection; deliberately avoid a
            # single chain and minimize repeated local structure.
            parent = names[rng.randrange(i)]
        mem.add_state(
            name=name,
            level=(rng.randrange(-37, 113) + rng.random()),
            cohesion=(rng.random() * 1.999983 - 0.999991),
            parent=parent,
            frequency=rng.randrange(1, 100003),
            resonance=(rng.random() * 200.0 - 100.0),
            sigma=entropy_string(rng, 23),
            created_at=base + timedelta(microseconds=rng.randrange(0, 10_000_000)),
        )
    return mem


def metrics(src: str, enc, baseline_tokens: int, exact: bool) -> dict:
    toks = len(enc.encode(src))
    chars = len(src)
    return {
        "chars": chars,
        "tokens": toks,
        "token_reduction_pct": (1 - toks / baseline_tokens) * 100.0,
        "exact": exact,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="glyphin_simulation21_result.json")
    args = ap.parse_args()
    enc = tiktoken.get_encoding(ENCODING)
    cases = []
    variants = ["sim17-compact", "structural-lineage", "state-columnar"]
    for n in SIZES:
        mem = build_memory(n, SEED + n)
        baseline = mem.to_json()
        base_tokens = len(enc.encode(baseline))
        encoded = {
            "sim17-compact": encode_sim17_compact(mem),
            "structural-lineage": encode_structural_lineage(mem),
            "state-columnar": encode_state_columnar(mem),
        }
        for variant in variants:
            text = encoded[variant]
            # Each existing decoder returns a fresh GlyphinMemory.
            if variant == "sim17-compact":
                from glyphin_simulation20 import decode_sim17_compact
                rebuilt = decode_sim17_compact(text)
            elif variant == "structural-lineage":
                from glyphin_simulation20 import decode_structural_lineage
                rebuilt = decode_structural_lineage(text)
            else:
                from glyphin_simulation20 import decode_state_columnar
                rebuilt = decode_state_columnar(text)
            ref = referee_memory(mem, rebuilt)
            m = metrics(text, enc, base_tokens, ref.exact)
            cases.append({"size": n, "variant": variant, "baseline_tokens": base_tokens, **m,
                          "missing_states": ref.missing_states, "extra_states": ref.extra_states,
                          "field_mismatches": ref.field_mismatches, "parameter_mismatches": ref.parameter_mismatches})
    summary = {}
    for v in variants:
        rows = [x for x in cases if x["variant"] == v]
        reds = [x["token_reduction_pct"] for x in rows]
        summary[v] = {
            "mean_token_reduction_pct": sum(reds) / len(reds),
            "median_token_reduction_pct": sorted(reds)[len(reds)//2],
            "exact_cases": sum(x["exact"] for x in rows),
            "total_cases": len(rows),
        }
    out = {
        "simulation": 21,
        "benchmark_version": "21.0",
        "purpose": "Adversarial high-entropy stress test of exact Glyphin memory encodings.",
        "seed": SEED,
        "sizes": SIZES,
        "tokenizer": ENCODING,
        "cases": cases,
        "summary": summary,
        "result_data_sha256": hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest(),
        "scope": "Benchmark-specific evidence only; not a global optimum or universal claim.",
    }
    Path(args.output).write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(out["summary"], indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
