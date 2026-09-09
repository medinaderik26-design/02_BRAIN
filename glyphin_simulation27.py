"""Glyphin Simulation 27.0 — downstream memory utility benchmark.

Tests whether queries over compressed/reconstructed state return the same
answers as queries over the original state. This is a deterministic
representation-level utility test, not an LLM semantic-equivalence test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics

import tiktoken

from glyphin_simulation21 import build_memory
from glyphin_simulation17 import encode_compact, decode_compact
from glyphin_simulation18 import encode_structural, decode_structural
from glyphin_simulation20 import encode_columnar, decode_columnar
from glyphin_state_referee import referee_memory

VERSION = "27.0"
SEEDS = (21092026, 31092026, 41092026, 51092026, 61092026)
SIZES = (256, 1024, 2048)
TOKENIZER = "cl100k_base"
VARIANTS = {
    "sim17-compact": (encode_compact, decode_compact),
    "structural-lineage": (encode_structural, decode_structural),
    "state-columnar": (encode_columnar, decode_columnar),
}
QUERY_TYPES = (
    "direct_attribute",
    "parent_lookup",
    "child_lookup",
    "multi_hop_traversal",
    "relationship_exists",
    "path_reconstruction",
    "temporal_ordering",
    "parameter_retrieval",
    "cross_state_comparison",
    "mixed_multi_hop",
)


def state_names(memory):
    return sorted(memory.states)


def answer_query(memory, query_type, i):
    names = state_names(memory)
    n = len(names)
    a = names[i % n]
    b = names[(i + n // 3) % n]
    c = names[(i + 2 * n // 3) % n]
    sa, sb, sc = memory.states[a], memory.states[b], memory.states[c]

    if query_type == "direct_attribute":
        return {"name": a, "level": sa.level, "cohesion": sa.cohesion,
                "frequency": sa.frequency, "resonance": sa.resonance,
                "sigma": sa.sigma, "created_at": sa.created_at}
    if query_type == "parent_lookup":
        return {"state": a, "parent": sa.parent}
    if query_type == "child_lookup":
        return {"state": a, "children": sorted(sa.children)}
    if query_type == "multi_hop_traversal":
        path = memory.get_path(a)
        return {"state": a, "path": path}
    if query_type == "relationship_exists":
        return {"a": a, "b": b, "a_parent_is_b": sa.parent == b,
                "b_parent_is_a": sb.parent == a}
    if query_type == "path_reconstruction":
        return {"a": a, "b": b, "path_a": memory.get_path(a), "path_b": memory.get_path(b)}
    if query_type == "temporal_ordering":
        ordered = sorted((s.created_at, s.name) for s in (sa, sb, sc))
        return {"states": [a, b, c], "chronological": [name for _, name in ordered]}
    if query_type == "parameter_retrieval":
        return {"decay_lambda": memory.decay_lambda, "alpha": memory.alpha, "beta": memory.beta}
    if query_type == "cross_state_comparison":
        return {"a": a, "b": b,
                "level_delta": sa.level - sb.level,
                "cohesion_delta": sa.cohesion - sb.cohesion,
                "frequency_delta": sa.frequency - sb.frequency,
                "same_parent": sa.parent == sb.parent}
    if query_type == "mixed_multi_hop":
        return {
            "start": a,
            "start_parent": sa.parent,
            "start_path": memory.get_path(a),
            "parent_children": sorted(memory.states[sa.parent].children) if sa.parent else [],
            "compare_to": c,
            "same_parent": sa.parent == sc.parent,
        }
    raise ValueError(query_type)


def query_text(query_type, i):
    return f"Q{query_type}:{i}"


def token_count(tokenizer, text):
    return len(tokenizer.encode(text, disallowed_special=()))


def evaluate(memory, variant, encoder, decoder, tokenizer, query_type, i):
    original = memory.to_json()
    encoded = encoder(memory)
    rebuilt = decoder(encoded)
    referee = referee_memory(memory, rebuilt)
    original_answer = answer_query(memory, query_type, i)
    rebuilt_answer = answer_query(rebuilt, query_type, i)
    answer_exact = original_answer == rebuilt_answer

    original_input_tokens = token_count(tokenizer, original + "\n" + query_text(query_type, i))
    compressed_input_tokens = token_count(tokenizer, encoded + "\n" + query_text(query_type, i))
    saved = original_input_tokens - compressed_input_tokens
    reduction = 100.0 * saved / original_input_tokens
    return {
        "variant": variant,
        "query_type": query_type,
        "query_index": i,
        "original_input_tokens": original_input_tokens,
        "compressed_input_tokens": compressed_input_tokens,
        "tokens_saved": saved,
        "token_reduction_pct": reduction,
        "state_exact": referee.exact,
        "answer_exact": answer_exact,
        "original_answer": original_answer,
        "compressed_answer": rebuilt_answer,
    }


def summarize(rows):
    exact_rows = [r for r in rows if r["answer_exact"] and r["state_exact"]]
    reductions = [r["token_reduction_pct"] for r in exact_rows]
    saved = [r["tokens_saved"] for r in exact_rows]
    return {
        "cases": len(rows),
        "state_exact_cases": sum(r["state_exact"] for r in rows),
        "answer_exact_cases": sum(r["answer_exact"] for r in rows),
        "utility_exact_rate_pct": 100.0 * sum(r["answer_exact"] and r["state_exact"] for r in rows) / len(rows),
        "mean_token_reduction_pct": statistics.mean(reductions) if reductions else None,
        "median_token_reduction_pct": statistics.median(reductions) if reductions else None,
        "stdev_token_reduction_pct": statistics.stdev(reductions) if len(reductions) > 1 else 0.0,
        "mean_tokens_saved": statistics.mean(saved) if saved else None,
        "min_token_reduction_pct": min(reductions) if reductions else None,
        "max_token_reduction_pct": max(reductions) if reductions else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="glyphin_simulation27_result.json")
    args = ap.parse_args()
    tokenizer = tiktoken.get_encoding(TOKENIZER)
    cases = []
    for seed in SEEDS:
        for size in SIZES:
            memory = build_memory(size, seed)
            for variant, (encoder, decoder) in VARIANTS.items():
                for qidx, query_type in enumerate(QUERY_TYPES):
                    row = evaluate(memory, variant, encoder, decoder, tokenizer, query_type, qidx)
                    row.update({"seed": seed, "size": size})
                    cases.append(row)

    summary = {v: summarize([r for r in cases if r["variant"] == v]) for v in VARIANTS}
    by_query = {q: summarize([r for r in cases if r["query_type"] == q]) for q in QUERY_TYPES}
    by_size = {str(s): {v: summarize([r for r in cases if r["size"] == s and r["variant"] == v]) for v in VARIANTS} for s in SIZES}
    failures = [r for r in cases if not (r["state_exact"] and r["answer_exact"])]

    out = {
        "simulation": 27,
        "benchmark_version": VERSION,
        "purpose": "Downstream deterministic query utility after compression/reconstruction.",
        "fixture_family": "Sim21 high-entropy memory",
        "seeds": list(SEEDS),
        "sizes": list(SIZES),
        "tokenizer": TOKENIZER,
        "query_types": list(QUERY_TYPES),
        "variants": list(VARIANTS),
        "total_cases": len(cases),
        "cases": cases,
        "summary": summary,
        "by_query_type": by_query,
        "by_size": by_size,
        "failures": failures,
        "scope": "Representation-level deterministic query evidence only; no LLM semantic-equivalence or universal-generalization claim.",
    }
    raw = json.dumps(out, sort_keys=True, separators=(",", ":")).encode()
    out["result_data_sha256"] = hashlib.sha256(raw).hexdigest()
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps(out["summary"], indent=2, sort_keys=True))
    print("failures:", len(failures))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
