"""Simulation 34 — resolver error propagation boundary.

Separates an upstream concept resolver from Glyphin's structural retrieval,
induced-state reconstruction, and exact downstream answer. The resolver is
synthetic and its outcomes are explicitly labeled: exact, miss, wrong-unique,
or ambiguous. No semantic model is evaluated here.
"""
from __future__ import annotations
import argparse, hashlib, json, random
from pathlib import Path
import tiktoken
from glyphin_simulation21 import build_memory
from glyphin_research_core import GlyphinMemory
from glyphin_simulation17 import encode_compact, decode_compact
from glyphin_simulation18 import encode_structural, decode_structural
from glyphin_simulation20 import encode_columnar, decode_columnar

SEEDS = (21092026, 31092026, 41092026, 51092026, 61092026)
SIZES = (256, 1024, 2048)
TARGETS_PER_STATE = 64
ENCODING = "cl100k_base"
VARIANTS = {
    "sim17-compact": (encode_compact, decode_compact),
    "structural-lineage": (encode_structural, decode_structural),
    "state-columnar": (encode_columnar, decode_columnar),
}
OUTCOMES = ("exact", "miss", "wrong_unique", "ambiguous")
VERSION = "34.0"


def parent_closure(mem, anchors):
    wanted = set(anchors)
    stack = list(anchors)
    while stack:
        name = stack.pop()
        state = mem.states.get(name)
        if state is None or state.parent is None:
            continue
        if state.parent not in wanted:
            wanted.add(state.parent)
            stack.append(state.parent)
    return wanted


def induced_memory(mem, names):
    out = GlyphinMemory(decay_lambda=mem.decay_lambda, alpha=mem.alpha, beta=mem.beta)
    names = set(names)
    for name in sorted(names):
        s = mem.states[name]
        parent = s.parent if s.parent in names else None
        out.add_state(name=s.name, level=s.level, cohesion=s.cohesion, parent=parent,
                      frequency=s.frequency, resonance=s.resonance, sigma=s.sigma,
                      created_at=s.created_at)
    return out


def resolver_outcome(mem, target, outcome, rng):
    names = sorted(mem.states)
    if outcome == "exact":
        return {"status": "resolved", "anchors": [target], "candidate_count": 1}
    if outcome == "miss":
        return {"status": "miss", "anchors": [], "candidate_count": 0}
    if outcome == "wrong_unique":
        choices = [n for n in names if n != target]
        wrong = choices[rng.randrange(len(choices))]
        return {"status": "resolved", "anchors": [wrong], "candidate_count": 1}
    # Ambiguous deliberately exposes two candidates; downstream retrieval must reject.
    choices = [n for n in names if n != target]
    other = choices[rng.randrange(len(choices))]
    return {"status": "ambiguous", "anchors": [target, other], "candidate_count": 2}


def target_answer(mem, target):
    # Deterministic downstream answer: the target's own state record plus its
    # parent chain. This measures representation-level correctness only.
    names = parent_closure(mem, [target])
    return [(n, mem.states[n].parent) for n in sorted(names)]


def run_case(mem, target, outcome, rng, variant, tokenizer):
    resolver = resolver_outcome(mem, target, outcome, rng)
    if resolver["status"] != "resolved":
        selected = set()
        accepted = False
    else:
        selected = parent_closure(mem, resolver["anchors"])
        accepted = True
    truth = parent_closure(mem, [target])
    closure_exact = selected == truth
    answer_exact = accepted and closure_exact and target_answer(mem, target) == target_answer(induced_memory(mem, selected), target)

    encoder, decoder = VARIANTS[variant]
    if accepted:
        subset = induced_memory(mem, selected)
        encoded = encoder(subset)
        rebuilt = decoder(encoded)
        reconstructed_names = set(rebuilt.states)
        transport_exact = reconstructed_names == selected
        if target in rebuilt.states:
            rebuilt_answer = target_answer(rebuilt, target)
            transport_answer_exact = transport_exact and rebuilt_answer == target_answer(mem, target)
        else:
            transport_answer_exact = False
        full_tokens = len(tokenizer.encode(mem.to_json(), disallowed_special=()))
        selected_tokens = len(tokenizer.encode(encoded, disallowed_special=()))
        reduction = (1 - selected_tokens / full_tokens) * 100.0
    else:
        full_tokens = len(tokenizer.encode(mem.to_json(), disallowed_special=()))
        selected_tokens = 0
        reduction = None
        transport_exact = True
        transport_answer_exact = False

    return {
        "target": target, "resolver_outcome": outcome,
        "resolver_status": resolver["status"], "candidate_count": resolver["candidate_count"],
        "resolver_anchor_exact": resolver["anchors"] == [target],
        "accepted": accepted, "truth_closure_size": len(truth), "selected_closure_size": len(selected),
        "closure_exact": closure_exact, "answer_exact": answer_exact,
        "transport_exact": transport_exact, "transport_answer_exact": transport_answer_exact,
        "full_tokens": full_tokens, "selected_tokens": selected_tokens,
        "selected_token_reduction_pct": reduction,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="glyphin_simulation34_result.json")
    args = ap.parse_args()
    tokenizer = tiktoken.get_encoding(ENCODING)
    cases = []
    for seed in SEEDS:
        for size in SIZES:
            mem = build_memory(size, seed + size)
            targets = sorted(mem.states)[:TARGETS_PER_STATE]
            for ti, target in enumerate(targets):
                for outcome in OUTCOMES:
                    rng = random.Random(seed * 1000003 + size * 1009 + ti * 97 + OUTCOMES.index(outcome))
                    for variant in VARIANTS:
                        row = run_case(mem, target, outcome, rng, variant, tokenizer)
                        cases.append({"seed": seed, "size": size, "variant": variant, **row})

    summary = {}
    for variant in VARIANTS:
        summary[variant] = {}
        for outcome in OUTCOMES:
            rows = [r for r in cases if r["variant"] == variant and r["resolver_outcome"] == outcome]
            summary[variant][outcome] = {
                "cases": len(rows),
                "resolver_anchor_exact": sum(r["resolver_anchor_exact"] for r in rows),
                "accepted": sum(r["accepted"] for r in rows),
                "closure_exact": sum(r["closure_exact"] for r in rows),
                "answer_exact": sum(r["answer_exact"] for r in rows),
                "transport_exact": sum(r["transport_exact"] for r in rows),
                "transport_answer_exact": sum(r["transport_answer_exact"] for r in rows),
                "mean_selected_closure_size": sum(r["selected_closure_size"] for r in rows) / len(rows),
                "mean_selected_token_reduction_pct": (
                    sum(r["selected_token_reduction_pct"] for r in rows if r["selected_token_reduction_pct"] is not None)
                    / max(1, sum(r["selected_token_reduction_pct"] is not None for r in rows))
                ),
            }

    out = {
        "simulation": 34, "benchmark_version": VERSION, "seeds": SEEDS, "sizes": SIZES,
        "targets_per_state": TARGETS_PER_STATE, "outcomes": OUTCOMES,
        "tokenizer": ENCODING, "variants": list(VARIANTS), "cases": cases, "summary": summary,
        "result_data_sha256": hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest(),
        "scope": "Synthetic resolver-error propagation benchmark. Exact/miss/wrong-unique/ambiguous resolver outcomes are injected upstream; Glyphin structural closure, induced reconstruction, deterministic transport, and deterministic answer checks are then measured. This does not measure semantic resolver quality, natural-language understanding, learned retrieval, LLM equivalence, latency, universal generalization, consciousness, or optimality."
    }
    Path(args.output).write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
