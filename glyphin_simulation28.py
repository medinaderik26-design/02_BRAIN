"""Glyphin Simulation 28.0 — query-conditioned retrieval benchmark.

Tests the hypothesis that a query can be answered from only the structural
neighborhood required by that query, rather than from the entire compressed
memory. This is deterministic representation-level evidence only.
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

VERSION = "28.0"
SEEDS = (21092026, 31092026, 41092026, 51092026, 61092026)
SIZES = (256, 1024, 2048)
TOKENIZER = "cl100k_base"
VARIANTS = {
    "sim17-compact": (encode_compact, decode_compact),
    "structural-lineage": (encode_structural, decode_structural),
    "state-columnar": (encode_columnar, decode_columnar),
}
QUERY_TYPES = (
    "direct_attribute", "parent_lookup", "child_lookup", "multi_hop_traversal",
    "relationship_exists", "path_reconstruction", "temporal_ordering",
    "parameter_retrieval", "cross_state_comparison", "mixed_multi_hop",
)


def names(memory):
    return sorted(memory.states)


def query_spec(memory, q, i):
    ns = names(memory); n = len(ns)
    a = ns[i % n]; b = ns[(i + n // 3) % n]; c = ns[(i + 2*n // 3) % n]
    sa, sb, sc = memory.states[a], memory.states[b], memory.states[c]
    required = {a, b, c}
    if q == "direct_attribute": answer = {"name": a, "level": sa.level, "cohesion": sa.cohesion, "frequency": sa.frequency, "resonance": sa.resonance, "sigma": sa.sigma, "created_at": sa.created_at}
    elif q == "parent_lookup": answer = {"state": a, "parent": sa.parent}; required = {a} | ({sa.parent} if sa.parent else set())
    elif q == "child_lookup": answer = {"state": a, "children": sorted(sa.children)}; required = {a} | set(sa.children)
    elif q == "multi_hop_traversal": answer = {"state": a, "path": memory.get_path(a)}; required = set(memory.get_path(a))
    elif q == "relationship_exists": answer = {"a": a, "b": b, "a_parent_is_b": sa.parent == b, "b_parent_is_a": sb.parent == a}; required = {a,b}
    elif q == "path_reconstruction": answer = {"a": a, "b": b, "path_a": memory.get_path(a), "path_b": memory.get_path(b)}; required = set(memory.get_path(a)) | set(memory.get_path(b))
    elif q == "temporal_ordering":
        ordered = sorted((s.created_at, s.name) for s in (sa,sb,sc)); answer = {"states":[a,b,c],"chronological":[x[1] for x in ordered]}; required = {a,b,c}
    elif q == "parameter_retrieval": answer = {"decay_lambda":memory.decay_lambda,"alpha":memory.alpha,"beta":memory.beta}; required = set()
    elif q == "cross_state_comparison": answer = {"a":a,"b":b,"level_delta":sa.level-sb.level,"cohesion_delta":sa.cohesion-sb.cohesion,"frequency_delta":sa.frequency-sb.frequency,"same_parent":sa.parent==sb.parent}; required = {a,b}
    elif q == "mixed_multi_hop":
        answer = {"start":a,"start_parent":sa.parent,"start_path":memory.get_path(a),"parent_children":sorted(memory.states[sa.parent].children) if sa.parent else [],"compare_to":c,"same_parent":sa.parent==sc.parent}
        required = set(memory.get_path(a)) | ({sa.parent} if sa.parent else set()) | (set(memory.states[sa.parent].children) if sa.parent else set()) | {c}
    else: raise ValueError(q)
    return answer, required


def induced_memory(memory, required):
    """Materialize exactly the state nodes needed by the query plus their
    parent/child references. For parameter-only queries this is the empty set.
    """
    from glyphin_research_core import GlyphinMemory
    out = GlyphinMemory(decay_lambda=memory.decay_lambda, alpha=memory.alpha, beta=memory.beta)
    for name in sorted(required):
        if name in memory.states:
            s = memory.states[name]
            out.add_state(s.name, s.level, s.cohesion, s.parent, s.frequency, s.resonance, s.sigma, s.created_at)
    return out


def token_count(enc, text):
    return len(enc.encode(text, disallowed_special=()))


def evaluate(memory, variant, encoder, decoder, tok, q, i):
    encoded = encoder(memory); rebuilt = decoder(encoded)
    state_ref = referee_memory(memory, rebuilt)
    answer, required = query_spec(memory, q, i)
    subset = induced_memory(rebuilt, required)
    subset_encoded = encoder(subset)
    # Query execution is against the reconstructed subset; for parameter-only
    # queries the subset is empty but parameters are retained by induced_memory.
    got, _ = query_spec(subset, q, i) if required or q == "parameter_retrieval" else (None, set())
    answer_exact = got == answer
    full_cost = token_count(tok, encoded)
    conditioned_cost = token_count(tok, subset_encoded)
    query_cost = token_count(tok, f"Q{q}:{i}")
    full_with_query = full_cost + query_cost
    conditioned_with_query = conditioned_cost + query_cost
    retrieved_pct = 100.0 * len(required) / len(memory.states)
    return {
        "variant":variant,"query_type":q,"query_index":i,
        "state_exact":state_ref.exact,"answer_exact":answer_exact,
        "total_states":len(memory.states),"required_states":len(required),
        "retrieved_state_pct":retrieved_pct,
        "full_memory_tokens":full_cost,"conditioned_tokens":conditioned_cost,
        "full_input_tokens":full_with_query,"conditioned_input_tokens":conditioned_with_query,
        "tokens_saved_by_conditioning":full_with_query-conditioned_with_query,
        "conditioned_token_reduction_pct":100.0*(full_with_query-conditioned_with_query)/full_with_query if full_with_query else 0.0,
        "unnecessary_state_pct":100.0*(len(memory.states)-len(required))/len(memory.states),
        "multi_hop_recall_pct":100.0 if required.issubset(set(subset.states)) else 0.0,
    }


def summarize(rows):
    good=[r for r in rows if r["state_exact"] and r["answer_exact"]]
    def mean(k): return statistics.mean([r[k] for r in good]) if good else None
    return {"cases":len(rows),"state_exact_cases":sum(r["state_exact"] for r in rows),"answer_exact_cases":sum(r["answer_exact"] for r in rows),"utility_exact_rate_pct":100.0*sum(r["state_exact"] and r["answer_exact"] for r in rows)/len(rows),"mean_retrieved_state_pct":mean("retrieved_state_pct"),"mean_conditioned_token_reduction_pct":mean("conditioned_token_reduction_pct"),"median_conditioned_token_reduction_pct":statistics.median([r["conditioned_token_reduction_pct"] for r in good]) if good else None,"mean_unnecessary_state_pct":mean("unnecessary_state_pct"),"mean_multi_hop_recall_pct":mean("multi_hop_recall_pct")}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output",default="glyphin_simulation28_result.json"); args=ap.parse_args()
    tok=tiktoken.get_encoding(TOKENIZER); rows=[]
    for seed in SEEDS:
        for size in SIZES:
            mem=build_memory(size,seed)
            for variant,(enc,dec) in VARIANTS.items():
                for i,q in enumerate(QUERY_TYPES):
                    r=evaluate(mem,variant,enc,dec,tok,q,i); r.update(seed=seed,size=size); rows.append(r)
    out={"simulation":28,"benchmark_version":VERSION,"purpose":"Query-conditioned retrieval from reconstructed state.","fixture_family":"Sim21 high-entropy memory","seeds":list(SEEDS),"sizes":list(SIZES),"tokenizer":TOKENIZER,"query_types":list(QUERY_TYPES),"variants":list(VARIANTS),"total_cases":len(rows),"cases":rows,"summary":{v:summarize([r for r in rows if r["variant"]==v]) for v in VARIANTS},"by_query_type":{q:summarize([r for r in rows if r["query_type"]==q]) for q in QUERY_TYPES},"scope":"Deterministic retrieval/representation evidence only; no LLM semantic-equivalence, latency, universal-generalization, or optimality claim."}
    raw=json.dumps(out,sort_keys=True,separators=(",",":")).encode(); out["result_data_sha256"]=hashlib.sha256(raw).hexdigest()
    with open(args.output,"w",encoding="utf-8") as f: json.dump(out,f,indent=2,sort_keys=True); f.write("\n")
    print(json.dumps(out["summary"],indent=2,sort_keys=True)); return 0 if all(r["state_exact"] and r["answer_exact"] for r in rows) else 1

if __name__=="__main__": raise SystemExit(main())
