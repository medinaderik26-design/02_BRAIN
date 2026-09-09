"""Glyphin Simulation 29.1: index-discovered query-conditioned retrieval."""
from __future__ import annotations
import argparse, hashlib, json, statistics
import tiktoken
from glyphin_simulation21 import build_memory
from glyphin_simulation17 import encode_compact, decode_compact
from glyphin_simulation18 import encode_structural, decode_structural
from glyphin_simulation20 import encode_columnar, decode_columnar
from glyphin_state_referee import referee_memory
from glyphin_research_core import GlyphinMemory
VERSION="29.1"; SEEDS=(21092026,31092026,41092026,51092026,61092026); SIZES=(256,1024,2048); BATCH_SIZES=(1,4,16,64,256); TOKENIZER="cl100k_base"
VARIANTS={"sim17-compact":(encode_compact,decode_compact),"structural-lineage":(encode_structural,decode_structural),"state-columnar":(encode_columnar,decode_columnar)}
QUERY_TYPES=("direct_attribute","parent_lookup","child_lookup","multi_hop_traversal","relationship_exists","path_reconstruction","temporal_ordering","parameter_retrieval","cross_state_comparison","mixed_multi_hop")

def names(memory): return sorted(memory.states)
def query_spec(memory,q,a,b,c):
    if q=="parameter_retrieval": return {"decay_lambda":memory.decay_lambda,"alpha":memory.alpha,"beta":memory.beta}
    sa=memory.states[a]
    if q=="direct_attribute": return {"name":a,"level":sa.level,"cohesion":sa.cohesion,"frequency":sa.frequency,"resonance":sa.resonance,"sigma":sa.sigma,"created_at":sa.created_at}
    if q=="parent_lookup": return {"state":a,"parent":sa.parent}
    if q=="child_lookup": return {"state":a,"children":sorted(sa.children)}
    if q=="multi_hop_traversal": return {"state":a,"path":memory.get_path(a)}
    sb=memory.states[b]
    if q=="relationship_exists": return {"a":a,"b":b,"a_parent_is_b":sa.parent==b,"b_parent_is_a":sb.parent==a}
    if q=="path_reconstruction": return {"a":a,"b":b,"path_a":memory.get_path(a),"path_b":memory.get_path(b)}
    if q=="temporal_ordering":
        sc=memory.states[c]; return {"states":[a,b,c],"chronological":[x[1] for x in sorted((s.created_at,s.name) for s in (sa,sb,sc))]}
    if q=="cross_state_comparison": return {"a":a,"b":b,"level_delta":sa.level-sb.level,"cohesion_delta":sa.cohesion-sb.cohesion,"frequency_delta":sa.frequency-sb.frequency,"same_parent":sa.parent==sb.parent}
    if q=="mixed_multi_hop":
        sc=memory.states[c]; return {"start":a,"start_parent":sa.parent,"start_path":memory.get_path(a),"parent_children":sorted(memory.states[sa.parent].children) if sa.parent else [],"compare_to":c,"same_parent":sa.parent==sc.parent}
    raise ValueError(q)

def build_index(memory): return {n:{"parent":memory.states[n].parent,"children":sorted(memory.states[n].children)} for n in names(memory)}
def ancestors_from_parent(parent_map,node):
    out=[]; seen=set(); cur=node
    while cur is not None and cur not in seen: seen.add(cur); out.append(cur); cur=parent_map[cur]
    return out

def ground_truth_closure(memory,q,a,b,c):
    if q=="parameter_retrieval": return set()
    p={n:memory.states[n].parent for n in names(memory)}; children={n:set(memory.states[n].children) for n in names(memory)}
    if q=="direct_attribute": return {a}
    if q=="parent_lookup": return {a}|({p[a]} if p[a] else set())
    if q=="child_lookup": return {a}|children[a]
    if q=="multi_hop_traversal": return set(ancestors_from_parent(p,a))
    if q=="relationship_exists": return {a,b}|({p[a]} if p[a] else set())|({p[b]} if p[b] else set())
    if q=="path_reconstruction": return set(ancestors_from_parent(p,a))|set(ancestors_from_parent(p,b))
    if q=="temporal_ordering": return {a,b,c}
    if q=="cross_state_comparison": return {a,b}|({p[a]} if p[a] else set())|({p[b]} if p[b] else set())
    if q=="mixed_multi_hop":
        out=set(ancestors_from_parent(p,a))|{c}; parent=p[a]
        if parent: out.add(parent); out.update(children[parent])
        if p[c]: out.add(p[c])
        return out
    raise ValueError(q)

def index_closure(index,q,a,b,c):
    if q=="parameter_retrieval": return set()
    p={n:index[n]["parent"] for n in index}
    if q=="direct_attribute": return {a}
    if q=="parent_lookup": return {a}|({index[a]["parent"]} if index[a]["parent"] else set())
    if q=="child_lookup": return {a}|set(index[a]["children"])
    if q=="multi_hop_traversal": return set(ancestors_from_parent(p,a))
    if q=="relationship_exists": return {a,b}|({index[a]["parent"]} if index[a]["parent"] else set())|({index[b]["parent"]} if index[b]["parent"] else set())
    if q=="path_reconstruction": return set(ancestors_from_parent(p,a))|set(ancestors_from_parent(p,b))
    if q=="temporal_ordering": return {a,b,c}
    if q=="cross_state_comparison": return {a,b}|({index[a]["parent"]} if index[a]["parent"] else set())|({index[b]["parent"]} if index[b]["parent"] else set())
    if q=="mixed_multi_hop":
        out=set(ancestors_from_parent(p,a))|{c}; parent=index[a]["parent"]
        if parent: out.add(parent); out.update(index[parent]["children"])
        if index[c]["parent"]: out.add(index[c]["parent"])
        return out
    raise ValueError(q)

def induced_memory(memory,required):
    out=GlyphinMemory(decay_lambda=memory.decay_lambda,alpha=memory.alpha,beta=memory.beta)
    for n in sorted(required):
        s=memory.states[n]; out.add_state(name=s.name,level=s.level,cohesion=s.cohesion,parent=None,frequency=s.frequency,resonance=s.resonance,sigma=s.sigma,created_at=s.created_at)
    for n in sorted(required):
        p=memory.states[n].parent
        if p in out.states: out.link_state(p,n)
    return out

def token_count(enc,text): return len(enc.encode(text,disallowed_special=()))
def query_payload(q,a,b,c): return json.dumps({"q":q,"a":a,"b":b,"c":c},sort_keys=True,separators=(",",":"))

def evaluate(memory,variant,encoder,decoder,tok,q,i,index):
    encoded=encoder(memory); rebuilt=decoder(encoded); state_ref=referee_memory(memory,rebuilt)
    ns=names(memory); n=len(ns); a,b,c=ns[i%n],ns[(i+n//3)%n],ns[(i+2*n//3)%n]
    answer=query_spec(memory,q,a,b,c); truth=ground_truth_closure(memory,q,a,b,c); required=index_closure(index,q,a,b,c)
    subset=induced_memory(rebuilt,required); got=query_spec(subset,q,a,b,c); qtext=query_payload(q,a,b,c)
    full=token_count(tok,encoded)+token_count(tok,qtext); selected=(0 if q=="parameter_retrieval" else token_count(tok,encoder(subset)))+token_count(tok,qtext)
    return {"variant":variant,"query_type":q,"query_index":i,"state_exact":state_ref.exact,"index_exact":required==truth,"answer_exact":got==answer,"total_states":n,"required_states":len(required),"retrieved_state_pct":100*len(required)/n,"full_input_tokens":full,"selected_input_tokens":selected,"tokens_saved":full-selected,"selected_token_reduction_pct":100*(full-selected)/full if full else 0}

def summarize(rows):
    good=[r for r in rows if r["state_exact"] and r["index_exact"] and r["answer_exact"]]
    return {"cases":len(rows),"state_exact_cases":sum(r["state_exact"] for r in rows),"index_exact_cases":sum(r["index_exact"] for r in rows),"answer_exact_cases":sum(r["answer_exact"] for r in rows),"utility_exact_rate_pct":100*len(good)/len(rows),"mean_retrieved_state_pct":statistics.mean(r["retrieved_state_pct"] for r in good),"mean_selected_token_reduction_pct":statistics.mean(r["selected_token_reduction_pct"] for r in good)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output",default="glyphin_simulation29_result.json"); args=ap.parse_args(); tok=tiktoken.get_encoding(TOKENIZER); rows=[]; index_tokens={v:[] for v in VARIANTS}
    for seed in SEEDS:
        for size in SIZES:
            mem=build_memory(size,seed)
            for v,(enc,dec) in VARIANTS.items():
                rebuilt=dec(enc(mem)); index=build_index(rebuilt); index_text=json.dumps(index,sort_keys=True,separators=(",",":")); index_tokens[v].append(token_count(tok,index_text))
                for i,q in enumerate(QUERY_TYPES):
                    r=evaluate(mem,v,enc,dec,tok,q,i,index); r.update(seed=seed,size=size); rows.append(r)
    mean_index_tokens={v:statistics.mean(x) for v,x in index_tokens.items()}; batches=[]
    for v in VARIANTS:
        vr=[r for r in rows if r["variant"]==v]
        for b in BATCH_SIZES:
            repeated=(vr*b)[:b]; full=sum(r["full_input_tokens"] for r in repeated); indexed=mean_index_tokens[v]+sum(r["selected_input_tokens"] for r in repeated)
            batches.append({"variant":v,"batch_size":b,"index_tokens_one_time":mean_index_tokens[v],"full_tokens":full,"indexed_selective_tokens":indexed,"tokens_saved_after_index":full-indexed,"indexed_total_reduction_pct":100*(full-indexed)/full if full else 0,"break_even_reached":indexed<full})
    out={"simulation":29,"benchmark_version":VERSION,"purpose":"Index-discovered query-conditioned retrieval using explicit query anchors and a deterministic structural index.","fixture_family":"Sim21 high-entropy memory","seeds":list(SEEDS),"sizes":list(SIZES),"tokenizer":TOKENIZER,"query_types":list(QUERY_TYPES),"variants":list(VARIANTS),"batch_sizes":list(BATCH_SIZES),"total_cases":len(rows),"cases":rows,"summary":{v:summarize([r for r in rows if r["variant"]==v]) for v in VARIANTS},"mean_index_tokens_by_variant":mean_index_tokens,"batch_amortization":batches,"scope":"Deterministic structural retrieval only. Query anchors are explicit state IDs; semantic query understanding is excluded. Dependency closure is discovered from the index, not supplied by the query specification. Ground-truth closure is used only for post-hoc index correctness validation. Index storage is reported as a one-time persistent cost and amortized over repeated queries. No latency, learned retrieval, LLM semantic equivalence, universal generalization, or optimality claim."}
    raw=json.dumps(out,sort_keys=True,separators=(",",":")).encode(); out["result_data_sha256"]=hashlib.sha256(raw).hexdigest()
    with open(args.output,"w",encoding="utf-8") as f: json.dump(out,f,indent=2,sort_keys=True); f.write("\n")
    print(json.dumps({"summary":out["summary"],"mean_index_tokens_by_variant":mean_index_tokens,"batch_amortization":batches},indent=2,sort_keys=True))
    return 0 if all(r["state_exact"] and r["index_exact"] and r["answer_exact"] for r in rows) else 1
if __name__=="__main__": raise SystemExit(main())
