"""Glyphin Simulation 30: deterministic query-to-anchor retrieval.

This benchmark removes the explicit a/b/c state-ID fields used in Sim29.
A query is represented as text containing state-name mentions. A deterministic
lexical resolver maps those mentions to index anchors; the structural index
then discovers the dependency closure. Semantic/NLP query understanding is
intentionally excluded.
"""
from __future__ import annotations
import argparse, hashlib, json, statistics
import tiktoken
from glyphin_simulation21 import build_memory
from glyphin_simulation17 import encode_compact, decode_compact
from glyphin_simulation18 import encode_structural, decode_structural
from glyphin_simulation20 import encode_columnar, decode_columnar
from glyphin_state_referee import referee_memory
from glyphin_research_core import GlyphinMemory
VERSION="30.1"
SEEDS=(21092026,31092026,41092026,51092026,61092026); SIZES=(256,1024,2048); BATCH_SIZES=(1,4,16,64,256); TOKENIZER="cl100k_base"
VARIANTS={"sim17-compact":(encode_compact,decode_compact),"structural-lineage":(encode_structural,decode_structural),"state-columnar":(encode_columnar,decode_columnar)}
QUERY_TYPES=("direct_attribute","parent_lookup","child_lookup","multi_hop_traversal","relationship_exists","path_reconstruction","temporal_ordering","parameter_retrieval","cross_state_comparison","mixed_multi_hop")
def names(memory): return sorted(memory.states)
def build_index(memory): return {n:{"parent":memory.states[n].parent,"children":sorted(memory.states[n].children)} for n in names(memory)}
def ancestors(parent_map,node):
    out=[]; seen=set(); cur=node
    while cur is not None and cur not in seen: seen.add(cur); out.append(cur); cur=parent_map[cur]
    return out
def ground_truth_closure(memory,q,a,b,c):
    if q=="parameter_retrieval": return set()
    p={n:memory.states[n].parent for n in names(memory)}; children={n:set(memory.states[n].children) for n in names(memory)}
    if q=="direct_attribute": return {a}
    if q=="parent_lookup": return {a}|({p[a]} if p[a] else set())
    if q=="child_lookup": return {a}|children[a]
    if q=="multi_hop_traversal": return set(ancestors(p,a))
    if q=="relationship_exists": return {a,b}|({p[a]} if p[a] else set())|({p[b]} if p[b] else set())
    if q=="path_reconstruction": return set(ancestors(p,a))|set(ancestors(p,b))
    if q=="temporal_ordering": return {a,b,c}
    if q=="cross_state_comparison": return {a,b}|({p[a]} if p[a] else set())|({p[b]} if p[b] else set())
    if q=="mixed_multi_hop":
        out=set(ancestors(p,a))|{c}; parent=p[a]
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
    if q=="multi_hop_traversal": return set(ancestors(p,a))
    if q=="relationship_exists": return {a,b}|({index[a]["parent"]} if index[a]["parent"] else set())|({index[b]["parent"]} if index[b]["parent"] else set())
    if q=="path_reconstruction": return set(ancestors(p,a))|set(ancestors(p,b))
    if q=="temporal_ordering": return {a,b,c}
    if q=="cross_state_comparison": return {a,b}|({index[a]["parent"]} if index[a]["parent"] else set())|({index[b]["parent"]} if index[b]["parent"] else set())
    if q=="mixed_multi_hop":
        out=set(ancestors(p,a))|{c}; parent=index[a]["parent"]
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
def make_query(q,a,b,c):
    if q=="parameter_retrieval": return "return memory parameters"
    if q in ("direct_attribute","parent_lookup","child_lookup","multi_hop_traversal"): return f"query state {a}"
    if q in ("relationship_exists","path_reconstruction","cross_state_comparison"): return f"compare states {a} and {b}"
    if q=="temporal_ordering": return f"order states {a} {b} {c}"
    if q=="mixed_multi_hop": return f"trace state {a} and compare with {c}"
    raise ValueError(q)
def resolve_anchors(index,qtext,q,expected_count):
    hits=sorted([n for n in index if n in qtext],key=lambda x:qtext.index(x))
    if q=="parameter_retrieval": return []
    if len(hits)!=expected_count: return None
    return hits
def query_spec(memory,q,a,b,c):
    if q=="parameter_retrieval": return {"decay_lambda":memory.decay_lambda,"alpha":memory.alpha,"beta":memory.beta}
    sa=memory.states[a]
    if q=="direct_attribute": return {"name":a,"level":sa.level,"cohesion":sa.cohesion,"frequency":sa.frequency,"resonance":sa.resonance,"sigma":sa.sigma,"created_at":sa.created_at}
    if q=="parent_lookup": return {"state":a,"parent":sa.parent}
    if q=="child_lookup": return {"state":a,"children":sorted(sa.children)}
    if q=="multi_hop_traversal": return {"state":a,"path":memory.get_path(a)}
    if q=="relationship_exists":
        sb=memory.states[b]; return {"a":a,"b":b,"a_parent_is_b":sa.parent==b,"b_parent_is_a":sb.parent==a}
    if q=="path_reconstruction": return {"a":a,"b":b,"path_a":memory.get_path(a),"path_b":memory.get_path(b)}
    if q=="temporal_ordering":
        sb,sc=memory.states[b],memory.states[c]; return {"states":[a,b,c],"chronological":[x[1] for x in sorted((s.created_at,s.name) for s in (sa,sb,sc))]}
    if q=="cross_state_comparison":
        sb=memory.states[b]; return {"a":a,"b":b,"level_delta":sa.level-sb.level,"cohesion_delta":sa.cohesion-sb.cohesion,"frequency_delta":sa.frequency-sb.frequency,"same_parent":sa.parent==sb.parent}
    if q=="mixed_multi_hop":
        sc=memory.states[c]; return {"start":a,"start_parent":sa.parent,"start_path":memory.get_path(a),"parent_children":sorted(memory.states[sa.parent].children) if sa.parent else [],"compare_to":c,"same_parent":sa.parent==sc.parent}
    raise ValueError(q)
def token_count(tok,text): return len(tok.encode(text,disallowed_special=()))
def evaluate(mem,v,enc,dec,tok,q,i,index):
    encoded=enc(mem); rebuilt=dec(encoded); state_ref=referee_memory(mem,rebuilt); ns=names(mem); n=len(ns); a,b,c=ns[i%n],ns[(i+n//3)%n],ns[(i+2*n//3)%n]
    qtext=make_query(q,a,b,c); expected=0 if q=="parameter_retrieval" else (3 if q=="temporal_ordering" else 2 if q in ("relationship_exists","path_reconstruction","cross_state_comparison","mixed_multi_hop") else 1)
    anchors=resolve_anchors(index,qtext,q,expected)
    expected_anchors=[] if q=="parameter_retrieval" else [a,b,c] if q=="temporal_ordering" else [a,c] if q=="mixed_multi_hop" else [a,b] if expected==2 else [a]
    anchor_exact=anchors is not None and anchors==expected_anchors
    truth=ground_truth_closure(mem,q,a,b,c); required=index_closure(index,q,a,b,c)
    if anchors is None: selected=rebuilt; answer_exact=False
    else:
        aa=anchors[0] if anchors else None
        bb=anchors[1] if q in ("relationship_exists","path_reconstruction","cross_state_comparison") else None
        cc=anchors[1] if q=="mixed_multi_hop" else anchors[2] if q=="temporal_ordering" else None
        selected=induced_memory(rebuilt,required); answer_exact=(query_spec(selected,q,aa,bb,cc)==query_spec(mem,q,a,b,c))
    full=token_count(tok,encoded)+token_count(tok,qtext); selected_tokens=(token_count(tok,enc(selected)) if q!="parameter_retrieval" else 0)+token_count(tok,qtext)
    return {"variant":v,"query_type":q,"query_index":i,"state_exact":state_ref.exact,"anchor_exact":anchor_exact,"index_exact":required==truth,"answer_exact":answer_exact,"total_states":n,"required_states":len(required),"retrieved_state_pct":100*len(required)/n,"full_input_tokens":full,"selected_input_tokens":selected_tokens,"tokens_saved":full-selected_tokens,"selected_token_reduction_pct":100*(full-selected_tokens)/full if full else 0,"query_text":qtext}
def summarize(rows):
    good=[r for r in rows if r["state_exact"] and r["anchor_exact"] and r["index_exact"] and r["answer_exact"]]
    return {"cases":len(rows),"state_exact_cases":sum(r["state_exact"] for r in rows),"anchor_exact_cases":sum(r["anchor_exact"] for r in rows),"index_exact_cases":sum(r["index_exact"] for r in rows),"answer_exact_cases":sum(r["answer_exact"] for r in rows),"utility_exact_rate_pct":100*len(good)/len(rows),"mean_retrieved_state_pct":statistics.mean(r["retrieved_state_pct"] for r in good),"mean_selected_token_reduction_pct":statistics.mean(r["selected_token_reduction_pct"] for r in good)}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output",default="glyphin_simulation30_result.json"); args=ap.parse_args(); tok=tiktoken.get_encoding(TOKENIZER); rows=[]; index_tokens={v:[] for v in VARIANTS}
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
    out={"simulation":30,"benchmark_version":VERSION,"purpose":"Deterministic query-to-anchor lexical resolution followed by index-discovered dependency retrieval; explicit a/b/c fields are removed from the query payload schema.","fixture_family":"Sim21 high-entropy memory","seeds":list(SEEDS),"sizes":list(SIZES),"tokenizer":TOKENIZER,"query_types":list(QUERY_TYPES),"variants":list(VARIANTS),"batch_sizes":list(BATCH_SIZES),"total_cases":len(rows),"cases":rows,"summary":{v:summarize([r for r in rows if r["variant"]==v]) for v in VARIANTS},"mean_index_tokens_by_variant":mean_index_tokens,"batch_amortization":batches,"scope":"Deterministic lexical anchor resolution only. State names are literal query-bearing tokens; no natural-language semantic understanding is measured. The structural index performs dependency discovery after anchor resolution. Index storage is a one-time persistent cost amortized over repeated queries. No latency, learned retrieval, LLM semantic equivalence, universal generalization, or optimality claim."}
    raw=json.dumps(out,sort_keys=True,separators=(",",":")).encode(); out["result_data_sha256"]=hashlib.sha256(raw).hexdigest()
    with open(args.output,"w",encoding="utf-8") as f: json.dump(out,f,indent=2,sort_keys=True); f.write("\n")
    print(json.dumps({"summary":out["summary"],"mean_index_tokens_by_variant":mean_index_tokens,"batch_amortization":batches},indent=2,sort_keys=True))
    return 0 if all(r["state_exact"] and r["anchor_exact"] and r["index_exact"] and r["answer_exact"] for r in rows) else 1
if __name__=="__main__": raise SystemExit(main())
