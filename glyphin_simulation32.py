"""Glyphin Simulation 32: controlled paraphrase query-intent resolution.

Sim32 isolates the query-interpretation boundary. Queries no longer contain
canonical state names or opaque aliases. Each state receives an independent,
controlled descriptor vocabulary. Query text uses exact descriptor variants,
word-order changes, and distractors. A deterministic normalizer resolves
those controlled variants to candidate states; the structural index then
performs dependency closure and the reconstructed subset answers the query.

This is NOT natural-language understanding. It is a bounded paraphrase
benchmark whose resolver vocabulary and gold anchors are explicit and
independently measurable. Semantic interpretation, learned retrieval,
latency, and LLM answer quality remain outside scope.
"""
from __future__ import annotations
import argparse, hashlib, json, re, statistics
from pathlib import Path
import tiktoken
from glyphin_simulation21 import build_memory
from glyphin_simulation17 import encode_compact, decode_compact
from glyphin_simulation18 import encode_structural, decode_structural
from glyphin_simulation20 import encode_columnar, decode_columnar
from glyphin_state_referee import referee_memory
from glyphin_research_core import GlyphinMemory
VERSION="32.0"
SEEDS=(21092026,31092026,41092026,51092026,61092026)
SIZES=(256,1024,2048)
TOKENIZER="cl100k_base"
QUERY_TYPES=("direct_attribute","parent_lookup","child_lookup","multi_hop_traversal","relationship_exists","path_reconstruction","temporal_ordering","parameter_retrieval","cross_state_comparison","mixed_multi_hop")
VARIANTS={"sim17-compact":(encode_compact,decode_compact),"structural-lineage":(encode_structural,decode_structural),"state-columnar":(encode_columnar,decode_columnar)}
CONCEPTS=(("alpha","first","primary"),("beta","second","secondary"),("gamma","third","tertiary"),("delta","fourth","quaternary"),("amber","gold","yellow"),("azure","blue","cyan"),("crimson","red","scarlet"),("emerald","green","jade"),("north","northern","upward"),("south","southern","downward"),("east","eastern","rightward"),("west","western","leftward"),("calm","quiet","steady"),("rapid","fast","quick"),("dense","compact","thick"),("sparse","thin","scattered"))
def names(memory): return sorted(memory.states)
def build_index(memory): return {n:{"parent":memory.states[n].parent,"children":sorted(memory.states[n].children)} for n in names(memory)}
def ancestors(parent_map,node):
    out=[]; seen=set(); cur=node
    while cur is not None and cur not in seen: seen.add(cur); out.append(cur); cur=parent_map[cur]
    return out
def ground_truth_closure(memory,q,a,b,c):
    if q=="parameter_retrieval": return set()
    p={n:memory.states[n].parent for n in names(memory)}; ch={n:set(memory.states[n].children) for n in names(memory)}
    if q=="direct_attribute": return {a}
    if q=="parent_lookup": return {a}|({p[a]} if p[a] else set())
    if q=="child_lookup": return {a}|ch[a]
    if q=="multi_hop_traversal": return set(ancestors(p,a))
    if q=="relationship_exists": return {a,b}|({p[a]} if p[a] else set())|({p[b]} if p[b] else set())
    if q=="path_reconstruction": return set(ancestors(p,a))|set(ancestors(p,b))
    if q=="temporal_ordering": return {a,b,c}
    if q=="cross_state_comparison": return {a,b}|({p[a]} if p[a] else set())|({p[b]} if p[b] else set())
    if q=="mixed_multi_hop":
        out=set(ancestors(p,a))|{c}
        if p[a]: out.add(p[a]); out.update(ch[p[a]])
        if p[c]: out.add(p[c])
        return out
    raise ValueError(q)
def index_closure(index,q,a,b,c):
    p={n:index[n]["parent"] for n in index}; ch={n:set(index[n]["children"]) for n in index}
    if q=="parameter_retrieval": return set()
    if q=="direct_attribute": return {a}
    if q=="parent_lookup": return {a}|({p[a]} if p[a] else set())
    if q=="child_lookup": return {a}|ch[a]
    if q=="multi_hop_traversal": return set(ancestors(p,a))
    if q=="relationship_exists": return {a,b}|({p[a]} if p[a] else set())|({p[b]} if p[b] else set())
    if q=="path_reconstruction": return set(ancestors(p,a))|set(ancestors(p,b))
    if q=="temporal_ordering": return {a,b,c}
    if q=="cross_state_comparison": return {a,b}|({p[a]} if p[a] else set())|({p[b]} if p[b] else set())
    if q=="mixed_multi_hop":
        out=set(ancestors(p,a))|{c}
        if p[a]: out.add(p[a]); out.update(ch[p[a]])
        if p[c]: out.add(p[c])
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
def concept_for_index(i): return CONCEPTS[i % len(CONCEPTS)]
def descriptor_map(memory):
    ns=names(memory); return {n:{"canonical":concept_for_index(i)[0],"synonyms":list(concept_for_index(i))} for i,n in enumerate(ns)}
def build_descriptor_index(dmap):
    idx={}
    for state,info in dmap.items():
        for term in info["synonyms"]: idx.setdefault(term,[]).append(state)
    return {k:sorted(v) for k,v in sorted(idx.items())}
def choose_descriptor(dmap,state,style):
    vals=dmap[state]["synonyms"]; return vals[0] if style==0 else vals[1] if style==1 else vals[2]
def make_query(q,a,b,c,dmap,style,distractor=None):
    if q=="parameter_retrieval": return "return memory parameters"
    da,db,dc=choose_descriptor(dmap,a,style),choose_descriptor(dmap,b,style),choose_descriptor(dmap,c,style); extra=(" while ignoring "+distractor) if distractor else ""
    if q=="direct_attribute": return f"show the {da} state{extra}"
    if q=="parent_lookup": return f"find the parent of the {da} state{extra}"
    if q=="child_lookup": return f"list children of the {da} state{extra}"
    if q=="multi_hop_traversal": return f"trace upward from the {da} state{extra}"
    if q=="relationship_exists": return f"compare the {da} state with the {db} state{extra}"
    if q=="path_reconstruction": return f"reconstruct paths for the {da} and {db} states{extra}"
    if q=="temporal_ordering": return f"put the {da}, {db}, and {dc} states in time order{extra}"
    if q=="cross_state_comparison": return f"compare properties of the {da} and {db} states{extra}"
    if q=="mixed_multi_hop": return f"trace the {da} state and compare it with {dc}{extra}"
    raise ValueError(q)
def gold_anchors(q,a,b,c):
    if q=="parameter_retrieval": return []
    if q in ("direct_attribute","parent_lookup","child_lookup","multi_hop_traversal"): return [a]
    if q in ("relationship_exists","path_reconstruction","cross_state_comparison"): return [a,b]
    if q=="temporal_ordering": return [a,b,c]
    if q=="mixed_multi_hop": return [a,c]
    raise ValueError(q)
def resolve_controlled(text,descriptor_index):
    hits=[]
    for term,cands in descriptor_index.items():
        m=re.search(r"\b"+re.escape(term)+r"\b",text.lower())
        if m: hits.append((m.start(),term,cands))
    hits.sort(key=lambda x:(x[0],x[1])); resolved=[]
    for _,term,cands in hits:
        if len(cands)!=1: return None,"ambiguous_descriptor",len(cands)
        state=cands[0]
        if state not in resolved: resolved.append(state)
    return resolved,"unique",1
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
def tokens(tok,text): return len(tok.encode(text,disallowed_special=()))
def evaluate(mem,v,enc,dec,tok,q,i,structural_index,descriptor_index,dmap,style):
    encoded=enc(mem); rebuilt=dec(encoded); state_ref=referee_memory(mem,rebuilt); ns=names(mem); n=len(ns); a,b,c=ns[i%n],ns[(i+n//3)%n],ns[(i+2*n//3)%n]
    distractor=None
    if q!="parameter_retrieval":
        di=(i+n//2)%n
        if ns[di] not in {a,b,c}: distractor=choose_descriptor(dmap,ns[di],(style+1)%3)
    qtext=make_query(q,a,b,c,dmap,style,distractor); resolved,reason,candidate_count=resolve_controlled(qtext,descriptor_index)
    expected=gold_anchors(q,a,b,c); anchor_exact=(resolved==expected if q!="parameter_retrieval" else resolved==[])
    truth=ground_truth_closure(mem,q,a,b,c); required=index_closure(structural_index,q,a,b,c); index_exact=(truth==required)
    if resolved is not None and anchor_exact:
        selected=rebuilt if not required else induced_memory(rebuilt,required)
        if q=="temporal_ordering": selected_args=(a,b,c)
        elif q=="mixed_multi_hop": selected_args=(a,None,c)
        elif len(expected)==2: selected_args=(a,b,None)
        elif len(expected)==1: selected_args=(a,None,None)
        else: selected_args=(None,None,None)
        answer_exact=(query_spec(selected,q,*selected_args)==query_spec(mem,q,a,b,c))
    else: selected=rebuilt; answer_exact=False
    full=tokens(tok,encoded)+tokens(tok,qtext); selected_tokens=tokens(tok,enc(selected))+tokens(tok,qtext)
    return {"variant":v,"query_type":q,"query_index":i,"style":style,"state_exact":state_ref.exact,"anchor_exact":anchor_exact,"unique_resolution":resolved is not None and reason=="unique","resolution_reason":reason,"anchor_candidates":candidate_count,"index_exact":index_exact,"answer_exact":answer_exact,"total_states":n,"required_states":len(required),"retrieved_state_pct":100*len(required)/n,"full_input_tokens":full,"selected_input_tokens":selected_tokens,"tokens_saved":full-selected_tokens,"selected_token_reduction_pct":100*(full-selected_tokens)/full if full else 0,"query_text":qtext,"gold_anchors":expected,"resolved_anchors":resolved}
def summarize(rows):
    return {"cases":len(rows),"state_exact_cases":sum(r["state_exact"] for r in rows),"unique_resolution_cases":sum(r["unique_resolution"] for r in rows),"anchor_exact_cases":sum(r["anchor_exact"] for r in rows),"index_exact_cases":sum(r["index_exact"] for r in rows),"answer_exact_cases":sum(r["answer_exact"] for r in rows),"utility_exact_rate_pct":100*sum(r["state_exact"] and r["anchor_exact"] and r["index_exact"] and r["answer_exact"] for r in rows)/len(rows) if rows else 0,"mean_selected_token_reduction_pct":statistics.mean(r["selected_token_reduction_pct"] for r in rows) if rows else 0,"median_selected_token_reduction_pct":statistics.median(r["selected_token_reduction_pct"] for r in rows) if rows else 0,"mean_retrieved_state_pct":statistics.mean(r["retrieved_state_pct"] for r in rows) if rows else 0}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output",default="glyphin_simulation32_result.json"); args=ap.parse_args(); tok=tiktoken.get_encoding(TOKENIZER); rows=[]
    for seed in SEEDS:
        for size in SIZES:
            mem=build_memory(size,seed); dmap=descriptor_map(mem); descriptor_index=build_descriptor_index(dmap)
            for v,(enc,dec) in VARIANTS.items():
                rebuilt=dec(enc(mem)); structural_index=build_index(rebuilt)
                for i,q in enumerate(QUERY_TYPES):
                    for style in range(3): rows.append({**evaluate(mem,v,enc,dec,tok,q,i,structural_index,descriptor_index,dmap,style),"seed":seed,"size":size})
    out={"simulation":32,"benchmark_version":VERSION,"purpose":"Controlled paraphrase query-intent resolution followed by persistent structural dependency retrieval.","fixture_family":"Sim21 high-entropy memory","seeds":list(SEEDS),"sizes":list(SIZES),"styles":["canonical_term","synonym_1","synonym_2"],"tokenizer":TOKENIZER,"query_types":list(QUERY_TYPES),"variants":list(VARIANTS),"cases":rows,"summary":{v:summarize([r for r in rows if r["variant"]==v]) for v in VARIANTS},"result_data_sha256":hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest(),"scope":"Bounded controlled-language benchmark only. Descriptor vocabulary, paraphrase variants, and gold anchors are explicit. No natural-language semantic understanding, learned retrieval, LLM answer equivalence, latency, universal generalization, consciousness, or optimality claim."}
    Path(args.output).write_text(json.dumps(out,indent=2,sort_keys=True),encoding="utf-8"); print(json.dumps(out["summary"],indent=2,sort_keys=True))
if __name__=="__main__": main()
