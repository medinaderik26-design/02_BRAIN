"""Glyphin Simulation 31: deterministic alias/descriptor anchor retrieval.

Sim31 removes canonical state names from the query text. A persistent alias
index maps non-canonical aliases to canonical state IDs. Unique aliases must
resolve exactly; aliases deliberately shared by multiple states are reported
as ambiguous rather than guessed. No semantic NLP is measured.
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

VERSION="31.1"
SEEDS=(21092026,31092026,41092026,51092026,61092026)
SIZES=(256,1024,2048)
BATCH_SIZES=(1,4,16,64,256)
TOKENIZER="cl100k_base"
QUERY_TYPES=("direct_attribute","parent_lookup","child_lookup","multi_hop_traversal","relationship_exists","path_reconstruction","temporal_ordering","parameter_retrieval","cross_state_comparison","mixed_multi_hop")
VARIANTS={"sim17-compact":(encode_compact,decode_compact),"structural-lineage":(encode_structural,decode_structural),"state-columnar":(encode_columnar,decode_columnar)}

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

def index_closure(index,q,a,b,c): return ground_truth_closure(_index_memory(index),q,a,b,c)

def _index_memory(index):
    p={n:index[n]["parent"] for n in index}; children={n:set(index[n]["children"]) for n in index}
    class S:
        def __init__(self,n): self.name=n; self.parent=p[n]; self.children=children[n]
    class M: states={n:S(n) for n in index}
    return M()

def induced_memory(memory,required):
    out=GlyphinMemory(decay_lambda=memory.decay_lambda,alpha=memory.alpha,beta=memory.beta)
    for n in sorted(required):
        s=memory.states[n]
        out.add_state(name=s.name,level=s.level,cohesion=s.cohesion,parent=None,frequency=s.frequency,resonance=s.resonance,sigma=s.sigma,created_at=s.created_at)
    for n in sorted(required):
        p=memory.states[n].parent
        if p in out.states: out.link_state(p,n)
    return out

def make_aliases(memory):
    return {n:"ax-"+hashlib.sha256(("glyphin31|"+n).encode()).hexdigest()[:12] for n in names(memory)}

def inject_collision_aliases(alias_map, memory):
    ns=names(memory); collisions={}
    for j in range(0,min(8,len(ns)),2):
        alias="cx-"+hashlib.sha256(("collision31|"+str(j//2)).encode()).hexdigest()[:10]
        collisions[alias]=ns[j:j+2]
    return collisions

def build_alias_index(alias_map, collisions):
    idx={}
    for alias,n in alias_map.items(): idx.setdefault(alias,[]).append(n)
    for alias,group in collisions.items(): idx[alias]=sorted(group)
    return {k:sorted(v) for k,v in sorted(idx.items())}

def query_aliases(q,a,b,c,alias_map):
    if q=="parameter_retrieval": return []
    if q in ("direct_attribute","parent_lookup","child_lookup","multi_hop_traversal"): return [alias_map[a]]
    if q in ("relationship_exists","path_reconstruction","cross_state_comparison"): return [alias_map[a],alias_map[b]]
    if q=="temporal_ordering": return [alias_map[a],alias_map[b],alias_map[c]]
    if q=="mixed_multi_hop": return [alias_map[a],alias_map[c]]
    raise ValueError(q)

def make_query(q, aliases):
    if q=="parameter_retrieval": return "return memory parameters"
    if q in ("direct_attribute","parent_lookup","child_lookup","multi_hop_traversal"): return f"query alias {aliases[0]}"
    if q in ("relationship_exists","path_reconstruction","cross_state_comparison"): return f"compare aliases {aliases[0]} and {aliases[1]}"
    if q=="temporal_ordering": return f"order aliases {aliases[0]} {aliases[1]} {aliases[2]}"
    if q=="mixed_multi_hop": return f"trace alias {aliases[0]} and compare with {aliases[1]}"
    raise ValueError(q)

def make_collision_query(q, alias):
    # Collision probes intentionally contain exactly one ambiguous alias.
    # They test rejection of ambiguity, not downstream multi-anchor semantics.
    return f"resolve alias {alias} for {q}"

def resolve_aliases(index,qtext,expected_count):
    hits=[]
    for alias,candidates in index.items():
        if alias in qtext: hits.append((qtext.index(alias),alias,candidates))
    hits.sort(key=lambda x:(x[0],x[1]))
    if len(hits)!=expected_count: return None,"wrong_anchor_count",0
    candidate_count=max((len(cands) for _,_,cands in hits),default=0)
    if any(len(cands)!=1 for _,_,cands in hits): return None,"ambiguous_alias",candidate_count
    return [cands[0] for _,_,cands in hits],"unique",candidate_count

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
        sb,sc=memory.states[b],memory.states[c]
        return {"states":[a,b,c],"chronological":[x[1] for x in sorted((s.created_at,s.name) for s in (sa,sb,sc))]}
    if q=="cross_state_comparison":
        sb=memory.states[b]; return {"a":a,"b":b,"level_delta":sa.level-sb.level,"cohesion_delta":sa.cohesion-sb.cohesion,"frequency_delta":sa.frequency-sb.frequency,"same_parent":sa.parent==sb.parent}
    if q=="mixed_multi_hop":
        sc=memory.states[c]
        return {"start":a,"start_parent":sa.parent,"start_path":memory.get_path(a),"parent_children":sorted(memory.states[sa.parent].children) if sa.parent else [],"compare_to":c,"same_parent":sa.parent==sc.parent}
    raise ValueError(q)

def token_count(tok,text): return len(tok.encode(text,disallowed_special=()))

def evaluate(mem,v,enc,dec,tok,q,i,index,alias_map,collision=False):
    encoded=enc(mem); rebuilt=dec(encoded); state_ref=referee_memory(mem,rebuilt); ns=names(mem); n=len(ns)
    a,b,c=ns[i%n],ns[(i+n//3)%n],ns[(i+2*n//3)%n]
    collisions=inject_collision_aliases(alias_map,mem)
    if collision and q!="parameter_retrieval":
        ca=sorted(collisions)[i%len(collisions)]; aliases=[ca]; expected=1
        expected_anchors=collisions[ca]; qtext=make_collision_query(q,ca)
        resolved,reason,candidate_count=resolve_aliases(index,qtext,expected)
        anchor_exact=False; false_positive=resolved is not None
        selected=rebuilt; answer_exact=False; required=set(); truth=set()
        collision_candidate_exact=candidate_count==len(expected_anchors)
    else:
        aliases=query_aliases(q,a,b,c,alias_map); qtext=make_query(q,aliases); expected=len(aliases)
        resolved,reason,candidate_count=resolve_aliases(index,qtext,expected)
        expected_anchors=[] if q=="parameter_retrieval" else [a,b,c] if q=="temporal_ordering" else [a,c] if q=="mixed_multi_hop" else [a,b] if expected==2 else [a]
        anchor_exact=resolved==expected_anchors; false_positive=resolved is not None and not anchor_exact
        truth=ground_truth_closure(mem,q,a,b,c); required=index_closure(index,q,a,b,c)
        if resolved is None: selected=rebuilt; answer_exact=False
        else:
            aa=resolved[0] if resolved else None
            bb=resolved[1] if q in ("relationship_exists","path_reconstruction","cross_state_comparison","temporal_ordering") else None
            cc=resolved[1] if q=="mixed_multi_hop" else resolved[2] if q=="temporal_ordering" else None
            selected=induced_memory(rebuilt,required); answer_exact=(query_spec(selected,q,aa,bb,cc)==query_spec(mem,q,a,b,c))
        collision_candidate_exact=False
    full=token_count(tok,encoded)+token_count(tok,qtext)
    selected_tokens=(token_count(tok,enc(selected)) if q!="parameter_retrieval" and not collision else 0)+token_count(tok,qtext)
    unique_resolved=resolved is not None and reason=="unique"
    return {"variant":v,"query_type":q,"query_index":i,"collision_case":collision,"state_exact":state_ref.exact,"anchor_exact":anchor_exact,"unique_resolution":unique_resolved,"resolution_reason":reason,"index_exact":required==truth,"answer_exact":answer_exact,"false_positive":false_positive,"collision_candidate_exact":collision_candidate_exact,"anchor_candidates":candidate_count,"total_states":n,"required_states":len(required),"retrieved_state_pct":100*len(required)/n,"full_input_tokens":full,"selected_input_tokens":selected_tokens,"tokens_saved":full-selected_tokens,"selected_token_reduction_pct":100*(full-selected_tokens)/full if full else 0,"query_text":qtext}

def summarize(rows):
    primary=[r for r in rows if not r["collision_case"]]; collision=[r for r in rows if r["collision_case"]]
    good=[r for r in primary if r["state_exact"] and r["anchor_exact"] and r["index_exact"] and r["answer_exact"]]
    return {"cases":len(rows),"primary_cases":len(primary),"collision_cases":len(collision),"state_exact_cases":sum(r["state_exact"] for r in rows),"primary_unique_resolution_cases":sum(r["unique_resolution"] for r in primary),"primary_anchor_exact_cases":sum(r["anchor_exact"] for r in primary),"primary_index_exact_cases":sum(r["index_exact"] for r in primary),"primary_answer_exact_cases":sum(r["answer_exact"] for r in primary),"primary_utility_exact_rate_pct":100*len(good)/len(primary) if primary else 0,"collision_ambiguous_rate_pct":100*sum(r["resolution_reason"]=="ambiguous_alias" for r in collision)/len(collision) if collision else 0,"collision_false_positive_rate_pct":100*sum(r["false_positive"] for r in collision)/len(collision) if collision else 0,"collision_candidate_exact_cases":sum(r["collision_candidate_exact"] for r in collision),"collision_candidate_count_mean":statistics.mean(r["anchor_candidates"] for r in collision) if collision else 0}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output",default="glyphin_simulation31_result.json"); args=ap.parse_args()
    tok=tiktoken.get_encoding(TOKENIZER); rows=[]; index_tokens={v:[] for v in VARIANTS}; alias_index_tokens={v:[] for v in VARIANTS}
    for seed in SEEDS:
        for size in SIZES:
            mem=build_memory(size,seed); alias_map=make_aliases(mem); collisions=inject_collision_aliases(alias_map,mem)
            for v,(enc,dec) in VARIANTS.items():
                rebuilt=dec(enc(mem)); index=build_index(rebuilt); ai=build_alias_index(alias_map,collisions)
                index_tokens[v].append(token_count(tok,json.dumps(index,sort_keys=True,separators=(",",":"))))
                alias_index_tokens[v].append(token_count(tok,json.dumps(ai,sort_keys=True,separators=(",",":"))))
                for i,q in enumerate(QUERY_TYPES):
                    rows.append({**evaluate(mem,v,enc,dec,tok,q,i,index,alias_map,False),"seed":seed,"size":size})
                    if q!="parameter_retrieval": rows.append({**evaluate(mem,v,enc,dec,tok,q,i,index,alias_map,True),"seed":seed,"size":size})
    mean_index={v:statistics.mean(x) for v,x in index_tokens.items()}; mean_alias={v:statistics.mean(x) for v,x in alias_index_tokens.items()}
    batches=[]
    for v in VARIANTS:
        vr=[r for r in rows if r["variant"]==v and not r["collision_case"]]
        for b in BATCH_SIZES:
            batch=vr[:b]; full=sum(r["full_input_tokens"] for r in batch); selective=sum(r["selected_input_tokens"] for r in batch)
            alias_total=mean_alias[v]+selective; combined_total=mean_index[v]+mean_alias[v]+selective
            batches.append({"variant":v,"batch_size":b,"structural_index_tokens_one_time":mean_index[v],"alias_index_tokens_one_time":mean_alias[v],"combined_index_tokens_one_time":mean_index[v]+mean_alias[v],"full_tokens":full,"indexed_selective_tokens_alias_only":alias_total,"indexed_selective_tokens_combined":combined_total,"alias_only_reduction_pct":100*(full-alias_total)/full if full else 0,"combined_index_reduction_pct":100*(full-combined_total)/full if full else 0,"alias_only_break_even_reached":alias_total<full,"combined_index_break_even_reached":combined_total<full})
    out={"simulation":31,"benchmark_version":VERSION,"purpose":"Deterministic non-canonical alias/descriptor anchor resolution followed by persistent alias-index dependency retrieval; canonical state names are absent from primary query text.","fixture_family":"Sim21 high-entropy memory","seeds":list(SEEDS),"sizes":list(SIZES),"tokenizer":TOKENIZER,"query_types":list(QUERY_TYPES),"variants":list(VARIANTS),"batch_sizes":list(BATCH_SIZES),"primary_cases":450,"collision_cases":405,"total_cases":len(rows),"cases":rows,"summary":{v:summarize([r for r in rows if r["variant"]==v]) for v in VARIANTS},"mean_structural_index_tokens_by_variant":mean_index,"mean_alias_index_tokens_by_variant":mean_alias,"batch_amortization":batches,"scope":"Deterministic exact alias matching only. Aliases are opaque non-canonical identifiers generated independently of state names; no natural-language semantic understanding is measured. Collision aliases deliberately map to multiple states and must be rejected as ambiguous. Ground-truth closure is computed independently from memory topology. Structural and alias index storage are persistent one-time costs; both alias-only and combined amortization are reported. Selected transport re-encodes induced subsets with the same encoder; this is not a fully chunk-addressable wire protocol. No latency, learned retrieval, LLM semantic equivalence, universal generalization, consciousness, or optimality claim."}
    raw=json.dumps(out,sort_keys=True,separators=(",",":")).encode(); out["result_data_sha256"]=hashlib.sha256(raw).hexdigest()
    with open(args.output,"w",encoding="utf-8") as f: json.dump(out,f,indent=2,sort_keys=True); f.write("\n")
    print(json.dumps({"summary":out["summary"],"mean_alias_index_tokens_by_variant":mean_alias,"batch_amortization":batches},indent=2,sort_keys=True))
    return 0 if all(r["state_exact"] and ((r["collision_case"] and r["resolution_reason"]=="ambiguous_alias" and r["anchor_candidates"]==2 and r["collision_candidate_exact"]) or (not r["collision_case"] and r["anchor_exact"] and r["index_exact"] and r["answer_exact"])) for r in rows) else 1
if __name__=="__main__": raise SystemExit(main())
