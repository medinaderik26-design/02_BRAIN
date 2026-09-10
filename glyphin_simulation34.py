"""Simulation 34 — resolver error propagation boundary.

Separates an upstream concept resolver from Glyphin's structural retrieval,
induced-state reconstruction, and exact downstream answer.
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

SEEDS=(21092026,31092026,41092026,51092026,61092026)
SIZES=(256,1024,2048)
TARGETS_PER_STATE=64
ENCODING="cl100k_base"
VARIANTS={"sim17-compact":(encode_compact,decode_compact),"structural-lineage":(encode_structural,decode_structural),"state-columnar":(encode_columnar,decode_columnar)}
OUTCOMES=("exact","miss","wrong_unique","ambiguous")
VERSION="34.2"

def parent_closure(mem,anchors):
    wanted=set(anchors); stack=list(anchors)
    while stack:
        s=mem.states.get(stack.pop())
        if s is None or s.parent is None: continue
        if s.parent not in wanted: wanted.add(s.parent); stack.append(s.parent)
    return wanted

def induced_memory(mem,names):
    out=GlyphinMemory(decay_lambda=mem.decay_lambda,alpha=mem.alpha,beta=mem.beta)
    pending=set(names)
    while pending:
        ready=sorted(n for n in pending if mem.states[n].parent is None or mem.states[n].parent not in pending)
        if not ready: raise AssertionError("induced dependency graph is cyclic")
        for name in ready:
            s=mem.states[name]; parent=s.parent if s.parent in names else None
            out.add_state(name=s.name,level=s.level,cohesion=s.cohesion,parent=parent,frequency=s.frequency,resonance=s.resonance,sigma=s.sigma,created_at=s.created_at)
            pending.remove(name)
    return out

def resolver_outcome(mem,target,outcome,rng):
    if outcome=="exact": return {"status":"resolved","anchors":[target],"candidate_count":1}
    if outcome=="miss": return {"status":"miss","anchors":[],"candidate_count":0}
    choices=[n for n in sorted(mem.states) if n!=target]; other=choices[rng.randrange(len(choices))]
    if outcome=="wrong_unique": return {"status":"resolved","anchors":[other],"candidate_count":1}
    return {"status":"ambiguous","anchors":[target,other],"candidate_count":2}

def target_answer(mem,target):
    return [(n,mem.states[n].parent) for n in sorted(parent_closure(mem,[target]))]

def run_case(mem,target,outcome,rng,variant,tok):
    res=resolver_outcome(mem,target,outcome,rng)
    accepted=res["status"]=="resolved"
    selected=parent_closure(mem,res["anchors"]) if accepted else set()
    truth=parent_closure(mem,[target])
    closure_exact=selected==truth
    answer_exact=accepted and closure_exact and target_answer(mem,target)==target_answer(induced_memory(mem,selected),target)
    enc,dec=VARIANTS[variant]
    full_tokens=len(tok.encode(mem.to_json(),disallowed_special=()))
    if accepted:
        induced=induced_memory(mem,selected)
        rebuilt=dec(enc(induced))
        transport_exact=set(rebuilt.states)==selected
        transport_answer_exact=transport_exact and target in rebuilt.states and target_answer(rebuilt,target)==target_answer(mem,target)
        selected_tokens=len(tok.encode(enc(induced),disallowed_special=()))
        reduction=(1-selected_tokens/full_tokens)*100.0
    else:
        transport_exact=True; transport_answer_exact=False; selected_tokens=0; reduction=None
    return {"target":target,"resolver_outcome":outcome,"resolver_status":res["status"],"candidate_count":res["candidate_count"],"resolver_anchor_exact":res["anchors"]==[target],"accepted":accepted,"truth_closure_size":len(truth),"selected_closure_size":len(selected),"closure_exact":closure_exact,"answer_exact":answer_exact,"transport_exact":transport_exact,"transport_answer_exact":transport_answer_exact,"full_tokens":full_tokens,"selected_tokens":selected_tokens,"selected_token_reduction_pct":reduction}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output",default="glyphin_simulation34_result.json"); args=ap.parse_args(); tok=tiktoken.get_encoding(ENCODING); cases=[]
    for seed in SEEDS:
      for size in SIZES:
       mem=build_memory(size,seed+size)
       for ti,target in enumerate(sorted(mem.states)[:TARGETS_PER_STATE]):
        for oi,outcome in enumerate(OUTCOMES):
         for variant in VARIANTS:
          rng=random.Random(seed*1000003+size*1009+ti*97+oi)
          cases.append({"seed":seed,"size":size,"variant":variant,**run_case(mem,target,outcome,rng,variant,tok)})
    summary={}
    for v in VARIANTS:
      summary[v]={}
      for o in OUTCOMES:
       rows=[r for r in cases if r["variant"]==v and r["resolver_outcome"]==o]; vals=[r["selected_token_reduction_pct"] for r in rows if r["selected_token_reduction_pct"] is not None]
       summary[v][o]={"cases":len(rows),"resolver_anchor_exact":sum(r["resolver_anchor_exact"] for r in rows),"accepted":sum(r["accepted"] for r in rows),"closure_exact":sum(r["closure_exact"] for r in rows),"answer_exact":sum(r["answer_exact"] for r in rows),"transport_exact":sum(r["transport_exact"] for r in rows),"transport_answer_exact":sum(r["transport_answer_exact"] for r in rows),"mean_selected_closure_size":sum(r["selected_closure_size"] for r in rows)/len(rows),"mean_selected_token_reduction_pct":sum(vals)/len(vals) if vals else None}
    out={"simulation":34,"benchmark_version":VERSION,"seeds":SEEDS,"sizes":SIZES,"targets_per_state":TARGETS_PER_STATE,"outcomes":OUTCOMES,"tokenizer":ENCODING,"variants":list(VARIANTS),"cases":cases,"summary":summary,"result_data_sha256":hashlib.sha256(json.dumps(cases,sort_keys=True).encode()).hexdigest(),"scope":"Synthetic resolver-error propagation benchmark. Resolver outcomes are injected upstream; Glyphin structural closure, induced reconstruction, deterministic transport, and deterministic answer checks are measured. No semantic resolver quality, natural-language understanding, learned retrieval, LLM equivalence, latency, universal generalization, consciousness, or optimality claim."}
    Path(args.output).write_text(json.dumps(out,indent=2,sort_keys=True),encoding="utf-8"); print(json.dumps(summary,indent=2,sort_keys=True))

if __name__=="__main__": main()
