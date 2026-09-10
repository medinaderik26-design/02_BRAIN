"""Simulation 35 — confidence/coverage policy boundary.

Synthetic resolver scores are injected upstream. The benchmark varies an
acceptance threshold and measures coverage, false acceptance, structural
fidelity, answer fidelity, and token cost. It does not evaluate semantic
resolution.
"""
from __future__ import annotations
import argparse, hashlib, json, random
from pathlib import Path
import tiktoken
from glyphin_simulation21 import build_memory
from glyphin_research_core import GlyphinMemory
from glyphin_simulation17 import encode_compact
from glyphin_simulation18 import encode_structural
from glyphin_simulation20 import encode_columnar

VERSION="35.2"
SEEDS=(21092026,31092026,41092026,51092026,61092026)
SIZES=(256,1024,2048)
TARGETS_PER_STATE=64
ENCODING="cl100k_base"
VARIANTS={"sim17-compact":encode_compact,"structural-lineage":encode_structural,"state-columnar":encode_columnar}
THRESHOLDS=(0.0,0.25,0.5,0.75,0.9,0.99)
OUTCOMES=("correct","wrong_unique","ambiguous","miss")
CASES_PER_OUTCOME=5*3*64

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

def resolver(mem,target,outcome,rng):
    if outcome=="correct": return {"status":"resolved","anchors":[target],"confidence":0.99}
    if outcome=="miss": return {"status":"miss","anchors":[],"confidence":0.0}
    other=sorted(n for n in mem.states if n!=target)[rng.randrange(len(mem.states)-1)]
    if outcome=="wrong_unique": return {"status":"resolved","anchors":[other],"confidence":0.90}
    return {"status":"ambiguous","anchors":[target,other],"confidence":0.50}

def run_case(mem,target,outcome,rng,variant,tok,threshold):
    r=resolver(mem,target,outcome,rng); accepted=r["status"]=="resolved" and r["confidence"]>=threshold
    truth=parent_closure(mem,[target]); selected=parent_closure(mem,r["anchors"]) if accepted else set()
    closure_exact=accepted and selected==truth
    answer_exact=False; transport_exact=False; transport_answer_exact=False; selected_tokens=0; reduction=None
    if accepted:
        answer_exact=target in selected and target in mem.states and target_answer(mem,target)==target_answer(induced_memory(mem,selected),target)
        induced=induced_memory(mem,selected); encoded=VARIANTS[variant](induced)
        rebuilt=decode_for_variant(variant,encoded)
        transport_exact=set(rebuilt.states)==selected
        transport_answer_exact=transport_exact and target in rebuilt.states and target_answer(rebuilt,target)==target_answer(mem,target)
        selected_tokens=len(tok.encode(encoded,disallowed_special=()))
        full_tokens=len(tok.encode(mem.to_json(),disallowed_special=()))
        reduction=(1-selected_tokens/full_tokens)*100.0
    return {"target":target,"resolver_outcome":outcome,"confidence":r["confidence"],"resolver_status":r["status"],"accepted":accepted,"resolver_anchor_exact":r["anchors"]==[target],"candidate_count":len(r["anchors"]),"truth_closure_size":len(truth),"selected_closure_size":len(selected),"closure_exact":closure_exact,"answer_exact":answer_exact,"transport_exact":transport_exact,"transport_answer_exact":transport_answer_exact,"selected_tokens":selected_tokens,"selected_token_reduction_pct":reduction}

def target_answer(mem,target):
    return [(n,mem.states[n].parent) for n in sorted(parent_closure(mem,[target]))]

def decode_for_variant(variant,text):
    if variant=="sim17-compact":
        from glyphin_simulation17 import decode_compact; return decode_compact(text)
    if variant=="structural-lineage":
        from glyphin_simulation18 import decode_structural; return decode_structural(text)
    from glyphin_simulation20 import decode_columnar; return decode_columnar(text)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output",default="glyphin_simulation35_result.json"); args=ap.parse_args(); tok=tiktoken.get_encoding(ENCODING); cases=[]
    for seed in SEEDS:
      for size in SIZES:
       mem=build_memory(size,seed+size)
       for ti,target in enumerate(sorted(mem.states)[:TARGETS_PER_STATE]):
        for oi,outcome in enumerate(OUTCOMES):
         for threshold in THRESHOLDS:
          for variant in VARIANTS:
           rng=random.Random(seed*1000003+size*1009+ti*97+oi)
           cases.append({"seed":seed,"size":size,"variant":variant,"threshold":threshold,**run_case(mem,target,outcome,rng,variant,tok,threshold)})
    summary={}
    for v in VARIANTS:
      summary[v]={}
      for t in THRESHOLDS:
       rows=[r for r in cases if r["variant"]==v and r["threshold"]==t]
       accepted=[r for r in rows if r["accepted"]]
       correct=[r for r in accepted if r["resolver_outcome"]=="correct"]
       wrong=[r for r in accepted if r["resolver_outcome"]=="wrong_unique"]
       ambiguous=[r for r in accepted if r["resolver_outcome"]=="ambiguous"]
       summary[v][str(t)]={"cases":len(rows),"accepted":len(accepted),"coverage_pct":100*len(accepted)/len(rows),"correct_acceptance_pct":100*len(correct)/CASES_PER_OUTCOME,"wrong_unique_false_acceptance_pct":100*len(wrong)/CASES_PER_OUTCOME,"ambiguous_false_acceptance_pct":100*len(ambiguous)/CASES_PER_OUTCOME,"closure_exact_pct":100*sum(r["closure_exact"] for r in rows)/len(rows),"answer_exact_pct":100*sum(r["answer_exact"] for r in rows)/len(rows),"transport_exact_pct":100*sum(r["transport_exact"] for r in rows)/len(rows),"mean_selected_token_reduction_pct":sum(r["selected_token_reduction_pct"] for r in accepted)/len(accepted) if accepted else None}
    out={"simulation":35,"benchmark_version":VERSION,"seeds":SEEDS,"sizes":SIZES,"targets_per_state":TARGETS_PER_STATE,"outcomes":OUTCOMES,"thresholds":THRESHOLDS,"tokenizer":ENCODING,"variants":list(VARIANTS),"cases":cases,"summary":summary,"result_data_sha256":hashlib.sha256(json.dumps(cases,sort_keys=True).encode()).hexdigest(),"scope":"Synthetic confidence/coverage policy benchmark. Resolver confidence and outcomes are injected upstream; thresholding, structural closure, deterministic reconstruction, transport, and exact answer checks are measured. Confidence is not semantic resolver accuracy. No natural-language understanding, learned retrieval, LLM equivalence, latency, universal generalization, consciousness, or optimality claim."}
    Path(args.output).write_text(json.dumps(out,indent=2,sort_keys=True),encoding="utf-8"); print(json.dumps(summary,indent=2,sort_keys=True))
if __name__=="__main__": main()
