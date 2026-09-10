"""Glyphin Simulation 33: surface-form robustness boundary.

Sim33 is designed to measure the boundary between deterministic Glyphin retrieval
and an external semantic/concept resolver. It intentionally does NOT claim
natural-language understanding. Cases contain a canonical descriptor, controlled
synonyms, reordered surface forms, punctuation variation, and unsupported free
paraphrases. The deterministic resolver is scored only on exact descriptor-family
recognition; unsupported forms are expected to require an external resolver.
"""
from __future__ import annotations
import argparse, hashlib, json, re
from pathlib import Path
from glyphin_simulation21 import build_memory

VERSION="33.0"
SEEDS=(21092026,31092026,41092026,51092026,61092026)
SIZES=(256,1024,2048)


def names(memory):
    return sorted(memory.states)


def descriptor(state_index):
    # Stable compositional concept coordinate. This is an address-like fixture,
    # not a semantic embedding.
    a=state_index//256; b=(state_index//16)%16; c=state_index%16
    vocab=(
      ("alpha","first","primary"),("beta","second","secondary"),
      ("gamma","third","tertiary"),("delta","fourth","quaternary"),
      ("amber","gold","yellow"),("azure","blue","cyan"),
      ("crimson","red","scarlet"),("emerald","green","jade"),
      ("north","northern","upward"),("south","southern","downward"),
      ("east","eastern","rightward"),("west","western","leftward"),
      ("calm","quiet","steady"),("rapid","fast","quick"),
      ("dense","compact","thick"),("sparse","thin","scattered"))
    return (vocab[a%16][0],vocab[b][0],vocab[c][0])


def build_fixture(memory):
    return {n: descriptor(i) for i,n in enumerate(names(memory))}


def deterministic_resolve(query, index):
    q=query.lower()
    hits=[]
    for phrase,state in index.items():
        if re.search(r"(?<!\\w)"+re.escape(" ".join(phrase))+r"(?!\\w)",q):
            hits.append(state)
    return sorted(set(hits))


def make_cases(memory, fixture):
    rows=[]; ns=names(memory)
    for i,state in enumerate(ns[:min(len(ns),32)]):
        words=fixture[state]
        canonical=" ".join(words)
        rows.extend([
          {"state":state,"surface":"canonical","query":f"retrieve {canonical}","expected":[state],"deterministic_supported":True},
          {"state":state,"surface":"punctuation","query":f"retrieve ({words[0]}, {words[1]}, {words[2]})","expected":[state],"deterministic_supported":False},
          {"state":state,"surface":"reordered","query":f"retrieve {words[2]} {words[0]} {words[1]}","expected":[state],"deterministic_supported":False},
          {"state":state,"surface":"free_paraphrase","query":f"bring back the concept associated with {canonical}","expected":[state],"deterministic_supported":False},
        ])
    return rows


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',default='glyphin_simulation33_result.json'); args=ap.parse_args()
    rows=[]
    for seed in SEEDS:
        for size in SIZES:
            mem=build_memory(size,seed); fixture=build_fixture(mem)
            idx={v:k for k,v in fixture.items()}
            for case in make_cases(mem,fixture):
                resolved=deterministic_resolve(case['query'],idx)
                rows.append({**case,'seed':seed,'size':size,'resolved':resolved,'exact':resolved==case['expected']})
    out={"simulation":33,"benchmark_version":VERSION,"cases":rows,"result_data_sha256":hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest(),"scope":"Boundary benchmark only. Measures deterministic surface-form robustness and identifies cases requiring an external semantic resolver. No natural-language understanding, learned retrieval, LLM equivalence, universal generalization, latency, consciousness, or optimality claim."}
    Path(args.output).write_text(json.dumps(out,indent=2,sort_keys=True),encoding='utf-8')
    print(json.dumps({"cases":len(rows),"exact":sum(r['exact'] for r in rows),"unsupported_surface_cases":sum(not r['deterministic_supported'] for r in rows)},indent=2))

if __name__=='__main__': main()
