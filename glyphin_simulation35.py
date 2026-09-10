"""Glyphin Simulation 35: confidence coverage measurement.

Sim35 measures how comprehensively the encoding/decoding system covers different
confidence distributions and state coverage metrics. Tests various confidence
levels and measure loss/accuracy across different memory configurations.
"""
from __future__ import annotations
import argparse, hashlib, json, statistics, random
import tiktoken
from glyphin_simulation21 import build_memory
from glyphin_simulation17 import encode_compact, decode_compact
from glyphin_simulation18 import encode_structural, decode_structural
from glyphin_simulation20 import encode_columnar, decode_columnar
from glyphin_state_referee import referee_memory
from glyphin_research_core import GlyphinMemory

VERSION="35.0"
SEEDS=(21092026,31092026,41092026,51092026,61092026)
SIZES=(256,512,1024,2048)
CONFIDENCE_LEVELS=(0.5,0.6,0.7,0.8,0.9,0.95,0.99)
BATCH_SIZES=(1,4,16,64)
TOKENIZER="cl100k_base"
VARIANTS={"sim17-compact":(encode_compact,decode_compact),"structural-lineage":(encode_structural,decode_structural),"state-columnar":(encode_columnar,decode_columnar)}

def token_count(tok,text): return len(tok.encode(text,disallowed_special=()))

def measure_confidence_coverage(mem,confidence_level):
    """Measure how well the system covers memory at a given confidence threshold."""
    ns=sorted(mem.states)
    if not ns: return {"coverage":0,"state_count":0,"avg_confidence":0}
    
    confidences=[]
    for name in ns:
        state=mem.states[name]
        # Simulate confidence as combination of state properties
        base_conf=min(confidence_level,(state.resonance+state.frequency)/200.0)
        confidences.append(base_conf)
    
    avg_conf=statistics.mean(confidences) if confidences else 0
    covered=sum(1 for c in confidences if c>=confidence_level)
    coverage_ratio=covered/len(ns) if ns else 0
    
    return {
        "confidence_threshold":confidence_level,
        "coverage_ratio":coverage_ratio,
        "covered_states":covered,
        "total_states":len(ns),
        "avg_confidence":avg_conf
    }

def evaluate(mem,v,enc,dec,tok,confidence_level,seed,size):
    """Evaluate encoding/decoding at a specific confidence level."""
    encoded=enc(mem)
    rebuilt=dec(encoded)
    state_ref=referee_memory(mem,rebuilt)
    
    # Measure coverage
    coverage=measure_confidence_coverage(rebuilt,confidence_level)
    
    # Token measurements
    full_tokens=token_count(tok,encoded)
    
    return {
        "variant":v,
        "confidence_level":confidence_level,
        "seed":seed,
        "size":size,
        "state_exact":state_ref.exact,
        "coverage_ratio":coverage["coverage_ratio"],
        "covered_states":coverage["covered_states"],
        "total_states":coverage["total_states"],
        "avg_confidence":coverage["avg_confidence"],
        "tokens":full_tokens
    }

def summarize(rows):
    """Summarize results across all confidence levels."""
    variants={v:[] for v in VARIANTS}
    conf_levels={c:[] for c in CONFIDENCE_LEVELS}
    
    for r in rows:
        if r["state_exact"]:
            variants[r["variant"]].append(r)
            conf_levels[r["confidence_level"]].append(r)
    
    variant_summary={}
    for v,vrows in variants.items():
        if vrows:
            variant_summary[v]={
                "cases":len(vrows),
                "avg_coverage":statistics.mean(r["coverage_ratio"] for r in vrows),
                "avg_confidence":statistics.mean(r["avg_confidence"] for r in vrows),
                "avg_tokens":statistics.mean(r["tokens"] for r in vrows)
            }
    
    conf_summary={}
    for c,crows in conf_levels.items():
        if crows:
            conf_summary[f"level_{c}"]={
                "cases":len(crows),
                "avg_coverage":statistics.mean(r["coverage_ratio"] for r in crows),
                "avg_confidence":statistics.mean(r["avg_confidence"] for r in crows),
                "avg_tokens":statistics.mean(r["tokens"] for r in crows)
            }
    
    return {
        "total_cases":len(rows),
        "exact_state_cases":sum(1 for r in rows if r["state_exact"]),
        "variant_summary":variant_summary,
        "confidence_summary":conf_summary
    }

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output",default="glyphin_simulation35_result.json"); args=ap.parse_args()
    tok=tiktoken.get_encoding(TOKENIZER); rows=[]
    
    for seed in SEEDS:
        for size in SIZES:
            mem=build_memory(size,seed)
            for v,(enc,dec) in VARIANTS.items():
                for conf in CONFIDENCE_LEVELS:
                    result=evaluate(mem,v,enc,dec,tok,conf,seed,size)
                    rows.append(result)
    
    summary=summarize(rows)
    out={
        "simulation":35,
        "benchmark_version":VERSION,
        "purpose":"Confidence coverage measurement across different memory sizes and encoding variants",
        "summary":summary,
        "results":rows
    }
    
    raw=json.dumps(out,sort_keys=True,separators=(",",":")).encode()
    out["result_data_sha256"]=hashlib.sha256(raw).hexdigest()
    
    with open(args.output,"w",encoding="utf-8") as f:
        json.dump(out,f,indent=2,sort_keys=True)
        f.write("\n")
    
    print(json.dumps({"summary":out["summary"]},indent=2,sort_keys=True))
    return 0

if __name__=="__main__": raise SystemExit(main())
