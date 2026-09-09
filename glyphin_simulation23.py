"""Glyphin Simulation 23 — Information Attribution / Factorization Ablation.

Measures which generic factorization choices contribute to exact compression.
Each ablation preserves the complete GlyphinMemory state and is independently
refereed. Only one factorization choice is disabled per ablation relative to
state-columnar. Evidence is benchmark-specific; no optimality claim.
"""
from __future__ import annotations
import argparse, hashlib, json, statistics
from datetime import datetime, timedelta, timezone
import tiktoken
from glyphin_simulation21 import build_memory, SEED as SIM21_SEED
from glyphin_simulation19 import esc, split_escaped
from glyphin_state_referee import referee_memory

VERSION="23.0"
SIZES=(4,8,16,32,64,128,256)
TOKENIZER="cl100k_base"
SEP=";"
SAFE={",",";","|",":","~","@"}


def parts(memory):
    names=sorted(memory.states); idx={n:i for i,n in enumerate(names)}
    states=[memory.states[n] for n in names]
    times=[datetime.fromisoformat(s.created_at) for s in states]
    if any(t.utcoffset() is None for t in times): raise ValueError("timestamps must be timezone-aware")
    offsets=[int(t.utcoffset().total_seconds()//60) for t in times]
    utc=[int(t.astimezone(timezone.utc).timestamp())*1_000_000+t.astimezone(timezone.utc).microsecond for t in times]
    base=min(utc); deltas=[u-base for u in utc]
    return names,idx,states,offsets,base,deltas


def enc_reference(m):
    names,idx,s,offs,base,deltas=parts(m)
    return SEP.join([
        f"{m.decay_lambda!r},{m.alpha!r},{m.beta!r}",
        ",".join(esc(n,SAFE) for n in names),
        ",".join("" if x.parent is None else str(idx[x.parent]) for x in s),
        ",".join(str(x.level) for x in s),", ".join(repr(x.cohesion) for x in s),
        ",".join(str(x.frequency) for x in s),", ".join(repr(x.resonance) for x in s),
        ",".join(esc(x.sigma,SAFE) for x in s),
        f"{base}:{','.join(map(str,offs))}",", ".join(map(str,deltas))
    ])

# The following variants disable exactly one generic factorization choice.
def enc_no_param_factor(m):
    names,idx,s,offs,base,deltas=parts(m)
    rows=[]
    for i,x in enumerate(s):
        p="" if x.parent is None else str(idx[x.parent])
        rows.append("|".join([f"{m.decay_lambda!r}",f"{m.alpha!r}",f"{m.beta!r}",esc(names[i],SAFE),p,str(x.level),repr(x.cohesion),str(x.frequency),repr(x.resonance),esc(x.sigma,SAFE),str(offs[i]),str(base+deltas[i])]))
    return "R"+SEP+SEP.join(rows)


def enc_no_name_dictionary(m):
    names,idx,s,offs,base,deltas=parts(m)
    return SEP.join([f"{m.decay_lambda!r},{m.alpha!r},{m.beta!r}",
        ",".join(esc(n,SAFE) for n in names),
        ",".join("" if x.parent is None else esc(x.parent,SAFE) for x in s),
        ",".join(str(x.level) for x in s),", ".join(repr(x.cohesion) for x in s),
        ",".join(str(x.frequency) for x in s),", ".join(repr(x.resonance) for x in s),
        ",".join(esc(x.sigma,SAFE) for x in s),f"{base}:{','.join(map(str,offs))}",", ".join(map(str,deltas))
    ])


def enc_no_time_delta(m):
    names,idx,s,offs,base,deltas=parts(m)
    return SEP.join([f"{m.decay_lambda!r},{m.alpha!r},{m.beta!r}",", ".join(esc(n,SAFE) for n in names),
        ",".join("" if x.parent is None else str(idx[x.parent]) for x in s),", ".join(str(x.level) for x in s),", ".join(repr(x.cohesion) for x in s),
        ",".join(str(x.frequency) for x in s),", ".join(repr(x.resonance) for x in s),", ".join(esc(x.sigma,SAFE) for x in s),
        ",".join(esc(x.created_at,SAFE) for x in s)])


def enc_no_columnar(m):
    names,idx,s,offs,base,deltas=parts(m)
    rows=[]
    for i,x in enumerate(s):
        p="" if x.parent is None else str(idx[x.parent])
        rows.append("|".join([esc(names[i],SAFE),p,str(x.level),repr(x.cohesion),str(x.frequency),repr(x.resonance),esc(x.sigma,SAFE),esc(x.created_at,SAFE)]))
    return SEP.join([f"{m.decay_lambda!r},{m.alpha!r},{m.beta!r}"]+rows)


def decode_rowwise(text):
    sections=split_escaped(text,SEP,True)
    if not sections or sections[0]!="R": raise ValueError("not rowwise-param encoding")
    rows=sections[1:];
    if not rows: return None
    vals=[split_escaped(r,"|",False) for r in rows]
    mem=None
    pending=list(range(len(vals)))
    while pending:
        progress=False
        for i in list(pending):
            v=vals[i]; parent=None if v[4]=="" else None
            if v[4]!="":
                p=int(v[4]); parent=vals[p][3] if p<len(vals) and p not in pending else None
                if p in pending: continue
            if mem is None: mem=__import__('glyphin_research_core').glyphin_research_core.GlyphinMemory(float(v[1]),float(v[2]),float(v[3]))
            mem.add_state(v[3],level=int(v[5]),cohesion=float(v[6]),parent=parent,frequency=int(v[7]),resonance=float(v[8]),sigma=v[9],created_at=datetime.fromtimestamp(int(v[11])/1_000_000,tz=timezone.utc).astimezone(timezone(timedelta(minutes=int(v[10])))).isoformat())
            pending.remove(i); progress=True
        if not progress: raise ValueError("unresolvable rows")
    return mem


def decode_common(text, mode):
    r=split_escaped(text,SEP,True)
    if mode=="no-time":
        if len(r)!=9: raise ValueError("bad no-time record")
    else:
        if len(r)!=10: raise ValueError("bad columnar record")
    from glyphin_research_core import GlyphinMemory
    p=split_escaped(r[0],",")
    mem=GlyphinMemory(float(p[0]),float(p[1]),float(p[2]))
    names=split_escaped(r[1],",") if r[1] else []
    parents=split_escaped(r[2],",") if r[2] else []
    levels=[int(x) for x in r[3].split(",")]
    coh=[float(x) for x in r[4].split(",")]
    freq=[int(x) for x in r[5].split(",")]
    res=[float(x) for x in r[6].split(",")]
    sig=split_escaped(r[7],",") if r[7] else []
    if mode=="no-name": parent_names=parents
    if mode=="no-time": times=[x for x in split_escaped(r[8],",")]
    else:
        b,os=r[8].split(":",1); base=int(b); offs=[int(x) for x in os.split(",") if x]; ds=[int(x) for x in r[9].split(",") if x]
    if not names or any(len(v)!=len(names) for v in (parents,levels,coh,freq,res,sig)): raise ValueError("column mismatch")
    pending=set(range(len(names)))
    while pending:
        progress=False
        for i in sorted(pending):
            if mode=="no-name":
                parent=None if parent_names[i]=="" else parent_names[i]
                if parent is not None and parent not in mem.states: continue
            else:
                pi=parents[i]
                if pi and int(pi) in pending: continue
                parent=None if pi=="" else names[int(pi)]
            if mode=="no-time": created=times[i]
            else:
                u=datetime.fromtimestamp(base//1_000_000,tz=timezone.utc)+timedelta(microseconds=(base%1_000_000)+ds[i]); created=u.astimezone(timezone(timedelta(minutes=offs[i]))).isoformat()
            mem.add_state(names[i],level=levels[i],cohesion=coh[i],parent=parent,frequency=freq[i],resonance=res[i],sigma=sig[i],created_at=created); pending.remove(i); progress=True
        if not progress: raise ValueError("unresolvable parent graph")
    return mem


def dec_ref(t): return decode_common(t,"ref")
def dec_no_name(t): return decode_common(t,"no-name")
def dec_no_time(t): return decode_common(t,"no-time")
def dec_no_columnar(t):
    r=split_escaped(t,SEP,True); from glyphin_research_core import GlyphinMemory
    p=split_escaped(r[0],","); mem=GlyphinMemory(float(p[0]),float(p[1]),float(p[2])); vals=[split_escaped(x,"|",False) for x in r[1:]]
    pending=set(range(len(vals)))
    while pending:
        progress=False
        for i in sorted(pending):
            v=vals[i]; parent=None if v[1]=="" else vals[int(v[1])][0]
            if parent is not None and parent not in mem.states: continue
            mem.add_state(v[0],level=int(v[2]),cohesion=float(v[3]),parent=parent,frequency=int(v[4]),resonance=float(v[5]),sigma=v[6],created_at=v[7]); pending.remove(i); progress=True
        if not progress: raise ValueError("unresolvable row graph")
    return mem

VARIANTS={"state-columnar-reference":(enc_reference,dec_ref),"no-param-factor":(enc_no_param_factor,decode_rowwise),"no-name-dictionary":(enc_no_name_dictionary,dec_no_name),"no-time-delta":(enc_no_time_delta,dec_no_time),"no-columnarization":(enc_no_columnar,dec_no_columnar)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output",default="glyphin_simulation23_result.json"); a=ap.parse_args(); tok=tiktoken.get_encoding(TOKENIZER)
    cases=[]
    for n in SIZES:
        mem=build_memory(n,SIM21_SEED+n); base=mem.to_json(); bt=len(tok.encode(base,disallowed_special=()))
        for name,(enc,dec) in VARIANTS.items():
            try:
                text=enc(mem); rebuilt=dec(text); ref=referee_memory(mem,rebuilt); et=len(tok.encode(text,disallowed_special=()));
                cases.append({"size":n,"variant":name,"baseline_tokens":bt,"encoded_tokens":et,"token_reduction_pct":(1-et/bt)*100,"chars":len(text),"exact":ref.exact,"field_mismatches":ref.field_mismatches,"parameter_mismatches":ref.parameter_mismatches})
            except Exception as e: cases.append({"size":n,"variant":name,"baseline_tokens":bt,"encoded_tokens":None,"token_reduction_pct":None,"chars":None,"exact":False,"error":type(e).__name__+":"+str(e)})
    summary={}
    for v in VARIANTS:
        rows=[x for x in cases if x["variant"]==v]; good=[x["token_reduction_pct"] for x in rows if x["exact"]]
        summary[v]={"total_cases":len(rows),"exact_cases":sum(x["exact"] for x in rows),"exact_rate_pct":100*sum(x["exact"] for x in rows)/len(rows),"mean_token_reduction_pct":statistics.mean(good) if good else None,"median_token_reduction_pct":statistics.median(good) if good else None}
    out={"simulation":23,"benchmark_version":VERSION,"purpose":"Information attribution via one-factor-at-a-time factorization ablation on deterministic Sim21 high-entropy memories.","seed":SIM21_SEED,"sizes":list(SIZES),"tokenizer":TOKENIZER,"variants":list(VARIANTS),"cases":cases,"summary":summary,"scope":"Benchmark-specific evidence only; not a global optimum, universal claim, or proof of cross-model semantic equivalence."}
    raw=json.dumps(out,sort_keys=True,separators=(",",":")).encode(); out["result_data_sha256"]=hashlib.sha256(raw).hexdigest(); Path(a.output).write_text(json.dumps(out,indent=2,sort_keys=True)+"\n",encoding="utf-8"); print(json.dumps(summary,indent=2,sort_keys=True))
    return 0 if all(x["exact"] for x in cases) else 1
if __name__=="__main__": raise SystemExit(main())
