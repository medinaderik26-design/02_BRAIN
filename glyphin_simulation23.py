"""Simulation 23 — Information Attribution / Factorization Ablation.

One-factor-at-a-time ablation of generic state-columnar compression choices.
All GlyphinMemory fields remain represented and are checked by the independent
state referee. Benchmark-specific evidence only; no optimality claim.
"""
from __future__ import annotations
import argparse, hashlib, json, statistics
from datetime import datetime, timedelta, timezone
import tiktoken
from glyphin_simulation21 import build_memory, SEED as SEED21
from glyphin_simulation19 import esc, split_escaped
from glyphin_state_referee import referee_memory
from glyphin_research_core import GlyphinMemory

VERSION="23.1"; SIZES=(4,8,16,32,64,128,256); TOKENIZER="cl100k_base"; SAFE={",",";","|",":","~","@"}

def parts(m):
    names=sorted(m.states); idx={n:i for i,n in enumerate(names)}; states=[m.states[n] for n in names]
    ts=[datetime.fromisoformat(s.created_at) for s in states]
    if any(t.utcoffset() is None for t in ts): raise ValueError("timestamps must be timezone-aware")
    offs=[int(t.utcoffset().total_seconds()//60) for t in ts]
    utc=[int(t.astimezone(timezone.utc).timestamp())*1_000_000+t.astimezone(timezone.utc).microsecond for t in ts]
    base=min(utc); deltas=[u-base for u in utc]
    return names,idx,states,offs,base,deltas

def encode_reference(m):
    names,idx,s,offs,base,deltas=parts(m)
    return ";".join([f"{m.decay_lambda!r},{m.alpha!r},{m.beta!r}",",".join(esc(n,SAFE) for n in names),",".join("" if x.parent is None else str(idx[x.parent]) for x in s),",".join(str(x.level) for x in s),",".join(repr(x.cohesion) for x in s),",".join(str(x.frequency) for x in s),",".join(repr(x.resonance) for x in s),",".join(esc(x.sigma,SAFE) for x in s),f"{base}:{','.join(map(str,offs))}",",".join(map(str,deltas))])

def encode_no_params(m):
    names,idx,s,offs,base,deltas=parts(m); rows=[]
    for i,x in enumerate(s):
        rows.append("|".join([esc(names[i],SAFE),"" if x.parent is None else str(idx[x.parent]),str(x.level),repr(x.cohesion),str(x.frequency),repr(x.resonance),esc(x.sigma,SAFE),str(offs[i]),str(base+deltas[i]),repr(m.decay_lambda),repr(m.alpha),repr(m.beta)]))
    return "R"+";"+";".join(rows)

def encode_no_names(m):
    names,idx,s,offs,base,deltas=parts(m)
    return ";".join([f"{m.decay_lambda!r},{m.alpha!r},{m.beta!r}",",".join(esc(n,SAFE) for n in names),",".join("" if x.parent is None else esc(x.parent,SAFE) for x in s),",".join(str(x.level) for x in s),",".join(repr(x.cohesion) for x in s),",".join(str(x.frequency) for x in s),",".join(repr(x.resonance) for x in s),",".join(esc(x.sigma,SAFE) for x in s),f"{base}:{','.join(map(str,offs))}",",".join(map(str,deltas))])

def encode_no_time_delta(m):
    names,idx,s,offs,base,deltas=parts(m)
    return ";".join([f"{m.decay_lambda!r},{m.alpha!r},{m.beta!r}",",".join(esc(n,SAFE) for n in names),",".join("" if x.parent is None else str(idx[x.parent]) for x in s),",".join(str(x.level) for x in s),",".join(repr(x.cohesion) for x in s),",".join(str(x.frequency) for x in s),",".join(repr(x.resonance) for x in s),",".join(esc(x.sigma,SAFE) for x in s),",".join(esc(x.created_at,SAFE) for x in s)])

def encode_no_columnar(m):
    names,idx,s,offs,base,deltas=parts(m); rows=[]
    for i,x in enumerate(s): rows.append("|".join([esc(names[i],SAFE),"" if x.parent is None else str(idx[x.parent]),str(x.level),repr(x.cohesion),str(x.frequency),repr(x.resonance),esc(x.sigma,SAFE),esc(x.created_at,SAFE)]))
    return ";".join([f"{m.decay_lambda!r},{m.alpha!r},{m.beta!r}"]+rows)

def decode_no_params(text):
    rows=split_escaped(text,";",True)[1:]; vals=[split_escaped(r,"|",False) for r in rows]
    if not vals: raise ValueError("empty row encoding")
    mem=GlyphinMemory(float(vals[0][9]),float(vals[0][10]),float(vals[0][11])); pending=set(range(len(vals)))
    while pending:
        progress=False
        for i in sorted(pending):
            v=vals[i]; p=v[1]
            if p and int(p) in pending: continue
            parent=None if p=="" else vals[int(p)][0]
            t=datetime.fromtimestamp(int(v[8])/1_000_000,tz=timezone.utc).astimezone(timezone(timedelta(minutes=int(v[7])))).isoformat()
            mem.add_state(v[0],level=int(v[2]),cohesion=float(v[3]),parent=parent,frequency=int(v[4]),resonance=float(v[5]),sigma=v[6],created_at=t); pending.remove(i); progress=True
        if not progress: raise ValueError("unresolvable parent rows")
    return mem

def decode_columnar(text, mode):
    r=split_escaped(text,";",True); expected=9 if mode=="no-time" else 10
    if len(r)!=expected: raise ValueError(f"expected {expected} sections, got {len(r)}")
    p=split_escaped(r[0],","); mem=GlyphinMemory(float(p[0]),float(p[1]),float(p[2])); names=split_escaped(r[1],","); parents=split_escaped(r[2],","); levels=[int(x) for x in r[3].split(",")]; coh=[float(x) for x in r[4].split(",")]; freq=[int(x) for x in r[5].split(",")]; res=[float(x) for x in r[6].split(",")]; sig=split_escaped(r[7],",")
    if mode=="no-time": times=split_escaped(r[8],",")
    else: base_s,off_s=r[8].split(":",1); base=int(base_s); offs=[int(x) for x in off_s.split(",") if x]; ds=[int(x) for x in r[9].split(",") if x]
    if any(len(v)!=len(names) for v in (parents,levels,coh,freq,res,sig)): raise ValueError("column mismatch")
    pending=set(range(len(names)))
    while pending:
        progress=False
        for i in sorted(pending):
            if mode=="no-names":
                parent=None if parents[i]=="" else parents[i]
                if parent is not None and parent not in mem.states: continue
            else:
                pi=parents[i]
                if pi and int(pi) in pending: continue
                parent=None if pi=="" else names[int(pi)]
            if mode=="no-time": created=times[i]
            else:
                u=datetime.fromtimestamp(base//1_000_000,tz=timezone.utc)+timedelta(microseconds=base%1_000_000+ds[i]); created=u.astimezone(timezone(timedelta(minutes=offs[i]))).isoformat()
            mem.add_state(names[i],level=levels[i],cohesion=coh[i],parent=parent,frequency=freq[i],resonance=res[i],sigma=sig[i],created_at=created); pending.remove(i); progress=True
        if not progress: raise ValueError("unresolvable parent graph")
    return mem

def decode_ref(t): return decode_columnar(t,"ref")
def decode_no_names(t): return decode_columnar(t,"no-names")
def decode_no_time(t): return decode_columnar(t,"no-time")
def decode_no_columnar(t):
    r=split_escaped(t,";",True); p=split_escaped(r[0],","); mem=GlyphinMemory(float(p[0]),float(p[1]),float(p[2])); vals=[split_escaped(x,"|",False) for x in r[1:]]; pending=set(range(len(vals)))
    while pending:
        progress=False
        for i in sorted(pending):
            v=vals[i]; parent=None if v[1]=="" else vals[int(v[1])][0]
            if parent is not None and parent not in mem.states: continue
            mem.add_state(v[0],level=int(v[2]),cohesion=float(v[3]),parent=parent,frequency=int(v[4]),resonance=float(v[5]),sigma=v[6],created_at=v[7]); pending.remove(i); progress=True
        if not progress: raise ValueError("unresolvable row graph")
    return mem

VARIANTS={"state-columnar-reference":(encode_reference,decode_ref),"no-param-factor":(encode_no_params,decode_no_params),"no-name-dictionary":(encode_no_names,decode_no_names),"no-time-delta":(encode_no_time_delta,decode_no_time),"no-columnarization":(encode_no_columnar,decode_no_columnar)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output",default="glyphin_simulation23_result.json"); a=ap.parse_args(); tok=tiktoken.get_encoding(TOKENIZER); cases=[]
    for n in SIZES:
        m=build_memory(n,SEED21+n); base=m.to_json(); bt=len(tok.encode(base,disallowed_special=()))
        for name,(enc,dec) in VARIANTS.items():
            try:
                text=enc(m); rebuilt=dec(text); ref=referee_memory(m,rebuilt); et=len(tok.encode(text,disallowed_special=()))
                cases.append({"size":n,"variant":name,"baseline_tokens":bt,"encoded_tokens":et,"token_reduction_pct":(1-et/bt)*100,"chars":len(text),"exact":ref.exact,"field_mismatches":ref.field_mismatches,"parameter_mismatches":ref.parameter_mismatches})
            except Exception as e: cases.append({"size":n,"variant":name,"baseline_tokens":bt,"encoded_tokens":None,"token_reduction_pct":None,"chars":None,"exact":False,"error":type(e).__name__+":"+str(e)})
    summary={}
    for v in VARIANTS:
        rows=[x for x in cases if x["variant"]==v]; good=[x["token_reduction_pct"] for x in rows if x["exact"]]; summary[v]={"total_cases":len(rows),"exact_cases":sum(x["exact"] for x in rows),"exact_rate_pct":100*sum(x["exact"] for x in rows)/len(rows),"mean_token_reduction_pct":statistics.mean(good) if good else None,"median_token_reduction_pct":statistics.median(good) if good else None}
    out={"simulation":23,"benchmark_version":VERSION,"purpose":"One-factor-at-a-time information attribution for generic state-columnar compression.","seed":SEED21,"sizes":list(SIZES),"tokenizer":TOKENIZER,"variants":list(VARIANTS),"cases":cases,"summary":summary,"scope":"Benchmark-specific evidence only; not a global optimum or universal claim."}
    raw=json.dumps(out,sort_keys=True,separators=(",",":")).encode(); out["result_data_sha256"]=hashlib.sha256(raw).hexdigest(); open(a.output,"w",encoding="utf-8").write(json.dumps(out,indent=2,sort_keys=True)+"\n"); print(json.dumps(summary,indent=2,sort_keys=True)); return 0 if all(x["exact"] for x in cases) else 1
if __name__=="__main__": raise SystemExit(main())
