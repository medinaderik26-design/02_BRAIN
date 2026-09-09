"""Glyphin Simulation 15 — compact semantic symbolic encoding."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from glyphin_compression import measure
from glyphin_research_core import GlyphinMemory
from glyphin_state_referee import referee_memory

VERSION = "15.3"
NULL = ""
ESCAPE = "\\"


def esc(value: str) -> str:
    out = []
    for char in str(value):
        if char in (ESCAPE, "|", ";", ","):
            out.append(ESCAPE)
        out.append(char)
    return "".join(out)


def split_escaped(text: str, delimiter: str, preserve_escapes: bool = False) -> list[str]:
    """Split one grammar layer; optionally retain escapes for a nested layer."""
    parts: list[str] = []
    current: list[str] = []
    escaped = False
    for char in text:
        if escaped:
            if preserve_escapes:
                current.extend((ESCAPE, char))
            else:
                current.append(char)
            escaped = False
        elif char == ESCAPE:
            escaped = True
        elif char == delimiter:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
    if escaped:
        raise ValueError("dangling escape")
    parts.append("".join(current))
    return parts


def encode_semantic(memory: GlyphinMemory) -> str:
    params = f"P{memory.decay_lambda!r},{memory.alpha!r},{memory.beta!r}"
    records = [params]
    for name in sorted(memory.states):
        state = memory.states[name]
        parent = "" if state.parent is None else state.parent
        fields = [state.name, str(state.level), repr(state.cohesion), parent,
                  str(state.frequency), repr(state.resonance), state.sigma,
                  state.created_at]
        records.append("S" + "|".join(esc(field) for field in fields))
    return ";".join(records)


def reconstruct_semantic(encoded: str) -> GlyphinMemory:
    records = split_escaped(encoded, ";", preserve_escapes=True)
    if not records or not records[0].startswith("P"):
        raise ValueError("missing parameter record")
    params = records[0][1:].split(",")
    if len(params) != 3:
        raise ValueError("parameter record must contain three values")
    memory = GlyphinMemory(decay_lambda=float(params[0]), alpha=float(params[1]), beta=float(params[2]))
    rows: list[tuple[str, int, float, str | None, int, float, str, str]] = []
    for record in records[1:]:
        if not record.startswith("S"):
            raise ValueError(f"unknown record: {record[:10]!r}")
        fields = split_escaped(record[1:], "|", preserve_escapes=False)
        if len(fields) != 8:
            raise ValueError("state record must contain eight fields")
        parent = None if fields[3] == NULL else fields[3]
        rows.append((fields[0], int(fields[1]), float(fields[2]), parent,
                     int(fields[4]), float(fields[5]), fields[6], fields[7]))
    remaining = {row[0]: row for row in rows}
    while remaining:
        progressed = False
        for name in sorted(tuple(remaining)):
            row = remaining[name]
            parent = row[3]
            if parent is None or parent in memory.states:
                memory.add_state(row[0], level=row[1], cohesion=row[2], parent=parent,
                                 frequency=row[4], resonance=row[5], sigma=row[6],
                                 created_at=row[7])
                del remaining[name]
                progressed = True
        if not progressed:
            raise ValueError("unresolvable parent relationships (cycle or missing parent)")
    return memory


@dataclass(frozen=True)
class CaseResult:
    name: str
    source_chars: int
    encoded_chars: int
    source_words: int
    encoded_words: int
    char_reduction_pct: float
    exact: bool
    source_count: int
    candidate_count: int
    field_mismatches: tuple[dict, ...]
    parameter_mismatches: tuple[dict, ...]


def build_fixture(name: str) -> GlyphinMemory:
    memory = GlyphinMemory(decay_lambda=0.02, alpha=0.30, beta=0.20)
    if name == "chain":
        rows = [("root",0,.90,None,4,.80,"echo"),("a",1,.70,"root",3,.65,"echo"),("b",2,.55,"a",2,.60,"echo"),("c",3,.40,"b",1,.50,"seed")]
    elif name == "branch":
        rows = [("root",0,.90,None,5,.90,"echo"),("left",1,.70,"root",3,.70,"echo"),("right",1,.60,"root",2,.60,"echo"),("leaf",2,.30,"left",1,.40,"seed")]
    elif name == "mixed":
        rows = [("alpha",0,.91,None,6,.88,"echo"),("beta",1,.73,"alpha",4,.77,"echo"),("gamma",1,.52,"alpha",2,.61,"echo"),("delta",2,.31,"beta",1,.44,"seed"),("epsilon",2,.49,"gamma",2,.57,"echo")]
    elif name == "escaping":
        rows = [("r|oot",0,.91,None,2,.88,"seed;root"),("a,b",1,.73,"r|oot",2,.77,"echo\\x"),("t~z",2,.52,"a,b",1,.61,"sigma|;")]
    else:
        raise ValueError(f"unknown fixture: {name}")
    for i, (node, level, cohesion, parent, frequency, resonance, sigma) in enumerate(rows):
        memory.add_state(node, level=level, cohesion=cohesion, parent=parent,
                         frequency=frequency, resonance=resonance, sigma=sigma,
                         created_at=f"2026-01-01T00:00:{i:02d}+00:00")
    return memory


def run_case(name: str) -> CaseResult:
    source = build_fixture(name)
    source_text = source.to_json()
    encoded = encode_semantic(source)
    candidate = reconstruct_semantic(encoded)
    verdict = referee_memory(source, candidate)
    metrics = measure(source_text, encoded)
    return CaseResult(name, metrics.source_chars, metrics.encoded_chars,
                      metrics.source_words, metrics.encoded_words,
                      metrics.char_reduction_pct, verdict.exact,
                      verdict.source_count, verdict.candidate_count,
                      verdict.field_mismatches, verdict.parameter_mismatches)


def run(output: Path) -> dict:
    cases = [run_case(name) for name in ("chain", "branch", "mixed", "escaping")]
    data = {"benchmark_version": VERSION, "cases": [asdict(c) for c in cases]}
    data["summary"] = {
        "total_cases": len(cases),
        "exact_cases": sum(c.exact for c in cases),
        "exact_rate_pct": sum(c.exact for c in cases) / len(cases) * 100,
        "mean_char_reduction_pct": sum(c.char_reduction_pct for c in cases) / len(cases),
    }
    raw = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    data["data_sha256"] = hashlib.sha256(raw).hexdigest()
    output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("glyphin_simulation15_result.json"))
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0 if result["summary"]["exact_cases"] == result["summary"]["total_cases"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
