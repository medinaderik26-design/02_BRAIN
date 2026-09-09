"""Glyphin Simulation 14 — semantic/state-preserving compression.

This experiment deliberately goes beyond topology. It encodes the complete
research-core GlyphState payload plus memory parameters, reconstructs a fresh
GlyphinMemory, and lets the independent state referee decide fidelity.

No optimizer/global-minimum claim is made. The representation is a transparent
semantic baseline designed to establish whether state can survive symbolic
compression before optimization is attempted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from glyphin_compression import measure
from glyphin_research_core import GlyphinMemory
from glyphin_state_referee import referee_memory

VERSION = "14.0"


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
    base = "2026-01-01T00:00:00+00:00"
    if name == "chain":
        rows = [
            ("root", 0, 0.90, None, 4, 0.80, "echo"),
            ("a", 1, 0.70, "root", 3, 0.65, "echo"),
            ("b", 2, 0.55, "a", 2, 0.60, "echo"),
            ("c", 3, 0.40, "b", 1, 0.50, "seed"),
        ]
    elif name == "branch":
        rows = [
            ("root", 0, 0.90, None, 5, 0.90, "echo"),
            ("left", 1, 0.70, "root", 3, 0.70, "echo"),
            ("right", 1, 0.60, "root", 2, 0.60, "echo"),
            ("leaf", 2, 0.30, "left", 1, 0.40, "seed"),
        ]
    elif name == "mixed":
        rows = [
            ("alpha", 0, 0.91, None, 6, 0.88, "echo"),
            ("beta", 1, 0.73, "alpha", 4, 0.77, "echo"),
            ("gamma", 1, 0.52, "alpha", 2, 0.61, "echo"),
            ("delta", 2, 0.31, "beta", 1, 0.44, "seed"),
            ("epsilon", 2, 0.49, "gamma", 2, 0.57, "echo"),
        ]
    else:
        raise ValueError(f"unknown fixture: {name}")
    for i, (node, level, cohesion, parent, frequency, resonance, sigma) in enumerate(rows):
        timestamp = f"2026-01-01T00:00:{i:02d}+00:00"
        memory.add_state(node, level=level, cohesion=cohesion, parent=parent,
                         frequency=frequency, resonance=resonance, sigma=sigma,
                         created_at=timestamp)
    return memory


def encode_semantic(memory: GlyphinMemory) -> str:
    payload = memory.to_dict()
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def reconstruct_semantic(encoded: str) -> GlyphinMemory:
    return GlyphinMemory.from_json(encoded)


def run_case(name: str) -> CaseResult:
    source = build_fixture(name)
    source_text = source.to_json()
    encoded = encode_semantic(source)
    candidate = reconstruct_semantic(encoded)
    verdict = referee_memory(source, candidate)
    metrics = measure(source_text, encoded)
    return CaseResult(
        name=name,
        source_chars=metrics.source_chars,
        encoded_chars=metrics.encoded_chars,
        source_words=metrics.source_words,
        encoded_words=metrics.encoded_words,
        char_reduction_pct=metrics.char_reduction_pct,
        exact=verdict.exact,
        source_count=verdict.source_count,
        candidate_count=verdict.candidate_count,
        field_mismatches=verdict.field_mismatches,
        parameter_mismatches=verdict.parameter_mismatches,
    )


def run(output: Path) -> dict:
    cases = [run_case(name) for name in ("chain", "branch", "mixed")]
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
    parser.add_argument("--output", type=Path, default=Path("glyphin_simulation14_result.json"))
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0 if result["summary"]["exact_cases"] == result["summary"]["total_cases"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
