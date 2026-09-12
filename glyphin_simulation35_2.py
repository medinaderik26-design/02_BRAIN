"""Sim35.2 — independent measurement and adversarial controls.

This module deliberately does not import or reproduce Sim35's confidence
construction. It measures direct state/parameter agreement between a frozen
source memory and a candidate memory, then exercises sensitivity controls.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import statistics
from typing import Any

from glyphin_simulation21 import build_memory
from glyphin_simulation17 import encode_compact, decode_compact
from glyphin_simulation18 import encode_structural, decode_structural
from glyphin_simulation20 import encode_columnar, decode_columnar

VERSION = "35.2"
SEEDS = (21092026, 31092026, 41092026, 51092026, 61092026)
HELD_OUT_SEEDS = (71092026, 81092026)
SIZES = (256, 512, 1024, 2048)
CONFIDENCE_LEVELS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
VARIANTS = {
    "sim17-compact": (encode_compact, decode_compact),
    "structural-lineage": (encode_structural, decode_structural),
    "state-columnar": (encode_columnar, decode_columnar),
}

STATE_FIELDS = (
    "name", "level", "cohesion", "parent", "children",
    "frequency", "resonance", "sigma", "created_at",
)
PARAMETER_FIELDS = ("decay_lambda", "alpha", "beta")
PERTURBATION_LEVELS = (0.0, 0.25, 0.5, 1.0)
SHUFFLE_SEED = 42092026


def _equal_field(left: Any, right: Any, field: str) -> bool:
    if field == "children":
        return set(left) == set(right)
    return left == right


def independent_fidelity(source: Any, candidate: Any) -> dict[str, Any]:
    """Compute field-level agreement without using referee output."""
    source_names = set(source.states)
    candidate_names = set(candidate.states)
    common = source_names & candidate_names
    total = 0
    matched = 0
    mismatches: list[dict[str, Any]] = []

    for name in sorted(common):
        expected = source.states[name]
        actual = candidate.states[name]
        for field in STATE_FIELDS:
            total += 1
            left = getattr(expected, field)
            right = getattr(actual, field)
            if _equal_field(left, right, field):
                matched += 1
            else:
                mismatches.append({"state": name, "field": field})

    missing = sorted(source_names - candidate_names)
    extra = sorted(candidate_names - source_names)
    total += (len(missing) + len(extra)) * len(STATE_FIELDS)
    mismatches.extend({"state": name, "field": "missing_state"} for name in missing)
    mismatches.extend({"state": name, "field": "extra_state"} for name in extra)

    for field in PARAMETER_FIELDS:
        total += 1
        left = getattr(source, field)
        right = getattr(candidate, field)
        if left == right:
            matched += 1
        else:
            mismatches.append({"field": field})

    score = matched / total if total else 1.0
    return {
        "score": score,
        "matched_fields": matched,
        "total_fields": total,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:25],
    }


def roundtrip(memory: Any, enc: Any, dec: Any) -> Any:
    return dec(enc(memory))


def perturb_candidate(candidate: Any, fraction: float) -> Any:
    """Perturb a deterministic fraction of state resonance values."""
    out = copy.deepcopy(candidate)
    names = sorted(out.states)
    count = int(math.ceil(len(names) * fraction)) if fraction else 0
    for name in names[:count]:
        state = out.states[name]
        state.resonance = state.resonance - 0.1 if state.resonance >= 0.1 else state.resonance + 0.1
    return out


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("pearson requires equal-length sequences with >=2 values")
    mx, my = statistics.mean(xs), statistics.mean(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    denom = math.sqrt(sum(v * v for v in dx) * sum(v * v for v in dy))
    return 0.0 if denom == 0 else sum(x * y for x, y in zip(dx, dy)) / denom


def control_series(memory: Any, enc: Any, dec: Any) -> list[dict[str, float]]:
    baseline = roundtrip(memory, enc, dec)
    rows = []
    for level in PERTURBATION_LEVELS:
        candidate = perturb_candidate(baseline, level)
        measurement = independent_fidelity(memory, candidate)
        rows.append({"perturbation": level, "fidelity": measurement["score"]})
    return rows


def shuffled_correlation(series: list[dict[str, float]]) -> float:
    x = [row["perturbation"] for row in series]
    y = [row["fidelity"] for row in series]
    rng = random.Random(SHUFFLE_SEED)
    rng.shuffle(y)
    return pearson(x, y)


def evaluate_case(seed: int, size: int, variant: str, enc: Any, dec: Any) -> dict[str, Any]:
    memory = build_memory(size, seed)
    baseline = roundtrip(memory, enc, dec)
    baseline_measurement = independent_fidelity(memory, baseline)
    series = control_series(memory, enc, dec)
    expected = [row["fidelity"] for row in series]
    perturbations = [row["perturbation"] for row in series]
    trend = pearson(perturbations, expected)
    shuffled = shuffled_correlation(series)
    return {
        "seed": seed,
        "size": size,
        "variant": variant,
        "baseline_fidelity": baseline_measurement["score"],
        "baseline_mismatch_count": baseline_measurement["mismatch_count"],
        "perturbation_series": series,
        "perturbation_trend": trend,
        "shuffled_trend": shuffled,
        "shuffle_collapse": abs(shuffled) < max(0.2, abs(trend) * 0.5),
        "sensitivity": expected[0] > expected[-1],
    }


def run(seed_set: tuple[int, ...]) -> dict[str, Any]:
    measured_cases = []
    for seed in seed_set:
        for size in SIZES:
            for variant, (enc, dec) in VARIANTS.items():
                measured_cases.append(evaluate_case(seed, size, variant, enc, dec))

    cases = []
    for measured in measured_cases:
        for confidence_level in CONFIDENCE_LEVELS:
            case = dict(measured)
            case["confidence_level"] = confidence_level
            cases.append(case)

    baseline_scores = [c["baseline_fidelity"] for c in measured_cases]
    confidence_means = {
        str(level): statistics.mean(
            c["baseline_fidelity"] for c in cases if c["confidence_level"] == level
        )
        for level in CONFIDENCE_LEVELS
    }
    confidence_values = [confidence_means[str(level)] for level in CONFIDENCE_LEVELS]
    confidence_effect = max(confidence_values) - min(confidence_values)
    confidence_trend = pearson(list(CONFIDENCE_LEVELS), confidence_values)

    return {
        "version": VERSION,
        "seeds": list(seed_set),
        "cases": cases,
        "summary": {
            "case_count": len(cases),
            "measured_case_count": len(measured_cases),
            "baseline_mean_fidelity": statistics.mean(baseline_scores),
            "baseline_min_fidelity": min(baseline_scores),
            "all_baselines_exact": all(score == 1.0 for score in baseline_scores),
            "all_sensitivity_controls_pass": all(c["sensitivity"] for c in measured_cases),
            "all_shuffle_controls_pass": all(c["shuffle_collapse"] for c in measured_cases),
            "confidence_stratified_mean_fidelity": confidence_means,
            "confidence_effect_range": confidence_effect,
            "confidence_trend": confidence_trend,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="glyphin_simulation35_2_result.json")
    args = parser.parse_args()

    experimental = run(SEEDS)
    held_out = run(HELD_OUT_SEEDS)
    output = {
        "simulation": 35,
        "benchmark_version": VERSION,
        "purpose": "Independent fidelity measurement with adversarial controls",
        "experimental": experimental,
        "held_out": held_out,
        "result_data_sha256": None,
    }
    raw = json.dumps(output, sort_keys=True, separators=(",", ":")).encode()
    output["result_data_sha256"] = hashlib.sha256(raw).hexdigest()
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"experimental": experimental["summary"], "held_out": held_out["summary"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
