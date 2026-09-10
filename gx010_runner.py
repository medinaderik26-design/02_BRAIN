"""GX-010 experiment harness.

This module defines a provider-neutral runner for the long-term memory experiment.
It deliberately does not call a model provider. A condition adapter supplies the
memory write/query behavior and a reader adapter supplies model-mediated answers.

The runner records configuration and per-question evidence without changing the
scientific measurement definitions in GX-010_SPEC.md.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
import time
from typing import Any, Dict, Iterable, List, Optional, Protocol


@dataclass(frozen=True)
class Question:
    question_id: str
    prompt: str
    expected: Any
    category: str
    session_id: Optional[str] = None


@dataclass(frozen=True)
class MemoryEvidence:
    context: str
    memory_store_size: int = 0
    retrieval_operations: int = 0
    reconstruction_exact: Optional[bool] = None
    state_referee_exact: Optional[bool] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReaderResult:
    answer: Any
    output_tokens: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class MemoryAdapter(Protocol):
    def reset(self) -> None: ...
    def insert(self, item: Any) -> None: ...
    def query(self, question: Question) -> MemoryEvidence: ...


class ReaderAdapter(Protocol):
    def answer(self, question: Question, evidence: MemoryEvidence) -> ReaderResult: ...


@dataclass
class GX010Run:
    experiment_id: str
    run_id: str
    git_sha: str
    model_name: str
    model_version: str
    prompt_version: str
    tokenizer_name: str
    tokenizer_version: str
    condition: str
    ablation: Optional[str]
    fixture_seed: int
    memory_size: int
    question_count: int
    correct_count: int
    accuracy: float
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    memory_store_size: int
    retrieval_latency_ms: float
    end_to_end_latency_ms: float
    reconstruction_exact: Optional[bool]
    state_referee_exact: Optional[bool]
    failure_count: int
    artifact_sha256: str
    failures: List[Dict[str, Any]] = field(default_factory=list)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _answer_equal(observed: Any, expected: Any) -> bool:
    return _canonical(observed) == _canonical(expected)


def run_condition(
    *,
    questions: Iterable[Question],
    memory_items: Iterable[Any],
    memory: MemoryAdapter,
    reader: ReaderAdapter,
    experiment_id: str,
    run_id: str,
    git_sha: str,
    model_name: str,
    model_version: str,
    prompt_version: str,
    tokenizer_name: str,
    tokenizer_version: str,
    condition: str,
    fixture_seed: int,
    ablation: Optional[str] = None,
) -> GX010Run:
    """Run one controlled condition and return a serializable evidence record."""
    memory.reset()
    items = list(memory_items)
    for item in items:
        memory.insert(item)

    question_list = list(questions)
    correct = 0
    input_tokens: Optional[int] = 0
    output_tokens: Optional[int] = 0
    retrieval_ms = 0.0
    end_to_end_ms = 0.0
    retrieval_ops = 0
    store_size = 0
    reconstruction_values: List[bool] = []
    referee_values: List[bool] = []
    failures: List[Dict[str, Any]] = []

    for question in question_list:
        started = time.perf_counter()
        retrieval_started = time.perf_counter()
        evidence = memory.query(question)
        retrieval_ms += (time.perf_counter() - retrieval_started) * 1000.0
        result = reader.answer(question, evidence)
        end_to_end_ms += (time.perf_counter() - started) * 1000.0

        retrieval_ops += evidence.retrieval_operations
        store_size = max(store_size, evidence.memory_store_size)
        if evidence.reconstruction_exact is not None:
            reconstruction_values.append(evidence.reconstruction_exact)
        if evidence.state_referee_exact is not None:
            referee_values.append(evidence.state_referee_exact)

        if result.output_tokens is not None:
            output_tokens += result.output_tokens
        else:
            output_tokens = None

        measured_input = evidence.metadata.get("input_tokens")
        if measured_input is not None:
            if input_tokens is not None:
                input_tokens += int(measured_input)
        else:
            input_tokens = None

        if _answer_equal(result.answer, question.expected):
            correct += 1
        else:
            failures.append({
                "question_id": question.question_id,
                "category": question.category,
                "expected": question.expected,
                "observed": result.answer,
                "retrieved_evidence": evidence.context,
                "reconstruction_exact": evidence.reconstruction_exact,
                "state_referee_exact": evidence.state_referee_exact,
                "failure_category": result.metadata.get("failure_category", "answer_mismatch"),
            })

    count = len(question_list)
    accuracy = correct / count if count else 0.0
    reconstruction_exact = all(reconstruction_values) if reconstruction_values else None
    referee_exact = all(referee_values) if referee_values else None

    artifact_payload = {
        "experiment_id": experiment_id,
        "run_id": run_id,
        "git_sha": git_sha,
        "condition": condition,
        "ablation": ablation,
        "fixture_seed": fixture_seed,
        "questions": [asdict(q) for q in question_list],
        "failures": failures,
    }
    artifact_sha = sha256(_canonical(artifact_payload).encode("utf-8")).hexdigest()

    return GX010Run(
        experiment_id=experiment_id,
        run_id=run_id,
        git_sha=git_sha,
        model_name=model_name,
        model_version=model_version,
        prompt_version=prompt_version,
        tokenizer_name=tokenizer_name,
        tokenizer_version=tokenizer_version,
        condition=condition,
        ablation=ablation,
        fixture_seed=fixture_seed,
        memory_size=len(items),
        question_count=count,
        correct_count=correct,
        accuracy=accuracy,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        memory_store_size=store_size,
        retrieval_latency_ms=retrieval_ms,
        end_to_end_latency_ms=end_to_end_ms,
        reconstruction_exact=reconstruction_exact,
        state_referee_exact=referee_exact,
        failure_count=len(failures),
        artifact_sha256=artifact_sha,
        failures=failures,
    )


def write_jsonl(path: str, runs: Iterable[GX010Run]) -> None:
    """Write immutable run records as JSON Lines."""
    with open(path, "w", encoding="utf-8") as handle:
        for run in runs:
            handle.write(json.dumps(asdict(run), sort_keys=True, ensure_ascii=False) + "\n")
