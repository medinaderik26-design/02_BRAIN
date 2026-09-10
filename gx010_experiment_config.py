"""GX-010 controlled experiment configuration.

This module keeps experimental configuration explicit and deterministic. It does
not contact a model provider. A future CLI can consume these definitions when
the local LLM reader is connected.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Any, Dict, List

from gx010_baselines import FullHistoryMemory, LexicalRetrievalMemory, make_fixture
from gx010_glyphin_adapter import GlyphinMemoryAdapter, DeterministicContextReader
from gx010_runner import run_condition, GX010Run


@dataclass(frozen=True)
class GX010Config:
    experiment_id: str = "GX-010"
    fixture_seed: int = 1001
    model_name: str = "NOT_CONNECTED"
    model_version: str = "NOT_CONNECTED"
    prompt_version: str = "gx010-reader-v1"
    tokenizer_name: str = "NOT_MEASURED"
    tokenizer_version: str = "NOT_MEASURED"
    git_sha: str = "LOCAL_RUN"


def run_smoke(config: GX010Config | None = None) -> List[GX010Run]:
    """Run A/B/C through the common harness using the deterministic reader."""
    cfg = config or GX010Config()
    items, questions = make_fixture()
    reader = DeterministicContextReader()

    conditions = [
        ("A_FULL_HISTORY", FullHistoryMemory()),
        ("B_RETRIEVAL_MEMORY", LexicalRetrievalMemory(top_k=5)),
        ("C_GLYPHIN", GlyphinMemoryAdapter()),
    ]

    runs: List[GX010Run] = []
    for condition, memory in conditions:
        runs.append(
            run_condition(
                questions=questions,
                memory_items=items,
                memory=memory,
                reader=reader,
                experiment_id=cfg.experiment_id,
                run_id=f"{cfg.experiment_id}-{condition}-smoke",
                git_sha=cfg.git_sha,
                model_name=cfg.model_name,
                model_version=cfg.model_version,
                prompt_version=cfg.prompt_version,
                tokenizer_name=cfg.tokenizer_name,
                tokenizer_version=cfg.tokenizer_version,
                condition=condition,
                fixture_seed=cfg.fixture_seed,
            )
        )
    return runs


def write_config(path: str | Path, config: GX010Config | None = None) -> None:
    cfg = config or GX010Config()
    Path(path).write_text(
        json.dumps(asdict(cfg), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    for run in run_smoke():
        print(json.dumps(asdict(run), sort_keys=True))
