"""Provider-neutral LLM reader interface for GX-010.

The reader is intentionally separated from memory conditions. This keeps the
A/B/C experiment fair: the same reader implementation receives evidence from
Full History, Conventional Retrieval, or Glyphin.

This module provides a deterministic prompt builder and a small OpenAI-compatible
HTTP client. It does not hard-code a provider or require a cloud service. A local
server can be used by supplying its base URL and model name.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Callable, Dict, Optional
from urllib.request import Request, urlopen

from gx010_runner import MemoryEvidence, Question, ReaderResult


@dataclass(frozen=True)
class PromptConfig:
    system_prompt: str = (
        "Answer the user's question using only the supplied memory evidence. "
        "If the evidence does not contain enough information, answer UNKNOWN. "
        "Return only the answer, with no explanation."
    )


def build_prompt(question: Question, evidence: MemoryEvidence, config: PromptConfig | None = None) -> str:
    cfg = config or PromptConfig()
    return (
        f"SYSTEM:\n{cfg.system_prompt}\n\n"
        f"MEMORY EVIDENCE:\n{evidence.context}\n\n"
        f"QUESTION:\n{question.prompt}\n\n"
        "ANSWER:"
    )


class ReaderTransport:
    """Callable transport so tests can avoid network access."""

    def __call__(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


@dataclass
class OpenAICompatibleReader:
    """Minimal chat-completions reader for local OpenAI-compatible servers."""

    base_url: str
    model: str
    api_key: Optional[str] = None
    temperature: float = 0.0
    max_output_tokens: int = 128
    prompt_config: PromptConfig = PromptConfig()
    transport: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None

    def _request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.transport is not None:
            return self.transport(payload)
        url = self.base_url.rstrip("/") + "/v1/chat/completions"
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(url, data=body, headers=headers, method="POST")
        with urlopen(request, timeout=300) as response:
            return json.loads(response.read().decode("utf-8"))

    def answer(self, question: Question, evidence: MemoryEvidence) -> ReaderResult:
        prompt = build_prompt(question, evidence, self.prompt_config)
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
        }
        response = self._request(payload)
        choice = response["choices"][0]
        message = choice.get("message", {})
        answer = message.get("content", "").strip()
        usage = response.get("usage", {}) or {}
        return ReaderResult(
            answer=answer,
            output_tokens=usage.get("completion_tokens"),
            metadata={"input_tokens": usage.get("prompt_tokens"), "raw_model": self.model},
        )


__all__ = ["PromptConfig", "build_prompt", "OpenAICompatibleReader"]
