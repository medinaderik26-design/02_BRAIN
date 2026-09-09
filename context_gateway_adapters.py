"""Provider-neutral downstream adapter contracts and a deterministic local adapter."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Protocol

@dataclass(frozen=True)
class ModelRequest:
    request_id: str
    query: str
    context: str

@dataclass(frozen=True)
class ModelResponse:
    request_id: str
    text: str
    input_chars: int
    output_chars: int

class DownstreamAdapter(Protocol):
    def send(self, request: ModelRequest) -> ModelResponse: ...

class EchoAdapter:
    """Local test adapter; never contacts a provider."""
    def send(self, request: ModelRequest) -> ModelResponse:
        text = request.context
        return ModelResponse(request.request_id, text, len(request.query)+len(request.context), len(text))

__all__ = ["ModelRequest", "ModelResponse", "DownstreamAdapter", "EchoAdapter"]
