"""Provider-neutral model adapter boundary.

Adapters receive a ModelRequest and return a ModelResponse. Real provider
credentials and network clients intentionally live outside this core module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from context_protocol import ModelRequest


@dataclass(frozen=True)
class ModelResponse:
    provider: str
    text: str
    request_fingerprint: str
    usage: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


class ProviderAdapter(Protocol):
    provider_name: str

    def generate(self, request: ModelRequest) -> ModelResponse:
        ...


class DryRunAdapter:
    """Deterministic adapter for integration tests without network access."""

    provider_name = "dry-run"

    def generate(self, request: ModelRequest) -> ModelResponse:
        text = request.envelope.context
        return ModelResponse(
            provider=self.provider_name,
            text=text,
            request_fingerprint=request.fingerprint(),
            usage={"network_call": False},
            metadata={"mode": "dry-run"},
        )
