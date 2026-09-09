"""End-to-end user-to-model gateway runtime."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from context_cache import ContextCache
from context_gateway import ContextGateway
from context_protocol import ModelRequest, envelope_from_gateway_packet
from provider_adapter import ModelResponse, ProviderAdapter


@dataclass(frozen=True)
class RuntimeResult:
    response: ModelResponse
    cache_hit: bool
    packet_fingerprint: str
    request_fingerprint: str
    cache_key: str


class GatewayRuntime:
    """Connect context selection, protocol, cache, and provider execution."""

    def __init__(self, gateway: ContextGateway, cache: ContextCache, provider: ProviderAdapter,
                 policy: Mapping[str, Any] | None = None) -> None:
        self.gateway = gateway
        self.cache = cache
        self.provider = provider
        self.policy = dict(policy or {})

    def prepare(self, query: str) -> ModelRequest:
        packet = self.gateway.build(query)
        envelope = envelope_from_gateway_packet(packet, self.policy)
        return ModelRequest(envelope=envelope)

    def run(self, query: str) -> RuntimeResult:
        request = self.prepare(query)
        key = request.cache_key()
        cached = self.cache.get(key)
        if cached is not None:
            return RuntimeResult(cached, True, request.envelope.fingerprint(), request.fingerprint(), key)

        response = self.provider.generate(request)
        self.cache.put(key, response)
        return RuntimeResult(response, False, request.envelope.fingerprint(), request.fingerprint(), key)
