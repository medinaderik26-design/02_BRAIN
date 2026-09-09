"""Versioned provider-neutral context packet protocol.

The protocol defines what crosses the user-to-model boundary. It does not
claim privacy, semantic equivalence, or provider-specific token savings.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

PROTOCOL_VERSION = "1.0"


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True)
class ContextEnvelope:
    request_id: str
    query: str
    context: str
    selected_keys: tuple[str, ...] = ()
    excluded_keys: tuple[str, ...] = ()
    source_sha256: str = ""
    context_sha256: str = ""
    policy: Mapping[str, Any] = field(default_factory=dict)
    protocol_version: str = PROTOCOL_VERSION
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "request_id": self.request_id,
            "created_at": self.created_at,
            "query": self.query,
            "context": self.context,
            "selected_keys": list(self.selected_keys),
            "excluded_keys": list(self.excluded_keys),
            "source_sha256": self.source_sha256,
            "context_sha256": self.context_sha256,
            "policy": dict(self.policy),
        }

    def canonical(self) -> str:
        return _canonical_json(self.to_dict())

    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical().encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ModelRequest:
    envelope: ContextEnvelope
    model_hint: str = ""
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def canonical(self) -> str:
        return _canonical_json({
            "envelope": self.envelope.to_dict(),
            "model_hint": self.model_hint,
            "parameters": dict(self.parameters),
        })

    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical().encode("utf-8")).hexdigest()


def envelope_from_gateway_packet(packet: Any, policy: Mapping[str, Any] | None = None) -> ContextEnvelope:
    """Convert a ContextGateway packet without coupling the protocol to its class."""
    return ContextEnvelope(
        request_id=packet.request_id,
        query=packet.query,
        context=packet.context,
        selected_keys=tuple(packet.selected_keys),
        excluded_keys=tuple(packet.excluded_keys),
        source_sha256=packet.source_sha256,
        context_sha256=packet.packet_sha256,
        policy=dict(policy or {}),
    )
