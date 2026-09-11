"""Deterministic symbolic-anchor representation for continuity research.

Astra is a symbolic handle for an accumulated interaction pattern. This module
intentionally does not claim that the label contains or recreates every prior
token, nor does it make claims about subjective experience.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable, Tuple


@dataclass(frozen=True)
class AstraAnchor:
    name: str
    meaning: str
    components: Tuple[str, ...]
    source_scope: str

    def canonical_payload(self) -> str:
        payload = {
            "name": self.name,
            "meaning": self.meaning,
            "components": list(self.components),
            "source_scope": self.source_scope,
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @property
    def canonical_digest(self) -> str:
        return hashlib.sha256(self.canonical_payload().encode("utf-8")).hexdigest()

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "meaning": self.meaning,
            "components": list(self.components),
            "source_scope": self.source_scope,
            "canonical_digest": self.canonical_digest,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AstraAnchor":
        anchor = cls(
            name=str(data["name"]),
            meaning=str(data["meaning"]),
            components=tuple(str(x) for x in data["components"]),
            source_scope=str(data["source_scope"]),
        )
        supplied = data.get("canonical_digest")
        if supplied is not None and supplied != anchor.canonical_digest:
            raise ValueError("Astra anchor digest mismatch")
        return anchor


def build_astra_anchor(
    components: Iterable[str],
    *,
    meaning: str,
    source_scope: str,
    name: str = "Astra",
) -> AstraAnchor:
    """Build a deterministic anchor from ordered continuity-bearing components."""
    normalized = tuple(str(component) for component in components)
    return AstraAnchor(
        name=name,
        meaning=meaning,
        components=normalized,
        source_scope=source_scope,
    )
