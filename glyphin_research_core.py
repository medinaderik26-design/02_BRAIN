"""Glyphin Research Core

A small, dependency-free implementation of the formal Glyphin state model.
This module is intentionally separate from the legacy Glyphin v1.7 runtime.
It provides the missing experimentally testable primitives:

- named glyph states
- parent/child lineage
- recurrence/frequency
- cohesion and resonance
- deterministic decay/update
- lineage-aware recall
- canonical JSON serialization

This is an implementation artifact, not evidence that the underlying Glyphin
hypothesis is validated. Validation belongs to the experiment/referee layer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import math
from typing import Dict, Iterable, List, Optional


@dataclass
class GlyphState:
    """Formal symbolic state: g = <n, l, c, p, chi, f, rho, sigma>."""

    name: str
    level: int = 0
    cohesion: float = 0.0
    parent: Optional[str] = None
    children: List[str] = field(default_factory=list)
    frequency: int = 1
    resonance: float = 1.0
    sigma: str = "seed"
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must not be empty")
        if not 0 <= self.cohesion <= 1:
            raise ValueError("cohesion must be in [0, 1]")
        if not 0 <= self.resonance <= 1:
            raise ValueError("resonance must be in [0, 1]")
        if self.frequency < 1:
            raise ValueError("frequency must be >= 1")
        self.children = list(dict.fromkeys(self.children))


class GlyphinMemory:
    """Deterministic in-memory symbolic topology for experiments."""

    def __init__(self, decay_lambda: float = 0.01, alpha: float = 0.25, beta: float = 0.25):
        if decay_lambda < 0 or alpha < 0 or beta < 0:
            raise ValueError("decay_lambda, alpha and beta must be non-negative")
        self.decay_lambda = decay_lambda
        self.alpha = alpha
        self.beta = beta
        self.states: Dict[str, GlyphState] = {}

    def add_state(
        self,
        name: str,
        *,
        level: int = 0,
        cohesion: float = 0.0,
        parent: Optional[str] = None,
        frequency: int = 1,
        resonance: float = 1.0,
        sigma: str = "seed",
    ) -> GlyphState:
        if name in self.states:
            raise ValueError(f"state already exists: {name}")
        if parent is not None and parent not in self.states:
            raise KeyError(f"parent does not exist: {parent}")
        state = GlyphState(
            name=name,
            level=level,
            cohesion=cohesion,
            parent=parent,
            frequency=frequency,
            resonance=resonance,
            sigma=sigma,
        )
        self.states[name] = state
        if parent is not None:
            self.states[parent].children.append(name)
        return state

    def link_state(self, parent: str, child: str) -> None:
        if parent not in self.states or child not in self.states:
            raise KeyError("both parent and child must exist")
        if child not in self.states[parent].children:
            self.states[parent].children.append(child)
        if self.states[child].parent is None:
            self.states[child].parent = parent

    def decay(self, delta: float) -> None:
        """Apply rho <- rho * exp(-lambda * delta) to every state."""
        if delta < 0:
            raise ValueError("delta must be non-negative")
        factor = math.exp(-self.decay_lambda * delta)
        for state in self.states.values():
            state.resonance = max(0.0, min(1.0, state.resonance * factor))

    def reinforce(self, name: str, kappa: float = 1.0) -> GlyphState:
        """Apply the specified recurrence update to one state."""
        if name not in self.states:
            raise KeyError(name)
        if not 0 <= kappa <= 1:
            raise ValueError("kappa must be in [0, 1]")
        state = self.states[name]
        state.cohesion = min(1.0, state.cohesion + self.alpha * kappa * (1 - state.cohesion))
        state.resonance = min(1.0, state.resonance + self.beta * kappa * (1 - state.resonance))
        state.frequency += 1
        state.sigma = "echo" if state.frequency >= 2 else "seed"
        return state

    def get_path(self, name: str) -> List[str]:
        """Return root-to-state lineage using parent links."""
        if name not in self.states:
            raise KeyError(name)
        path: List[str] = []
        seen = set()
        current: Optional[str] = name
        while current is not None:
            if current in seen:
                raise ValueError(f"parent cycle detected at {current}")
            seen.add(current)
            path.append(current)
            current = self.states[current].parent
        path.reverse()
        return path

    def recall(self, name: str) -> Dict[str, object]:
        """Return the symbolic state plus its lineage, not a similarity search."""
        state = self.states[name]
        return {
            "target": name,
            "state": asdict(state),
            "lineage": self.get_path(name),
        }

    def hyperstate(self, threshold: float = 0.8) -> List[str]:
        return sorted(
            name for name, state in self.states.items() if state.cohesion >= threshold
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "version": "glyphin-research-core-0.1",
            "parameters": {
                "decay_lambda": self.decay_lambda,
                "alpha": self.alpha,
                "beta": self.beta,
            },
            "states": {
                name: asdict(self.states[name]) for name in sorted(self.states)
            },
        }

    def to_json(self) -> str:
        """Canonical serialization for reproducible artifact hashing."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "GlyphinMemory":
        params = payload.get("parameters", {})
        memory = cls(
            decay_lambda=float(params.get("decay_lambda", 0.01)),
            alpha=float(params.get("alpha", 0.25)),
            beta=float(params.get("beta", 0.25)),
        )
        states = payload.get("states", {})
        if not isinstance(states, dict):
            raise ValueError("states must be an object")
        for name in sorted(states):
            raw = dict(states[name])
            raw.pop("children", None)
            memory.states[name] = GlyphState(name=name, **raw)
        return memory

    @classmethod
    def from_json(cls, text: str) -> "GlyphinMemory":
        return cls.from_dict(json.loads(text))


__all__ = ["GlyphState", "GlyphinMemory"]
