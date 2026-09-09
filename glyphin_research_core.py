"""Glyphin Research Core.

The executable, dependency-free implementation of the formal Glyphin state
model. Legacy ``glyphin.py`` remains separate and is not the research engine.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Dict, List, Optional


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
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

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
    """Persistent-capable symbolic memory for Glyphin experiments."""

    VERSION = "glyphin-research-core-0.3"

    def __init__(self, decay_lambda: float = 0.01, alpha: float = 0.25, beta: float = 0.25):
        if decay_lambda < 0 or alpha < 0 or beta < 0:
            raise ValueError("decay_lambda, alpha and beta must be non-negative")
        self.decay_lambda = decay_lambda
        self.alpha = alpha
        self.beta = beta
        self.states: Dict[str, GlyphState] = {}

    def add_state(self, name: str, *, level: int = 0, cohesion: float = 0.0,
                  parent: Optional[str] = None, frequency: int = 1,
                  resonance: float = 1.0, sigma: str = "seed") -> GlyphState:
        if name in self.states:
            raise ValueError(f"state already exists: {name}")
        if parent is not None and parent not in self.states:
            raise KeyError(f"parent does not exist: {parent}")
        if parent == name:
            raise ValueError("a state cannot be its own parent")
        state = GlyphState(name=name, level=level, cohesion=cohesion, parent=parent,
                           frequency=frequency, resonance=resonance, sigma=sigma)
        self.states[name] = state
        if parent is not None:
            self._add_child(parent, name)
        return state

    def _add_child(self, parent: str, child: str) -> None:
        if child not in self.states[parent].children:
            self.states[parent].children.append(child)

    def _rebuild_children(self) -> None:
        for state in self.states.values():
            state.children = []
        for name, state in self.states.items():
            if state.parent is not None:
                if state.parent not in self.states:
                    raise ValueError(f"state {name!r} references missing parent {state.parent!r}")
                if state.parent == name:
                    raise ValueError(f"state {name!r} cannot be its own parent")
                self._add_child(state.parent, name)

    def link_state(self, parent: str, child: str) -> None:
        """Create a single-parent lineage edge without allowing inconsistent children lists."""
        if parent not in self.states or child not in self.states:
            raise KeyError("both parent and child must exist")
        if parent == child:
            raise ValueError("a state cannot be its own parent")
        existing = self.states[child].parent
        if existing is not None and existing != parent:
            raise ValueError(f"child {child!r} already has parent {existing!r}")
        self.states[child].parent = parent
        self._add_child(parent, child)
        if parent in self.get_path(child):
            # get_path would now detect the cycle; reject it before leaving the
            # in-memory object in an invalid state.
            self.states[child].parent = existing
            if existing is None:
                self.states[parent].children.remove(child)
            raise ValueError(f"link would create a parent cycle: {parent!r} -> {child!r}")

    def decay(self, delta: float) -> None:
        if delta < 0:
            raise ValueError("delta must be non-negative")
        factor = math.exp(-self.decay_lambda * delta)
        for state in self.states.values():
            state.resonance = max(0.0, min(1.0, state.resonance * factor))

    def reinforce(self, name: str, kappa: float = 1.0) -> GlyphState:
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
        return {"target": name, "state": asdict(self.states[name]), "lineage": self.get_path(name)}

    def hyperstate(self, threshold: float = 0.8) -> List[str]:
        return sorted(name for name, state in self.states.items() if state.cohesion >= threshold)

    def to_dict(self) -> Dict[str, object]:
        return {
            "version": self.VERSION,
            "parameters": {"decay_lambda": self.decay_lambda, "alpha": self.alpha, "beta": self.beta},
            "states": {name: asdict(self.states[name]) for name in sorted(self.states)},
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json() + "\n", encoding="utf-8")
        return target

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "GlyphinMemory":
        params = payload.get("parameters", {})
        if not isinstance(params, dict):
            raise ValueError("parameters must be an object")
        memory = cls(decay_lambda=float(params.get("decay_lambda", 0.01)),
                     alpha=float(params.get("alpha", 0.25)),
                     beta=float(params.get("beta", 0.25)))
        states = payload.get("states", {})
        if not isinstance(states, dict):
            raise ValueError("states must be an object")
        for name in sorted(states):
            raw = dict(states[name])
            raw["name"] = name
            memory.states[name] = GlyphState(**raw)
        # Children are derived from parent pointers. This prevents stale or
        # contradictory serialized child lists from becoming authoritative.
        memory._rebuild_children()
        return memory

    @classmethod
    def from_json(cls, text: str) -> "GlyphinMemory":
        return cls.from_dict(json.loads(text))

    @classmethod
    def load(cls, path: str | Path) -> "GlyphinMemory":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


__all__ = ["GlyphState", "GlyphinMemory"]
