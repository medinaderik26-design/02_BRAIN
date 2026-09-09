"""Formal GlyphState/GlyphinMemory research core."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional


_LEVELS = {
    "seed": 0,
    "echo": 1,
    "resonant": 2,
    "wave": 3,
    "hyperstate": 4,
    "monumentalthic": 5,
}


@dataclass
class GlyphState:
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
        if not 0.0 <= self.cohesion <= 1.0:
            raise ValueError("cohesion must be in [0, 1]")
        if not 0.0 <= self.resonance <= 1.0:
            raise ValueError("resonance must be in [0, 1]")
        if self.frequency < 1:
            raise ValueError("frequency must be >= 1")


class GlyphinMemory:
    """Single-parent, state-rich symbolic memory."""

    def __init__(self, *, decay_lambda: float = 0.1, alpha: float = 0.25, beta: float = 0.25) -> None:
        if decay_lambda < 0 or alpha < 0 or beta < 0:
            raise ValueError("decay_lambda, alpha, and beta must be non-negative")
        self.decay_lambda = decay_lambda
        self.alpha = alpha
        self.beta = beta
        self.states: Dict[str, GlyphState] = {}

    def _add_child(self, parent: str, child: str) -> None:
        if child not in self.states[parent].children:
            self.states[parent].children.append(child)
            self.states[parent].children.sort()

    def _rebuild_children(self) -> None:
        for state in self.states.values():
            state.children = []
        for name in sorted(self.states):
            parent = self.states[name].parent
            if parent is not None:
                if parent not in self.states:
                    raise ValueError(f"state {name!r} references missing parent {parent!r}")
                if parent == name:
                    raise ValueError(f"state {name!r} cannot be its own parent")
                self._add_child(parent, name)
        for name in sorted(self.states):
            self.get_path(name)

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
        created_at: Optional[str] = None,
    ) -> GlyphState:
        if name in self.states:
            raise ValueError(f"state already exists: {name!r}")
        if parent is not None and parent not in self.states:
            raise KeyError(f"parent does not exist: {parent!r}")
        state = GlyphState(
            name=name,
            level=level,
            cohesion=cohesion,
            parent=parent,
            frequency=frequency,
            resonance=resonance,
            sigma=sigma,
            created_at=created_at or datetime.now(timezone.utc).isoformat(),
        )
        self.states[name] = state
        if parent is not None:
            self._add_child(parent, name)
        return state

    def link_state(self, parent: str, child: str) -> None:
        """Create a single-parent lineage edge without leaving invalid state."""
        if parent not in self.states or child not in self.states:
            raise KeyError("both parent and child must exist")
        if parent == child:
            raise ValueError("a state cannot be its own parent")
        existing = self.states[child].parent
        if existing is not None and existing != parent:
            raise ValueError(f"child {child!r} already has parent {existing!r}")

        # The proposed edge parent -> child is cyclic exactly when child is
        # already an ancestor of parent. Check before mutating either state.
        if child in self.get_path(parent):
            raise ValueError(f"link would create a parent cycle: {parent!r} -> {child!r}")

        self.states[child].parent = parent
        self._add_child(parent, child)

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

    def recall(self, name: str) -> Dict[str, Any]:
        state = self.states.get(name)
        if state is None:
            raise KeyError(name)
        return {"target": name, "lineage": self.get_path(name), "state": asdict(state)}

    def hyperstate(self, threshold: float = 0.8) -> List[str]:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        return sorted(name for name, state in self.states.items() if state.cohesion >= threshold)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": "glyphin-research-core-1.0",
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
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GlyphinMemory":
        parameters = dict(data.get("parameters", {}))
        memory = cls(
            decay_lambda=parameters.get("decay_lambda", 0.1),
            alpha=parameters.get("alpha", 0.25),
            beta=parameters.get("beta", 0.25),
        )
        states = data.get("states", {})
        for name in sorted(states):
            raw = dict(states[name])
            raw.pop("children", None)
            raw["name"] = name
            memory.states[name] = GlyphState(**raw)
        memory._rebuild_children()
        return memory

    @classmethod
    def from_json(cls, text: str) -> "GlyphinMemory":
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("Glyphin memory JSON must contain an object")
        return cls.from_dict(data)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json() + "\n", encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> "GlyphinMemory":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


__all__ = ["GlyphState", "GlyphinMemory"]
