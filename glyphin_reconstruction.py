"""Glyphin reconstruction layer.

Parses the bounded symbolic grammar used by the research experiments and
reconstructs an exact directed topology. This module intentionally contains
no semantic inference: syntax is converted to edges, then independently
refereed against a source topology.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple

from glyphin_topology import DirectedTopology


TOKEN_RE = re.compile(r"\s*(->|\{|\}|,|;|[A-Za-z0-9_]+)")


@dataclass(frozen=True)
class Token:
    kind: str
    value: str


class ReconstructionError(ValueError):
    pass


class SymbolicParser:
    def __init__(self, text: str):
        self.tokens = self._lex(text)
        self.i = 0

    @staticmethod
    def _lex(text: str) -> List[Token]:
        tokens: List[Token] = []
        pos = 0
        while pos < len(text):
            m = TOKEN_RE.match(text, pos)
            if not m:
                if text[pos:].strip() == "":
                    break
                raise ReconstructionError(f"invalid character at offset {pos}: {text[pos]!r}")
            value = m.group(1)
            kind = "ID" if value[0].isalnum() or "_" in value else value
            tokens.append(Token(kind, value))
            pos = m.end()
        return tokens

    def peek(self) -> Token | None:
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def take(self, kind: str | None = None) -> Token:
        tok = self.peek()
        if tok is None:
            raise ReconstructionError("unexpected end of input")
        if kind is not None and tok.kind != kind:
            raise ReconstructionError(f"expected {kind}, got {tok.kind}:{tok.value}")
        self.i += 1
        return tok

    def parse(self) -> DirectedTopology:
        graph = DirectedTopology()
        while self.peek() is not None:
            self._statement(graph)
            if self.peek() is not None:
                self.take(";")
        return graph

    def _statement(self, graph: DirectedTopology) -> None:
        first = self.take("ID").value
        graph.add_node(first)
        self._chain_from(graph, first)

    def _chain_from(self, graph: DirectedTopology, parent: str) -> None:
        while self.peek() is not None and self.peek().kind == "->":
            self.take("->")
            if self.peek().kind == "{":
                self.take("{")
                self._fanout(graph, parent)
                self.take("}")
                continue
            child = self.take("ID").value
            graph.add_edge(parent, child)
            parent = child

    def _fanout(self, graph: DirectedTopology, parent: str) -> None:
        first = True
        while True:
            if not first:
                self.take(",")
            first = False
            item = self.take("ID").value
            graph.add_edge(parent, item)
            self._chain_from(graph, item)
            if self.peek() is None or self.peek().kind != ",":
                break


def reconstruct(text: str) -> DirectedTopology:
    return SymbolicParser(text).parse()


def compare(source: DirectedTopology, encoded: str) -> Tuple[bool, dict]:
    """Reconstruct and return a machine-readable exact comparison."""
    candidate = reconstruct(encoded)
    report = source.compare(candidate)
    return bool(report["exact"]), report


__all__ = ["Token", "ReconstructionError", "SymbolicParser", "reconstruct", "compare"]
