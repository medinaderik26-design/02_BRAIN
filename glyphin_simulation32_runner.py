"""Corrected Sim32 runner: unique multi-token controlled descriptors.

The original Sim32 benchmark reused a 16-item descriptor vocabulary across
all states, so exact descriptor matching became ambiguous at size > 16.
This runner keeps the benchmark deterministic but assigns each state a unique
three-token descriptor phrase from the same 16x16x16 controlled vocabulary.
Each phrase still has three controlled paraphrase variants. This is a
controlled-language resolver, not natural-language understanding.
"""
import sys
import glyphin_simulation32 as sim


def _code_digits(i):
    base = len(sim.CONCEPTS)
    return ((i // (base * base)) % base, (i // base) % base, i % base)


def descriptor_map(memory):
    ns = sim.names(memory)
    out = {}
    for i, n in enumerate(ns):
        a, b, c = _code_digits(i)
        out[n] = {
            "canonical": tuple(sim.CONCEPTS[j][0] for j in (a, b, c)),
            "synonyms": tuple(
                tuple(sim.CONCEPTS[j][style] for j in (a, b, c))
                for style in range(3)
            ),
        }
    return out


def build_descriptor_index(dmap):
    idx = {}
    for state, info in dmap.items():
        for phrase in info["synonyms"]:
            key = " ".join(phrase)
            idx.setdefault(key, []).append(state)
    return {k: sorted(v) for k, v in sorted(idx.items())}


def choose_descriptor(dmap, state, style):
    return " ".join(dmap[state]["synonyms"][style])


sim.VERSION = "32.1"
sim.descriptor_map = descriptor_map
sim.build_descriptor_index = build_descriptor_index
sim.choose_descriptor = choose_descriptor

if __name__ == "__main__":
    sim.main()
