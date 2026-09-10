"""Corrected Sim32 runner: unique multi-token controlled descriptors.

The benchmark uses a 16x16x16 controlled vocabulary. Each state gets a unique
three-token descriptor phrase and three controlled synonym variants. This is
a deterministic controlled-language resolver, not natural-language
understanding.
"""
import glyphin_simulation32 as sim


def _code_digits(i):
    base = len(sim.VOCAB)
    return ((i // (base * base)) % base, (i // base) % base, i % base)


def descriptor_map(memory):
    ns = sim.names(memory)
    out = {}
    for i, n in enumerate(ns):
        a, b, c = _code_digits(i)
        out[n] = {
            "canonical": tuple(sim.VOCAB[j][0] for j in (a, b, c)),
            "variants": tuple(
                tuple(sim.VOCAB[j][style] for j in (a, b, c))
                for style in range(3)
            ),
        }
    return out


def build_descriptor_index(dmap):
    idx = {}
    for state, info in dmap.items():
        for phrase in info["variants"]:
            key = " ".join(phrase)
            idx.setdefault(key, []).append(state)
    return {k: sorted(v) for k, v in sorted(idx.items())}


def choose_descriptor(dmap, state, style):
    return " ".join(dmap[state]["variants"][style])


sim.VERSION = "32.1"
sim.descriptor_map = descriptor_map
sim.build_descriptor_index = build_descriptor_index
sim.choose_descriptor = choose_descriptor

if __name__ == "__main__":
    sim.main()
