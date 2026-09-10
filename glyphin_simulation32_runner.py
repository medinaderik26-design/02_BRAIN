"""Corrected Sim32 runner: unique compositional descriptors and query parsing."""
import re
import glyphin_simulation32 as sim


def _code_digits(i):
    base = len(sim.VOCAB)
    return ((i // (base * base)) % base, (i // base) % base, i % base)


def descriptor_map(memory):
    ns = sim.names(memory)
    out = {}
    for i, n in enumerate(ns):
        a, b, c = _code_digits(i)
        out[n] = {"canonical": tuple(sim.VOCAB[j][0] for j in (a, b, c)),
                  "variants": tuple(tuple(sim.VOCAB[j][style] for j in (a, b, c)) for style in range(3))}
    return out


def build_descriptor_index(dmap):
    idx = {}
    for state, info in dmap.items():
        for phrase in info["variants"]:
            idx.setdefault(" ".join(phrase), []).append(state)
    return {k: sorted(v) for k, v in sorted(idx.items())}


def choose_descriptor(dmap, state, style):
    return " ".join(dmap[state]["variants"][style])


def resolve_controlled(text, index):
    # The benchmark deliberately includes a distractor after this delimiter.
    # It must not become an anchor merely because its descriptor is present.
    query = text.lower().split(" while ignoring ", 1)[0]
    hits = []
    for phrase, state in index.items():
        m = re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", query)
        if m:
            hits.append((m.start(), phrase, state))
    hits.sort(key=lambda x: (x[0], x[1]))
    resolved = []
    for _, _, state in hits:
        if state not in resolved:
            resolved.append(state)
    return (resolved, "unique", 1) if resolved else ([], "none", 0)


sim.VERSION = "32.2"
sim.descriptor_map = descriptor_map
sim.build_descriptor_index = build_descriptor_index
sim.choose_descriptor = choose_descriptor
sim.resolve_controlled = resolve_controlled

if __name__ == "__main__":
    sim.main()
