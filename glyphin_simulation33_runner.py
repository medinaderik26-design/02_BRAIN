"""Sim33 runner: controlled surface-form resolver with an explicit semantic boundary."""
import re
import glyphin_simulation33 as sim

A = (("alpha", "first", "primary"), ("beta", "second", "secondary"),
     ("gamma", "third", "tertiary"), ("delta", "fourth", "quaternary"),
     ("epsilon", "fifth", "quinary"), ("zeta", "sixth", "senary"),
     ("eta", "seventh", "septenary"), ("theta", "eighth", "octonary"),
     ("iota", "ninth", "nonary"), ("kappa", "tenth", "denary"),
     ("lambda", "eleventh", "undecimal"), ("mu", "twelfth", "duodecimal"),
     ("nu", "thirteenth", "tridecimal"), ("xi", "fourteenth", "tetradecimal"),
     ("omicron", "fifteenth", "quindecimal"), ("pi", "sixteenth", "hexadecimal"))
B = (("amber", "gold", "yellow"), ("azure", "blue", "cyan"),
     ("crimson", "red", "scarlet"), ("emerald", "green", "jade"),
     ("violet", "purple", "mauve"), ("ivory", "white", "cream"),
     ("charcoal", "black", "ebony"), ("coral", "orange", "tangerine"),
     ("teal", "turquoise", "aqua"), ("indigo", "navy", "cobalt"),
     ("bronze", "copper", "russet"), ("silver", "gray", "slate"),
     ("rose", "pink", "blush"), ("lime", "chartreuse", "olive"),
     ("maroon", "burgundy", "wine"), ("beige", "tan", "khaki"))
C = (("north", "northern", "upward"), ("south", "southern", "downward"),
     ("east", "eastern", "rightward"), ("west", "western", "leftward"),
     ("inside", "internal", "within"), ("outside", "external", "beyond"),
     ("near", "nearby", "close"), ("far", "distant", "remote"),
     ("above", "upper", "overhead"), ("below", "lower", "underneath"),
     ("before", "prior", "earlier"), ("after", "later", "subsequent"),
     ("calm", "quiet", "steady"), ("rapid", "fast", "quick"),
     ("dense", "compact", "thick"), ("sparse", "thin", "scattered"))

VERSION = "33.2"
VOCABS = (A, B, C)


def _digits(i):
    base = 16
    return ((i // (base * base)) % base, (i // base) % base, i % base)


def descriptor_family(i):
    a, b, c = _digits(i)
    return (A[a], B[b], C[c])


def build_fixture(memory):
    return {n: descriptor_family(i) for i, n in enumerate(sorted(memory.states))}


def build_index(fixture):
    index = {}
    for state, family in fixture.items():
        for style in range(3):
            phrase = tuple(words[style] for words in family)
            key = tuple(sorted(phrase))
            if key in index and index[key] != state:
                raise AssertionError(f"controlled descriptor collision: {key}")
            index[key] = state
    return dict(sorted(index.items()))


def _tokens(text):
    return re.findall(r"[a-z]+", text.lower())


def controlled_resolve(query, index):
    # Controlled resolver: punctuation-insensitive, order-insensitive, and
    # synonym-aware through an explicit finite vocabulary. This is not NLP.
    toks = _tokens(query)
    hits = []
    for key, state in index.items():
        remaining = list(toks)
        if all(token in remaining and not remaining.remove(token) for token in key):
            hits.append(state)
    return sorted(set(hits))


def make_cases(memory, fixture):
    rows = []
    for state, family in list(fixture.items())[:64]:
        canonical = tuple(words[0] for words in family)
        synonyms = tuple(words[1] for words in family)
        punctuation = f"({canonical[0]}, {canonical[1]}; {canonical[2]})"
        reordered = f"{canonical[2]} {canonical[0]} {canonical[1]}"
        synonymized = " ".join(synonyms)
        free = "bring back the idea connected to this state"
        rows.extend([
            {"state": state, "surface": "canonical", "query": "retrieve " + " ".join(canonical), "expected": [state], "resolver_expected": "controlled"},
            {"state": state, "surface": "punctuation", "query": "retrieve " + punctuation, "expected": [state], "resolver_expected": "controlled"},
            {"state": state, "surface": "reordered", "query": "retrieve " + reordered, "expected": [state], "resolver_expected": "controlled"},
            {"state": state, "surface": "synonymized", "query": "retrieve " + synonymized, "expected": [state], "resolver_expected": "controlled"},
            {"state": state, "surface": "free_paraphrase", "query": free, "expected": [state], "resolver_expected": "external"},
        ])
    return rows


def main():
    import argparse, hashlib, json
    from pathlib import Path
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="glyphin_simulation33_result.json")
    args = ap.parse_args()
    rows = []
    for seed in sim.SEEDS:
        for size in sim.SIZES:
            memory = sim.build_memory(size, seed)
            fixture = build_fixture(memory)
            index = build_index(fixture)
            for case in make_cases(memory, fixture):
                resolved = controlled_resolve(case["query"], index)
                exact = resolved == case["expected"]
                rows.append({**case, "seed": seed, "size": size, "resolved": resolved, "exact": exact,
                             "external_required": case["resolver_expected"] == "external"})
    summary = {}
    for surface in sorted({r["surface"] for r in rows}):
        part = [r for r in rows if r["surface"] == surface]
        summary[surface] = {"cases": len(part), "exact": sum(r["exact"] for r in part),
                           "exact_rate_pct": 100.0 * sum(r["exact"] for r in part) / len(part),
                           "external_required_cases": sum(r["external_required"] for r in part)}
    out = {"simulation": 33, "benchmark_version": VERSION, "cases": rows, "summary": summary,
           "result_data_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
           "scope": "Controlled surface-form boundary benchmark only. Punctuation, word-order, and explicit synonym variation are resolved by a deterministic vocabulary/index. Free paraphrase has no lexical anchor and is intentionally marked as requiring an external semantic/concept resolver. No natural-language understanding, learned retrieval, LLM equivalence, latency, universal generalization, consciousness, or optimality claim."}
    Path(args.output).write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("cases:", len(rows))
    print("exact:", sum(r["exact"] for r in rows))

if __name__ == "__main__":
    main()
