"""Run Simulation 31 with the corrected state->alias index orientation.

Sim31's alias_map is {canonical_state: opaque_alias}. The resolver requires
{opaque_alias: [canonical_state, ...]}. Keep this correction isolated so the
benchmark's original implementation remains auditable while CI executes the
corrected path.
"""
import glyphin_simulation31 as sim


def corrected_build_alias_index(alias_map, collisions):
    idx = {}
    for state_name, alias in alias_map.items():
        idx.setdefault(alias, []).append(state_name)
    for alias, group in collisions.items():
        idx[alias] = sorted(group)
    return {key: sorted(values) for key, values in sorted(idx.items())}


sim.build_alias_index = corrected_build_alias_index
sim.main()
