"""Tests for structural Glyphin round-trip verification."""
from __future__ import annotations

import unittest

from glyphin_research_core import GlyphinMemory
from glyphin_roundtrip import round_trip_memory, round_trip_topology
from glyphin_topology import DirectedTopology


class GlyphinRoundTripTests(unittest.TestCase):
    def test_memory_forest_round_trip_is_structurally_exact(self):
        memory = GlyphinMemory()
        memory.add_state("root", cohesion=0.7, resonance=0.8)
        memory.add_state("left", parent="root")
        memory.add_state("right", parent="root")
        memory.add_state("leaf", parent="left")

        result = round_trip_memory(memory)

        self.assertTrue(result.structural_exact)
        self.assertEqual(result.original_nodes, 4)
        self.assertEqual(result.original_edges, 3)
        self.assertEqual(result.round_trip_edges, 3)
        self.assertFalse(result.dynamics_verified)
        self.assertIn("cohesion", result.first_adaptation.lost_state_fields)

    def test_topology_fan_out_round_trip_is_exact(self):
        topology = DirectedTopology.from_edges(
            [("root", "a"), ("root", "b"), ("a", "c")],
            nodes=["root", "a", "b", "c"],
        )

        result = round_trip_topology(topology)

        self.assertTrue(result.structural_exact)
        self.assertEqual(result.missing_edges, [])
        self.assertEqual(result.extra_edges, [])

    def test_topology_fan_in_is_not_claimed_exact(self):
        topology = DirectedTopology.from_edges(
            [("a", "x"), ("b", "x")], nodes=["a", "b", "x"]
        )

        result = round_trip_topology(topology)

        self.assertFalse(result.structural_exact)
        self.assertEqual(
            set(result.first_adaptation.unsupported_edges), {("a", "x"), ("b", "x")}
        )
        self.assertEqual(result.round_trip_edges, 0)

    def test_topology_cycle_is_not_claimed_exact(self):
        topology = DirectedTopology.from_edges(
            [("a", "b"), ("b", "a")], nodes=["a", "b"]
        )

        result = round_trip_topology(topology)

        self.assertFalse(result.structural_exact)
        self.assertEqual(
            set(result.first_adaptation.unsupported_edges), {("a", "b"), ("b", "a")}
        )
        self.assertEqual(result.round_trip_edges, 0)


if __name__ == "__main__":
    unittest.main()
