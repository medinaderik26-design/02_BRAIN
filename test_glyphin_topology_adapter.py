"""Tests for explicit GlyphinMemory <-> DirectedTopology adaptation."""

import unittest

from glyphin_research_core import GlyphinMemory
from glyphin_topology import DirectedTopology
from glyphin_topology_adapter import memory_to_topology, topology_to_memory


class TopologyAdapterTests(unittest.TestCase):
    def test_lineage_conversion_preserves_nodes_and_edges(self):
        memory = GlyphinMemory()
        memory.add_state("root")
        memory.add_state("child", parent="root")
        memory.add_state("grandchild", parent="child")

        topology, report = memory_to_topology(memory)

        self.assertEqual(topology.nodes, {"root", "child", "grandchild"})
        self.assertEqual(topology.edges, {("root", "child"), ("child", "grandchild")})
        self.assertFalse(report.lossless)
        self.assertIn("cohesion", report.lost_state_fields)

    def test_fan_out_is_representable(self):
        topology = DirectedTopology.from_edges(
            [("root", "a"), ("root", "b")], nodes=["root", "a", "b"]
        )
        memory, report = topology_to_memory(topology)
        self.assertEqual(report.unsupported_edges, [])
        self.assertEqual(memory.states["a"].parent, "root")
        self.assertEqual(memory.states["b"].parent, "root")

    def test_fan_in_is_blocked_in_strict_mode(self):
        topology = DirectedTopology.from_edges(
            [("a", "x"), ("b", "x")], nodes=["a", "b", "x"]
        )
        memory, report = topology_to_memory(topology)
        self.assertEqual(
            set(report.unsupported_edges), {("a", "x"), ("b", "x")}
        )
        self.assertEqual(memory.states["x"].parent, None)
        self.assertEqual(report.target_edges, 0)

    def test_cycle_is_reported_and_blocked_before_relationship_mutation(self):
        topology = DirectedTopology.from_edges(
            [("a", "b"), ("b", "a")], nodes=["a", "b"]
        )
        memory, report = topology_to_memory(topology)
        self.assertEqual(set(report.unsupported_edges), {("a", "b"), ("b", "a")})
        self.assertEqual(memory.states["a"].parent, None)
        self.assertEqual(memory.states["b"].parent, None)
        self.assertEqual(report.target_edges, 0)

    def test_non_strict_mode_keeps_unambiguous_acyclic_edges(self):
        topology = DirectedTopology.from_edges(
            [("root", "a"), ("a", "b"), ("x", "b")],
            nodes=["root", "a", "b", "x"],
        )
        memory, report = topology_to_memory(topology, strict=False)
        self.assertEqual(report.unsupported_edges, [("a", "b"), ("x", "b")])
        self.assertEqual(memory.states["a"].parent, "root")
        self.assertEqual(memory.states["b"].parent, None)

    def test_isolated_node_survives_memory_to_topology(self):
        memory = GlyphinMemory()
        memory.add_state("isolated")
        topology, _ = memory_to_topology(memory)
        self.assertEqual(topology.nodes, {"isolated"})
        self.assertEqual(topology.edges, set())


if __name__ == "__main__":
    unittest.main()
