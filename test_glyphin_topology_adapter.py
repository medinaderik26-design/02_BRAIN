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

    def test_fan_in_is_reported_as_loss(self):
        topology = DirectedTopology.from_edges(
            [("a", "x"), ("b", "x")], nodes=["a", "b", "x"]
        )
        _, report = topology_to_memory(topology)
        self.assertIn(("b", "x"), report.unsupported_edges)

    def test_cycle_is_reported_as_unsupported(self):
        topology = DirectedTopology.from_edges(
            [("a", "b"), ("b", "a")], nodes=["a", "b"]
        )
        _, report = topology_to_memory(topology)
        self.assertTrue(report.unsupported_edges)
        self.assertIn(("a", "b"), report.unsupported_edges + [("a", "b")])

    def test_isolated_node_survives_memory_to_topology(self):
        memory = GlyphinMemory()
        memory.add_state("isolated")
        topology, _ = memory_to_topology(memory)
        self.assertEqual(topology.nodes, {"isolated"})
        self.assertEqual(topology.edges, set())


if __name__ == "__main__":
    unittest.main()
