"""Deterministic tests for glyphin_research_core."""

import json
import unittest

from glyphin_research_core import GlyphinMemory


class GlyphinResearchCoreTests(unittest.TestCase):
    def make_memory(self):
        m = GlyphinMemory(decay_lambda=0.1, alpha=0.25, beta=0.25)
        m.add_state("root", cohesion=0.5)
        m.add_state("child", parent="root", level=1)
        m.add_state("grandchild", parent="child", level=2)
        return m

    def test_lineage(self):
        m = self.make_memory()
        self.assertEqual(m.get_path("grandchild"), ["root", "child", "grandchild"])
        self.assertEqual(m.states["root"].children, ["child"])
        self.assertEqual(m.states["child"].children, ["grandchild"])

    def test_reinforcement_is_bounded_and_increments_frequency(self):
        m = self.make_memory()
        before = m.states["child"].cohesion
        m.reinforce("child", kappa=1.0)
        state = m.states["child"]
        self.assertGreater(state.cohesion, before)
        self.assertLessEqual(state.cohesion, 1.0)
        self.assertEqual(state.frequency, 2)
        self.assertEqual(state.sigma, "echo")

    def test_decay(self):
        m = self.make_memory()
        m.states["child"].resonance = 1.0
        m.decay(10.0)
        self.assertAlmostEqual(m.states["child"].resonance, 0.3678794412, places=8)

    def test_recall_is_lineage_aware(self):
        m = self.make_memory()
        result = m.recall("grandchild")
        self.assertEqual(result["target"], "grandchild")
        self.assertEqual(result["lineage"], ["root", "child", "grandchild"])
        self.assertEqual(result["state"]["name"], "grandchild")

    def test_canonical_serialization_round_trip(self):
        m = self.make_memory()
        m.reinforce("child", kappa=0.5)
        encoded = m.to_json()
        self.assertEqual(encoded, m.to_json())
        restored = GlyphinMemory.from_json(encoded)
        self.assertEqual(restored.to_json(), encoded)
        self.assertEqual(restored.get_path("grandchild"), m.get_path("grandchild"))

    def test_hyperstate(self):
        m = self.make_memory()
        self.assertEqual(m.hyperstate(0.5), ["root"])
        m.reinforce("child", kappa=1.0)
        self.assertEqual(m.hyperstate(0.1), ["child", "root"])


if __name__ == "__main__":
    unittest.main()
