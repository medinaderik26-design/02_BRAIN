"""Deterministic tests for the Glyphin execution layer."""

import unittest

from glyphin_engine import GlyphinExecutionEngine


class GlyphinExecutionEngineTests(unittest.TestCase):
    def test_full_execution_and_reload(self):
        engine = GlyphinExecutionEngine()
        engine.create("root", cohesion=0.5)
        engine.create("child", level=1)
        engine.create("grandchild", level=2)
        engine.link("root", "child")
        engine.link("child", "grandchild")
        engine.reinforce("child", kappa=1.0)
        engine.decay(2.0)
        recalled = engine.recall("grandchild")

        self.assertEqual(recalled["lineage"], ["root", "child", "grandchild"])
        report = engine.verify_reload(["grandchild"])
        self.assertTrue(report.reload_exact)
        self.assertTrue(report.state_referee_exact)
        self.assertEqual(report.state_referee_mismatches, [])
        self.assertEqual(report.state_count, 3)
        self.assertEqual(report.relationship_count, 2)
        self.assertEqual(
            report.lineage_checks["grandchild"],
            ["root", "child", "grandchild"],
        )
        self.assertEqual(len(report.events), 8)

    def test_operation_sequence_is_explicit(self):
        engine = GlyphinExecutionEngine()
        engine.execute([
            {"operation": "create", "arguments": {"name": "a"}},
            {"operation": "create", "arguments": {"name": "b"}},
            {"operation": "link", "arguments": {"parent": "a", "child": "b"}},
            {"operation": "reinforce", "arguments": {"name": "b", "kappa": 0.5}},
        ])
        self.assertEqual(engine.memory.get_path("b"), ["a", "b"])
        self.assertEqual([e.operation for e in engine.events], [
            "create", "create", "link", "reinforce"
        ])

    def test_invalid_operation_does_not_get_silently_accepted(self):
        engine = GlyphinExecutionEngine()
        with self.assertRaises(ValueError):
            engine.execute([{"operation": "invent", "arguments": {}}])


if __name__ == "__main__":
    unittest.main()
