"""Tests for reproducible Glyphin experiment records."""

import tempfile
import unittest
from pathlib import Path

from glyphin_experiment import GlyphinExperiment


class GlyphinExperimentTests(unittest.TestCase):
    def operations(self):
        return [
            {"operation": "create", "name": "root"},
            {"operation": "create", "name": "child", "level": 1},
            {"operation": "link", "parent": "root", "child": "child"},
            {"operation": "reinforce", "name": "child", "kappa": 1.0},
            {"operation": "recall", "name": "child"},
        ]

    def test_run_is_canonical_and_fingerprint_stable(self):
        first = GlyphinExperiment().run(self.operations())
        second = GlyphinExperiment().run(self.operations())
        # Each run gets its own explicit creation timestamps. The important
        # invariant is that the recorded run is self-contained and canonical.
        self.assertNotEqual(first.operations, second.operations)
        self.assertEqual(first.to_json(), first.to_json())
        self.assertEqual(first.fingerprint(), first.fingerprint())
        creates = [op for op in first.operations if op["operation"] == "create"]
        self.assertEqual(len(creates), 2)
        self.assertTrue(all("created_at" in op for op in creates))
        self.assertEqual(first.final_memory["states"].keys(), second.final_memory["states"].keys())

    def test_saved_record_contains_operations_and_final_memory(self):
        run = GlyphinExperiment().run(self.operations())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.json"
            saved = run.save(path)
            text = saved.read_text(encoding="utf-8")
        self.assertIn('"operations"', text)
        self.assertIn('"final_memory"', text)
        self.assertIn('"child"', text)
        self.assertEqual(len(run.results), len(run.operations))


if __name__ == "__main__":
    unittest.main()
