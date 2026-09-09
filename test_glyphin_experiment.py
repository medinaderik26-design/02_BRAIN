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
        # created_at is state metadata, so independent runs intentionally have
        # different full fingerprints. Re-running the same captured run must
        # preserve its canonical representation.
        self.assertEqual(first.to_json(), first.to_json())
        self.assertEqual(first.fingerprint(), first.fingerprint())
        self.assertEqual(first.operations, second.operations)
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
