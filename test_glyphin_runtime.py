"""Executable tests for the Glyphin research runtime."""

import json
import tempfile
import unittest
from pathlib import Path

from glyphin_research_core import GlyphinMemory
from glyphin_runtime import GlyphinRuntime


class GlyphinRuntimeTests(unittest.TestCase):
    def test_operation_sequence_updates_one_memory(self):
        runtime = GlyphinRuntime(GlyphinMemory(decay_lambda=0.1))
        results = runtime.run([
            {"operation": "create", "name": "root"},
            {"operation": "create", "name": "child"},
            {"operation": "link", "parent": "root", "child": "child"},
            {"operation": "reinforce", "name": "child", "kappa": 1.0},
            {"operation": "decay", "delta": 1.0},
            {"operation": "recall", "name": "child"},
        ])
        self.assertEqual([r["operation"] for r in results],
                         ["create", "create", "link", "reinforce", "decay", "recall"])
        self.assertEqual(runtime.memory.get_path("child"), ["root", "child"])
        self.assertEqual(runtime.memory.states["child"].frequency, 2)
        self.assertLess(runtime.memory.states["child"].resonance, 1.0)

    def test_save_and_load_preserve_runtime_memory(self):
        runtime = GlyphinRuntime()
        runtime.create("root")
        runtime.create("child")
        runtime.link("root", "child")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.json"
            runtime.save(path)
            restored = GlyphinRuntime.load(path)
        self.assertEqual(restored.memory.to_json(), runtime.memory.to_json())
        self.assertEqual(restored.memory.states["root"].children, ["child"])
        self.assertEqual(restored.memory.get_path("child"), ["root", "child"])

    def test_link_rejects_second_parent(self):
        runtime = GlyphinRuntime()
        runtime.create("a")
        runtime.create("b")
        runtime.create("child")
        runtime.link("a", "child")
        with self.assertRaises(ValueError):
            runtime.link("b", "child")
        self.assertEqual(runtime.memory.states["child"].parent, "a")
        self.assertEqual(runtime.memory.states["a"].children, ["child"])
        self.assertEqual(runtime.memory.states["b"].children, [])

    def test_link_rejects_cycle_without_mutating_relationships(self):
        runtime = GlyphinRuntime()
        runtime.create("a")
        runtime.create("b")
        runtime.link("a", "b")
        with self.assertRaises(ValueError):
            runtime.link("b", "a")
        self.assertEqual(runtime.memory.states["a"].parent, None)
        self.assertEqual(runtime.memory.states["b"].parent, "a")
        self.assertEqual(runtime.memory.states["a"].children, ["b"])
        self.assertEqual(runtime.memory.states["b"].children, [])

    def test_operation_file_shape(self):
        payload = json.dumps([
            {"operation": "create", "name": "x"},
            {"operation": "recall", "name": "x"},
        ])
        operations = json.loads(payload)
        runtime = GlyphinRuntime()
        results = runtime.run(operations)
        self.assertEqual(results[-1]["lineage"], ["x"])


if __name__ == "__main__":
    unittest.main()
