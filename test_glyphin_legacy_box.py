"""Characterization box for legacy glyphin.py.

Do not treat a pass as a design endorsement. These tests pin current behavior
before any repair of the original file. glyphin.py is not imported: the module
loads ../04_TOOLS/tools_core.py at import time.
"""
from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LEGACY = ROOT / "glyphin.py"


def legacy_source() -> str:
    return LEGACY.read_text(encoding="utf-8")


def load_variator_class():
    """Execute DualConeVariatorController from source without running module imports."""
    tree = ast.parse(legacy_source())
    class_node = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "DualConeVariatorController":
            class_node = node
            break
    if class_node is None:
        raise AssertionError("DualConeVariatorController missing from glyphin.py")
    module = ast.Module(body=[class_node], type_ignores=[])
    compiled = compile(module, str(LEGACY), "exec")
    namespace: dict = {}
    exec(compiled, namespace)
    return namespace["DualConeVariatorController"]


class LegacyGlyphinSourceContractTests(unittest.TestCase):
    def test_legacy_file_exists_and_is_untouched_entry(self):
        self.assertTrue(LEGACY.is_file())
        source = legacy_source()
        self.assertIn("04_TOOLS", source)
        self.assertIn("tools_core.py", source)
        self.assertIn("parent.parent", source)

    def test_hardcoded_pressure_is_still_point_four_five(self):
        self.assertIn("simulated_pressure = 0.45", legacy_source())

    def test_cli_surface_is_still_present(self):
        source = legacy_source()
        self.assertIn('startswith("remember ")', source)
        self.assertIn('startswith("recall ")', source)
        self.assertIn("Laws 001-006", source)
        self.assertIn('user_input.lower() == "exit"', source)

    def test_import_path_points_outside_this_repo(self):
        expected = 'Path(__file__).resolve().parent.parent / "04_TOOLS" / "tools_core.py"'
        self.assertIn(expected, legacy_source())


class DualConeVariatorCharacterizationTests(unittest.TestCase):
    def setUp(self):
        self.cls = load_variator_class()
        self.variator = self.cls()

    def test_default_momenta_and_start_fraction(self):
        self.assertEqual(self.variator.up_momentum, 0.35)
        self.assertEqual(self.variator.down_momentum, 0.15)
        self.assertEqual(self.variator.current_fraction, 1.0)

    def test_pressure_is_clamped(self):
        self.assertEqual(self.cls().calculate_aperture(-2.0), 0.85)
        high = self.cls()
        high.current_fraction = 0.0
        self.assertEqual(high.calculate_aperture(3.0), 0.35)

    def test_down_path_from_one_toward_point_four_five(self):
        first = self.variator.calculate_aperture(0.45)
        self.assertAlmostEqual(first, 1.0 - (1.0 - 0.45) * 0.15)
        second = self.variator.calculate_aperture(0.45)
        self.assertLess(second, first)
        self.assertGreater(second, 0.45)

    def test_up_path_from_low_fraction(self):
        self.variator.current_fraction = 0.10
        opened = self.variator.calculate_aperture(0.80)
        self.assertAlmostEqual(opened, 0.10 + (0.80 - 0.10) * 0.35)

    def test_output_stays_in_unit_interval(self):
        for pressure in (-1.0, 0.0, 0.45, 1.0, 2.0):
            value = self.cls().calculate_aperture(pressure)
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)


if __name__ == "__main__":
    unittest.main()
