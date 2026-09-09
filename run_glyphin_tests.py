"""Single-command test runner for the Glyphin research code.

The repository intentionally avoids a third-party test dependency. Existing
``unittest.TestCase`` tests and simple pytest-style ``test_*`` functions are
both collected here so the suite has one canonical invocation.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


def _load_module(path: Path):
    module_name = f"_glyphin_test_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load test module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_suite(root: Path | None = None) -> unittest.TestSuite:
    """Collect unittest classes and zero-argument ``test_*`` functions."""
    root = root or Path(__file__).resolve().parent
    suite = unittest.TestSuite()
    loader = unittest.defaultTestLoader

    for path in sorted(root.glob("test_*.py")):
        module = _load_module(path)
        suite.addTests(loader.loadTestsFromModule(module))
        for name in sorted(dir(module)):
            if not name.startswith("test_"):
                continue
            candidate = getattr(module, name)
            if callable(candidate) and not isinstance(candidate, type):
                suite.addTest(unittest.FunctionTestCase(candidate))

    return suite


def main() -> int:
    suite = build_suite()
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
