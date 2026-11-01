#!/usr/bin/env python3
"""
run_all_tests.py

Test runner executing all unit and integration tests under baseline/tests/.
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TESTS_DIR = Path(__file__).resolve().parent


def run_tests():
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(TESTS_DIR), pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    print("=" * 80)
    print("RUNNING ALL BASELINE TESTS")
    print("=" * 80)
    result = runner.run(suite)
    print("=" * 80)
    print(f"Tests run: {result.testsRun}, Errors: {len(result.errors)}, Failures: {len(result.failures)}, Skipped: {len(result.skipped)}")
    print("=" * 80)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
