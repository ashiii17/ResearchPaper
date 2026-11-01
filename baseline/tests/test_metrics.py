#!/usr/bin/env python3
"""
test_metrics.py

Unit tests for classification and change detection metrics:
- Precision, recall, F1, and IoU computation
- Exclusion of ignore_index (255) pixels
- Confusion matrix dimensions and counts
- Baseline reference comparison
"""

import sys
import unittest
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baseline.evaluate import compute_binary_metrics, IGNORE_INDEX


class TestMetrics(unittest.TestCase):

    def test_perfect_prediction(self):
        """Verifies perfect predictions yield 1.0 for precision, recall, F1, and IoU."""
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 1])
        m = compute_binary_metrics(y_pred, y_true)

        self.assertAlmostEqual(m["precision"], 1.0)
        self.assertAlmostEqual(m["recall"], 1.0)
        self.assertAlmostEqual(m["f1"], 1.0)
        self.assertAlmostEqual(m["iou"], 1.0)
        self.assertAlmostEqual(m["accuracy"], 1.0)
        self.assertEqual(m["confusion_matrix"], [[2, 0], [0, 3]])

    def test_ignore_index_exclusion(self):
        """Asserts pixels with ignore_index (255) are completely omitted from metrics."""
        # 4 valid pixels + 2 ignored pixels
        y_true = np.array([0, 1, 0, 1, IGNORE_INDEX, IGNORE_INDEX])
        # Model predicts wrong on ignored pixels; should NOT affect metrics
        y_pred = np.array([0, 1, 0, 1, 1, 0])
        m = compute_binary_metrics(y_pred, y_true)

        self.assertEqual(m["total_valid_pixels"], 4)
        self.assertAlmostEqual(m["iou"], 1.0)
        self.assertAlmostEqual(m["accuracy"], 1.0)

    def test_majority_class_baseline(self):
        """Verifies majority-class baseline (all 0) yields IoU=0 and F1=0 on imbalanced data."""
        y_true = np.array([0] * 99 + [1] * 1)  # 1% positive
        y_pred = np.zeros(100)  # all negative

        m = compute_binary_metrics(y_pred, y_true)
        self.assertEqual(m["tp"], 0)
        self.assertEqual(m["fp"], 0)
        self.assertEqual(m["tn"], 99)
        self.assertEqual(m["fn"], 1)

        self.assertAlmostEqual(m["precision"], 0.0)
        self.assertAlmostEqual(m["recall"], 0.0)
        self.assertAlmostEqual(m["f1"], 0.0)
        self.assertAlmostEqual(m["iou"], 0.0)
        self.assertAlmostEqual(m["accuracy"], 0.99)
        self.assertAlmostEqual(m["positive_class_pct"], 1.0)


if __name__ == "__main__":
    unittest.main()
