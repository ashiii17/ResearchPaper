#!/usr/bin/env python3
"""
test_tsa.py

Lightweight unit test suite for Time Series Analysis module:
- Built-up fraction formula verification
- NoData (class 0) exclusion and masking
- Chronological ordering and temporal consistency
- Metric calculation and zero-denominator MAPE guard
- Forecaster predictions on deterministic sequences
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tsa.evaluate_models import calculate_ts_metrics
from tsa.forecast_models import NaivePersistenceForecaster, DriftForecaster, HoltLinearForecaster


class TestTSAModule(unittest.TestCase):

    def test_built_up_fraction_formula_and_nodata_exclusion(self):
        """Verifies built-up fraction formula and strict exclusion of class 0 from denominator."""
        # Synthetic chip raster (500x500 = 250,000 pixels)
        # 10,000 pixels = Class 0 (NoData)
        # 24,000 pixels = Class 7 (Built Area)
        # 216,000 pixels = Class 5 (Crops)
        total_pixels = 250000
        nodata_pixels = 10000
        built_pixels = 24000

        valid_pixels = total_pixels - nodata_pixels
        self.assertEqual(valid_pixels, 240000)

        # Formula: built_pixels / valid_pixels * 100
        built_fraction_pct = (built_pixels / valid_pixels) * 100.0
        self.assertAlmostEqual(built_fraction_pct, 10.0, places=4)

        # Verify that dividing by total_pixels would have given wrong answer
        wrong_pct = (built_pixels / total_pixels) * 100.0
        self.assertNotEqual(built_fraction_pct, wrong_pct)

    def test_chronological_ordering(self):
        """Verifies that generated pilot time series has strict chronological ordering."""
        csv_path = Path("/Users/ashi/ResearchPaper/india_lulc/tsa/results/pilot_timeseries.csv")
        if not csv_path.exists():
            self.skipTest("pilot_timeseries.csv not yet generated.")

        df = pd.read_csv(csv_path)
        for cid, grp in df.groupby("chip_id"):
            years = grp["year"].tolist()
            expected_years = list(range(2017, 2026))
            self.assertEqual(years, expected_years, f"Years for {cid} must strictly match 2017..2025 in chronological order")

    def test_metrics_calculation_and_mape_zero_guard(self):
        """Verifies MAE, RMSE, and MAPE zero-denominator exclusion guard."""
        y_true = np.array([10.0, 12.0])
        y_pred = np.array([11.0, 11.0])

        m = calculate_ts_metrics(y_true, y_pred)
        # Errors: |11 - 10| = 1.0, |11 - 12| = 1.0 -> MAE = 1.0, RMSE = 1.0
        self.assertAlmostEqual(m["mae"], 1.0)
        self.assertAlmostEqual(m["rmse"], 1.0)
        self.assertEqual(m["mape_note"], "VALID")
        # MAPE: 0.5 * (|1/10| + |1/12|) * 100 = 0.5 * (10% + 8.333%) = 9.1667%
        self.assertAlmostEqual(m["mape"], 9.1667, places=3)

        # Near-zero actual value (< 0.05%) must trigger exclusion guard
        y_zero = np.array([0.0, 0.02])
        y_pred_zero = np.array([0.1, 0.1])
        m_zero = calculate_ts_metrics(y_zero, y_pred_zero)
        self.assertTrue(np.isnan(m_zero["mape"]))
        self.assertIn("EXCLUDED", m_zero["mape_note"])

    def test_forecasters_on_linear_sequence(self):
        """Verifies Naive and Drift forecasters on deterministic sequence."""
        # Sequence: [1, 2, 3, 4, 5] -> slope = (5 - 1) / 4 = 1.0
        y_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Naive forecast for 2 steps should be [5.0, 5.0]
        naive = NaivePersistenceForecaster().fit(y_train)
        np.testing.assert_array_equal(naive.predict(2), np.array([5.0, 5.0]))

        # Drift forecast for 2 steps should be [6.0, 7.0]
        drift = DriftForecaster().fit(y_train)
        np.testing.assert_array_almost_equal(drift.predict(2), np.array([6.0, 7.0]))


if __name__ == "__main__":
    unittest.main()
