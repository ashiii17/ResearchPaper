#!/usr/bin/env python3
"""
test_splits.py

Unit tests for split and data leakage audit:
- Split column integrity in chips.csv
- Disjoint splits (zero chip ID overlap)
- Strata representation across splits
"""

import sys
import unittest
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baseline.audit_splits import CHIPS_FILE, PILOT_IDS


class TestSplits(unittest.TestCase):

    def setUp(self):
        self.df = pd.read_csv(CHIPS_FILE)

    def test_total_chip_count(self):
        """Verifies exactly 1,500 chips in chips.csv."""
        self.assertEqual(len(self.df), 1500)

    def test_split_column_values(self):
        """Verifies split column contains strictly train, val, and test."""
        splits = set(self.df["split"].unique())
        self.assertEqual(splits, {"train", "val", "test"})

    def test_zero_leakage_across_splits(self):
        """Asserts zero chip ID overlap across train, val, and test splits."""
        train_ids = set(self.df[self.df["split"] == "train"]["chip_id"])
        val_ids = set(self.df[self.df["split"] == "val"]["chip_id"])
        test_ids = set(self.df[self.df["split"] == "test"]["chip_id"])

        self.assertEqual(len(train_ids.intersection(val_ids)), 0, "Train and Val must be disjoint")
        self.assertEqual(len(train_ids.intersection(test_ids)), 0, "Train and Test must be disjoint")
        self.assertEqual(len(val_ids.intersection(test_ids)), 0, "Val and Test must be disjoint")

    def test_strata_presence_in_all_splits(self):
        """Verifies all five strata (R, A, B, C, D) are represented in train, val, and test."""
        expected_strata = {"R", "A", "B", "C", "D"}
        for s in ["train", "val", "test"]:
            strata_in_split = set(self.df[self.df["split"] == s]["stratum"].unique())
            self.assertEqual(strata_in_split, expected_strata, f"All strata must be present in split '{s}'")

    def test_pilot_chips_presence_and_distribution(self):
        """Verifies the 5 pilot chips exist in chips.csv with 4 train and 1 val."""
        pilot_df = self.df[self.df["chip_id"].isin(PILOT_IDS)]
        self.assertEqual(len(pilot_df), 5, "All 5 pilot chips must exist in chips.csv")
        split_counts = pilot_df["split"].value_counts().to_dict()
        self.assertEqual(split_counts.get("train", 0), 4)
        self.assertEqual(split_counts.get("val", 0), 1)
        self.assertEqual(split_counts.get("test", 0), 0)


if __name__ == "__main__":
    unittest.main()
