#!/usr/bin/env python3
"""
test_dataset.py

Unit tests for Sentinel2SegmentationDataset and Sentinel2ChangeDataset:
- Sample shapes and data types
- Label values and ignore masks
- Missing-file handling
- Grid alignment and transform assertions
"""

import sys
import unittest
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baseline.dataset import (
    Sentinel2SegmentationDataset,
    Sentinel2ChangeDataset,
    verify_raster_pair,
    IGNORE_INDEX,
    BUILT_CLASS
)


class TestDataset(unittest.TestCase):

    def setUp(self):
        self.seg_ds = Sentinel2SegmentationDataset(split="train", patch_size=128)
        self.chg_ds = Sentinel2ChangeDataset(split="train", patch_size=128)

    def test_sample_shapes_and_dtypes(self):
        """Verifies tensor shapes and dtypes."""
        self.assertGreater(len(self.seg_ds), 0, "Segmentation dataset must contain samples")
        item = self.seg_ds[0]

        img = item["image"]
        tgt = item["target"]
        mask = item["valid_mask"]

        # Check shapes
        self.assertEqual(img.shape, (7, 128, 128), "Image should have 7 channels and patch size 128x128")
        self.assertEqual(tgt.shape, (128, 128), "Target shape must match patch size")
        self.assertEqual(mask.shape, (128, 128), "Mask shape must match patch size")

        # Check dtypes
        self.assertEqual(str(img.dtype), "float32", "Image must be float32")
        self.assertIn(str(tgt.dtype), ["int64", "int32"], "Target must be integer")
        self.assertEqual(str(mask.dtype), "float32", "Mask must be float32")

    def test_label_values_and_ignore_mask(self):
        """Verifies target labels are strictly {0, 1, 255} and ignore mask excludes 255."""
        item = self.seg_ds[0]
        tgt = item["target"]
        mask = item["valid_mask"]

        if hasattr(tgt, "cpu"):
            tgt = tgt.cpu().numpy()
            mask = mask.cpu().numpy()

        unique_vals = set(np.unique(tgt))
        allowed_vals = {0, 1, IGNORE_INDEX}
        self.assertTrue(unique_vals.issubset(allowed_vals), f"Unexpected label values found: {unique_vals - allowed_vals}")

        # Check mask consistency: mask is 0 exactly where target == IGNORE_INDEX
        ignored_pixels = (tgt == IGNORE_INDEX)
        masked_pixels = (mask == 0.0)
        np.testing.assert_array_equal(ignored_pixels, masked_pixels, "Valid mask must be 0 exactly where target is IGNORE_INDEX")

    def test_change_dataset_shapes(self):
        """Verifies consecutive change dataset features and binary targets."""
        self.assertGreater(len(self.chg_ds), 0, "Change dataset must contain samples")
        c_item = self.chg_ds[0]

        img = c_item["image"]
        tgt = c_item["target"]

        # 18 channels = 6 (t) + 6 (t+1) + 6 (|diff|)
        self.assertEqual(img.shape, (18, 128, 128))
        self.assertEqual(tgt.shape, (128, 128))

    def test_missing_file_handling(self):
        """Verifies graceful handling of non-existent chips or missing years."""
        empty_ds = Sentinel2SegmentationDataset(
            s2_dir="/tmp/non_existent_s2_directory_12345",
            split="train"
        )
        self.assertEqual(len(empty_ds), 0, "Dataset with non-existent directory must have 0 samples without crashing")

    def test_grid_alignment_assertion(self):
        """Verifies that raster pair verification raises error on shape or CRS mismatch."""
        class MockRaster:
            def __init__(self, shape, crs, transform):
                self.shape = shape
                self.crs = crs
                self.transform = transform

        class MockTransform:
            a, b, c, d, e, f = 10.0, 0.0, 500000.0, 0.0, -10.0, 2000000.0

        r1 = MockRaster((500, 500), "EPSG:32643", MockTransform())
        r2 = MockRaster((500, 500), "EPSG:32643", MockTransform())
        # Identical rasters should pass
        verify_raster_pair(r1, r2)

        # Mismatched CRS should raise ValueError
        r_bad_crs = MockRaster((500, 500), "EPSG:32644", MockTransform())
        with self.assertRaises(ValueError):
            verify_raster_pair(r1, r_bad_crs)

        # Mismatched shape should raise ValueError
        r_bad_shape = MockRaster((250, 250), "EPSG:32643", MockTransform())
        with self.assertRaises(ValueError):
            verify_raster_pair(r1, r_bad_shape)


if __name__ == "__main__":
    unittest.main()
