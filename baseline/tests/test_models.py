#!/usr/bin/env python3
"""
test_models.py

Unit tests for model architecture, loss functions, and reproducible seeding:
- Deterministic seeding verification
- Model parameter validation
- Checkpoint directory handling
"""

import sys
import unittest
from pathlib import Path
import random
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baseline.train_baseline import set_seed
from baseline.models import CompactUNet, ChangeDetectionUNet, HAS_TORCH


class TestModels(unittest.TestCase):

    def test_reproducible_seeding(self):
        """Verifies deterministic random seeding reproduces identical pseudo-random sequences."""
        set_seed(1234)
        seq1 = [random.random() for _ in range(5)]
        arr1 = np.random.randn(5)

        set_seed(1234)
        seq2 = [random.random() for _ in range(5)]
        arr2 = np.random.randn(5)

        self.assertEqual(seq1, seq2, "Random sequences must be identical with same seed")
        np.testing.assert_array_equal(arr1, arr2, "NumPy arrays must be identical with same seed")

    def test_model_initialization(self):
        """Verifies CompactUNet and ChangeDetectionUNet initialization."""
        seg_model = CompactUNet(in_channels=7, num_classes=2)
        chg_model = ChangeDetectionUNet(in_channels=18, num_classes=2)

        self.assertEqual(seg_model.in_channels, 7)
        self.assertEqual(seg_model.num_classes, 2)
        self.assertEqual(chg_model.in_channels, 18)
        self.assertEqual(chg_model.num_classes, 2)

    def test_forward_pass_if_torch_available(self):
        """If PyTorch is installed, tests forward pass and loss calculation."""
        if not HAS_TORCH:
            self.skipTest("PyTorch is not installed in the active environment.")
        import torch
        from baseline.models import MaskedCrossEntropyLoss

        model = CompactUNet(in_channels=7, num_classes=2)
        criterion = MaskedCrossEntropyLoss(ignore_index=255)

        x = torch.randn(2, 7, 64, 64)
        y = torch.randint(0, 2, (2, 64, 64))
        y[0, 0, 0] = 255  # test ignore index

        logits = model(x)
        self.assertEqual(logits.shape, (2, 2, 64, 64))

        loss = criterion(logits, y)
        self.assertTrue(torch.isfinite(loss), "Loss must be finite")


if __name__ == "__main__":
    unittest.main()
