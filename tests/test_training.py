"""
Tests for training functionality.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import random
import sys
from pathlib import Path

# Add project root and src to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from nfl_bdb.training.train import MaskedMSELoss, set_seed


class TestMaskedMSELoss:
    """Tests for MaskedMSELoss."""

    def test_masked_mse_loss_computation(self):
        """Test that MaskedMSELoss computes correctly."""
        criterion = MaskedMSELoss()

        pred = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])  # [1, 2, 2]
        target = torch.tensor([[[1.0, 2.0], [5.0, 6.0]]])  # [1, 2, 2]
        mask = torch.tensor([[True, True]])  # [1, 2]

        loss = criterion(pred, target, mask)

        # MSE per sample computed with mean(dim=2):
        # First position: mean([(1-1)^2, (2-2)^2]) = mean([0, 0]) = 0
        # Second position: mean([(3-5)^2, (4-6)^2]) = mean([4, 4]) = 4
        # loss = [0, 4], mask = [True, True]
        # Masked sum: 0 + 4 = 4, mask sum: 2
        # Final: 4 / 2 = 2
        expected = 2.0
        assert abs(loss.item() - expected) < 1e-5

    def test_masked_mse_loss_with_partial_mask(self):
        """Test MaskedMSELoss with partial masking."""
        criterion = MaskedMSELoss()

        pred = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])  # [1, 3, 2]
        target = torch.tensor([[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]])  # [1, 3, 2]
        mask = torch.tensor([[True, False, True]])  # [1, 3] - middle masked out

        loss = criterion(pred, target, mask)

        # First: (1^2 + 2^2) / 2 = 2.5 -> mean = 2.5
        # Third: (5^2 + 6^2) / 2 = 30.5 -> mean = 30.5
        # Masked mean: (2.5 + 30.5) / 2 = 16.5
        expected = 16.5
        assert abs(loss.item() - expected) < 1e-5

    def test_masked_mse_loss_all_masked(self):
        """Test MaskedMSELoss when all values are masked (edge case)."""
        criterion = MaskedMSELoss()

        pred = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        target = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]])
        mask = torch.tensor([[False, False]])

        loss = criterion(pred, target, mask)

        # With all masked, sum is 0, clamp ensures no division by zero
        assert loss.item() == 0.0


class TestReproducibility:
    """Tests for reproducibility functionality."""

    def test_set_seed_makes_random_reproducible(self):
        """Test that set_seed makes Python random reproducible."""
        set_seed(42)
        values1 = [random.random() for _ in range(10)]

        set_seed(42)
        values2 = [random.random() for _ in range(10)]

        assert values1 == values2

    def test_set_seed_makes_numpy_reproducible(self):
        """Test that set_seed makes NumPy random reproducible."""
        set_seed(42)
        arr1 = np.random.randn(10)

        set_seed(42)
        arr2 = np.random.randn(10)

        np.testing.assert_array_equal(arr1, arr2)

    def test_set_seed_makes_torch_reproducible(self):
        """Test that set_seed makes PyTorch random reproducible."""
        set_seed(42)
        tensor1 = torch.randn(10)

        set_seed(42)
        tensor2 = torch.randn(10)

        torch.testing.assert_close(tensor1, tensor2)

    def test_different_seeds_give_different_results(self):
        """Test that different seeds produce different random values."""
        set_seed(42)
        value1 = random.random()

        set_seed(123)
        value2 = random.random()

        assert value1 != value2


class TestCheckpointSaveLoad:
    """Tests for checkpoint saving and loading."""

    def test_checkpoint_save_format(self, tmp_path):
        """Test that checkpoints are saved in correct format."""
        from nfl_bdb.training.train import save_checkpoint

        # Create simple model and optimizer
        model = nn.Linear(10, 2)
        optimizer = torch.optim.Adam(model.parameters())

        save_checkpoint(model, optimizer, epoch=5, train_loss=0.5, val_loss=0.3, out_dir=tmp_path)

        # Check file exists
        ckpt_path = tmp_path / "checkpoint_epoch_5.pt"
        assert ckpt_path.exists()

        # Load and verify contents
        ckpt = torch.load(ckpt_path)
        assert "epoch" in ckpt
        assert "model_state_dict" in ckpt
        assert "optimizer_state_dict" in ckpt
        assert "train_loss" in ckpt
        assert "val_loss" in ckpt
        assert ckpt["epoch"] == 5
        assert ckpt["train_loss"] == 0.5
        assert ckpt["val_loss"] == 0.3

    def test_checkpoint_model_state_loadable(self, tmp_path):
        """Test that saved model state can be loaded back."""
        from nfl_bdb.training.train import save_checkpoint

        model = nn.Linear(10, 2)
        optimizer = torch.optim.Adam(model.parameters())

        # Get original weights
        original_weight = model.weight.clone()

        save_checkpoint(model, optimizer, epoch=1, train_loss=0.5, val_loss=0.3, out_dir=tmp_path)

        # Create new model and load
        new_model = nn.Linear(10, 2)
        ckpt = torch.load(tmp_path / "checkpoint_epoch_1.pt")
        new_model.load_state_dict(ckpt["model_state_dict"])

        torch.testing.assert_close(new_model.weight, original_weight)
