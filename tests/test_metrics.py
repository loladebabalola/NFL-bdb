"""
Tests for trajectory evaluation metrics.
"""

import pytest
import numpy as np
import torch
import sys
from pathlib import Path

# Add project root and src to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from nfl_bdb.evaluation.metrics import compute_ade, compute_fde, compute_metrics, compute_miss_rate


class TestADE:
    """Tests for Average Displacement Error."""

    def test_ade_perfect_prediction(self):
        """ADE should be 0 for perfect predictions."""
        pred = np.array([[0, 0], [1, 1], [2, 2]])
        target = np.array([[0, 0], [1, 1], [2, 2]])
        ade = compute_ade(pred, target)
        assert ade == 0.0

    def test_ade_constant_error(self):
        """ADE with constant displacement."""
        pred = np.array([[1, 0], [2, 0], [3, 0]])
        target = np.array([[0, 0], [1, 0], [2, 0]])
        ade = compute_ade(pred, target)
        assert abs(ade - 1.0) < 1e-6  # All displacements are 1.0

    def test_ade_diagonal_error(self):
        """ADE with diagonal displacement."""
        pred = np.array([[1, 1], [2, 2]])
        target = np.array([[0, 0], [1, 1]])
        ade = compute_ade(pred, target)
        expected = np.sqrt(2)  # sqrt(1^2 + 1^2)
        assert abs(ade - expected) < 1e-6

    def test_ade_batched(self):
        """ADE should work with batched inputs."""
        pred = np.array([
            [[0, 0], [1, 1]],
            [[0, 0], [2, 2]]
        ])  # [2, 2, 2]
        target = np.array([
            [[0, 0], [0, 0]],
            [[0, 0], [0, 0]]
        ])
        ade = compute_ade(pred, target)
        # Batch 1: sqrt(2), Batch 2: sqrt(8), average
        expected = (0 + np.sqrt(2) + 0 + np.sqrt(8)) / 4
        assert abs(ade - expected) < 1e-6

    def test_ade_with_torch_tensor(self):
        """ADE should work with PyTorch tensors."""
        pred = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        target = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
        ade = compute_ade(pred, target)
        assert abs(ade - 0.5) < 1e-6


class TestFDE:
    """Tests for Final Displacement Error."""

    def test_fde_perfect_prediction(self):
        """FDE should be 0 for perfect final prediction."""
        pred = np.array([[0, 0], [1, 1], [2, 2]])
        target = np.array([[0, 0], [1, 1], [2, 2]])
        fde = compute_fde(pred, target)
        assert fde == 0.0

    def test_fde_only_final_matters(self):
        """FDE only considers final timestep."""
        pred = np.array([[100, 100], [200, 200], [3, 3]])  # Bad early, good final
        target = np.array([[0, 0], [1, 1], [3, 3]])
        fde = compute_fde(pred, target)
        assert fde == 0.0

    def test_fde_final_error(self):
        """FDE with known final error."""
        pred = np.array([[0, 0], [1, 1], [5, 3]])
        target = np.array([[0, 0], [1, 1], [2, 3]])
        fde = compute_fde(pred, target)
        assert abs(fde - 3.0) < 1e-6  # |5-2| = 3, |3-3| = 0, sqrt(9+0) = 3

    def test_fde_batched(self):
        """FDE should work with batched inputs."""
        pred = np.array([
            [[0, 0], [3, 0]],  # FDE = 3
            [[0, 0], [0, 4]]   # FDE = 4
        ])
        target = np.array([
            [[0, 0], [0, 0]],
            [[0, 0], [0, 0]]
        ])
        fde = compute_fde(pred, target)
        expected = (3.0 + 4.0) / 2
        assert abs(fde - expected) < 1e-6


class TestMissRate:
    """Tests for miss rate computation."""

    def test_miss_rate_all_hits(self):
        """Miss rate should be 0% when all within threshold."""
        pred = np.array([
            [[0, 0], [0.5, 0.5]],
            [[0, 0], [1.0, 0.0]]
        ])
        target = np.array([
            [[0, 0], [0, 0]],
            [[0, 0], [0, 0]]
        ])
        miss_rate = compute_miss_rate(pred, target, threshold=2.0)
        assert miss_rate == 0.0

    def test_miss_rate_all_misses(self):
        """Miss rate should be 100% when all beyond threshold."""
        pred = np.array([
            [[0, 0], [10, 0]],
            [[0, 0], [0, 10]]
        ])
        target = np.array([
            [[0, 0], [0, 0]],
            [[0, 0], [0, 0]]
        ])
        miss_rate = compute_miss_rate(pred, target, threshold=2.0)
        assert miss_rate == 100.0

    def test_miss_rate_half(self):
        """Miss rate should be 50% when half miss."""
        pred = np.array([
            [[0, 0], [1, 0]],   # FDE = 1 < 2 (hit)
            [[0, 0], [5, 0]]    # FDE = 5 > 2 (miss)
        ])
        target = np.array([
            [[0, 0], [0, 0]],
            [[0, 0], [0, 0]]
        ])
        miss_rate = compute_miss_rate(pred, target, threshold=2.0)
        assert miss_rate == 50.0


class TestComputeMetrics:
    """Tests for combined metrics computation."""

    def test_compute_metrics_returns_dict(self):
        """compute_metrics should return dict with ade and fde."""
        pred = np.array([[0, 0], [1, 1]])
        target = np.array([[0, 0], [0, 0]])
        metrics = compute_metrics(pred, target)

        assert 'ade' in metrics
        assert 'fde' in metrics
        assert isinstance(metrics['ade'], float)
        assert isinstance(metrics['fde'], float)

    def test_compute_metrics_values(self):
        """Verify compute_metrics values."""
        pred = np.array([[0, 0], [2, 0]])
        target = np.array([[0, 0], [0, 0]])

        metrics = compute_metrics(pred, target)

        # ADE = (0 + 2) / 2 = 1
        assert abs(metrics['ade'] - 1.0) < 1e-6
        # FDE = 2
        assert abs(metrics['fde'] - 2.0) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
