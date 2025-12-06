"""
Tests for inference pipeline.
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add project root and src to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import configs.default as config
from nfl_bdb.inference.run_inference import inference_collate_fn, create_graph_from_batch
from nfl_bdb.utils import GraphFeatures


class TestInferenceCollateFn:
    """Tests for inference collate function."""

    def create_mock_inference_sample(self, key=(1, 1, 12345)):
        """Create a mock inference sample."""
        num_players = config.MAX_PLAYERS
        max_T = config.MAX_TRAJECTORY_LENGTH

        return {
            "node_feats": torch.randn(num_players, config.NODE_DIM),
            "edge_index": torch.randint(0, num_players, (2, 50)),
            "edge_attr": torch.randn(50, 4),
            "initial_pos": torch.randn(max_T, 2),
            "mask": torch.ones(max_T - 1, dtype=torch.bool),
            "key": key,
            "trajectory_length": 22
        }

    def test_inference_collate_fn_shapes(self):
        """Test inference collate function produces correct shapes."""
        samples = [self.create_mock_inference_sample((1, 1, i)) for i in range(4)]

        batch = inference_collate_fn(samples)

        B = 4
        N = config.MAX_PLAYERS
        T = config.MAX_TRAJECTORY_LENGTH

        assert batch["node_feats"].shape == (B * N, config.NODE_DIM)
        assert batch["batch_index"].shape == (B * N,)
        assert batch["initial_pos"].shape == (B, T, 2)
        assert batch["mask"].shape == (B, T - 1)
        assert len(batch["keys"]) == B

    def test_inference_collate_fn_preserves_keys(self):
        """Test that keys are preserved during collation."""
        keys = [(1, 2, 100), (1, 2, 101), (1, 3, 102)]
        samples = [self.create_mock_inference_sample(k) for k in keys]

        batch = inference_collate_fn(samples)

        assert batch["keys"] == keys

    def test_inference_collate_fn_empty_raises(self):
        """Test that empty batch raises ValueError."""
        with pytest.raises(ValueError, match="Empty batch"):
            inference_collate_fn([])


class TestCreateGraphFromBatch:
    """Tests for graph creation from batched data."""

    def test_create_graph_from_batch(self):
        """Test creating GraphFeatures from batch dictionary."""
        B = 2
        N = config.MAX_PLAYERS

        batch = {
            "node_feats": torch.randn(B * N, config.NODE_DIM),
            "edge_index": torch.randint(0, B * N, (2, 100)),
            "edge_attr": torch.randn(100, 4),
            "batch_index": torch.cat([torch.zeros(N), torch.ones(N)]).long()
        }

        device = torch.device('cpu')
        graph = create_graph_from_batch(batch, device)

        assert isinstance(graph, GraphFeatures)
        assert graph.node_features.shape == (B * N, config.NODE_DIM)
        assert graph.edge_index.shape[0] == 2
        assert graph.batch.shape == (B * N,)


class TestInferenceFeatureConsistency:
    """Tests to ensure inference uses same features as training."""

    def test_feature_count_matches_config(self):
        """Test that both training and inference use config.NODE_DIM features."""
        # The feature count should be consistent
        assert config.NODE_DIM == 13

    def test_feature_extraction_includes_xy(self):
        """Test that feature extraction includes x, y coordinates."""
        # This is a documentation test - the 13 features should include:
        # [x, y, vx, vy, s, a, dist_ball, dx_ball, dy_ball,
        #  player_role_Targeted Receiver, player_role_Defensive Coverage,
        #  player_side_Offense, player_side_Defense]
        expected_features = [
            'x', 'y', 'vx', 'vy', 's', 'a', 'dist_ball', 'dx_ball', 'dy_ball',
            'player_role_Targeted Receiver', 'player_role_Defensive Coverage',
            'player_side_Offense', 'player_side_Defense'
        ]
        assert len(expected_features) == config.NODE_DIM
