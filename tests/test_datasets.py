"""
Tests for data loading and dataset classes.
"""

import pytest
import torch
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add project root and src to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import configs.default as config
from nfl_bdb.utils import NFLTrajectoryDataset, collate_fn, build_graph, GraphFeatures


class TestGraphBuilder:
    """Tests for graph construction."""

    def test_build_graph_returns_graph_features(self):
        """Test that build_graph returns a GraphFeatures object."""
        input_row = {
            'x': 50.0, 'y': 26.5, 'vx': 1.0, 'vy': 0.5,
            's': 5.0, 'a': 0.1, 'dist_ball': 10.0,
            'dx_ball': 5.0, 'dy_ball': 3.0,
            'player_role_Targeted Receiver': 1.0,
            'player_role_Defensive Coverage': 0.0,
            'player_side_Offense': 1.0,
            'player_side_Defense': 0.0,
            'nfl_id': 12345
        }
        all_players = [input_row]

        graph = build_graph(input_row, all_players, max_distance=20.0)

        assert isinstance(graph, GraphFeatures)
        assert graph.node_features is not None
        assert graph.edge_index is not None
        assert graph.edge_attr is not None

    def test_build_graph_node_features_shape(self):
        """Test that node features have correct shape [N, 13]."""
        players = []
        for i in range(5):
            players.append({
                'x': 50.0 + i, 'y': 26.5, 'vx': 1.0, 'vy': 0.5,
                's': 5.0, 'a': 0.1, 'dist_ball': 10.0,
                'dx_ball': 5.0, 'dy_ball': 3.0,
                'player_role_Targeted Receiver': 0.0,
                'player_role_Defensive Coverage': 0.0,
                'player_side_Offense': 1.0,
                'player_side_Defense': 0.0,
                'nfl_id': 12345 + i
            })

        graph = build_graph(players[0], players, max_distance=20.0)

        assert graph.node_features.shape[0] == 5  # 5 players
        assert graph.node_features.shape[1] == config.NODE_DIM  # 13 features

    def test_build_graph_edge_index_shape(self):
        """Test that edge_index has shape [2, E]."""
        input_row = {
            'x': 50.0, 'y': 26.5, 'vx': 1.0, 'vy': 0.5,
            's': 5.0, 'a': 0.1, 'dist_ball': 10.0,
            'dx_ball': 5.0, 'dy_ball': 3.0,
            'player_role_Targeted Receiver': 0.0,
            'player_role_Defensive Coverage': 0.0,
            'player_side_Offense': 1.0,
            'player_side_Defense': 0.0,
            'nfl_id': 12345
        }

        graph = build_graph(input_row, [input_row], max_distance=20.0, include_self_loops=True)

        assert graph.edge_index.shape[0] == 2
        assert graph.edge_index.shape[1] >= 1  # At least self-loop

    def test_build_graph_handles_nan_values(self):
        """Test that build_graph handles NaN values gracefully."""
        input_row = {
            'x': 50.0, 'y': np.nan, 'vx': float('inf'), 'vy': 0.5,
            's': None, 'a': 0.1, 'dist_ball': 10.0,
            'dx_ball': 5.0, 'dy_ball': 3.0,
            'player_role_Targeted Receiver': 0.0,
            'player_role_Defensive Coverage': 0.0,
            'player_side_Offense': 1.0,
            'player_side_Defense': 0.0,
            'nfl_id': 12345
        }

        graph = build_graph(input_row, [input_row], max_distance=20.0)

        # Should not contain NaN or Inf
        assert not torch.isnan(graph.node_features).any()
        assert not torch.isinf(graph.node_features).any()


class TestGraphFeatures:
    """Tests for GraphFeatures dataclass."""

    def test_graph_features_x_property(self):
        """Test that .x property returns node_features."""
        node_features = torch.randn(10, 13)
        edge_index = torch.randint(0, 10, (2, 20))
        edge_attr = torch.randn(20, 4)
        batch = torch.zeros(10, dtype=torch.long)

        graph = GraphFeatures(
            node_features=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=batch
        )

        assert torch.equal(graph.x, graph.node_features)

    def test_graph_features_to_device(self):
        """Test that .to() moves all tensors to device."""
        graph = GraphFeatures(
            node_features=torch.randn(10, 13),
            edge_index=torch.randint(0, 10, (2, 20)),
            edge_attr=torch.randn(20, 4),
            batch=torch.zeros(10, dtype=torch.long)
        )

        # Move to CPU (should work on any system)
        graph_cpu = graph.to(torch.device('cpu'))

        assert graph_cpu.node_features.device == torch.device('cpu')
        assert graph_cpu.edge_index.device == torch.device('cpu')

    def test_graph_features_contiguous(self):
        """Test that .contiguous() returns contiguous tensors."""
        # Create non-contiguous tensor
        node_features = torch.randn(10, 13).t().t()  # Transpose twice may not be contiguous

        graph = GraphFeatures(
            node_features=node_features,
            edge_index=torch.randint(0, 10, (2, 20)),
            edge_attr=torch.randn(20, 4),
            batch=torch.zeros(10, dtype=torch.long)
        )

        graph_contig = graph.contiguous()

        assert graph_contig.node_features.is_contiguous()
        assert graph_contig.edge_index.is_contiguous()


class TestCollateFn:
    """Tests for batch collation."""

    def create_mock_sample(self, key=(1, 1, 12345)):
        """Create a mock dataset sample."""
        num_players = config.MAX_PLAYERS
        max_T = config.MAX_TRAJECTORY_LENGTH

        return {
            "node_feats": torch.randn(num_players, config.NODE_DIM),
            "edge_index": torch.randint(0, num_players, (2, 50)),
            "edge_attr": torch.randn(50, 4),
            "initial_pos": torch.randn(num_players, max_T, 2),
            "target": torch.randn(num_players, max_T - 1, 2),
            "mask": torch.ones(num_players, max_T - 1, dtype=torch.bool),
            "key": key,
            "trajectory_length": max_T
        }

    def test_collate_fn_batches_correctly(self):
        """Test that collate_fn creates correct batch shapes."""
        samples = [self.create_mock_sample((1, 1, i)) for i in range(4)]

        batch = collate_fn(samples)

        B = 4
        N = config.MAX_PLAYERS
        T = config.MAX_TRAJECTORY_LENGTH

        assert batch["node_feats"].shape == (B * N, config.NODE_DIM)
        assert batch["batch_index"].shape == (B * N,)
        assert batch["initial_pos"].shape == (B, N, T, 2)
        assert batch["target"].shape == (B, N, T - 1, 2)
        assert batch["mask"].shape == (B, N, T - 1)
        assert len(batch["keys"]) == B

    def test_collate_fn_edge_offset(self):
        """Test that edge indices are properly offset for batching."""
        samples = [self.create_mock_sample((1, 1, i)) for i in range(2)]

        batch = collate_fn(samples)

        # Edge indices should be in range [0, B*N)
        max_edge_idx = batch["edge_index"].max().item()
        assert max_edge_idx < 2 * config.MAX_PLAYERS

    def test_collate_fn_empty_batch_raises(self):
        """Test that empty batch raises ValueError."""
        with pytest.raises(ValueError, match="Empty batch"):
            collate_fn([])
