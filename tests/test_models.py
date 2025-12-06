"""
Tests for model architectures.
"""

import pytest
import torch
import sys
from pathlib import Path

# Add project root and src to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from nfl_bdb.models import STGNNRefine
from nfl_bdb.models.stgnn_refine import GATLayer
from nfl_bdb.utils import GraphFeatures


class TestSTGNNRefine:
    """Tests for STGNNRefine model."""

    def test_initialization(self):
        """Test STGNNRefine model initialization."""
        model = STGNNRefine(
            node_dim=13,
            hidden=256,
            graph_layers=3,
            temporal_layers=4,
            heads=8,
            max_len=100,
            dropout=0.1
        )
        assert model is not None
        assert model.hidden == 256
        assert model.max_len == 100

    def test_forward_training(self):
        """Test STGNNRefine forward pass in training mode."""
        model = STGNNRefine(node_dim=13, hidden=64, max_len=100)
        model.eval()

        # Create dummy graph
        batch_size = 2
        num_nodes = 22
        num_edges = 100

        graph = GraphFeatures(
            node_features=torch.randn(batch_size * num_nodes, 13),
            edge_index=torch.randint(0, batch_size * num_nodes, (2, num_edges)),
            edge_attr=torch.randn(num_edges, 4),
            batch=torch.cat([torch.zeros(num_nodes), torch.ones(num_nodes)]).long()
        )

        # Forward pass (training mode - no initial_pos)
        with torch.no_grad():
            output = model(graph, initial_pos=None)

        assert output.shape == (batch_size, 100, 2)  # [B, max_len, 2]

    def test_forward_inference(self):
        """Test STGNNRefine forward pass in inference mode."""
        model = STGNNRefine(node_dim=13, hidden=64, max_len=100)
        model.eval()

        # Create dummy graph
        batch_size = 2
        num_nodes = 22
        num_edges = 100

        graph = GraphFeatures(
            node_features=torch.randn(batch_size * num_nodes, 13),
            edge_index=torch.randint(0, batch_size * num_nodes, (2, num_edges)),
            edge_attr=torch.randn(num_edges, 4),
            batch=torch.cat([torch.zeros(num_nodes), torch.ones(num_nodes)]).long()
        )

        # Forward pass (inference mode - with initial_pos)
        initial_pos = torch.randn(batch_size, 2)  # [B, 2]

        with torch.no_grad():
            output = model(graph, initial_pos=initial_pos)

        assert output.shape == (batch_size, 100, 2)  # [B, max_len, 2]

    def test_output_no_nan_inf(self):
        """Test that model output does not contain NaN or Inf."""
        model = STGNNRefine(node_dim=13, hidden=64, max_len=100)
        model.eval()

        batch_size = 2
        num_nodes = 22
        num_edges = 100

        graph = GraphFeatures(
            node_features=torch.randn(batch_size * num_nodes, 13),
            edge_index=torch.randint(0, batch_size * num_nodes, (2, num_edges)),
            edge_attr=torch.randn(num_edges, 4),
            batch=torch.cat([torch.zeros(num_nodes), torch.ones(num_nodes)]).long()
        )

        with torch.no_grad():
            output = model(graph, initial_pos=None)

        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"


class TestGATLayer:
    """Tests for Graph Attention Layer."""

    def test_gat_layer_output_shape(self):
        """Test that GATLayer produces correct output shape."""
        hidden_dim = 64
        heads = 4
        num_nodes = 10
        num_edges = 30

        layer = GATLayer(hidden_dim=hidden_dim, heads=heads)

        x = torch.randn(num_nodes, hidden_dim)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))

        output = layer(x, edge_index)

        assert output.shape == (num_nodes, hidden_dim)

    def test_gat_attention_sums_to_one(self):
        """Test that attention weights sum to 1 per source node."""
        hidden_dim = 64
        heads = 4
        num_nodes = 5

        layer = GATLayer(hidden_dim=hidden_dim, heads=heads, dropout=0.0)

        x = torch.randn(num_nodes, hidden_dim)

        # Create fully connected graph (every node connects to every other)
        src = []
        dst = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                src.append(i)
                dst.append(j)
        edge_index = torch.tensor([src, dst], dtype=torch.long)

        # Manually compute attention weights using the internal method
        with torch.no_grad():
            N = x.size(0)
            row, col = edge_index

            Q = layer.Wq(x).view(N, layer.heads, layer.d_k)
            K = layer.Wk(x).view(N, layer.heads, layer.d_k)

            Q_i = Q[row]
            K_j = K[col]

            scores = (Q_i * K_j).sum(dim=-1) / (layer.d_k ** 0.5)

            # Use the scatter_softmax method
            attn = layer._scatter_softmax(scores, row, N)

            # For each source node, attention weights should sum to ~1
            for node_idx in range(num_nodes):
                mask = row == node_idx
                attn_sum = attn[mask].sum(dim=0)  # Sum per head
                for h in range(heads):
                    assert abs(attn_sum[h].item() - 1.0) < 1e-5, \
                        f"Attention for node {node_idx}, head {h} sums to {attn_sum[h].item()}"

    def test_gat_layer_residual_connection(self):
        """Test that GATLayer has residual connection (output != 0 when input is meaningful)."""
        hidden_dim = 64
        heads = 4

        layer = GATLayer(hidden_dim=hidden_dim, heads=heads)

        x = torch.randn(5, hidden_dim)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)

        output = layer(x, edge_index)

        # Output should not be all zeros (residual connection)
        assert not torch.allclose(output, torch.zeros_like(output))


# Legacy function-based tests for backwards compatibility
def test_stgnn_refine_initialization():
    """Test STGNNRefine model initialization."""
    TestSTGNNRefine().test_initialization()


def test_stgnn_refine_forward_training():
    """Test STGNNRefine forward pass in training mode."""
    TestSTGNNRefine().test_forward_training()


def test_stgnn_refine_forward_inference():
    """Test STGNNRefine forward pass in inference mode."""
    TestSTGNNRefine().test_forward_inference()

