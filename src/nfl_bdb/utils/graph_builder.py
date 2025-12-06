"""
Graph construction utilities for building spatial-temporal graphs from NFL tracking data.
"""

import numpy as np
import torch
from typing import Dict, Tuple, List
from dataclasses import dataclass


@dataclass
class GraphFeatures:
    """Container for graph features with lightweight `.to` support."""
    node_features: torch.Tensor  # [N, F_node]
    edge_index: torch.Tensor     # [2, E]
    edge_attr: torch.Tensor      # [E, F_edge]
    batch: torch.Tensor          # [N] - batch assignment for each node

    def to(self, device: torch.device, non_blocking: bool = False):
        """Return a copy of this graph on the specified device."""
        return GraphFeatures(
            node_features=self.node_features.to(device, non_blocking=non_blocking),
            edge_index=self.edge_index.to(device, non_blocking=non_blocking),
            edge_attr=self.edge_attr.to(device, non_blocking=non_blocking),
            batch=self.batch.to(device, non_blocking=non_blocking),
        )

    @property
    def x(self) -> torch.Tensor:
        """PyG-style alias used by the model."""
        return self.node_features

    def contiguous(self):
        """Ensure all tensors own their storage to avoid as_strided issues."""
        return GraphFeatures(
            node_features=self.node_features.contiguous(),
            edge_index=self.edge_index.contiguous(),
            edge_attr=self.edge_attr.contiguous(),
            batch=self.batch.contiguous(),
        )


def compute_distance(pos1: np.ndarray, pos2: np.ndarray) -> float:
    """Compute Euclidean distance between two positions."""
    return np.sqrt(np.sum((pos1 - pos2) ** 2))


def compute_angle(pos1: np.ndarray, pos2: np.ndarray) -> float:
    """Compute angle from pos1 to pos2 in radians."""
    dx = pos2[0] - pos1[0]
    dy = pos2[1] - pos1[1]
    return np.arctan2(dy, dx)


def build_graph(
    input_row: Dict,
    all_players: List[Dict],
    max_distance: float = 20.0,
    include_self_loops: bool = True
) -> GraphFeatures:
    """
    Build a spatial graph from player positions and features.
    
    Args:
        input_row: Dictionary containing features for the target player
        all_players: List of dictionaries for all players in the play
        max_distance: Maximum distance for edge connections (yards)
        include_self_loops: Whether to include self-loops
        
    Returns:
        GraphFeatures object with node features, edge indices, and edge attributes
    """
    n_players = len(all_players)
    
    # Extract positions and features
    positions = []
    node_features_list = []
    
    # Helper function to safely get float value, replacing NaN/Inf with 0.0
    def safe_get(player_dict, key, default=0.0):
        val = player_dict.get(key, default)
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            return default
        try:
            val_float = float(val)
            if np.isnan(val_float) or np.isinf(val_float):
                return default
            return val_float
        except (ValueError, TypeError):
            return default
    
    for player in all_players:
        pos = np.array([safe_get(player, 'x', 0.0), safe_get(player, 'y', 0.0)])
        positions.append(pos)
        
        # Node features: [x, y, vx, vy, s, a, dist_ball, dx_ball, dy_ball, ...]
        features = [
            safe_get(player, 'x', 0.0),
            safe_get(player, 'y', 0.0),
            safe_get(player, 'vx', 0.0),
            safe_get(player, 'vy', 0.0),
            safe_get(player, 's', 0.0),
            safe_get(player, 'a', 0.0),
            safe_get(player, 'dist_ball', 0.0),
            safe_get(player, 'dx_ball', 0.0),
            safe_get(player, 'dy_ball', 0.0),
        ]
        
        # Add one-hot encoded features if available
        for key in ['player_role_Targeted Receiver', 'player_role_Defensive Coverage',
                    'player_side_Offense', 'player_side_Defense']:
            features.append(safe_get(player, key, 0.0))
        
        node_features_list.append(features)
    
    # Convert to numpy first, then replace NaN/Inf, then to tensor
    node_features_np = np.array(node_features_list, dtype=np.float32)
    node_features_np = np.nan_to_num(node_features_np, nan=0.0, posinf=0.0, neginf=0.0)
    node_features = torch.tensor(node_features_np, dtype=torch.float32).contiguous()
    
    # Build edge connections (within max_distance)
    edge_list = []
    edge_attr_list = []
    
    for i in range(n_players):
        for j in range(n_players):
            if i == j and not include_self_loops:
                continue
            
            dist = compute_distance(positions[i], positions[j])
            
            if dist <= max_distance or i == j:
                edge_list.append([i, j])
                
                # Edge attributes: [distance, angle, relative_velocity_x, relative_velocity_y]
                angle = compute_angle(positions[i], positions[j])

                # Use safe_get for velocity to handle NaN/Inf values
                vx_i = safe_get(all_players[i], 'vx', 0.0)
                vy_i = safe_get(all_players[i], 'vy', 0.0)
                vx_j = safe_get(all_players[j], 'vx', 0.0)
                vy_j = safe_get(all_players[j], 'vy', 0.0)

                rel_vx = vx_j - vx_i
                rel_vy = vy_j - vy_i

                # Safety check for edge attributes
                if np.isnan(rel_vx) or np.isinf(rel_vx):
                    rel_vx = 0.0
                if np.isnan(rel_vy) or np.isinf(rel_vy):
                    rel_vy = 0.0

                edge_attr = [
                    dist,
                    angle,
                    rel_vx,
                    rel_vy,
                ]

                edge_attr_list.append(edge_attr)
    
    if len(edge_list) == 0:
        # Fallback: create self-loops only
        edge_list = [[i, i] for i in range(n_players)]
        edge_attr_list = [[0.0, 0.0, 0.0, 0.0] for _ in range(n_players)]
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32).contiguous()
    
    # Batch assignment (all nodes in same batch for single graph)
    batch = torch.zeros(n_players, dtype=torch.long).contiguous()
    
    return GraphFeatures(
        node_features=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        batch=batch
    )

