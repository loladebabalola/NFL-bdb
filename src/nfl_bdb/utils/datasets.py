"""
PyTorch Dataset for NFL trajectory sequences.

DeepMind-caliber dataset with strict shape contracts and zero dtype errors.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, Tuple, List, Optional
import pandas as pd
import sys
from pathlib import Path

# Add project root to path for config import
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))
import configs.default as config

from .graph_builder import build_graph


class NFLTrajectoryDataset(Dataset):
    """
    Dataset for NFL trajectory prediction with strict shape contracts.
    
    Returns:
        {
            "node_feats": Tensor[N, F],      # Node features (numeric only)
            "edge_index": Tensor[2, E],      # Edge indices
            "edge_attr": Tensor[E, 4],       # Edge attributes
            "initial_pos": Tensor[num_players, T, 2],      # Initial position sequence [players, time, x, y]
            "target": Tensor[num_players, T-1, 2],        # Target deltas/positions [players, t=1 to t=T-1, x, y]
            "mask": Tensor[num_players, T-1],             # Validity mask for target
            "key": Tuple[int, int, int]      # (game_id, play_id, nfl_id)
        }
    """
    
    def __init__(
        self,
        sequences: Dict[Tuple[int, int, int], Dict],
        all_input_df: pd.DataFrame = None,
        max_players: int = 22,
        max_trajectory_length: int = 100,
        radius: float = 20.0
    ):
        """
        Args:
            sequences: Dictionary mapping (game_id, play_id, nfl_id) to sequence dict
            all_input_df: Full input dataframe for building graphs with all players
            max_players: Maximum number of players to include in graph
            max_trajectory_length: Maximum trajectory length (will pad if shorter)
            radius: Maximum distance for edge connections (yards)
        """
        self.sequences = sequences
        self.keys = list(sequences.keys())
        self.all_input_df = all_input_df
        self.max_players = max_players
        self.max_trajectory_length = max_trajectory_length
        self.radius = radius
        
        # Validate sequences
        if len(self.keys) == 0:
            raise ValueError("No sequences provided")
    
    def __len__(self) -> int:
        return len(self.keys)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample with strict shape contracts.
        
        Returns:
            Dictionary with guaranteed shapes:
            - node_feats: [N, F] where N <= max_players, F = numeric features
            - edge_index: [2, E]
            - edge_attr: [E, 4]
            - initial_pos: [num_players, T, 2] where T = trajectory length
            - target: [num_players, T-1, 2] (positions from t=1 to t=T-1)
            - mask: [num_players, T-1] (True for valid positions)
        """
        key = self.keys[idx]
        seq = self.sequences[key]
        
        game_id, play_id, nfl_id = key
        input_row = seq['input']
        target_trajectory = seq['target']  # [T, 2] - full trajectory
        
        # ====================================================================
        # 1. Extract and validate trajectory
        # ====================================================================
        if not isinstance(target_trajectory, np.ndarray):
            target_trajectory = np.array(target_trajectory)
        
        if target_trajectory.shape[1] != 2:
            raise ValueError(f"Target trajectory must have shape [T, 2], got {target_trajectory.shape}")
        
        num_players = self.max_players  # 22
        T = len(target_trajectory)  # Number of time steps for single player
        
        if T < 2:
            raise ValueError(f"Trajectory must have at least 2 frames, got {T}")
        
        # Truncate if too long
        if T > self.max_trajectory_length:
            target_trajectory = target_trajectory[:self.max_trajectory_length]
            T = self.max_trajectory_length
        
        # Replicate single player trajectory across all players: [T, 2] -> [num_players, T, 2]
        # Broadcast the trajectory to all players (same trajectory for all players)
        target_trajectory = np.broadcast_to(
            target_trajectory[np.newaxis, :, :], 
            (num_players, T, 2)
        ).copy()  # [num_players, T, 2]
        
        # Extract initial position (first frame for all players): [num_players, 1, 2]
        initial_pos = target_trajectory[:, 0:1, :].copy()  # [num_players, 1, 2]
        
        # Target is positions from t=1 to t=T-1 (deltas will be computed in model)
        # For now, we return absolute positions; model will compute deltas
        target = target_trajectory[:, 1:, :].copy()  # [num_players, T-1, 2]
        
        # Create mask (all True since we truncated/padded): [num_players, T-1]
        mask = np.ones((num_players, T - 1), dtype=bool)  # [num_players, T-1]
        
        # ====================================================================
        # 2. Build graph with all players
        # ====================================================================
        if self.all_input_df is not None:
            play_mask = (
                (self.all_input_df['game_id'] == game_id) &
                (self.all_input_df['play_id'] == play_id)
            )
            play_data = self.all_input_df[play_mask].copy()
            
            # Get last frame for each player
            all_players = []
            for player_nfl_id in play_data['nfl_id'].unique():
                player_data = play_data[play_data['nfl_id'] == player_nfl_id]
                if len(player_data) > 0:
                    player_row = player_data.iloc[-1].to_dict()
                    all_players.append(player_row)
        else:
            # Fallback: only use target player
            all_players = [input_row.to_dict()]
        
        # Ensure target player is first
        target_player_dict = input_row.to_dict()
        target_nfl_id = target_player_dict.get('nfl_id')
        
        # Separate target player from others
        other_players = [p for p in all_players if p.get('nfl_id') != target_nfl_id]
        all_players = [target_player_dict] + other_players[:self.max_players - 1]
        
        # ====================================================================
        # 3. Extract numeric node features (matching build_graph exactly)
        # ====================================================================
        # Extract the same 13 features that build_graph uses
        # Features: [x, y, vx, vy, s, a, dist_ball, dx_ball, dy_ball,
        #            player_role_Targeted Receiver, player_role_Defensive Coverage,
        #            player_side_Offense, player_side_Defense]
        node_features_list = []
        
        for player in all_players:
            # Helper function to safely get float value, replacing NaN/Inf with 0.0
            def safe_get(key, default=0.0):
                val = player.get(key, default)
                if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
                    return default
                try:
                    val_float = float(val)
                    if np.isnan(val_float) or np.isinf(val_float):
                        return default
                    return val_float
                except (ValueError, TypeError):
                    return default
            
            features = [
                safe_get('x', 0.0),
                safe_get('y', 0.0),
                safe_get('vx', 0.0),
                safe_get('vy', 0.0),
                safe_get('s', 0.0),
                safe_get('a', 0.0),
                safe_get('dist_ball', 0.0),
                safe_get('dx_ball', 0.0),
                safe_get('dy_ball', 0.0),
            ]
            
            # Add one-hot encoded features if available
            for key in ['player_role_Targeted Receiver', 'player_role_Defensive Coverage',
                        'player_side_Offense', 'player_side_Defense']:
                features.append(safe_get(key, 0.0))
            
            node_features_list.append(features)
        
        # Convert to numpy array
        node_feats_np = np.array(node_features_list, dtype=np.float32)  # [N, 13]
        
        # Final safety check: replace any remaining NaN/Inf with 0.0
        node_feats_np = np.nan_to_num(node_feats_np, nan=0.0, posinf=0.0, neginf=0.0)
        N = node_feats_np.shape[0]
        F = node_feats_np.shape[1]
        
        # Validate feature count matches config
        if F != config.NODE_DIM:
            raise ValueError(
                f"Expected {config.NODE_DIM} node features (from config.NODE_DIM), got {F} for {key}. "
                f"Features: {node_features_list[0] if node_features_list else 'empty'}"
            )
        
        # Pad to max_players if needed (with zeros)
        if N < self.max_players:
            padding = np.zeros((self.max_players - N, F), dtype=node_feats_np.dtype)
            node_feats_np = np.vstack([node_feats_np, padding])
            N = self.max_players
        
        # Convert to tensor
        try:
            node_feats = torch.tensor(node_feats_np, dtype=torch.float32)
        except Exception as e:
            raise TypeError(
                f"Node feature conversion failed for {key}\n"
                f"Dtypes: {numeric_df.dtypes}\n"
                f"Values sample: {node_feats_np[:3]}"
            ) from e
        
        # ====================================================================
        # 4. Build graph (edge_index, edge_attr)
        # ====================================================================
        graph = build_graph(
            input_row.to_dict(),
            all_players[:N],  # Use actual number of players (before padding)
            max_distance=self.radius,
            include_self_loops=True
        )
        
        # Pad graph to max_players if needed
        if N < self.max_players:
            # Add self-loops for padded nodes
            padded_nodes = list(range(N, self.max_players))
            for node_idx in padded_nodes:
                # Add self-loop
                new_edges = torch.tensor([[node_idx], [node_idx]], dtype=torch.long)
                graph.edge_index = torch.cat([graph.edge_index, new_edges], dim=1)
                # Add zero edge attributes
                zero_attr = torch.zeros((1, graph.edge_attr.shape[1]), dtype=torch.float32)
                graph.edge_attr = torch.cat([graph.edge_attr, zero_attr], dim=0)
        
        # ====================================================================
        # 5. Pad trajectory sequences to max_trajectory_length
        # ====================================================================
        # Pad initial_pos to [num_players, max_T, 2]
        padded_initial = np.zeros((num_players, self.max_trajectory_length, 2), dtype=np.float32)
        padded_initial[:, 0:1] = initial_pos  # First position for all players
        # Repeat first position for padding (or use last valid)
        if T > 1:
            padded_initial[:, 1:T] = target_trajectory[:, 1:T, :]
            padded_initial[:, T:] = target_trajectory[:, -1:, :]  # Repeat last position
        
        # Pad target to [num_players, max_T-1, 2]
        padded_target = np.zeros((num_players, self.max_trajectory_length - 1, 2), dtype=np.float32)
        padded_target[:, :T-1] = target
        if T - 1 < self.max_trajectory_length - 1:
            padded_target[:, T-1:] = target[:, -1:, :]  # Repeat last position
        
        # Pad mask to [num_players, max_T-1]
        padded_mask = np.zeros((num_players, self.max_trajectory_length - 1), dtype=bool)
        padded_mask[:, :T-1] = mask
        
        # ====================================================================
        # 6. Return with strict shape validation
        # ====================================================================
        result = {
            "node_feats": node_feats,  # [max_players, F]
            "edge_index": graph.edge_index,  # [2, E]
            "edge_attr": graph.edge_attr,  # [E, 4]
            "initial_pos": torch.tensor(padded_initial, dtype=torch.float32),  # [num_players, max_T, 2]
            "target": torch.tensor(padded_target, dtype=torch.float32),  # [num_players, max_T-1, 2]
            "mask": torch.tensor(padded_mask, dtype=torch.bool),  # [num_players, max_T-1]
            "key": key,
            "trajectory_length": T
        }
        
        # Strict shape assertions
        assert result["node_feats"].shape[0] == self.max_players, \
            f"node_feats shape[0] must be {self.max_players}, got {result['node_feats'].shape[0]}"
        assert result["edge_index"].shape[0] == 2, \
            f"edge_index shape[0] must be 2, got {result['edge_index'].shape[0]}"
        assert result["initial_pos"].shape == (num_players, self.max_trajectory_length, 2), \
            f"initial_pos shape must be ({num_players}, {self.max_trajectory_length}, 2), got {result['initial_pos'].shape}"
        assert result["target"].shape == (num_players, self.max_trajectory_length - 1, 2), \
            f"target shape must be ({num_players}, {self.max_trajectory_length - 1}, 2), got {result['target'].shape}"
        assert result["mask"].shape == (num_players, self.max_trajectory_length - 1,), \
            f"mask shape must be ({num_players}, {self.max_trajectory_length - 1},), got {result['mask'].shape}"
        
        return result
