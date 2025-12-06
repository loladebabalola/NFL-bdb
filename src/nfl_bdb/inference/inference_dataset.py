"""
Inference Dataset for NFL trajectory prediction.

Mirrors NFLTrajectoryDataset structure exactly, but without target/target_mask.
Returns graph-structured input matching training pipeline.
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

from ..utils.graph_builder import build_graph


class InferenceDataset(Dataset):
    """
    Dataset for NFL trajectory inference with strict shape contracts matching training.
    
    Returns:
        {
            "node_feats": Tensor[N, F],      # Node features (numeric only)
            "edge_index": Tensor[2, E],      # Edge indices
            "edge_attr": Tensor[E, 4],       # Edge attributes
            "initial_pos": Tensor[T, 2],      # Initial position (last observed frame, repeated)
            "mask": Tensor[T-1],             # Validity mask (all ones for inference)
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
            max_trajectory_length: Maximum trajectory length (for padding)
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
        Get a single sample with strict shape contracts matching training.
        
        Returns:
            Dictionary with guaranteed shapes:
            - node_feats: [N, F] where N = max_players, F = numeric features
            - edge_index: [2, E]
            - edge_attr: [E, 4]
            - initial_pos: [max_T, 2] (last observed position, repeated)
            - mask: [max_T-1] (all ones for inference)
        """
        key = self.keys[idx]
        seq = self.sequences[key]
        
        game_id, play_id, nfl_id = key
        input_row = seq['input']  # pandas Series with last frame
        
        # ====================================================================
        # 1. Extract initial position (last observed frame)
        # ====================================================================
        # For inference, initial_pos is the last observed position
        # We need to predict 21 steps (t+1 to t+21), so we pad to max_trajectory_length
        last_x = float(input_row.get('x', 0.0))
        last_y = float(input_row.get('y', 0.0))
        initial_pos = np.array([[last_x, last_y]], dtype=np.float32)  # [1, 2]
        
        # Pad initial_pos to [max_T, 2] - match training padding pattern exactly
        padded_initial = np.zeros((self.max_trajectory_length, 2), dtype=np.float32)
        padded_initial[0] = initial_pos[0]  # First position
        # For inference, we don't have a trajectory, so repeat the last observed position
        # This matches training's pattern: padded_initial[T:] = target_trajectory[-1]
        padded_initial[1:] = initial_pos[0]  # Repeat last position for all remaining frames
        
        # Create mask (all ones for inference - we predict all 21 steps)
        # For inference: T = 22 (1 initial + 21 predictions)
        T = 22
        mask = np.ones(T - 1, dtype=bool)  # [21] - matches training pattern
        
        # Pad mask to [max_T-1] - match training padding pattern exactly
        padded_mask = np.zeros(self.max_trajectory_length - 1, dtype=bool)
        padded_mask[:T-1] = mask
        
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
        # 2.5. Convert categorical/boolean fields to numeric (like training)
        # ====================================================================
        # --- FIX: convert categorical directions/orientation and boolean fields into numeric like training ---
        def normalize_player_fields(player_dict):
            """Convert categorical and boolean fields to numeric values."""
            player_dict = player_dict.copy()
            
            # Convert 'dir'
            if 'dir' in player_dict:
                dir_val = player_dict['dir']
                if isinstance(dir_val, str):
                    if dir_val.lower() == "left":
                        player_dict['dir'] = 270.0
                    elif dir_val.lower() == "right":
                        player_dict['dir'] = 90.0
                    else:
                        player_dict['dir'] = 0.0  # fallback
                # If already numeric, leave it as is
            
            # Convert 'o'
            if 'o' in player_dict:
                o_val = player_dict['o']
                if isinstance(o_val, str):
                    if o_val.lower() == "left":
                        player_dict['o'] = 270.0
                    elif o_val.lower() == "right":
                        player_dict['o'] = 90.0
                    else:
                        player_dict['o'] = 0.0  # fallback
                # If already numeric, leave it as is
            
            # --- FIX: normalize boolean column to numeric ---
            if 'player_to_predict' in player_dict:
                pred_val = player_dict['player_to_predict']
                if isinstance(pred_val, str):
                    player_dict['player_to_predict'] = 1.0 if pred_val.lower() == "true" else 0.0
                elif isinstance(pred_val, bool):
                    player_dict['player_to_predict'] = float(pred_val)
                elif pred_val is None:
                    player_dict['player_to_predict'] = 0.0
                else:
                    # fallback: treat unknown values as not the predicted player
                    try:
                        player_dict['player_to_predict'] = float(pred_val)
                    except (ValueError, TypeError):
                        player_dict['player_to_predict'] = 0.0
            
            return player_dict
        
        # Apply conversion to all players (including target player)
        all_players = [normalize_player_fields(p) for p in all_players]
        
        # ====================================================================
        # 3. Extract numeric node features (MUST MATCH TRAINING EXACTLY)
        # ====================================================================
        # Extract the same 13 features that training uses in datasets.py
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

            # Extract the exact 13 features matching training (including x, y!)
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
            for feature_name in ['player_role_Targeted Receiver', 'player_role_Defensive Coverage',
                        'player_side_Offense', 'player_side_Defense']:
                features.append(safe_get(feature_name, 0.0))

            node_features_list.append(features)

        # Convert to numpy array
        node_feats_np = np.array(node_features_list, dtype=np.float32)  # [N, 13]

        # Final safety check: replace any remaining NaN/Inf with 0.0
        node_feats_np = np.nan_to_num(node_feats_np, nan=0.0, posinf=0.0, neginf=0.0)
        N = node_feats_np.shape[0]
        F = node_feats_np.shape[1]

        # Validate feature count matches config (must match training)
        if F != config.NODE_DIM:
            raise ValueError(
                f"Expected {config.NODE_DIM} node features (from config.NODE_DIM), got {F} for key={key}. "
                f"Features extracted: {node_features_list[0] if node_features_list else 'empty'}"
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
                f"Node feature conversion failed for key={key}\n"
                f"Dtypes: {numeric_df.dtypes}\n"
                f"Values sample: {node_feats_np[:3]}"
            ) from e
        
        # ====================================================================
        # 4. Build graph (edge_index, edge_attr)
        # ====================================================================
        graph = build_graph(
            input_row.to_dict(),
            all_players[:N],  # Use actual number of players (before padding) - matches training
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
        # 5. Return with strict shape validation
        # ====================================================================
        result = {
            "node_feats": node_feats,  # [max_players, F]
            "edge_index": graph.edge_index,  # [2, E]
            "edge_attr": graph.edge_attr,  # [E, 4]
            "initial_pos": torch.tensor(padded_initial, dtype=torch.float32),  # [max_T, 2]
            "mask": torch.tensor(padded_mask, dtype=torch.bool),  # [max_T-1]
            "key": key,
            "trajectory_length": 22,  # For inference: 1 initial + 21 predictions
        }
        
        # Strict shape assertions (matching training)
        assert result["node_feats"].shape[0] == self.max_players, \
            f"node_feats shape[0] must be {self.max_players}, got {result['node_feats'].shape[0]}"
        assert result["edge_index"].shape[0] == 2, \
            f"edge_index shape[0] must be 2, got {result['edge_index'].shape[0]}"
        assert result["initial_pos"].shape == (self.max_trajectory_length, 2), \
            f"initial_pos shape must be ({self.max_trajectory_length}, 2), got {result['initial_pos'].shape}"
        assert result["mask"].shape == (self.max_trajectory_length - 1,), \
            f"mask shape must be ({self.max_trajectory_length - 1},), got {result['mask'].shape}"
        
        return result

