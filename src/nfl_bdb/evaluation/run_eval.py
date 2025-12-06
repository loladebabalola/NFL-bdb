import os
import json
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add project root to path for config import
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))
import configs.default as config

from nfl_bdb.models import STGNNRefine
from nfl_bdb.utils import build_graph, GraphFeatures


# -------------------------------------------------------------
# Dataset for evaluation (same as training, but only for eval)
# -------------------------------------------------------------
class EvalDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, max_players=22, max_T=22):
        self.seq_keys = list(sequences.keys())
        self.sequences = sequences
        self.max_players = max_players
        self.max_T = max_T  # 1 initial + 21 predictions

    def __len__(self):
        return len(self.seq_keys)

    def __getitem__(self, idx):
        key = self.seq_keys[idx]
        item = self.sequences[key]

        input_df = item["input"].copy()
        target = item["target"]  # shape (21, 2)

        # --- Build node features (numeric only) ---
        # Ensure input_df is a DataFrame during type filtering
        if isinstance(input_df, pd.Series):
            df_numeric = input_df.to_frame().T
        else:
            df_numeric = input_df
        
        numeric_cols = df_numeric.select_dtypes(
            include=["number", "bool"]
        ).columns.tolist()
        df_numeric = df_numeric[numeric_cols].fillna(0)
        
        # Ensure input_df is DataFrame for subsequent operations
        if isinstance(input_df, pd.Series):
            input_df = input_df.to_frame().T

        # Pad to max_players
        node_feats = np.zeros((self.max_players, df_numeric.shape[1]), dtype=np.float32)
        num_players = min(len(df_numeric), self.max_players)
        node_feats[:num_players] = df_numeric.values[:num_players]

        # Build graph - match training dataset call exactly
        all_players = input_df.to_dict("records")
        all_players = all_players[:num_players]
        
        # Get target player row (same as training: input_row.to_dict())
        player_row = input_df[input_df["player_to_predict"] == True].iloc[0]
        input_row = player_row.to_dict()
        
        # Convert target to tensor first
        target_tensor = torch.tensor(target, dtype=torch.float32)

        # Extract initial position: USE FIRST FRAME OF TARGET (matching training exactly)
        # Training NFLTrajectoryDataset.__getitem__: initial_pos = target_trajectory[:, 0:1, :].copy()
        # This ensures train-eval data contract is aligned
        pred_steps = 21

        # Use target[0] as start position (same as training)
        if target_tensor.shape[0] > 0:
            start_pos = target_tensor[0:1, :]  # [1, 2] - first position of target
        else:
            # Fallback to input_row if target is empty (shouldn't happen)
            start_pos = torch.tensor([[input_row.get("x", 0.0), input_row.get("y", 0.0)]], dtype=torch.float32)

        # Construct initial_pos: [T, 2] = target[0] + target[1:T] for padding
        if target_tensor.shape[0] >= pred_steps:
            initial_pos = target_tensor[:pred_steps]  # [T, 2]
        else:
            # Pad if target is shorter
            remaining = pred_steps - target_tensor.shape[0]
            if target_tensor.shape[0] > 0:
                pad = target_tensor[-1:].repeat(remaining, 1)
                initial_pos = torch.cat([target_tensor, pad], dim=0)  # [T, 2]
            else:
                initial_pos = start_pos.repeat(pred_steps, 1)  # [T, 2]
        
        # Build graph exactly like training
        graph = build_graph(
            input_row=input_row,               # dict for target player
            all_players=all_players[:num_players],
            max_distance=20.0,
            include_self_loops=True
        )
        
        # Ensure tensors are safe for batching
        graph = graph.contiguous()
        
        # ---- Ensure target is always 21 steps (same as training) ----
        MAX_STEPS = 21
        num_steps = target_tensor.shape[0]
        if num_steps < MAX_STEPS:
            pad_amount = MAX_STEPS - num_steps
            pad = torch.zeros((pad_amount, 2), dtype=target_tensor.dtype)
            target_tensor = torch.cat([target_tensor, pad], dim=0)
        elif num_steps > MAX_STEPS:
            # Truncate extra steps (should be rare)
            target_tensor = target_tensor[:MAX_STEPS]
        # --------------------------------------------------------------

        return {
            "graph": graph,                    # full GraphFeatures object
            "initial_pos": initial_pos,        # shape (T, 2) where T=21
            "target": target_tensor,           # shape (21, 2)
        }


# -------------------------------------------------------------
# Collate Function
# -------------------------------------------------------------
def eval_collate(batch):
    graphs = [b["graph"] for b in batch]
    initial_pos = torch.stack([b["initial_pos"] for b in batch])  # [B, T, 2] where T=21
    # Do NOT squeeze - keep shape [B, T, 2]
    targets = torch.stack([b["target"] for b in batch])  # [B, 21, 2]
    
    # Batch graphs into a single GraphFeatures object
    B = len(graphs)
    N = graphs[0].node_features.shape[0]  # nodes per graph
    
    # Concatenate node features
    batched_node_features = torch.cat([g.node_features for g in graphs], dim=0)  # [B*N, F]
    
    # Batch edge indices with offsets
    batched_edge_index_list = []
    batched_edge_attr_list = []
    node_offset = 0
    
    for g in graphs:
        offset_edge_index = g.edge_index + node_offset
        batched_edge_index_list.append(offset_edge_index)
        batched_edge_attr_list.append(g.edge_attr)
        node_offset += N
    
    batched_edge_index = torch.cat(batched_edge_index_list, dim=1)  # [2, total_edges]
    batched_edge_attr = torch.cat(batched_edge_attr_list, dim=0)  # [total_edges, 4]
    
    # Create batch assignment vector
    batch_index = torch.cat([torch.full((N,), i, dtype=torch.long) for i in range(B)], dim=0)  # [B*N]
    
    # Create batched graph
    batched_graph = GraphFeatures(
        node_features=batched_node_features,
        edge_index=batched_edge_index,
        edge_attr=batched_edge_attr,
        batch=batch_index
    )
    
    return batched_graph, initial_pos, targets


# -------------------------------------------------------------
# RMSE Computation
# -------------------------------------------------------------
def compute_rmse(pred, target):
    mse = ((pred - target) ** 2).mean()
    return float(torch.sqrt(mse).item())


def compute_step_rmse(pred, target):
    rmse_per_step = []
    for t in range(21):
        mse_t = ((pred[:, t] - target[:, t]) ** 2).mean()
        rmse_per_step.append(float(torch.sqrt(mse_t).item()))
    return rmse_per_step


# -------------------------------------------------------------
# Evaluation Logic
# -------------------------------------------------------------
def run_eval(
    ckpt_path=None,
    seq_path=None,
    batch_size=None,
    device=None
):
    """Run evaluation on sequences."""
    # Use defaults from config if not provided
    if ckpt_path is None:
        ckpt_path = str(config.MODEL_DIR / "best_model.pt")
    if seq_path is None:
        seq_path = str(config.PROCESSED_DIR / "sequences.pkl")
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    if device is None:
        device = "cuda" if config.DEVICE.startswith("cuda") else "cpu"
    print("[+] Loading sequences:", seq_path)
    sequences = pickle.load(open(seq_path, "rb"))

    print("[+] Loading model:", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)

    model = STGNNRefine(
        node_dim=config.NODE_DIM,
        hidden=config.HIDDEN_DIM,
        graph_layers=config.GRAPH_LAYERS,
        temporal_layers=config.TEMPORAL_LAYERS,
        heads=config.HEADS,
        max_len=config.MAX_TRAJECTORY_LENGTH,
        dropout=config.DROPOUT
    )

    # Handle both dict format and direct state_dict format
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

    # Load with strict=False but capture missing/unexpected keys
    incompatible = model.load_state_dict(state_dict, strict=False)

    # Handle key mismatches
    if incompatible.missing_keys:
        raise ValueError(
            f"Checkpoint is incompatible with model architecture. "
            f"Missing {len(incompatible.missing_keys)} keys: {incompatible.missing_keys[:5]}... "
            f"Please ensure checkpoint was trained with the same model architecture."
        )
    if incompatible.unexpected_keys:
        print(f"[INFO] Unexpected keys in checkpoint (ignored): {incompatible.unexpected_keys}")

    model.to(device)
    model.eval()
    print("[✓] Model loaded successfully")

    print("[+] Building EvalDataset...")
    ds = EvalDataset(sequences, max_players=config.MAX_PLAYERS, max_T=21)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=eval_collate)

    all_preds = []
    all_targets = []

    print("[+] Running local evaluation...")
    with torch.no_grad():
        for graph, initial_pos, targets in tqdm(dl):
            graph = graph.to(device)
            initial_pos = initial_pos.to(device)  # [B, T, 2] where T=21
            
            # initial_pos is already [B, T, 2] - do NOT squeeze
            preds = model(graph, initial_pos)  # [B, max_len, 2] where max_len=100
            
            # Extract only first 21 steps (matching target shape) - use clone to ensure it's a new tensor
            preds = preds[:, :21, :].clone()  # [B, 21, 2]
            
            # Verify shapes match before appending
            if preds.shape[1] != 21:
                raise RuntimeError(f"Preds shape mismatch: expected [B, 21, 2], got {preds.shape}")
            if targets.shape[1] != 21:
                raise RuntimeError(f"Targets shape mismatch: expected [B, 21, 2], got {targets.shape}")

            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    preds = torch.cat(all_preds, dim=0)  # [total_samples, 21, 2]
    targets = torch.cat(all_targets, dim=0)  # [total_samples, 21, 2]
    
    # Final shape verification
    assert preds.shape == targets.shape, f"Shape mismatch: preds {preds.shape} vs targets {targets.shape}"

    # Compute metrics
    overall_rmse = compute_rmse(preds, targets)
    step_rmse = compute_step_rmse(preds, targets)

    print("\n[=== Evaluation Results ===]")
    print(f"Overall RMSE: {overall_rmse:.5f}")
    for i, r in enumerate(step_rmse):
        print(f"Step {i}: {r:.5f}")

    # Save metrics
    metrics = {
        "overall_rmse": overall_rmse,
        "step_rmse": step_rmse,
    }
    json.dump(metrics, open("eval_metrics.json", "w"), indent=4)
    print("[+] Saved eval_metrics.json")

    # Save predictions for further analysis
    np.save("eval_preds.npy", preds.numpy())
    np.save("eval_targets.npy", targets.numpy())
    print("[+] Saved eval_preds.npy and eval_targets.npy")

    return overall_rmse


if __name__ == "__main__":
    run_eval()
