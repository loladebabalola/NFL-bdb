"""
Inference script for NFL Big Data Bowl trajectory prediction.

Uses InferenceDataset to create graph-structured input matching training pipeline.
"""

import os
import pickle
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List, Dict
import sys
from pathlib import Path

# Add project root to path for config import
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))
import configs.default as config

from nfl_bdb.models import STGNNRefine
from nfl_bdb.utils import GraphFeatures
from .inference_dataset import InferenceDataset


def inference_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for inference batching (no target/trajectory_length).
    
    Args:
        batch: List of samples, each with:
            - node_feats: [N, F]
            - edge_index: [2, E]
            - edge_attr: [E, 4]
            - initial_pos: [T, 2]
            - mask: [T-1]
            - key: Tuple
    
    Returns:
        Batched dictionary with:
            - node_feats: [B*N, F] (concatenated, with batch_index)
            - edge_index: [2, total_edges] (offset indices)
            - edge_attr: [total_edges, 4] (concatenated)
            - batch_index: [B*N] (batch assignment for each node)
            - initial_pos: [B, T, 2] (stacked)
            - mask: [B, T-1] (stacked)
            - keys: List[Tuple]
    """
    if len(batch) == 0:
        raise ValueError("Empty batch")
    
    B = len(batch)
    
    # Extract and validate shapes from first sample
    first_sample = batch[0]
    N = first_sample["node_feats"].shape[0]  # max_players (same for all)
    F = first_sample["node_feats"].shape[1]  # feature dim (same for all)
    T = first_sample["initial_pos"].shape[0]  # max_trajectory_length (same for all)
    
    # Batch node features and create batch_index
    node_feats_list = []
    batch_index_list = []
    
    for b_idx, sample in enumerate(batch):
        node_feats = sample["node_feats"]  # [N, F]
        assert node_feats.shape == (N, F), \
            f"Sample {b_idx}: node_feats shape must be ({N}, {F}), got {node_feats.shape}"
        
        node_feats_list.append(node_feats)
        batch_index_list.append(torch.full((N,), b_idx, dtype=torch.long))
    
    batched_node_feats = torch.cat(node_feats_list, dim=0)  # [B*N, F]
    batch_index = torch.cat(batch_index_list, dim=0)  # [B*N]
    
    # Batch edge indices and attributes (with offset)
    edge_index_list = []
    edge_attr_list = []
    
    node_offset = 0
    for b_idx, sample in enumerate(batch):
        edge_index = sample["edge_index"]  # [2, E]
        edge_attr = sample["edge_attr"]  # [E, 4]
        
        assert edge_index.shape[0] == 2, \
            f"Sample {b_idx}: edge_index shape[0] must be 2, got {edge_index.shape[0]}"
        assert edge_attr.shape[1] == 4, \
            f"Sample {b_idx}: edge_attr shape[1] must be 4, got {edge_attr.shape[1]}"
        assert edge_index.shape[1] == edge_attr.shape[0], \
            f"Sample {b_idx}: edge_index and edge_attr must have matching edge count"
        
        # Offset edge indices by node_offset
        offset_edge_index = edge_index + node_offset
        edge_index_list.append(offset_edge_index)
        edge_attr_list.append(edge_attr)
        
        node_offset += N
    
    batched_edge_index = torch.cat(edge_index_list, dim=1)  # [2, total_edges]
    batched_edge_attr = torch.cat(edge_attr_list, dim=0)  # [total_edges, 4]
    
    # Batch temporal sequences (initial_pos, mask)
    initial_pos_list = []
    mask_list = []
    keys_list = []
    
    for sample in batch:
        initial_pos = sample["initial_pos"]  # [T, 2]
        mask = sample["mask"]  # [T-1]
        key = sample["key"]
        
        assert initial_pos.shape == (T, 2), \
            f"initial_pos shape must be ({T}, 2), got {initial_pos.shape}"
        assert mask.shape == (T - 1,), \
            f"mask shape must be ({T - 1},), got {mask.shape}"
        
        initial_pos_list.append(initial_pos)
        mask_list.append(mask)
        keys_list.append(key)
    
    batched_initial_pos = torch.stack(initial_pos_list, dim=0)  # [B, T, 2]
    batched_mask = torch.stack(mask_list, dim=0)  # [B, T-1]
    
    return {
        "node_feats": batched_node_feats.contiguous(),  # [B*N, F]
        "edge_index": batched_edge_index.contiguous(),  # [2, total_edges]
        "edge_attr": batched_edge_attr.contiguous(),  # [total_edges, 4]
        "batch_index": batch_index.contiguous(),  # [B*N]
        "initial_pos": batched_initial_pos.contiguous(),  # [B, T, 2]
        "mask": batched_mask.contiguous(),  # [B, T-1]
        "keys": keys_list,  # List[Tuple]
    }


def create_graph_from_batch(batch: dict, device: torch.device) -> GraphFeatures:
    """
    Create a GraphFeatures object from batched dictionary.
    
    Args:
        batch: Dictionary from collate_fn with:
            - node_feats: [B*N, F]
            - edge_index: [2, total_edges]
            - edge_attr: [total_edges, 4]
            - batch_index: [B*N]
        device: Target device
    
    Returns:
        GraphFeatures object with .x, .edge_index, .edge_attr, .batch
    """
    graph = GraphFeatures(
        node_features=batch["node_feats"].to(device),
        edge_index=batch["edge_index"].to(device),
        edge_attr=batch["edge_attr"].to(device),
        batch=batch["batch_index"].to(device)
    )
    return graph


def run_inference(
    ckpt_path="models/stgnn_refine_best.pt",
    seq_path="data/processed/test_sequences.pkl",
    submission_path="submission.csv",
    batch_size=32,
    device="cuda",
    all_input_df=None
):
    """
    Run inference on test sequences and generate submission file.
    
    Args:
        ckpt_path: Path to model checkpoint
        seq_path: Path to test_sequences.pkl
        submission_path: Output path for submission.csv
        batch_size: Batch size for inference
        device: Device to run inference on
        all_input_df: Optional full input dataframe for graph construction
    """
    print("[+] Loading checkpoint:", ckpt_path)
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

    # Get state dict from checkpoint (handle both formats)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    # Load with strict=False but capture missing/unexpected keys
    incompatible = model.load_state_dict(state_dict, strict=False)

    # Warn about key mismatches
    if incompatible.missing_keys:
        print(f"[WARNING] Missing keys in checkpoint: {incompatible.missing_keys}")
        print("[WARNING] Model parameters not loaded - predictions may be unreliable!")
    if incompatible.unexpected_keys:
        print(f"[INFO] Unexpected keys in checkpoint (ignored): {incompatible.unexpected_keys}")

    # Raise error if critical keys are missing
    if incompatible.missing_keys:
        raise ValueError(
            f"Checkpoint is incompatible with model architecture. "
            f"Missing {len(incompatible.missing_keys)} keys. "
            f"Please ensure checkpoint was trained with the same model architecture."
        )

    model.to(device)
    model.eval()
    print("[✓] Model loaded successfully")

    print("[+] Loading test sequences:", seq_path)
    with open(seq_path, "rb") as f:
        sequences = pickle.load(f)

    print(f"[+] Creating InferenceDataset with {len(sequences)} sequences")
    dataset = InferenceDataset(
        sequences=sequences,
        all_input_df=all_input_df,
        max_players=config.MAX_PLAYERS,
        max_trajectory_length=config.MAX_TRAJECTORY_LENGTH,
        radius=config.RADIUS
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=inference_collate_fn,
        num_workers=0  # Set to 0 to avoid multiprocessing issues
    )

    preds_rows = []

    print("[+] Running inference…")
    with torch.no_grad():
        for batch in tqdm(loader):
            # Move all tensors to device
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            
            # Create graph object from batch
            graph = create_graph_from_batch(batch, device)
            
            # Extract initial_pos: [B, max_T, 2] -> [B, 2] (first position, which is last observed frame)
            # For inference, initial_pos is the last observed frame, repeated across T
            # We take the first element as the starting position
            initial_pos = batch["initial_pos"][:, 0, :]  # [B, 2] - starting position
            
            # Forward pass
            # Model expects initial_pos as [B, 2] and returns absolute positions
            # [B, max_T, 2] where each row is absolute position at that time step
            positions = model(graph, initial_pos=initial_pos)  # [B, max_T, 2]
            
            # Extract only the first 21 predictions (steps 0-20, which correspond to t+1 to t+21)
            # Note: model returns positions for all max_T steps, but we only need 21
            positions = positions[:, :21, :]  # [B, 21, 2]
            positions = positions.cpu().numpy()
            
            # Unpack predictions
            keys = batch["keys"]
            for key, pred in zip(keys, positions):
                gameId, playId, nflId = key
                for step in range(21):
                    preds_rows.append({
                        "gameId": int(gameId),
                        "playId": int(playId),
                        "nflId": int(nflId),
                        "step": step,
                        "x": float(pred[step][0]),
                        "y": float(pred[step][1]),
                    })

    # Create submission DataFrame
    df = pd.DataFrame(preds_rows)
    
    # Ensure correct column order
    df = df[["gameId", "playId", "nflId", "step", "x", "y"]]
    
    # Save submission
    df.to_csv(submission_path, index=False)
    
    print(f"[✓] Saved submission: {submission_path}")
    print(f"[✓] Total predictions: {len(df)}")
    print(f"[✓] Unique (gameId, playId, nflId): {df[['gameId', 'playId', 'nflId']].drop_duplicates().shape[0]}")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
if __name__ == "__main__":
    # Optionally load all_input_df for better graph construction
    # For now, we'll use None (fallback to single player)
    all_input_df = None
    
    # If you have access to full test input data, you can load it here:
    # import pandas as pd
    # all_input_df = pd.read_csv("data/test_sample/test_input.csv")
    
    run_inference(
        ckpt_path="models/stgnn_refine_best.pt",
        seq_path="data/processed/test_sequences.pkl",
        submission_path="submission.csv",
        batch_size=32,
        device="cuda" if torch.cuda.is_available() else "cpu",
        all_input_df=all_input_df
    )
