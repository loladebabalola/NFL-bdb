"""
Unified Training Script for NFL Spatio-Temporal GNN + Transformer Model
Includes:
 - sequence loader
 - dataframe loader
 - dataset creation
 - checkpoint saving
 - training & validation loops
 - reproducibility (seeded random state)
 - learning rate scheduling
 - early stopping
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
import random
import pandas as pd
import pickle
import numpy as np
import sys
import os

# Add project root to path for config import
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))
import configs.default as config

from nfl_bdb.models import STGNNRefine
from nfl_bdb.utils import NFLTrajectoryDataset, collate_fn, GraphFeatures


# ============================================================
# 0. REPRODUCIBILITY
# ============================================================
def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For fully deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"[✓] Random seed set to {seed} for reproducibility")


# ============================================================
# 1. LOAD SEQUENCES
# ============================================================
def load_sequences():
    """Load preprocessed sequences.pkl."""
    seq_path = config.PROCESSED_DIR / "sequences.pkl"
    if not seq_path.exists():
        raise FileNotFoundError(f"❌ Sequences not found: {seq_path}")

    with open(seq_path, "rb") as f:
        seq = pickle.load(f)

    print(f"[✓] Loaded {len(seq)} sequences")
    return seq


# ============================================================
# 2. LOAD INPUT DATAFRAME (ALL PLAYER FRAMES)
# ============================================================
def load_all_input_df():
    """Load all input CSVs from data/train."""
    frames = []
    for csv in sorted(config.TRAIN_DIR.glob("*.csv")):
        print(f"  - reading {csv.name}")
        frames.append(pd.read_csv(csv))

    df = pd.concat(frames, ignore_index=True)
    print(f"[✓] Loaded all_input_df with {len(df)} rows")
    return df


# ============================================================
# 3. SPLIT + CREATE DATASETS
# ============================================================
def create_datasets(sequences, all_input_df):
    """Split sequences into train/val and create Dataset objects."""
    keys = list(sequences.keys())
    random.shuffle(keys)

    split = int(len(keys) * 0.90)
    train_keys = keys[:split]
    val_keys = keys[split:]

    train_seq = {k: sequences[k] for k in train_keys}
    val_seq = {k: sequences[k] for k in val_keys}

    train_dataset = NFLTrajectoryDataset(
        sequences=train_seq,
        all_input_df=all_input_df
    )

    val_dataset = NFLTrajectoryDataset(
        sequences=val_seq,
        all_input_df=all_input_df
    )

    return train_dataset, val_dataset


# ============================================================
# 4. MASKED MSE LOSS
# ============================================================
class MaskedMSELoss(nn.Module):
    """Compute MSE only where mask == True."""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, pred, target, mask):
        # pred: [B, T, 2]
        # target: [B, T, 2]
        loss = self.mse(pred, target).mean(dim=2)  # [B, T]
        loss = loss * mask
        return loss.sum() / mask.sum().clamp(min=1)


# ============================================================
# 5. SAVE CHECKPOINT
# ============================================================
def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, out_dir):
    """Write a full checkpoint file."""
    out_dir.mkdir(exist_ok=True)
    ckpt_path = out_dir / f"checkpoint_epoch_{epoch}.pt"

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss
    }, ckpt_path)

    print(f"[✓] Saved checkpoint → {ckpt_path}")


# ============================================================
# 6. TRAIN / VAL LOOPS
# ============================================================
def train_epoch(model, loader, optimizer, criterion, scaler, device, epoch=None):
    model.train()
    total_loss = 0

    for batch_idx, batch in enumerate(loader):
        optimizer.zero_grad()
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

        # Debug print for first batch of first epoch
        if epoch == 1 and batch_idx == 0:
            print("DEBUG initial_pos shape:", batch["initial_pos"].shape)

        # Construct GraphFeatures object from batch dictionary
        graph = GraphFeatures(
            node_features=batch["node_feats"],
            edge_index=batch["edge_index"],
            edge_attr=batch["edge_attr"],
            batch=batch["batch_index"]
        )
        
        # Validate input features
        if torch.isnan(graph.node_features).any() or torch.isinf(graph.node_features).any():
            print(f"Warning: NaN/Inf in node_features at batch {batch_idx}, skipping")
            continue

        with autocast():
            # Forward pass: model predicts velocities (no initial_pos during training)
            pred_vel = model(graph, initial_pos=None)  # [B, max_len, 2] = [B, 100, 2] - velocities
            
            # Check for NaN/Inf in predictions early
            if torch.isnan(pred_vel).any() or torch.isinf(pred_vel).any():
                print(f"Warning: NaN/Inf detected in predictions at batch {batch_idx}, skipping")
                print(f"  pred_vel stats: min={pred_vel.min().item():.4f}, max={pred_vel.max().item():.4f}, mean={pred_vel.mean().item():.4f}")
                continue
            
            # Velocity loss computation
            # batch["target"] is [B, 22, 99, 2], extract target player (first player)
            pos = batch["target"][:, 0, :, :]  # [B, 99, 2] - target player trajectory (positions)
            mask = batch["mask"][:, 0, :]  # [B, 99] - target player mask
            
            # Compute target velocities from positions
            target_vel = pos[:, 1:, :] - pos[:, :-1, :]  # [B, T-1, 2] = [B, 98, 2]
            
            # Model predicts velocities for each step, use first T-1 velocities
            pred_vel_aligned = pred_vel[:, :target_vel.shape[1], :]  # [B, 98, 2]
            
            # Velocity mask (shift mask by 1 since velocities are between timesteps)
            vel_mask = mask[:, 1:]  # [B, 98]
            
            # Safeguard: ensure mask has valid entries
            mask_sum = vel_mask.sum()
            if mask_sum == 0:
                print(f"Warning: Empty mask at batch {batch_idx}, skipping")
                continue
            
            # Masked velocity MSE
            mse = ((pred_vel_aligned - target_vel) ** 2).sum(dim=-1)  # [B, 98]
            loss = (mse * vel_mask).sum() / mask_sum.clamp(min=1)
            
            # Check for NaN/Inf in loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN/Inf loss at batch {batch_idx}, skipping")
                print(f"  mse stats: min={mse.min().item():.4f}, max={mse.max().item():.4f}")
                continue

        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
        
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        
        # Log progress periodically
        if (batch_idx + 1) % config.LOG_INTERVAL == 0:
            avg_loss = total_loss / (batch_idx + 1)
            print(f"  Batch {batch_idx + 1}/{len(loader)} | Loss: {avg_loss:.4f}")

    return {"loss": total_loss / len(loader)}


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

            # Construct GraphFeatures object from batch dictionary
            graph = GraphFeatures(
                node_features=batch["node_feats"],
                edge_index=batch["edge_index"],
                edge_attr=batch["edge_attr"],
                batch=batch["batch_index"]
            )
            
            # Validate input features
            if torch.isnan(graph.node_features).any() or torch.isinf(graph.node_features).any():
                continue

            # Forward pass: model predicts velocities (no initial_pos during validation)
            pred_vel = model(graph, initial_pos=None)  # [B, max_len, 2] = [B, 100, 2] - velocities
            
            # Check for NaN/Inf in predictions
            if torch.isnan(pred_vel).any() or torch.isinf(pred_vel).any():
                continue
            
            # Velocity loss computation (same as training)
            # batch["target"] is [B, 22, 99, 2], extract target player (first player)
            pos = batch["target"][:, 0, :, :]  # [B, 99, 2] - target player trajectory (positions)
            mask = batch["mask"][:, 0, :]  # [B, 99] - target player mask
            
            # Compute target velocities from positions
            target_vel = pos[:, 1:, :] - pos[:, :-1, :]  # [B, T-1, 2] = [B, 98, 2]
            
            # Model predicts velocities for each step, use first T-1 velocities
            pred_vel_aligned = pred_vel[:, :target_vel.shape[1], :]  # [B, 98, 2]
            
            # Velocity mask (shift mask by 1 since velocities are between timesteps)
            vel_mask = mask[:, 1:]  # [B, 98]
            
            # Safeguard: ensure mask has valid entries
            mask_sum = vel_mask.sum()
            if mask_sum == 0:
                continue
            
            # Masked velocity MSE
            mse = ((pred_vel_aligned - target_vel) ** 2).sum(dim=-1)  # [B, 98]
            loss = (mse * vel_mask).sum() / mask_sum.clamp(min=1)
            
            # Check for NaN/Inf in loss
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            total_loss += loss.item()

    return {"loss": total_loss / len(loader)}


# ============================================================
# 7. MAIN TRAIN FUNCTION
# ============================================================
def main():
    # Set seed for reproducibility
    seed = getattr(config, 'SEED', 42)
    set_seed(seed)

    # Ensure directories exist
    if hasattr(config, 'ensure_dirs'):
        config.ensure_dirs()
    else:
        config.MODEL_DIR.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[✓] Using device: {device}")

    # ----- Load data -----
    sequences = load_sequences()
    all_input_df = load_all_input_df()
    train_dataset, val_dataset = create_datasets(sequences, all_input_df)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=getattr(config, 'NUM_WORKERS', 0)  # Default to 0 if not in config
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=getattr(config, 'NUM_WORKERS', 0)  # Default to 0 if not in config
    )

    # ----- Model -----
    model = STGNNRefine(
        node_dim=config.NODE_DIM,
        hidden=config.HIDDEN_DIM,
        graph_layers=config.GRAPH_LAYERS,
        temporal_layers=config.TEMPORAL_LAYERS,
        heads=config.HEADS,
        max_len=config.MAX_TRAJECTORY_LENGTH,
        dropout=config.DROPOUT
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    criterion = MaskedMSELoss()
    scaler = GradScaler()

    # ----- Learning Rate Scheduler -----
    use_scheduler = getattr(config, 'USE_SCHEDULER', True)
    if use_scheduler:
        lr_min = getattr(config, 'LR_MIN', 1e-6)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.EPOCHS, eta_min=lr_min
        )
        print(f"[✓] Using CosineAnnealingLR scheduler (eta_min={lr_min})")
    else:
        scheduler = None

    best_val_loss = float("inf")
    best_path = config.MODEL_DIR / "best_model.pt"

    # ----- Early Stopping -----
    patience = getattr(config, 'EARLY_STOPPING_PATIENCE', 10)
    no_improve_count = 0
    print(f"[✓] Early stopping patience: {patience} epochs")

    # ----- Training -----
    for epoch in range(1, config.EPOCHS + 1):
        current_lr = optimizer.param_groups[0]['lr']

        train_metrics = train_epoch(model, train_loader, optimizer, criterion, scaler, device, epoch=epoch)
        val_metrics = validate(model, val_loader, criterion, device)

        print(f"\nEpoch {epoch}/{config.EPOCHS} | LR: {current_lr:.2e} | Train: {train_metrics['loss']:.4f} | Val: {val_metrics['loss']:.4f}")

        # Step scheduler
        if scheduler is not None:
            scheduler.step()

        # Save periodic checkpoint
        if epoch % config.CHECKPOINT_INTERVAL == 0:
            save_checkpoint(model, optimizer, epoch, train_metrics["loss"], val_metrics["loss"], config.MODEL_DIR)

        # Save best model and check early stopping
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            no_improve_count = 0
            torch.save(model.state_dict(), best_path)
            print(f"[✓] Saved BEST model → {best_path} (loss={best_val_loss:.4f})")
        else:
            no_improve_count += 1
            print(f"[!] No improvement for {no_improve_count}/{patience} epochs")

            if no_improve_count >= patience:
                print(f"\n[!] Early stopping triggered at epoch {epoch}")
                break

    print("\nTraining complete.")
    print(f"Best Validation Loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()


