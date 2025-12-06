"""
Trajectory prediction metrics.

Standard metrics for evaluating trajectory predictions:
- ADE (Average Displacement Error): Mean L2 distance across all timesteps
- FDE (Final Displacement Error): L2 distance at final timestep
- minADE/minFDE: Minimum over K predictions (for multi-modal predictions)
"""

import numpy as np
import torch
from typing import Union, Tuple, Optional


def compute_ade(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    mask: Optional[Union[np.ndarray, torch.Tensor]] = None
) -> float:
    """
    Compute Average Displacement Error (ADE).

    ADE = mean(||pred_t - target_t||_2) for all t

    Args:
        pred: Predicted positions [B, T, 2] or [T, 2]
        target: Ground truth positions [B, T, 2] or [T, 2]
        mask: Optional validity mask [B, T] or [T]

    Returns:
        ADE value (scalar)
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    if mask is not None and isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    # Compute L2 distance at each timestep
    displacement = np.linalg.norm(pred - target, axis=-1)  # [B, T] or [T]

    if mask is not None:
        # Apply mask
        displacement = displacement * mask
        ade = displacement.sum() / mask.sum()
    else:
        ade = displacement.mean()

    return float(ade)


def compute_fde(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    mask: Optional[Union[np.ndarray, torch.Tensor]] = None
) -> float:
    """
    Compute Final Displacement Error (FDE).

    FDE = ||pred_T - target_T||_2 (at final timestep)

    Args:
        pred: Predicted positions [B, T, 2] or [T, 2]
        target: Ground truth positions [B, T, 2] or [T, 2]
        mask: Optional validity mask [B, T] or [T] - uses last valid timestep

    Returns:
        FDE value (scalar)
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    if mask is not None and isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    if mask is not None:
        # Find last valid timestep for each sample
        if pred.ndim == 3:  # [B, T, 2]
            fde_list = []
            for i in range(pred.shape[0]):
                valid_idx = np.where(mask[i])[0]
                if len(valid_idx) > 0:
                    last_idx = valid_idx[-1]
                    fde_list.append(np.linalg.norm(pred[i, last_idx] - target[i, last_idx]))
            fde = np.mean(fde_list) if fde_list else 0.0
        else:  # [T, 2]
            valid_idx = np.where(mask)[0]
            if len(valid_idx) > 0:
                last_idx = valid_idx[-1]
                fde = np.linalg.norm(pred[last_idx] - target[last_idx])
            else:
                fde = 0.0
    else:
        # Use last timestep
        if pred.ndim == 3:  # [B, T, 2]
            final_displacement = np.linalg.norm(pred[:, -1] - target[:, -1], axis=-1)
            fde = final_displacement.mean()
        else:  # [T, 2]
            fde = np.linalg.norm(pred[-1] - target[-1])

    return float(fde)


def compute_metrics(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    mask: Optional[Union[np.ndarray, torch.Tensor]] = None
) -> dict:
    """
    Compute all trajectory metrics.

    Args:
        pred: Predicted positions [B, T, 2] or [T, 2]
        target: Ground truth positions [B, T, 2] or [T, 2]
        mask: Optional validity mask

    Returns:
        Dictionary with 'ade', 'fde' keys
    """
    return {
        'ade': compute_ade(pred, target, mask),
        'fde': compute_fde(pred, target, mask)
    }


def compute_miss_rate(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    threshold: float = 2.0
) -> float:
    """
    Compute miss rate (percentage of final predictions beyond threshold).

    Args:
        pred: Predicted positions [B, T, 2] or [T, 2]
        target: Ground truth positions [B, T, 2] or [T, 2]
        threshold: Distance threshold in yards (default 2.0)

    Returns:
        Miss rate as percentage (0-100)
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    # Final displacement error
    if pred.ndim == 3:
        final_error = np.linalg.norm(pred[:, -1] - target[:, -1], axis=-1)
    else:
        final_error = np.linalg.norm(pred[-1] - target[-1])
        final_error = np.array([final_error])

    miss_rate = (final_error > threshold).mean() * 100
    return float(miss_rate)
