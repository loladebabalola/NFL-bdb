"""
Attention visualization for the STGNN model.

Visualize:
- Graph attention weights between players
- Player importance heatmaps
- Trajectory predictions on field
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Optional, Tuple, List
import sys
from pathlib import Path


def draw_football_field(ax, xlim=(0, 120), ylim=(0, 53.3)):
    """
    Draw a football field background.

    Args:
        ax: Matplotlib axis
        xlim: X-axis limits (yard lines)
        ylim: Y-axis limits (field width)
    """
    # Field background
    ax.set_facecolor('#2e7d32')

    # Yard lines
    for yard in range(0, 121, 5):
        if xlim[0] <= yard <= xlim[1]:
            alpha = 0.5 if yard % 10 == 0 else 0.2
            ax.axvline(x=yard, color='white', alpha=alpha, linewidth=0.5)

    # Hash marks
    for y in [0, 53.3/3, 2*53.3/3, 53.3]:
        ax.axhline(y=y, color='white', alpha=0.2, linewidth=0.5)

    # End zones
    if xlim[0] <= 10:
        ax.add_patch(patches.Rectangle((0, 0), 10, 53.3,
                                        facecolor='#1565c0', alpha=0.3))
    if xlim[1] >= 110:
        ax.add_patch(patches.Rectangle((110, 0), 10, 53.3,
                                        facecolor='#c62828', alpha=0.3))

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('Yard Line', fontsize=10)
    ax.set_ylabel('Field Width (yards)', fontsize=10)


def plot_trajectories(
    predictions: np.ndarray,
    targets: Optional[np.ndarray] = None,
    initial_positions: Optional[np.ndarray] = None,
    player_labels: Optional[List[str]] = None,
    title: str = "Trajectory Predictions",
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
):
    """
    Plot predicted trajectories on football field.

    Args:
        predictions: Predicted trajectories [N, T, 2] or [T, 2]
        targets: Ground truth trajectories (same shape)
        initial_positions: Starting positions [N, 2] or [2]
        player_labels: Optional labels for each player
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Figure and axis objects
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Ensure 3D shape
    if predictions.ndim == 2:
        predictions = predictions[np.newaxis, :, :]
    if targets is not None and targets.ndim == 2:
        targets = targets[np.newaxis, :, :]

    N = predictions.shape[0]
    colors = plt.cm.tab10(np.linspace(0, 1, min(N, 10)))

    # Determine field bounds from data
    all_x = predictions[:, :, 0].flatten()
    all_y = predictions[:, :, 1].flatten()
    if targets is not None:
        all_x = np.concatenate([all_x, targets[:, :, 0].flatten()])
        all_y = np.concatenate([all_y, targets[:, :, 1].flatten()])

    x_margin = 5
    xlim = (max(0, all_x.min() - x_margin), min(120, all_x.max() + x_margin))
    ylim = (max(0, all_y.min() - 5), min(53.3, all_y.max() + 5))

    draw_football_field(ax, xlim=xlim, ylim=ylim)

    # Plot each player's trajectory
    for i in range(min(N, 10)):  # Limit to 10 for visibility
        color = colors[i]
        label = player_labels[i] if player_labels else f'Player {i+1}'

        # Predicted trajectory
        ax.plot(predictions[i, :, 0], predictions[i, :, 1],
                '-', color=color, linewidth=2, alpha=0.8, label=f'{label} (pred)')

        # Mark prediction endpoints
        ax.scatter(predictions[i, 0, 0], predictions[i, 0, 1],
                   s=100, color=color, marker='o', edgecolor='white', linewidth=2, zorder=5)
        ax.scatter(predictions[i, -1, 0], predictions[i, -1, 1],
                   s=100, color=color, marker='s', edgecolor='white', linewidth=2, zorder=5)

        # Ground truth if available
        if targets is not None:
            ax.plot(targets[i, :, 0], targets[i, :, 1],
                    '--', color=color, linewidth=1.5, alpha=0.5, label=f'{label} (true)')

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8, ncol=2)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved trajectory plot to {save_path}")

    return fig, ax


def plot_attention_heatmap(
    attention_weights: np.ndarray,
    node_labels: Optional[List[str]] = None,
    title: str = "Graph Attention Weights",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
):
    """
    Plot attention weights as a heatmap.

    Args:
        attention_weights: Attention matrix [N, N] or [N, N, H] (H=heads)
        node_labels: Labels for each node (player)
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Figure and axis objects
    """
    if attention_weights.ndim == 3:
        # Average over heads
        attention_weights = attention_weights.mean(axis=-1)

    N = attention_weights.shape[0]

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(attention_weights, cmap='YlOrRd', aspect='auto')
    plt.colorbar(im, ax=ax, label='Attention Weight')

    if node_labels is None:
        node_labels = [f'P{i+1}' for i in range(N)]

    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xticklabels(node_labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(node_labels, fontsize=8)

    ax.set_xlabel('Target Player', fontsize=12)
    ax.set_ylabel('Source Player', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved attention heatmap to {save_path}")

    return fig, ax


def plot_prediction_distribution(
    predictions_df,
    title: str = "Prediction Distribution",
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None
):
    """
    Plot distribution of predictions.

    Args:
        predictions_df: DataFrame with columns [gameId, playId, nflId, step, x, y]
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Figure and axes objects
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # X distribution
    ax1 = axes[0]
    ax1.hist(predictions_df['x'], bins=50, color='steelblue', alpha=0.7, edgecolor='white')
    ax1.set_xlabel('X Position (yards)', fontsize=10)
    ax1.set_ylabel('Count', fontsize=10)
    ax1.set_title('X Position Distribution', fontsize=12)
    ax1.axvline(predictions_df['x'].mean(), color='red', linestyle='--', label=f"Mean: {predictions_df['x'].mean():.1f}")
    ax1.legend()

    # Y distribution
    ax2 = axes[1]
    ax2.hist(predictions_df['y'], bins=50, color='forestgreen', alpha=0.7, edgecolor='white')
    ax2.set_xlabel('Y Position (yards)', fontsize=10)
    ax2.set_ylabel('Count', fontsize=10)
    ax2.set_title('Y Position Distribution', fontsize=12)
    ax2.axvline(predictions_df['y'].mean(), color='red', linestyle='--', label=f"Mean: {predictions_df['y'].mean():.1f}")
    ax2.legend()

    # Heatmap
    ax3 = axes[2]
    heatmap, xedges, yedges = np.histogram2d(predictions_df['x'], predictions_df['y'], bins=30)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax3.imshow(heatmap.T, extent=extent, origin='lower', aspect='auto', cmap='YlOrRd')
    plt.colorbar(im, ax=ax3, label='Density')
    ax3.set_xlabel('X Position (yards)', fontsize=10)
    ax3.set_ylabel('Y Position (yards)', fontsize=10)
    ax3.set_title('Position Heatmap', fontsize=12)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved distribution plot to {save_path}")

    return fig, axes
