# Data Contracts

This document specifies the data formats and contracts used throughout the NFL Big Data Bowl trajectory prediction pipeline.

## Node Features (13 features)

Both training and inference use exactly **13 node features** in the following order:

| Index | Feature Name | Type | Description |
|-------|-------------|------|-------------|
| 0 | `x` | float | Player x-coordinate on field (yards) |
| 1 | `y` | float | Player y-coordinate on field (yards) |
| 2 | `vx` | float | Velocity in x-direction (yards/sec) |
| 3 | `vy` | float | Velocity in y-direction (yards/sec) |
| 4 | `s` | float | Speed (yards/sec) |
| 5 | `a` | float | Acceleration (yards/sec²) |
| 6 | `dist_ball` | float | Distance to ball (yards) |
| 7 | `dx_ball` | float | x-offset from ball (yards) |
| 8 | `dy_ball` | float | y-offset from ball (yards) |
| 9 | `player_role_Targeted Receiver` | float | One-hot: is targeted receiver |
| 10 | `player_role_Defensive Coverage` | float | One-hot: is defensive coverage |
| 11 | `player_side_Offense` | float | One-hot: is on offense |
| 12 | `player_side_Defense` | float | One-hot: is on defense |

**Important**: This count is defined in `config.NODE_DIM = 13`.

## Edge Attributes (4 features)

Edge attributes encode the relationship between two players:

| Index | Feature Name | Type | Description |
|-------|-------------|------|-------------|
| 0 | `distance` | float | Euclidean distance between players (yards) |
| 1 | `angle` | float | Angle from source to target player (radians) |
| 2 | `rel_vx` | float | Relative velocity in x (target - source) |
| 3 | `rel_vy` | float | Relative velocity in y (target - source) |

## Graph Structure

- **Nodes**: Up to `MAX_PLAYERS = 22` players per play
- **Edges**: Fully connected within `RADIUS = 20.0` yards, plus self-loops
- **Batching**: Graphs are batched by concatenating node features and offsetting edge indices

## Sequence Format (sequences.pkl)

The preprocessed sequences are stored as a dictionary:

```python
{
    (game_id, play_id, nfl_id): {
        'input': pd.Series,  # Last observed frame for target player
        'target': np.ndarray  # [T, 2] trajectory positions
    },
    ...
}
```

## Dataset Output Shapes

### Training Dataset (NFLTrajectoryDataset)

```python
{
    "node_feats": Tensor[max_players, 13],       # Node features
    "edge_index": Tensor[2, E],                  # Edge indices
    "edge_attr": Tensor[E, 4],                   # Edge attributes
    "initial_pos": Tensor[max_players, T, 2],   # Initial positions
    "target": Tensor[max_players, T-1, 2],      # Target positions
    "mask": Tensor[max_players, T-1],           # Validity mask
    "key": Tuple[int, int, int],                # (game_id, play_id, nfl_id)
    "trajectory_length": int                    # Actual trajectory length
}
```

### Batched Training Data

```python
{
    "node_feats": Tensor[B*N, 13],              # Concatenated nodes
    "edge_index": Tensor[2, total_edges],       # Offset edge indices
    "edge_attr": Tensor[total_edges, 4],        # Concatenated edges
    "batch_index": Tensor[B*N],                 # Batch assignment
    "initial_pos": Tensor[B, N, T, 2],          # Stacked positions
    "target": Tensor[B, N, T-1, 2],             # Stacked targets
    "mask": Tensor[B, N, T-1],                  # Stacked masks
    "keys": List[Tuple],                        # Batch keys
    "trajectory_lengths": Tensor[B]             # Lengths
}
```

Where:
- `B` = batch size
- `N` = max_players (22)
- `T` = max_trajectory_length (100)

### Inference Dataset (InferenceDataset)

```python
{
    "node_feats": Tensor[max_players, 13],      # Node features
    "edge_index": Tensor[2, E],                 # Edge indices
    "edge_attr": Tensor[E, 4],                  # Edge attributes
    "initial_pos": Tensor[T, 2],                # Initial position sequence
    "mask": Tensor[T-1],                        # Validity mask (all ones)
    "key": Tuple[int, int, int],                # (game_id, play_id, nfl_id)
}
```

## Model I/O

### Training Mode (initial_pos=None)

- **Input**: GraphFeatures with batched nodes and edges
- **Output**: `Tensor[B, max_len, 2]` - Predicted velocities

### Inference Mode (initial_pos provided)

- **Input**: GraphFeatures + initial_pos `Tensor[B, 2]`
- **Output**: `Tensor[B, max_len, 2]` - Absolute positions (integrated velocities)

## Important Constraints

1. **Feature Consistency**: Training and inference MUST use identical feature extraction
2. **initial_pos Contract**: Training uses `target[:, 0:1, :]`, inference uses input x,y
3. **NaN/Inf Handling**: All NaN/Inf values are replaced with 0.0
4. **Padding**: Shorter sequences are padded to `MAX_TRAJECTORY_LENGTH`

## Configuration Reference

Key config values in `configs/default.py`:

```python
MAX_PLAYERS = 22
NODE_DIM = 13
EDGE_DIM = 4
RADIUS = 20.0
MAX_TRAJECTORY_LENGTH = 100
```
