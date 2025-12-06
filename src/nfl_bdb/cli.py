"""
Command-line interface for NFL Big Data Bowl 2026 project.

Provides entrypoints for training, inference, evaluation, and submission generation.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import configs.default as config
from nfl_bdb.training import train_main
from nfl_bdb.inference import run_inference
from nfl_bdb.evaluation import run_eval


def validate_positive(value, name):
    """Validate that a value is positive."""
    if value is not None and value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def train_cli(args):
    """CLI entrypoint for training."""
    # Validate arguments
    validate_positive(args.batch_size, "batch_size")
    validate_positive(args.learning_rate, "learning_rate")
    validate_positive(args.epochs, "epochs")

    # Get effective values (use args if provided, else config defaults)
    batch_size = args.batch_size or config.BATCH_SIZE
    learning_rate = args.learning_rate or config.LEARNING_RATE
    epochs = args.epochs or config.EPOCHS
    device = args.device or config.DEVICE

    # Override config with validated values (create local copies instead of mutating)
    # Note: We still mutate config here for compatibility, but log the effective values
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.learning_rate:
        config.LEARNING_RATE = args.learning_rate
        config.LR = args.learning_rate  # Also update alias
    if args.epochs:
        config.EPOCHS = args.epochs
    if args.device:
        config.DEVICE = args.device

    print("=" * 60)
    print("NFL Big Data Bowl 2026 - Training")
    print("=" * 60)
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {epochs}")
    print(f"Device: {device}")
    print("=" * 60)
    print()

    train_main()


def infer_cli(args):
    """CLI entrypoint for inference."""
    ckpt_path = args.checkpoint or str(config.MODEL_DIR / "best_model.pt")
    seq_path = args.sequences or str(config.PROCESSED_DIR / "test_sequences.pkl")
    output_path = args.output or "submission.csv"
    batch_size = args.batch_size or config.BATCH_SIZE
    device = args.device or ("cuda" if config.DEVICE.startswith("cuda") else "cpu")
    
    print("=" * 60)
    print("NFL Big Data Bowl 2026 - Inference")
    print("=" * 60)
    print(f"Checkpoint: {ckpt_path}")
    print(f"Sequences: {seq_path}")
    print(f"Output: {output_path}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")
    print("=" * 60)
    print()
    
    run_inference(
        ckpt_path=ckpt_path,
        seq_path=seq_path,
        submission_path=output_path,
        batch_size=batch_size,
        device=device,
        all_input_df=None
    )


def eval_cli(args):
    """CLI entrypoint for evaluation."""
    ckpt_path = args.checkpoint or str(config.MODEL_DIR / "best_model.pt")
    seq_path = args.sequences or str(config.PROCESSED_DIR / "sequences.pkl")
    batch_size = args.batch_size or config.BATCH_SIZE
    device = args.device or ("cuda" if config.DEVICE.startswith("cuda") else "cpu")
    
    print("=" * 60)
    print("NFL Big Data Bowl 2026 - Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {ckpt_path}")
    print(f"Sequences: {seq_path}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")
    print("=" * 60)
    print()
    
    run_eval(
        ckpt_path=ckpt_path,
        seq_path=seq_path,
        batch_size=batch_size,
        device=device
    )


def submit_cli(args):
    """CLI entrypoint for Kaggle submission generation with actual model predictions."""
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    import numpy as np
    import torch
    from nfl_bdb.models import STGNNRefine
    from nfl_bdb.utils import build_graph, GraphFeatures

    input_path = args.input or "/kaggle/input/nfl-big-data-bowl-2026-prediction/test_input.csv"
    output_path = args.output or "/kaggle/working/submission.parquet"
    checkpoint_path = getattr(args, 'checkpoint', None) or str(config.MODEL_DIR / "best_model.pt")

    # Fallback for local testing
    if not Path(input_path).exists():
        input_path = str(config.TEST_SAMPLE_DIR / "test_input.csv")

    print("=" * 60)
    print("NFL Big Data Bowl 2026 - Submission Generation")
    print("=" * 60)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print("=" * 60)
    print()

    # Check if checkpoint exists
    if not Path(checkpoint_path).exists():
        print(f"[WARNING] Checkpoint not found at {checkpoint_path}")
        print("[WARNING] Generating DUMMY predictions (all zeros) - NOT suitable for submission!")
        use_model = False
    else:
        use_model = True

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Using device: {device}")

    # Load model if checkpoint exists
    if use_model:
        print(f"[+] Loading model from {checkpoint_path}...")
        model = STGNNRefine(
            node_dim=config.NODE_DIM,
            hidden=config.HIDDEN_DIM,
            graph_layers=config.GRAPH_LAYERS,
            temporal_layers=config.TEMPORAL_LAYERS,
            heads=config.HEADS,
            max_len=config.MAX_TRAJECTORY_LENGTH,
            dropout=config.DROPOUT
        )
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print("[✓] Model loaded successfully")

    # Load and filter data
    print("[+] Loading test_input.csv...")
    df = pd.read_csv(input_path)
    print(f"[✓] Loaded {len(df):,} total rows")

    # Filter to player_to_predict == True
    print("\n[+] Filtering to player_to_predict == True...")
    if df['player_to_predict'].dtype == 'object':
        submission_df = df[df['player_to_predict'].astype(str).str.lower() == 'true'].copy()
    else:
        submission_df = df[df['player_to_predict'] == True].copy()
    print(f"[✓] Filtered to {len(submission_df):,} prediction targets")

    # Generate predictions
    predictions = []
    print("\n[+] Generating predictions...")

    # Group by (game_id, play_id, nfl_id) for batch processing
    grouped = submission_df.groupby(['game_id', 'play_id', 'nfl_id'])

    for (game_id, play_id, nfl_id), group in grouped:
        # Get the last frame for this player (input to model)
        last_frame = group.iloc[-1]

        if use_model:
            # Build graph for this player
            player_dict = last_frame.to_dict()

            # Get all players in this play for graph construction
            play_mask = (df['game_id'] == game_id) & (df['play_id'] == play_id)
            play_data = df[play_mask]
            all_players = []
            for pid in play_data['nfl_id'].unique():
                player_rows = play_data[play_data['nfl_id'] == pid]
                if len(player_rows) > 0:
                    all_players.append(player_rows.iloc[-1].to_dict())

            # Build graph
            graph = build_graph(
                player_dict,
                all_players[:config.MAX_PLAYERS],
                max_distance=config.RADIUS,
                include_self_loops=True
            )

            # Move to device
            graph = graph.to(device)

            # Get initial position
            initial_pos = torch.tensor([[player_dict.get('x', 0.0), player_dict.get('y', 0.0)]],
                                       dtype=torch.float32, device=device)

            # Run inference
            with torch.no_grad():
                pred_positions = model(graph, initial_pos)  # [1, max_len, 2]
                pred_positions = pred_positions[0, :21, :].cpu().numpy()  # [21, 2]

            # Extract predictions for each step
            for step in range(21):
                predictions.append({
                    'game_id': int(game_id),
                    'play_id': int(play_id),
                    'nfl_id': int(nfl_id),
                    'frame_id': int(last_frame['frame_id']) + step + 1,
                    'x': float(pred_positions[step, 0]),
                    'y': float(pred_positions[step, 1]),
                })
        else:
            # Dummy predictions (fallback when no model)
            for step in range(21):
                predictions.append({
                    'game_id': int(game_id),
                    'play_id': int(play_id),
                    'nfl_id': int(nfl_id),
                    'frame_id': int(last_frame['frame_id']) + step + 1,
                    'x': 0.0,
                    'y': 0.0,
                })

    print(f"[✓] Generated {len(predictions):,} predictions")

    # Create submission DataFrame
    pred_df = pd.DataFrame(predictions)

    # Construct row_id
    print("\n[+] Constructing row_id...")
    row_id = (
        pred_df['game_id'].astype(str) + '_' +
        pred_df['play_id'].astype(str) + '_' +
        pred_df['nfl_id'].astype(str) + '_' +
        pred_df['frame_id'].astype(str)
    ).astype(str).astype(object)

    # Create final submission
    submission = pd.DataFrame({
        'row_id': row_id,
        'x': pred_df['x'].astype('float64'),
        'y': pred_df['y'].astype('float64')
    })[['row_id', 'x', 'y']]

    # Define schema
    schema = pa.schema([
        pa.field('row_id', pa.string()),
        pa.field('x', pa.float64()),
        pa.field('y', pa.float64()),
    ])

    # Write parquet
    print("\n[+] Writing parquet file...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(submission, schema=schema)
    pq.write_table(
        table,
        output_path,
        compression='snappy',
        use_dictionary=False,
        write_statistics=False
    )

    print(f"[✓] Parquet file written to: {output_path}")
    print(f"[✓] Total rows: {len(submission):,}")
    if not use_model:
        print("\n[WARNING] Predictions are DUMMY values (all zeros) - train a model first!")


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="NFL Big Data Bowl 2026 - Trajectory Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--batch-size', type=int, help='Batch size')
    train_parser.add_argument('--learning-rate', type=float, help='Learning rate')
    train_parser.add_argument('--epochs', type=int, help='Number of epochs')
    train_parser.add_argument('--device', type=str, help='Device (cuda/cpu)')
    
    # Inference command
    infer_parser = subparsers.add_parser('infer', help='Run inference')
    infer_parser.add_argument('--checkpoint', type=str, help='Model checkpoint path')
    infer_parser.add_argument('--sequences', type=str, help='Test sequences pickle path')
    infer_parser.add_argument('--output', type=str, help='Output CSV path')
    infer_parser.add_argument('--batch-size', type=int, help='Batch size')
    infer_parser.add_argument('--device', type=str, help='Device (cuda/cpu)')
    
    # Evaluation command
    eval_parser = subparsers.add_parser('eval', help='Evaluate the model')
    eval_parser.add_argument('--checkpoint', type=str, help='Model checkpoint path')
    eval_parser.add_argument('--sequences', type=str, help='Sequences pickle path')
    eval_parser.add_argument('--batch-size', type=int, help='Batch size')
    eval_parser.add_argument('--device', type=str, help='Device (cuda/cpu)')
    
    # Submission command
    submit_parser = subparsers.add_parser('submit', help='Generate Kaggle submission')
    submit_parser.add_argument('--input', type=str, help='Input CSV path')
    submit_parser.add_argument('--output', type=str, help='Output parquet path')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    if args.command == 'train':
        train_cli(args)
    elif args.command == 'infer':
        infer_cli(args)
    elif args.command == 'eval':
        eval_cli(args)
    elif args.command == 'submit':
        submit_cli(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

