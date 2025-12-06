#!/usr/bin/env python3
"""
CLI script to regenerate sequences.pkl without running notebooks.

Usage:
    python scripts/build_sequences.py
"""

import sys
from pathlib import Path

# Add project root and src to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import pandas as pd
import numpy as np
from nfl_bdb.preprocessing import (
    load_all_train_inputs,
    load_all_train_outputs,
    normalize_play_direction,
    add_features,
    build_sequences,
    save_sequences
)
import configs.default as config


def main():
    """Run the full preprocessing pipeline and save sequences."""
    print("=" * 60)
    print("NFL Big Data Bowl 2026 - Build Sequences Pipeline")
    print("=" * 60)
    print()
    
    # Step 1: Load data
    print("Step 1: Loading data...")
    print("-" * 60)
    input_df = load_all_train_inputs(config.DATA_PATH)
    output_df = load_all_train_outputs(config.DATA_PATH)
    print(f"✓ Loaded {len(input_df):,} input rows and {len(output_df):,} output rows")
    print()
    
    # Step 2: Normalize coordinates
    print("Step 2: Normalizing coordinates...")
    print("-" * 60)
    input_df = normalize_play_direction(input_df)
    if 'play_direction' in output_df.columns:
        output_df = normalize_play_direction(output_df)
    else:
        # Map play_direction from input to output
        play_dir_map = input_df.groupby(['game_id', 'play_id'])['play_direction'].first().to_dict()
        output_df['play_direction'] = output_df.apply(
            lambda row: play_dir_map.get((row['game_id'], row['play_id']), 'right'),
            axis=1
        )
        output_df = normalize_play_direction(output_df)
    print("✓ Normalized coordinates (offense → right)")
    print()
    
    # Step 3: Feature engineering
    print("Step 3: Adding features...")
    print("-" * 60)
    input_df = add_features(input_df)
    output_df = add_features(output_df)
    print(f"✓ Added features (input: {input_df.shape}, output: {output_df.shape})")
    print()
    
    # Step 4: Build sequences
    print("Step 4: Building sequences...")
    print("-" * 60)
    sequences = build_sequences(input_df, output_df)
    print(f"✓ Built {len(sequences):,} sequences")
    print()
    
    # Step 5: Save sequences
    print("Step 5: Saving sequences...")
    print("-" * 60)
    sequences_path = save_sequences(sequences)
    print()
    
    print("=" * 60)
    print("✓ Pipeline complete!")
    print("=" * 60)
    print(f"Sequences saved to: {sequences_path}")
    print(f"Ready for model training.")
    print()


if __name__ == "__main__":
    main()




