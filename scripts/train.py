#!/usr/bin/env python3
"""
Training CLI entrypoint.

Usage:
    python scripts/train.py [--batch-size BATCH_SIZE] [--learning-rate LR] [--epochs EPOCHS] [--device DEVICE]
"""

import sys
from pathlib import Path

# Add project root and src to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from nfl_bdb.cli import train_cli
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NFL trajectory prediction model")
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--learning-rate', type=float, help='Learning rate')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--device', type=str, help='Device (cuda/cpu)')
    args = parser.parse_args()
    
    train_cli(args)

