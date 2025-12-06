#!/usr/bin/env python3
"""
Inference CLI entrypoint.

Usage:
    python scripts/infer.py [--checkpoint PATH] [--sequences PATH] [--output PATH] [--batch-size SIZE] [--device DEVICE]
"""

import sys
from pathlib import Path

# Add project root and src to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from nfl_bdb.cli import infer_cli
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference for NFL trajectory prediction")
    parser.add_argument('--checkpoint', type=str, help='Model checkpoint path')
    parser.add_argument('--sequences', type=str, help='Test sequences pickle path')
    parser.add_argument('--output', type=str, help='Output CSV path')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--device', type=str, help='Device (cuda/cpu)')
    args = parser.parse_args()
    
    infer_cli(args)

