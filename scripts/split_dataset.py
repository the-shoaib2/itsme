#!/usr/bin/env python3
"""
Dataset Splitting Script.
Splits verified transcripts into train, val, and test manifests in data/manifests/.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from itsme.dataset.splitter import split_dataset_files


def main():
    parser = argparse.ArgumentParser(description="Split verified dataset into train/val/test splits.")
    parser.add_argument("--input", default="data/transcripts/verified.jsonl", help="Input verified.jsonl path")
    parser.add_argument("--output-dir", default="data/manifests", help="Output manifests directory")
    parser.add_argument("--train-ratio", type=float, default=0.90, help="Train ratio (default 0.90)")
    parser.add_argument("--val-ratio", type=float, default=0.05, help="Validation ratio (default 0.05)")
    parser.add_argument("--test-ratio", type=float, default=0.05, help="Test ratio (default 0.05)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    args = parser.parse_args()

    splits = split_dataset_files(
        verified_jsonl=args.input,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    print(f"Dataset split complete: train={len(splits['train'])}, val={len(splits['validation'])}, test={len(splits['test'])}")

if __name__ == "__main__":
    main()
