#!/usr/bin/env python3
"""
Checkpoint Comparison Script.
Generates and compares identical test sentences across multiple checkpoints.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from itsme.evaluation.comparer import compare_checkpoints


def main():
    parser = argparse.ArgumentParser(description="Compare audio output across multiple model checkpoints.")
    parser.add_argument("--checkpoints", nargs="+", required=True, help="List of checkpoint directories")
    parser.add_argument("--prompts-dir", default="evaluation/prompts", help="Directory containing test prompts")
    parser.add_argument("--output-dir", default="evaluation/comparison", help="Output comparison directory")
    args = parser.parse_args()

    results = compare_checkpoints(
        checkpoint_dirs=args.checkpoints,
        prompts_dir=args.prompts_dir,
        output_dir=args.output_dir
    )
    print(f"Compared {len(args.checkpoints)} checkpoints across evaluation prompts. Results in {args.output_dir}")

if __name__ == "__main__":
    main()
