#!/usr/bin/env python3
"""
Model Fine-Tuning Entry Point Script.
Usage: python scripts/train.py --config configs/training.yaml [--resume runs/itsme/latest]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from itsme.config.config import load_yaml
from itsme.training.trainer import CosyVoiceTrainer


def main():
    parser = argparse.ArgumentParser(description="Train/Fine-tune CosyVoice 3 model on personal voice dataset.")
    parser.add_argument("--config", default="configs/training.yaml", help="Path to training configuration YAML")
    parser.add_argument("--run-dir", default="runs/itsme", help="Run output directory")
    parser.add_argument("--resume", default=None, help="Path to checkpoint or run directory to resume from")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    trainer = CosyVoiceTrainer(config=cfg, run_dir=args.run_dir, resume_from=args.resume)
    
    try:
        run_path = trainer.train()
        print(f"Training completed successfully! Output stored in {run_path}")
    except KeyboardInterrupt:
        print("\nTraining interrupted gracefully by user. Latest checkpoint preserved.")
        sys.exit(0)
    except Exception as e:
        print(f"\nTraining Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
