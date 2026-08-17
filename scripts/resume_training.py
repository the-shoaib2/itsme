#!/usr/bin/env python3
"""
Resume Training Script.
Resumes fine-tuning run from latest checkpoint in runs/itsme/latest or specified path.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from itsme.config.config import load_yaml
from itsme.training.trainer import CosyVoiceTrainer


def main():
    parser = argparse.ArgumentParser(description="Resume training from latest checkpoint.")
    parser.add_argument("--config", default="configs/training.yaml", help="Path to training configuration YAML")
    parser.add_argument("--run-dir", default="runs/itsme", help="Run directory")
    parser.add_argument("--checkpoint", default="runs/itsme/latest", help="Checkpoint or run directory to resume")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    trainer = CosyVoiceTrainer(config=cfg, run_dir=args.run_dir, resume_from=args.checkpoint)
    
    try:
        run_path = trainer.train()
        print(f"Resumed training completed successfully! Output in {run_path}")
    except KeyboardInterrupt:
        print("\nResume training interrupted gracefully by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nResume Training Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
