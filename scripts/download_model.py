#!/usr/bin/env python3
"""
Model Download Script for CosyVoice 3 Foundation Model.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from itsme.training.model_downloader import download_cosyvoice_model


def main():
    parser = argparse.ArgumentParser(description="Download CosyVoice 3 foundation model.")
    parser.add_argument("--model-name", default="FunAudioLLM/Fun-CosyVoice3-0.5B-2512", help="Model name or repository ID")
    parser.add_argument("--model-dir", default="models/base", help="Output model directory")
    parser.add_argument("--force", action="store_true", help="Force redownload")
    args = parser.parse_args()

    path = download_cosyvoice_model(
        model_name=args.model_name,
        models_dir=args.model_dir,
        force_redownload=args.force
    )
    print(f"Model downloaded and verified at {path}")

if __name__ == "__main__":
    main()
