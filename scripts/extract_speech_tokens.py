#!/usr/bin/env python3
"""
Speech Token Extraction Script.
Extracts utt2speech_token.pt using CosyVoice speech tokenizer.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from itsme.features.speech_tokens import extract_speech_tokens


def main():
    parser = argparse.ArgumentParser(description="Extract discrete speech tokens.")
    parser.add_argument("--cosyvoice-dir", default="data/cosyvoice", help="CosyVoice directory")
    parser.add_argument("--model-dir", default="models/base", help="Model directory")
    parser.add_argument("--device", default="auto", help="Device (cuda, mps, cpu, auto)")
    parser.add_argument("--force", action="store_true", help="Force recomputation")
    args = parser.parse_args()

    res = extract_speech_tokens(
        cosyvoice_dir=args.cosyvoice_dir,
        model_dir=args.model_dir,
        device=args.device,
        force_recompute=args.force
    )
    print(f"Extracted speech tokens: {res}")

if __name__ == "__main__":
    main()
