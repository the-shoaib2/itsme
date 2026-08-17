#!/usr/bin/env python3
"""
Speaker Embedding Extraction Script.
Extracts utt2embedding.pt and spk2embedding.pt.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from itsme.features.embeddings import extract_speaker_embeddings


def main():
    parser = argparse.ArgumentParser(description="Extract speaker embeddings.")
    parser.add_argument("--cosyvoice-dir", default="data/cosyvoice", help="CosyVoice directory")
    parser.add_argument("--model-dir", default="models/base", help="Model directory")
    parser.add_argument("--device", default="auto", help="Device (cuda, mps, cpu, auto)")
    parser.add_argument("--force", action="store_true", help="Force recomputation")
    args = parser.parse_args()

    res = extract_speaker_embeddings(
        cosyvoice_dir=args.cosyvoice_dir,
        model_dir=args.model_dir,
        device=args.device,
        force_recompute=args.force
    )
    print(f"Extracted speaker embeddings: {res}")

if __name__ == "__main__":
    main()
