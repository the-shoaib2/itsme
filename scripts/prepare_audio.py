#!/usr/bin/env python3
"""
Audio Preprocessing Script.
Converts raw files to 24kHz mono WAV in data/processed/.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from itsme.audio.preprocessor import prepare_all_audio


def main():
    parser = argparse.ArgumentParser(description="Preprocess raw audio files.")
    parser.add_argument("--raw-dir", default="data/raw", help="Path to raw audio directory")
    parser.add_argument("--output-dir", default="data/processed", help="Path to processed audio directory")
    parser.add_argument("--sample-rate", type=int, default=24000, help="Target sample rate")
    args = parser.parse_args()

    recs = prepare_all_audio(raw_dir=args.raw_dir, output_dir=args.output_dir, target_sr=args.sample_rate)
    print(f"Preprocessed {len(recs)} audio files into {args.output_dir}")

if __name__ == "__main__":
    main()
