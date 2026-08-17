#!/usr/bin/env python3
"""
CosyVoice Kaldi-style Metadata Preparation Script.
Generates wav.scp, text, utt2spk, spk2utt under data/cosyvoice/.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from itsme.dataset.cosyvoice_prep import prepare_all_cosyvoice_metadata


def main():
    parser = argparse.ArgumentParser(description="Prepare official CosyVoice Kaldi metadata files.")
    parser.add_argument("--manifests-dir", default="data/manifests", help="Path to manifests directory")
    parser.add_argument("--cosyvoice-dir", default="data/cosyvoice", help="Path to cosyvoice dataset output directory")
    parser.add_argument("--speaker-id", default="itsme", help="Speaker ID")
    args = parser.parse_args()

    res = prepare_all_cosyvoice_metadata(
        manifests_dir=args.manifests_dir,
        cosyvoice_dir=args.cosyvoice_dir,
        speaker_id=args.speaker_id
    )
    print(f"Prepared CosyVoice Kaldi metadata in {args.cosyvoice_dir}")

if __name__ == "__main__":
    main()
