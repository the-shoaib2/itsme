#!/usr/bin/env python3
"""
Parquet Dataset Generation Script.
Builds data.parquet for CosyVoice training splits under data/cosyvoice/.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from itsme.features.parquet_builder import build_all_parquets


def main():
    parser = argparse.ArgumentParser(description="Build Parquet datasets for CosyVoice training.")
    parser.add_argument("--cosyvoice-dir", default="data/cosyvoice", help="CosyVoice dataset directory")
    args = parser.parse_args()

    res = build_all_parquets(cosyvoice_dir=args.cosyvoice_dir)
    print(f"Built Parquet datasets: {res}")

if __name__ == "__main__":
    main()
