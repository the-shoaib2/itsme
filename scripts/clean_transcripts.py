#!/usr/bin/env python3
"""
Transcript Cleaning Script.
Cleans raw.jsonl into data/transcripts/clean.jsonl.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from itsme.transcription.cleaner import clean_transcripts_file


def main():
    parser = argparse.ArgumentParser(description="Clean raw Whisper transcripts.")
    parser.add_argument("--input", default="data/transcripts/raw.jsonl", help="Input raw.jsonl path")
    parser.add_argument("--output", default="data/transcripts/clean.jsonl", help="Output clean.jsonl path")
    args = parser.parse_args()

    recs = clean_transcripts_file(input_jsonl=args.input, output_jsonl=args.output)
    print(f"Cleaned {len(recs)} transcripts -> {args.output}")

if __name__ == "__main__":
    main()
