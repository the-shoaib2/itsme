#!/usr/bin/env python3
"""
Human Transcript Review CLI Script.
Interactive terminal review producing data/transcripts/verified.jsonl.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from itsme.transcription.reviewer import review_transcripts_cli


def main():
    parser = argparse.ArgumentParser(description="Human transcript review tool.")
    parser.add_argument("--input", default="data/transcripts/clean.jsonl", help="Input clean.jsonl path")
    parser.add_argument("--output", default="data/transcripts/verified.jsonl", help="Output verified.jsonl path")
    parser.add_argument("--auto-accept", action="store_true", help="Auto-accept all clean transcripts")
    args = parser.parse_args()

    recs = review_transcripts_cli(
        input_jsonl=args.input,
        output_jsonl=args.output,
        auto_accept=args.auto_accept
    )
    print(f"Verified {len(recs)} transcripts -> {args.output}")

if __name__ == "__main__":
    main()
