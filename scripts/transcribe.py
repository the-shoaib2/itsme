#!/usr/bin/env python3
"""
Whisper Transcription Script.
Transcribes audio segments into data/transcripts/raw.jsonl.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from itsme.transcription.whisper import transcribe_segments


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio segments with Whisper.")
    parser.add_argument("--segments-dir", default="data/segments", help="Path to speech segments directory")
    parser.add_argument("--output", default="data/transcripts/raw.jsonl", help="Output raw.jsonl path")
    parser.add_argument("--model", default="large-v3", help="Whisper model size/name")
    parser.add_argument("--language", default="en", help="Language code")
    parser.add_argument("--device", default="auto", help="Device (cuda, mps, cpu, auto)")
    args = parser.parse_args()

    recs = transcribe_segments(
        segments_dir=args.segments_dir,
        output_jsonl=args.output,
        model_name=args.model,
        language=args.language,
        device=args.device
    )
    print(f"Transcribed {len(recs)} segments -> {args.output}")

if __name__ == "__main__":
    main()
