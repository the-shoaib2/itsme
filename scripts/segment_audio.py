#!/usr/bin/env python3
"""
VAD Segmentation and Quality Filtering Script.
Segments audio from data/processed/ to data/segments/ and filters quality.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from itsme.audio.filter import filter_and_report_segments
from itsme.audio.vad import segment_all_audio


def main():
    parser = argparse.ArgumentParser(description="Segment and quality filter audio.")
    parser.add_argument("--processed-dir", default="data/processed", help="Path to processed audio directory")
    parser.add_argument("--segments-dir", default="data/segments", help="Path to segments output directory")
    parser.add_argument("--min-duration", type=float, default=2.0, help="Minimum utterance duration (s)")
    parser.add_argument("--max-duration", type=float, default=15.0, help="Maximum utterance duration (s)")
    args = parser.parse_args()

    recs = segment_all_audio(
        processed_dir=args.processed_dir,
        output_dir=args.segments_dir,
        min_duration_s=args.min_duration,
        max_duration_s=args.max_duration
    )
    report = filter_and_report_segments(
        segments_dir=args.segments_dir,
        min_duration_s=args.min_duration,
        max_duration_s=args.max_duration
    )
    print(f"Segmented and filtered {report['accepted']}/{report['total_files']} valid segments into {args.segments_dir}")

if __name__ == "__main__":
    main()
