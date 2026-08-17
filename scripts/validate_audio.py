#!/usr/bin/env python3
"""
Audio Validation Script.
Inspects raw audio files and generates data/reports/audio_quality.json.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from itsme.audio.validator import validate_audio_dir


def main():
    parser = argparse.ArgumentParser(description="Validate raw audio quality.")
    parser.add_argument("--raw-dir", default="data/raw", help="Path to raw audio directory")
    parser.add_argument("--output-report", default="data/reports/audio_quality.json", help="Path to output report JSON")
    args = parser.parse_args()

    reports = validate_audio_dir(raw_dir=args.raw_dir, output_report_path=args.output_report)
    print(f"Validated {len(reports)} files. Report saved to {args.output_report}")

if __name__ == "__main__":
    main()
