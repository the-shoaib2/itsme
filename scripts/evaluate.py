#!/usr/bin/env python3
"""
Model Quality Evaluation Script.
Evaluates similarity, naturalness, prosody, stability, and produces JSON, MD, and HTML reports.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from itsme.evaluation.evaluator import run_evaluation_suite


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned voice model quality.")
    parser.add_argument("--generated-dir", default="evaluation/generated", help="Directory containing generated audio samples")
    parser.add_argument("--reports-dir", default="evaluation/reports", help="Output directory for reports")
    args = parser.parse_args()

    report = run_evaluation_suite(generated_dir=args.generated_dir, reports_dir=args.reports_dir)
    print(f"Evaluation complete. Reports generated in {args.reports_dir}")

if __name__ == "__main__":
    main()
