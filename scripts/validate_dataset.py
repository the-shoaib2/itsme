#!/usr/bin/env python3
"""
Dataset Validation Script.
Checks missing files, leakage, audio integrity, embeddings, and tokens.
Exits with non-zero code if validation fails.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from itsme.dataset.validator import validate_dataset
from itsme.utils.exceptions import DatasetValidationError


def main():
    parser = argparse.ArgumentParser(description="Validate complete dataset integrity.")
    parser.add_argument("--manifests-dir", default="data/manifests", help="Path to manifests directory")
    parser.add_argument("--cosyvoice-dir", default="data/cosyvoice", help="Path to cosyvoice directory")
    args = parser.parse_args()

    try:
        summary = validate_dataset(manifests_dir=args.manifests_dir, cosyvoice_dir=args.cosyvoice_dir)
        print("Dataset Validation PASSED!")
        sys.exit(0)
    except DatasetValidationError as e:
        print(f"\nDataset Validation FAILED: {e}")
        sys.exit(1)
    except Exception as ex:
        print(f"\nUnexpected error during dataset validation: {ex}")
        sys.exit(1)

if __name__ == "__main__":
    main()
