#!/usr/bin/env python3
"""
System Environment and Hardware Check Script for ItsMe.
"""

import sys
from pathlib import Path

# Add src to python path if executing directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from itsme.utils.hardware import get_system_status, print_system_check_report


def main():
    status = get_system_status()
    print_system_check_report(status)
    
    # Return exit code 0 if critical requirements pass
    if status["python"]["status"] == "OK" and status["pytorch"]["status"] == "OK":
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
