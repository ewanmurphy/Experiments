#!/usr/bin/env python3
"""Example post-processing script for experiment results.

This script is called by the experiment runner with the summary.csv path as the first argument.
The script runs from the run directory (where summary.csv is located).

Usage:
    python postprocess_example.py /path/to/summary.csv
"""

import sys
import csv
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("Error: summary.csv path required as argument", file=sys.stderr)
        return 1

    summary_csv = Path(sys.argv[1])

    if not summary_csv.exists():
        print(f"Error: {summary_csv} not found", file=sys.stderr)
        return 1

    print(f"Post-processing summary: {summary_csv}")
    print("-" * 60)

    # Read the summary CSV
    with open(summary_csv, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Found {len(rows)} experiment results")
    print()

    # Example: Calculate statistics and write to output file
    if rows:
        # Print all rows
        fieldnames = reader.fieldnames
        print("Results:")
        for row in rows:
            print(f"  {row}")

        # Example: Write a summary report
        report_file = summary_csv.parent / "postprocess_report.txt"
        with open(report_file, 'w') as f:
            f.write("Post-Processing Report\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Total experiments: {len(rows)}\n")
            f.write(f"Results file: {summary_csv}\n\n")
            f.write("All results:\n")
            for row in rows:
                f.write(f"  {row}\n")

        print(f"\nReport saved to: {report_file}")

    return 0

if __name__ == '__main__':
    sys.exit(main())
