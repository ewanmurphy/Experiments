#!/usr/bin/env python3
"""Post-process the bundled demo experiment summary."""

import csv
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python postprocess.py /path/to/summary.csv", file=sys.stderr)
        return 1

    summary_path = Path(sys.argv[1])
    if not summary_path.exists():
        print(f"Summary file not found: {summary_path}", file=sys.stderr)
        return 1

    with open(summary_path, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("No rows found in summary.csv", file=sys.stderr)
        return 1

    best_row = max(rows, key=lambda row: float(row["score"]))
    avg_score = sum(float(row["score"]) for row in rows) / len(rows)

    report_path = summary_path.parent / "demo_report.txt"
    with open(report_path, "w") as f:
        f.write("Demo Experiment Report\n")
        f.write("======================\n\n")
        f.write(f"Runs: {len(rows)}\n")
        f.write(f"Average score: {avg_score:.4f}\n")
        f.write("Best run:\n")
        f.write(f"  learning_rate={best_row['learning_rate']}\n")
        f.write(f"  batch_size={best_row['batch_size']}\n")
        f.write(f"  epochs={best_row['epochs']}\n")
        f.write(f"  score={best_row['score']}\n")
        f.write(f"  loss={best_row['loss']}\n")

    print(f"Processed {len(rows)} runs")
    print(f"Average score: {avg_score:.4f}")
    print(f"Best score: {best_row['score']}")
    print(f"Report saved to: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
