#!/usr/bin/env python3
"""Small demo experiment used to showcase the CLI workflow."""

import argparse
import json
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    args = parser.parse_args()

    print("Starting demo experiment")
    print(f"learning_rate={args.learning_rate}")
    print(f"batch_size={args.batch_size}")
    print(f"epochs={args.epochs}")

    # Short sleep so users can see progress output without making the demo slow.
    time.sleep(0.15)

    lr_penalty = abs(args.learning_rate - 0.005) * 40
    batch_penalty = (args.batch_size - 16) / 40
    score = max(0.0, round(0.95 - lr_penalty - batch_penalty, 4))
    loss = round(1.0 - score / 1.2, 4)

    results = {
        "score": score,
        "loss": loss,
        "trained_epochs": args.epochs,
    }

    with open(Path("results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("Wrote results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
