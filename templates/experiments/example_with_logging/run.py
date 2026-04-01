#!/usr/bin/env python
"""Example experiment script demonstrating logging usage.

This script shows how to use the logging_config utility in your experiments.
Logging output goes to both console (stdout) and experiment.log file.

The log file ensures all output is captured even if the job times out.
"""

import sys
import time
from pathlib import Path

# Add parent directory to path to import experiment module
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiment.logging_config import setup_logging


def main():
    """Run example experiment with logging."""
    # Setup logging - logs to both console and experiment.log
    logger = setup_logging("experiment.log")

    logger.info("=" * 60)
    logger.info("Starting example experiment")
    logger.info("=" * 60)

    # Log experiment parameters (would come from command-line in real use)
    logger.info("Parameters: learning_rate=0.01, batch_size=32")

    # Simulate some training iterations
    try:
        for iteration in range(10):
            logger.debug(f"Iteration {iteration}: Processing batch")

            # Simulate work
            time.sleep(0.1)

            if iteration % 3 == 0:
                logger.info(f"Iteration {iteration}: Loss = {0.5 - iteration * 0.04:.4f}")

            if iteration == 7:
                logger.warning(f"Iteration {iteration}: Loss plateauing - consider adjusting learning rate")

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        return 1

    logger.info("=" * 60)
    logger.info("Training complete")
    logger.info("Results saved to results.json")
    logger.info("=" * 60)

    # Simulate saving results
    import json
    with open("results.json", "w") as f:
        json.dump({"final_loss": 0.14, "iterations": 10}, f)

    return 0


if __name__ == "__main__":
    sys.exit(main())
