"""Logging configuration utility for experiment scripts."""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    log_file: Optional[str] = "experiment.log",
    level: int = logging.INFO,
    name: Optional[str] = None,
) -> logging.Logger:
    """Configure logging for experiment scripts.

    Sets up both file and console output with consistent formatting.
    Output is written to disk immediately (no buffering).

    Args:
        log_file: Path to log file. If None, only log to console.
        level: Logging level (logging.DEBUG, INFO, WARNING, ERROR, CRITICAL)
        name: Logger name (default: __name__ of caller)

    Returns:
        Configured logger instance

    Example:
        from experiment.logging_config import setup_logging

        logger = setup_logging("my_experiment.log")
        logger.info("Starting experiment")
        logger.warning("Something unexpected")
        logger.error("Error occurred")
    """
    # Get or create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear any existing handlers (avoid duplicates)
    logger.handlers.clear()

    # Formatter with timestamp
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # File handler (if log_file specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def get_logger(
    name: Optional[str] = None,
    log_file: Optional[str] = "experiment.log",
    level: int = logging.INFO,
) -> logging.Logger:
    """Get a logger with automatic setup.

    Simpler alternative to setup_logging() for quick use.

    Args:
        name: Logger name (default: caller's module name)
        log_file: Path to log file
        level: Logging level

    Returns:
        Logger instance

    Example:
        from experiment.logging_config import get_logger

        logger = get_logger()
        logger.info("Message")
    """
    logger = logging.getLogger(name)

    # Only setup if this logger has no handlers yet
    if not logger.handlers:
        return setup_logging(log_file=log_file, level=level, name=name)

    return logger
