"""Experiment logging and metadata tracking."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class ExperimentLogger:
    """Logs experiment execution with metadata tracking."""

    def __init__(self, log_dir: str = "logs", script_name: Optional[str] = None):
        """Initialize the experiment logger.

        Args:
            log_dir: Directory to store log files
            script_name: Name of the script being run (used for log filename)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y_%b_%d_%Hh%Mm%Ss")
        script_base = Path(script_name).stem if script_name else "experiment"
        log_filename = f"{script_base}_{timestamp}.log"
        self.log_path = self.log_dir / log_filename

        metadata_filename = f"{script_base}_{timestamp}.json"
        self.metadata_path = self.log_dir / metadata_filename

        # Configure logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        handler = logging.FileHandler(self.log_path)
        handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        # Initialize metadata
        self.metadata: Dict[str, Any] = {
            "start_time": datetime.now().isoformat(),
            "script": script_name,
        }

    def record_params(self, params: Dict[str, Any]) -> None:
        """Record experiment parameters."""
        self.metadata["parameters"] = params
        self.logger.info(f"Parameters: {json.dumps(params, indent=2)}")

    def finalize(self, exit_code: int) -> None:
        """Finalize logging with exit code and save metadata."""
        self.metadata["end_time"] = datetime.now().isoformat()
        self.metadata["exit_code"] = exit_code
        self.metadata["log_file"] = str(self.log_path)

        with open(self.metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

        self.logger.info(f"Experiment completed with exit code: {exit_code}")

    def get_log_path(self) -> Path:
        """Get the path to the log file."""
        return self.log_path

    def get_metadata_path(self) -> Path:
        """Get the path to the metadata file."""
        return self.metadata_path
