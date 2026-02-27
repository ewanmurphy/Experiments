"""Job state management and metadata tracking."""

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any


@dataclass
class ClusterMetadata:
    """Cluster-specific metadata for a job."""
    host: str
    slurm_job_id: str
    num_experiments: int
    user: Optional[str] = None  # None when using SSH config alias
    partition: Optional[str] = None  # None uses cluster default


@dataclass
class RunMetadata:
    """Metadata for a submitted cluster run."""
    run_id: str
    experiment_name: str
    timestamp: str
    local_dir: str
    remote_dir: str
    cluster: ClusterMetadata
    status: str  # submitted, running, completed, collected, failed, cancelled
    submitted_at: str
    collected_at: Optional[str] = None
    config_file: Optional[str] = None
    script: Optional[str] = None


class RunStateManager:
    """Manages run metadata and state."""

    def __init__(self, metadata_dir: Optional[str] = None):
        """Initialize state manager.

        Args:
            metadata_dir: Directory to store metadata files (default: .experiment_runs)
        """
        if metadata_dir is None:
            self.metadata_dir = Path(".experiment_runs")
        else:
            self.metadata_dir = Path(metadata_dir)

        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def save_run(self, metadata: RunMetadata) -> None:
        """Save run metadata to file.

        Args:
            metadata: RunMetadata object to save
        """
        file_path = self.metadata_dir / f"{metadata.run_id}.json"
        with open(file_path, "w") as f:
            json.dump(asdict(metadata), f, indent=2)

    def load_run(self, run_id: str) -> RunMetadata:
        """Load run metadata from file.

        Args:
            run_id: Run identifier

        Returns:
            RunMetadata object

        Raises:
            FileNotFoundError: If run metadata doesn't exist
        """
        file_path = self.metadata_dir / f"{run_id}.json"
        if not file_path.exists():
            raise FileNotFoundError(f"Run metadata not found: {run_id}")

        with open(file_path) as f:
            data = json.load(f)

        # Reconstruct nested dataclass
        cluster_data = data.pop("cluster")
        cluster = ClusterMetadata(**cluster_data)

        return RunMetadata(cluster=cluster, **data)

    def update_status(self, run_id: str, status: str) -> None:
        """Update run status.

        Args:
            run_id: Run identifier
            status: New status (submitted, running, completed, collected, failed, cancelled)
        """
        metadata = self.load_run(run_id)
        metadata.status = status

        if status == "collected":
            metadata.collected_at = datetime.now().isoformat()

        self.save_run(metadata)

    def list_runs(
        self,
        status_filter: Optional[str] = None,
        include_collected: bool = False
    ) -> List[RunMetadata]:
        """List all runs.

        Args:
            status_filter: Filter by status (e.g., "submitted", "running", "completed")
            include_collected: Include already collected runs

        Returns:
            List of RunMetadata objects
        """
        runs = []

        for file_path in sorted(self.metadata_dir.glob("*.json")):
            try:
                metadata = self.load_run(file_path.stem)
                if status_filter and metadata.status != status_filter:
                    continue
                if metadata.status == "collected" and not include_collected:
                    continue
                runs.append(metadata)
            except Exception:
                # Skip corrupted metadata files
                continue

        return runs

    def delete_run_metadata(self, run_id: str) -> None:
        """Delete run metadata file.

        Args:
            run_id: Run identifier
        """
        file_path = self.metadata_dir / f"{run_id}.json"
        if file_path.exists():
            file_path.unlink()

    def get_run_metadata_path(self, run_id: str) -> str:
        """Get path to run metadata file.

        Args:
            run_id: Run identifier

        Returns:
            Absolute path to metadata file
        """
        return str(self.metadata_dir / f"{run_id}.json")
