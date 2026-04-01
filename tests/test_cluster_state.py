"""Tests for cluster state management module."""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime
from experiment.cluster import (
    RunStateManager,
    RunMetadata,
    ClusterMetadata,
)


def test_save_and_load_run_metadata():
    """Test saving and loading run metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        state_manager = RunStateManager(tmpdir)

        metadata = RunMetadata(
            run_id="test_exp_2026_Jan_15_10h30m00s",
            experiment_name="test_exp",
            timestamp="2026_Jan_15_10h30m00s",
            local_dir="/path/to/local",
            remote_dir="/home/user/experiments/test_exp/2026_Jan_15_10h30m00s",
            cluster=ClusterMetadata(
                host="cluster.edu",
                user="testuser",
                slurm_job_id="12345",
                partition="standard",
                num_experiments=10
            ),
            status="submitted",
            submitted_at=datetime.now().isoformat(),
            config_file="config.yaml",
            script="train.py"
        )

        # Save
        state_manager.save_run(metadata)

        # Load
        loaded = state_manager.load_run("test_exp_2026_Jan_15_10h30m00s")

        assert loaded.run_id == metadata.run_id
        assert loaded.experiment_name == "test_exp"
        assert loaded.cluster.slurm_job_id == "12345"
        assert loaded.status == "submitted"


def test_update_status():
    """Test updating run status."""
    with tempfile.TemporaryDirectory() as tmpdir:
        state_manager = RunStateManager(tmpdir)

        metadata = RunMetadata(
            run_id="test_run",
            experiment_name="test",
            timestamp="2026_Jan_15_10h30m00s",
            local_dir="/local",
            remote_dir="/remote",
            cluster=ClusterMetadata(
                host="host",
                user="user",
                slurm_job_id="123",
                partition="std",
                num_experiments=5
            ),
            status="submitted",
            submitted_at=datetime.now().isoformat()
        )

        state_manager.save_run(metadata)
        state_manager.update_status("test_run", "completed")

        loaded = state_manager.load_run("test_run")
        assert loaded.status == "completed"


def test_list_runs():
    """Test listing runs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        state_manager = RunStateManager(tmpdir)

        # Create multiple runs
        for i in range(3):
            metadata = RunMetadata(
                run_id=f"test_run_{i}",
                experiment_name=f"exp_{i}",
                timestamp=f"2026_Jan_15_{i:02d}h00m00s",
                local_dir="/local",
                remote_dir="/remote",
                cluster=ClusterMetadata(
                    host="host",
                    user="user",
                    slurm_job_id=str(100 + i),
                    partition="std",
                    num_experiments=5
                ),
                status="submitted",
                submitted_at=datetime.now().isoformat()
            )
            state_manager.save_run(metadata)

        runs = state_manager.list_runs()
        assert len(runs) == 3


def test_load_nonexistent_run():
    """Test loading run that doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        state_manager = RunStateManager(tmpdir)

        with pytest.raises(FileNotFoundError):
            state_manager.load_run("nonexistent_run")
