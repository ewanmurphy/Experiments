"""Integration tests for cluster operations with mocked SSH."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json

from experiment.cluster import (
    load_cluster_config,
    RunStateManager,
    generate_slurm_script,
)


@pytest.fixture
def mock_cluster_config(tmp_path):
    """Create a test cluster configuration."""
    import yaml

    config_file = tmp_path / "cluster.yaml"
    config_data = {
        "ssh": {
            "host": "test.cluster.edu",
            "user": "testuser",
            "remote_base_dir": "/home/testuser/experiments"
        },
        "slurm": {
            "partition": "standard",
            "time_limit": "01:00:00",
            "memory": "4G",
            "cpus": 1,
            "gpus": 0,
            "max_concurrent": 0
        },
        "modules": ["python/3.9"],
        "sync": {
            "to_cluster": ["*.py", "*.yaml"],
            "from_cluster": ["exp_*/results.json", "exp_*/logs/**"]
        }
    }

    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    return config_file


@pytest.fixture
def state_manager(tmp_path):
    """Create a RunStateManager with temporary directory."""
    return RunStateManager(str(tmp_path / ".experiment_runs"))


def test_load_cluster_config_full(mock_cluster_config):
    """Test loading full cluster config with all fields."""
    config = load_cluster_config(str(mock_cluster_config))

    assert config.ssh.host == "test.cluster.edu"
    assert config.ssh.user == "testuser"
    assert config.ssh.remote_base_dir == "/home/testuser/experiments"
    assert config.slurm.partition == "standard"
    assert config.slurm.time_limit == "01:00:00"
    assert config.slurm.cpus == 1
    assert "python/3.9" in config.modules
    assert len(config.sync.to_cluster) > 0


def test_save_and_retrieve_run_metadata(state_manager):
    """Test complete save and retrieve workflow."""
    from experiment.cluster import RunMetadata, ClusterMetadata
    from datetime import datetime

    # Create metadata
    metadata = RunMetadata(
        run_id="integration_test_run",
        experiment_name="test_exp",
        timestamp="2026_Jan_15_10h30m00s",
        local_dir="/local/path",
        remote_dir="/remote/path",
        cluster=ClusterMetadata(
            host="test.cluster.edu",
            user="testuser",
            slurm_job_id="99999",
            partition="standard",
            num_experiments=20
        ),
        status="submitted",
        submitted_at=datetime.now().isoformat(),
        config_file="config.yaml",
        script="train.py"
    )

    # Save
    state_manager.save_run(metadata)

    # Retrieve
    retrieved = state_manager.load_run("integration_test_run")

    # Verify all fields
    assert retrieved.run_id == "integration_test_run"
    assert retrieved.experiment_name == "test_exp"
    assert retrieved.cluster.slurm_job_id == "99999"
    assert retrieved.cluster.num_experiments == 20
    assert retrieved.status == "submitted"


def test_update_status_workflow(state_manager):
    """Test status update workflow."""
    from experiment.cluster import RunMetadata, ClusterMetadata
    from datetime import datetime

    # Create and save initial metadata
    metadata = RunMetadata(
        run_id="status_test",
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

    # Update through lifecycle
    state_manager.update_status("status_test", "running")
    run = state_manager.load_run("status_test")
    assert run.status == "running"

    state_manager.update_status("status_test", "completed")
    run = state_manager.load_run("status_test")
    assert run.status == "completed"

    state_manager.update_status("status_test", "collected")
    run = state_manager.load_run("status_test")
    assert run.status == "collected"
    assert run.collected_at is not None


def test_slurm_script_generation_with_config(mock_cluster_config):
    """Test SLURM script generation using loaded config."""
    config = load_cluster_config(str(mock_cluster_config))

    script = generate_slurm_script(
        script_path="experiment.py",
        num_experiments=15,
        partition=config.slurm.partition,
        account=config.slurm.account,
        time_limit=config.slurm.time_limit,
        memory=config.slurm.memory,
        cpus=config.slurm.cpus,
        gpus=config.slurm.gpus,
        max_concurrent=config.slurm.max_concurrent,
        modules=config.modules,
        environment=config.environment,
        remote_run_dir=f"{config.ssh.remote_base_dir}/test/2026_Jan_15_10h30m00s",
        experiment_name="test_exp",
        timestamp="2026_Jan_15_10h30m00s"
    )

    # Verify critical components
    assert "#SBATCH --array=1-15" in script
    assert "#SBATCH --partition=standard" in script
    assert "module load python/3.9" in script
    assert "export PYTHONUNBUFFERED=1" in script
    assert "python -u ../experiment.py" in script


def test_multiple_runs_in_state_manager(state_manager):
    """Test managing multiple runs."""
    from experiment.cluster import RunMetadata, ClusterMetadata
    from datetime import datetime

    # Create 3 runs
    for i in range(3):
        metadata = RunMetadata(
            run_id=f"run_{i}",
            experiment_name=f"exp_{i}",
            timestamp=f"2026_Jan_15_{i:02d}h00m00s",
            local_dir=f"/local/exp_{i}",
            remote_dir=f"/remote/exp_{i}",
            cluster=ClusterMetadata(
                host="host",
                user="user",
                slurm_job_id=str(100 + i),
                partition="std",
                num_experiments=5 + i
            ),
            status="submitted",
            submitted_at=datetime.now().isoformat()
        )
        state_manager.save_run(metadata)

    # List all
    all_runs = state_manager.list_runs()
    assert len(all_runs) == 3

    # List by status
    submitted_runs = state_manager.list_runs(status_filter="submitted")
    assert len(submitted_runs) == 3

    # Update one to completed
    state_manager.update_status("run_0", "completed")
    completed_runs = state_manager.list_runs(status_filter="completed")
    assert len(completed_runs) == 1
    assert completed_runs[0].run_id == "run_0"


@patch('experiment.cluster.ssh.subprocess.run')
def test_ssh_connection_success(mock_subprocess):
    """Test successful SSH connection."""
    from experiment.cluster.ssh import test_ssh_connection

    # Mock successful connection
    mock_subprocess.return_value = MagicMock(returncode=0)

    result = test_ssh_connection("test.host", "user")
    assert result is True
    mock_subprocess.assert_called_once()


@patch('experiment.cluster.ssh.subprocess.run')
def test_ssh_connection_failure(mock_subprocess):
    """Test failed SSH connection."""
    from experiment.cluster.ssh import test_ssh_connection

    # Mock failed connection
    mock_subprocess.return_value = MagicMock(returncode=1)

    result = test_ssh_connection("bad.host", "user")
    assert result is False


@patch('experiment.cluster.ssh.subprocess.run')
def test_execute_remote_command_success(mock_subprocess):
    """Test successful remote command execution."""
    from experiment.cluster.ssh import execute_remote_command

    mock_subprocess.return_value = MagicMock(
        returncode=0,
        stdout="command output"
    )

    return_code, output = execute_remote_command("host", "user", "ls -la")
    assert return_code == 0
    assert output == "command output"


@patch('experiment.cluster.ssh.subprocess.run')
def test_execute_remote_command_failure(mock_subprocess):
    """Test failed remote command execution."""
    from experiment.cluster.ssh import execute_remote_command

    mock_subprocess.return_value = MagicMock(
        returncode=127,
        stdout="command not found"
    )

    return_code, output = execute_remote_command("host", "user", "bad_cmd")
    assert return_code == 127


def test_ssh_error_exception():
    """Test SSHError exception."""
    from experiment.cluster.ssh import SSHError

    with pytest.raises(SSHError):
        raise SSHError("Test error message")
