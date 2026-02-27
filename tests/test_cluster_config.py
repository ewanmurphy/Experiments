"""Tests for cluster configuration module."""

import pytest
import tempfile
from pathlib import Path
import yaml
from experiment.cluster import (
    load_cluster_config,
    merge_experiment_cluster_config,
    ClusterConfig,
)


def test_load_cluster_config_valid():
    """Test loading valid cluster config."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "cluster.yaml"
        config_data = {
            "ssh": {
                "host": "cluster.example.edu",
                "user": "testuser",
                "remote_base_dir": "/home/testuser/experiments"
            },
            "slurm": {
                "partition": "gpu",
                "time_limit": "04:00:00",
                "memory": "16G",
                "cpus": 4,
                "gpus": 1
            }
        }
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        config = load_cluster_config(str(config_path))

        assert config.ssh.host == "cluster.example.edu"
        assert config.ssh.user == "testuser"
        assert config.slurm.partition == "gpu"
        assert config.slurm.gpus == 1


def test_load_cluster_config_ssh_alias():
    """Test loading cluster config with SSH alias (no explicit user)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "cluster.yaml"
        config_data = {
            "ssh": {
                "host": "my-cluster",  # SSH alias from ~/.ssh/config
                "remote_base_dir": "/scratch/myusername/experiments"
                # Note: user is omitted, will be resolved from SSH config
            },
            "slurm": {
                "time_limit": "02:00:00",
                "memory": "8G",
                "cpus": 2
            }
        }
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        config = load_cluster_config(str(config_path))

        assert config.ssh.host == "my-cluster"
        assert config.ssh.user is None  # User comes from SSH config
        assert config.ssh.remote_base_dir == "/scratch/myusername/experiments"
        assert config.slurm.partition is None  # Uses cluster default


def test_load_cluster_config_missing_required_fields():
    """Test loading config with missing required fields."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "cluster.yaml"
        config_data = {
            "ssh": {
                "host": "cluster.example.edu"
                # Missing remote_base_dir
            }
        }
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        with pytest.raises(ValueError):
            load_cluster_config(str(config_path))


def test_load_cluster_config_file_not_found():
    """Test loading config file that doesn't exist."""
    with pytest.raises(FileNotFoundError):
        load_cluster_config("/nonexistent/path/cluster.yaml")


def test_merge_experiment_cluster_config():
    """Test merging experiment overrides with base config."""
    base_config = ClusterConfig(
        ssh=pytest.importorskip("experiment.cluster").SSHConfig(
            host="cluster.example.edu",
            user="testuser",
            remote_base_dir="/home/testuser/experiments"
        ),
        modules=["python/3.9"]
    )

    override = {
        "partition": "gpu",
        "time_limit": "08:00:00",
        "modules": ["cuda/11.8"],
        "sync": {
            "to_cluster": ["models/**"]
        }
    }

    merged = merge_experiment_cluster_config(base_config, override)

    assert merged.slurm.partition == "gpu"
    assert merged.slurm.time_limit == "08:00:00"
    assert "python/3.9" in merged.modules
    assert "cuda/11.8" in merged.modules
    assert "models/**" in merged.sync.to_cluster


def test_merge_with_no_override():
    """Test merge with None override."""
    base_config = ClusterConfig(
        ssh=pytest.importorskip("experiment.cluster").SSHConfig(
            host="cluster.example.edu",
            user="testuser",
            remote_base_dir="/home/testuser/experiments"
        )
    )

    merged = merge_experiment_cluster_config(base_config, None)

    assert merged.ssh.host == base_config.ssh.host
    assert merged.slurm.partition == base_config.slurm.partition
