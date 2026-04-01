"""Tests for cluster config discovery (find_cluster_config and find_cluster_configs functions)."""

import pytest
import tempfile
from pathlib import Path
import yaml

from experiment.cluster import find_cluster_config, find_cluster_configs


def test_find_cluster_config_per_experiment():
    """Test finding per-experiment cluster config."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create experiment directory with cluster.yaml
        exp_dir = tmpdir / "experiments" / "my_exp"
        exp_dir.mkdir(parents=True)
        cluster_config = exp_dir / "cluster.yaml"
        with open(cluster_config, "w") as f:
            yaml.dump({"ssh": {"host": "test", "user": "user", "remote_base_dir": "/home"}}, f)

        # Change to temp directory
        import os
        orig_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)

            # Find should return per-experiment config
            found = find_cluster_config("my_exp")
            assert found == str(cluster_config)
        finally:
            os.chdir(orig_cwd)


def test_find_cluster_config_project_root():
    """Test finding project root cluster config."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create project root cluster.yaml
        root_config = tmpdir / "cluster.yaml"
        with open(root_config, "w") as f:
            yaml.dump({"ssh": {"host": "test", "user": "user", "remote_base_dir": "/home"}}, f)

        # Create experiments directory (but no per-experiment config)
        exp_dir = tmpdir / "experiments" / "my_exp"
        exp_dir.mkdir(parents=True)

        import os
        orig_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)

            # Find should return project root config
            found = find_cluster_config("my_exp")
            assert found == str(root_config)
        finally:
            os.chdir(orig_cwd)


def test_find_cluster_config_explicit_override():
    """Test explicit config path override."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create custom config file
        custom_config = tmpdir / "my_custom_cluster.yaml"
        with open(custom_config, "w") as f:
            yaml.dump({"ssh": {"host": "test", "user": "user", "remote_base_dir": "/home"}}, f)

        # Find with explicit path
        found = find_cluster_config("any_exp", str(custom_config))
        assert found == str(custom_config)


def test_find_cluster_config_search_order():
    """Test that per-experiment config takes precedence over project root."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create both per-experiment and project root configs
        root_config = tmpdir / "cluster.yaml"
        with open(root_config, "w") as f:
            yaml.dump({"ssh": {"host": "root_cluster", "user": "user", "remote_base_dir": "/home"}}, f)

        exp_dir = tmpdir / "experiments" / "my_exp"
        exp_dir.mkdir(parents=True)
        exp_config = exp_dir / "cluster.yaml"
        with open(exp_config, "w") as f:
            yaml.dump({"ssh": {"host": "exp_cluster", "user": "user", "remote_base_dir": "/home"}}, f)

        import os
        orig_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)

            # Should return per-experiment config (takes precedence)
            found = find_cluster_config("my_exp")
            assert found == str(exp_config)
        finally:
            os.chdir(orig_cwd)


def test_find_cluster_config_not_found():
    """Test that FileNotFoundError is raised when config not found."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create experiments directory but no config files
        exp_dir = tmpdir / "experiments" / "my_exp"
        exp_dir.mkdir(parents=True)

        import os
        orig_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)

            # Should raise FileNotFoundError
            with pytest.raises(FileNotFoundError) as exc_info:
                find_cluster_config("my_exp")

            # Check error message contains helpful info
            error_msg = str(exc_info.value)
            assert "experiments/my_exp/cluster.yaml" in error_msg
            assert "cluster.yaml" in error_msg
        finally:
            os.chdir(orig_cwd)


def test_find_cluster_config_explicit_not_found():
    """Test that explicit config path that doesn't exist raises error."""
    with pytest.raises(FileNotFoundError) as exc_info:
        find_cluster_config("any_exp", "/nonexistent/path/cluster.yaml")

    assert "/nonexistent/path/cluster.yaml" in str(exc_info.value)


# Tests for find_cluster_configs (multi-config discovery)

def test_find_cluster_configs_single_default():
    """Test finding single default cluster config."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create single default cluster.yaml
        exp_dir = tmpdir / "experiments" / "my_exp"
        exp_dir.mkdir(parents=True)
        cluster_config = exp_dir / "cluster.yaml"
        with open(cluster_config, "w") as f:
            yaml.dump({"ssh": {"host": "test", "user": "user", "remote_base_dir": "/home"}}, f)

        import os
        orig_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            configs = find_cluster_configs("my_exp")
            assert len(configs) == 1
            assert configs[0] == ("default", str(cluster_config))
        finally:
            os.chdir(orig_cwd)


def test_find_cluster_configs_multiple_named():
    """Test finding multiple named cluster configs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create multiple configs in per-experiment directory
        exp_dir = tmpdir / "experiments" / "my_exp"
        exp_dir.mkdir(parents=True)

        # Create default and named configs
        default_config = exp_dir / "cluster.yaml"
        with open(default_config, "w") as f:
            yaml.dump({"ssh": {"host": "test", "user": "user", "remote_base_dir": "/home"}}, f)

        gpu_config = exp_dir / "cluster_gpu.yaml"
        with open(gpu_config, "w") as f:
            yaml.dump({"ssh": {"host": "test", "user": "user", "remote_base_dir": "/home"}}, f)

        hpc_config = exp_dir / "cluster_hpc.yaml"
        with open(hpc_config, "w") as f:
            yaml.dump({"ssh": {"host": "test", "user": "user", "remote_base_dir": "/home"}}, f)

        import os
        orig_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            configs = find_cluster_configs("my_exp")
            assert len(configs) == 3

            # Check that display names are correct
            display_names = [name for name, _ in configs]
            assert "default" in display_names
            assert "gpu" in display_names
            assert "hpc" in display_names

            # Check order: default comes first, then alphabetical by name
            assert configs[0][0] == "default"
            assert configs[1][0] == "gpu"
            assert configs[2][0] == "hpc"
        finally:
            os.chdir(orig_cwd)


def test_find_cluster_configs_per_experiment_takes_precedence():
    """Test that per-experiment configs take precedence over root configs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create root config
        root_config = tmpdir / "cluster.yaml"
        with open(root_config, "w") as f:
            yaml.dump({"ssh": {"host": "root_host", "user": "user", "remote_base_dir": "/home"}}, f)

        root_gpu = tmpdir / "cluster_gpu.yaml"
        with open(root_gpu, "w") as f:
            yaml.dump({"ssh": {"host": "root_host", "user": "user", "remote_base_dir": "/home"}}, f)

        # Create per-experiment config (same name, should take precedence)
        exp_dir = tmpdir / "experiments" / "my_exp"
        exp_dir.mkdir(parents=True)
        exp_config = exp_dir / "cluster.yaml"
        with open(exp_config, "w") as f:
            yaml.dump({"ssh": {"host": "exp_host", "user": "user", "remote_base_dir": "/home"}}, f)

        import os
        orig_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            configs = find_cluster_configs("my_exp")

            # Should find 2 configs: per-experiment default and root gpu
            assert len(configs) == 2
            assert configs[0] == ("default", str(exp_config))
            assert configs[1] == ("gpu", str(root_gpu))
        finally:
            os.chdir(orig_cwd)


def test_find_cluster_configs_explicit_path():
    """Test finding configs with explicit path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create custom config file
        custom_config = tmpdir / "my_custom_cluster.yaml"
        with open(custom_config, "w") as f:
            yaml.dump({"ssh": {"host": "test", "user": "user", "remote_base_dir": "/home"}}, f)

        # Find with explicit path
        configs = find_cluster_configs("any_exp", str(custom_config))
        assert len(configs) == 1
        assert configs[0] == ("my_custom_cluster.yaml", str(custom_config))


def test_find_cluster_configs_explicit_not_found():
    """Test that explicit config path that doesn't exist raises error."""
    with pytest.raises(FileNotFoundError) as exc_info:
        find_cluster_configs("any_exp", "/nonexistent/path/cluster.yaml")

    assert "/nonexistent/path/cluster.yaml" in str(exc_info.value)


def test_find_cluster_configs_empty():
    """Test that empty list is returned when no configs found."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create experiments directory but no config files
        exp_dir = tmpdir / "experiments" / "my_exp"
        exp_dir.mkdir(parents=True)

        import os
        orig_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            configs = find_cluster_configs("my_exp")
            assert configs == []
        finally:
            os.chdir(orig_cwd)


# Tests for find_cluster_config with multiple configs

def test_find_cluster_config_multiple_raises_error():
    """Test that find_cluster_config raises error when multiple configs found."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create multiple configs
        exp_dir = tmpdir / "experiments" / "my_exp"
        exp_dir.mkdir(parents=True)

        default_config = exp_dir / "cluster.yaml"
        with open(default_config, "w") as f:
            yaml.dump({"ssh": {"host": "test", "user": "user", "remote_base_dir": "/home"}}, f)

        gpu_config = exp_dir / "cluster_gpu.yaml"
        with open(gpu_config, "w") as f:
            yaml.dump({"ssh": {"host": "test", "user": "user", "remote_base_dir": "/home"}}, f)

        import os
        orig_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)

            # Should raise FileNotFoundError with list of available configs
            with pytest.raises(FileNotFoundError) as exc_info:
                find_cluster_config("my_exp")

            error_msg = str(exc_info.value)
            assert "Multiple cluster configs found" in error_msg
            assert "default" in error_msg
            assert "gpu" in error_msg
            assert "--cluster-config" in error_msg
        finally:
            os.chdir(orig_cwd)
