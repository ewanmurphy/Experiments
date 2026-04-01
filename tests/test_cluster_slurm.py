"""Tests for SLURM script generation and job management."""

import pytest
from experiment.cluster.slurm import (
    generate_slurm_script,
    JobStatus,
)


def test_generate_slurm_script_basic():
    """Test basic SLURM script generation."""
    script = generate_slurm_script(
        script_path="train.py",
        num_experiments=10,
        partition="standard",
        account=None,
        time_limit="01:00:00",
        memory="4G",
        cpus=1,
        gpus=0,
        max_concurrent=0,
        modules=[],
        environment=None,
        remote_run_dir="/home/user/experiments/test/2026_Jan_15_10h30m00s",
        experiment_name="test_exp",
        timestamp="2026_Jan_15_10h30m00s"
    )

    # Verify script contains required SLURM directives
    assert "#!/bin/bash" in script
    assert "#SBATCH --job-name=test_exp_2026_Jan_15_10h30m00s" in script
    assert "#SBATCH --array=1-10" in script
    assert "#SBATCH --partition=standard" in script
    assert "#SBATCH --time=01:00:00" in script
    assert "#SBATCH --mem=4G" in script
    assert "#SBATCH --cpus-per-task=1" in script
    assert "#SBATCH --gres=gpu:0" not in script  # GPU line omitted when gpus=0

    # Verify script contains experiment execution logic
    assert "cd /home/user/experiments/test/2026_Jan_15_10h30m00s" in script
    assert "exp_$EXP_NUM" in script
    assert "config_generated.csv" in script
    assert "export PYTHONUNBUFFERED=1" in script
    assert "python -u ../train.py" in script


def test_generate_slurm_script_with_gpus():
    """Test SLURM script generation with GPU allocation."""
    script = generate_slurm_script(
        script_path="train_gpu.py",
        num_experiments=5,
        partition="gpu",
        account="gpu-project",
        time_limit="04:00:00",
        memory="16G",
        cpus=4,
        gpus=1,
        max_concurrent=2,
        modules=["cuda/11.8", "python/3.9"],
        environment="source ~/venv/bin/activate",
        remote_run_dir="/home/user/exp",
        experiment_name="gpu_exp",
        timestamp="2026_Jan_15_10h30m00s"
    )

    assert "#SBATCH --partition=gpu" in script
    assert "#SBATCH --account=gpu-project" in script
    assert "#SBATCH --gres=gpu:1" in script
    assert "#SBATCH --cpus-per-task=4" in script
    assert "#SBATCH --array=1-5%2" in script  # Max concurrent limit
    assert "module load cuda/11.8" in script
    assert "module load python/3.9" in script
    assert "source ~/venv/bin/activate" in script


def test_generate_slurm_script_no_max_concurrent():
    """Test SLURM script without max concurrent limit."""
    script = generate_slurm_script(
        script_path="script.py",
        num_experiments=100,
        partition="standard",
        account=None,
        time_limit="02:00:00",
        memory="8G",
        cpus=2,
        gpus=0,
        max_concurrent=0,  # No limit
        modules=[],
        environment=None,
        remote_run_dir="/home/user/exp",
        experiment_name="large_exp",
        timestamp="2026_Jan_15_10h30m00s"
    )

    # Should have unbounded array job
    assert "#SBATCH --array=1-100" in script
    assert "#SBATCH --array=1-100%" not in script  # No % limit


def test_generate_slurm_script_parameter_parsing():
    """Test that SLURM script includes parameter parsing logic."""
    script = generate_slurm_script(
        script_path="experiment.py",
        num_experiments=10,
        partition="standard",
        account=None,
        time_limit="01:00:00",
        memory="4G",
        cpus=1,
        gpus=0,
        max_concurrent=0,
        modules=[],
        environment=None,
        remote_run_dir="/home/user/exp",
        experiment_name="test",
        timestamp="2026_Jan_15_10h30m00s"
    )

    # Verify AWK parsing is included
    assert "awk" in script
    assert "config_generated.csv" in script
    assert "$SLURM_ARRAY_TASK_ID" in script
    assert "${PARAMS[@]}" in script


def test_generate_slurm_script_optional_partition():
    """Test SLURM script generation without specifying partition (uses cluster default)."""
    script = generate_slurm_script(
        script_path="experiment.py",
        num_experiments=5,
        partition=None,  # No partition specified, will use cluster default
        account=None,
        time_limit="01:00:00",
        memory="4G",
        cpus=1,
        gpus=0,
        max_concurrent=0,
        modules=[],
        environment=None,
        remote_run_dir="/home/user/exp",
        experiment_name="test",
        timestamp="2026_Jan_15_10h30m00s"
    )

    # Verify partition directive is NOT included when partition is None
    assert "#SBATCH --partition=" not in script
    # But other SBATCH directives should still be there
    assert "#SBATCH --array=1-5" in script
    assert "#SBATCH --time=01:00:00" in script
    assert "#SBATCH --mem=4G" in script


def test_generate_slurm_script_optional_account():
    """Test SLURM script generation without specifying account (uses user default)."""
    script = generate_slurm_script(
        script_path="experiment.py",
        num_experiments=5,
        partition="standard",
        account=None,  # No account specified, will use user's default
        time_limit="01:00:00",
        memory="4G",
        cpus=1,
        gpus=0,
        max_concurrent=0,
        modules=[],
        environment=None,
        remote_run_dir="/home/user/exp",
        experiment_name="test",
        timestamp="2026_Jan_15_10h30m00s"
    )

    # Verify account directive is NOT included when account is None
    assert "#SBATCH --account=" not in script
    # But other SBATCH directives should still be there
    assert "#SBATCH --partition=standard" in script
    assert "#SBATCH --array=1-5" in script
    assert "#SBATCH --time=01:00:00" in script


def test_job_status_dataclass():
    """Test JobStatus dataclass."""
    status = JobStatus(
        job_id="12345",
        state="RUNNING",
        completed_tasks=5,
        total_tasks=10,
        elapsed_time="00:05:30",
        time_limit="01:00:00"
    )

    assert status.job_id == "12345"
    assert status.state == "RUNNING"
    assert status.completed_tasks == 5
    assert status.total_tasks == 10


def test_job_status_completed():
    """Test JobStatus for completed job."""
    status = JobStatus(
        job_id="12345",
        state="COMPLETED",
        completed_tasks=10,
        total_tasks=10
    )

    assert status.state == "COMPLETED"
    assert status.completed_tasks == status.total_tasks


def test_job_status_failed():
    """Test JobStatus for failed job."""
    status = JobStatus(
        job_id="12345",
        state="FAILED",
        completed_tasks=5,
        total_tasks=10
    )

    assert status.state == "FAILED"
    assert status.completed_tasks < status.total_tasks
