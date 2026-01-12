"""Experiment runner for executing Python scripts with parameters."""

import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import typer
from experiment.logger import ExperimentLogger


def run_experiment(
    script_path: str,
    params: Dict[str, Any],
    experiment_dir: Optional[str] = None,
    output_to_console: bool = True,
    verbose: bool = False,
) -> int:
    """Run a Python experiment script with parameters.

    Args:
        script_path: Path to the Python script to run
        params: Dictionary of parameters to pass to the script
        experiment_dir: Directory for this experiment. Logs will go into
                       experiment_dir/logs, and script will run from experiment_dir
        output_to_console: Whether to stream output to console
        verbose: Whether to show detailed logging and metadata information

    Returns:
        Exit code from the experiment script
    """
    script_file = Path(script_path)
    if not script_file.exists():
        typer.echo(f"Error: Script not found: {script_path}", err=True)
        return 1

    # Get absolute path to script
    script_file = script_file.resolve()

    # Set up experiment directory and log directory
    if experiment_dir:
        exp_dir = Path(experiment_dir)
        exp_dir.mkdir(parents=True, exist_ok=True)
        log_dir = exp_dir / "logs"
        working_dir = exp_dir
    else:
        log_dir = Path("logs")
        working_dir = Path.cwd()

    # Initialize logger
    logger = ExperimentLogger(log_dir=str(log_dir), script_name=script_file.name)
    logger.record_params(params)

    if verbose:
        typer.echo(f"Starting experiment: {script_file.name}")
        if experiment_dir:
            typer.echo(f"Experiment directory: {exp_dir}")
        typer.echo(f"Log file: {logger.get_log_path()}")
        typer.echo(f"Metadata file: {logger.get_metadata_path()}")
        typer.echo("-" * 60)

    # Convert parameters to command-line arguments
    cmd = [sys.executable, str(script_file)]
    for key, value in params.items():
        cmd.append(f"--{key}")
        cmd.append(str(value))

    try:
        # Run script as subprocess from experiment directory
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(working_dir),
        )

        # Stream output in real-time
        for line in process.stdout:
            line = line.rstrip()
            if output_to_console:
                typer.echo(line)
            logger.logger.info(line)

        exit_code = process.wait()

    except Exception as e:
        typer.echo(f"Error running experiment: {e}", err=True)
        logger.logger.error(f"Exception: {e}")
        exit_code = 1

    # Finalize logging
    logger.finalize(exit_code)

    if verbose:
        typer.echo("-" * 60)
        typer.echo(f"Experiment completed with exit code: {exit_code}")

    return exit_code
