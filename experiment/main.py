"""Main CLI entry point."""

import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
from typing import Any, List, Optional, Tuple
import typer
from experiment import __version__
from experiment.config import (
    generate_csv_from_yaml,
    load_csv,
    merge_params,
    yaml_to_csv,
)
from experiment.runner import run_experiment

app = typer.Typer()


def get_available_experiments(experiments_dir: Path) -> List[str]:
    """Get list of available experiments from experiments/ directory.

    Args:
        experiments_dir: Path to experiments directory

    Returns:
        List of experiment names (subdirectory names with config.yaml)
    """
    if not experiments_dir.exists():
        return []

    experiments = []
    for item in experiments_dir.iterdir():
        if item.is_dir():
            config_file = item / "config.yaml"
            if config_file.exists():
                experiments.append(item.name)

    return sorted(experiments)


def create_results_summary(run_dir: Path, experiments: List[dict], verbose: bool = False) -> None:
    """Create a summary CSV combining parameters and results.

    Args:
        run_dir: Path to the run directory
        experiments: List of parameter dictionaries for each experiment
        verbose: Whether to show detailed output
    """
    summary = []

    for i, params in enumerate(experiments, 1):
        exp_subdir = run_dir / f"exp_{i:03d}"
        results_file = exp_subdir / "results.json"

        row = dict(params)  # Start with parameters

        # Add results if they exist
        if results_file.exists():
            try:
                with open(results_file) as f:
                    results = json.load(f)
                    row.update(results)
            except (json.JSONDecodeError, IOError) as e:
                typer.echo(f"Warning: Could not read {results_file}: {e}", err=True)

        summary.append(row)

    # Write summary CSV
    if summary:
        summary_file = run_dir / "summary.csv"
        fieldnames = list(summary[0].keys())

        with open(summary_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary)

        if verbose:
            typer.echo(f"Summary saved to: {summary_file}")


def run_post_processing(script_path: str, summary_csv_path: Path, run_dir: Path, verbose: bool = False) -> None:
    """Run post-processing script with summary.csv as input.

    The post-processing script is called with the summary.csv path as its first argument.
    The script runs from the run_dir, so any output files are saved there.
    If the script fails or doesn't exist, a warning is logged but execution continues.

    Args:
        script_path: Path to post-processing script (relative to cwd or absolute)
        summary_csv_path: Path to the summary.csv file
        run_dir: Run directory where script output will be saved
    """
    # Resolve script path
    script_file = Path(script_path)
    if not script_file.is_absolute():
        script_file = Path.cwd() / script_file

    if not script_file.exists():
        typer.echo(f"Warning: Post-processing script not found: {script_path}", err=True)
        return

    if verbose:
        typer.echo("-" * 60)
        typer.echo("Running post-processing script...")
        typer.echo(f"Script: {script_file}")
        typer.echo(f"Input: {summary_csv_path}")
        typer.echo(f"Output directory: {run_dir}")

    # Run post-processing script from the run directory
    cmd = [sys.executable, str(script_file), str(summary_csv_path)]

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(run_dir),
        )

        # Stream output in real-time
        for line in process.stdout:
            typer.echo(line.rstrip())

        exit_code = process.wait()

        if exit_code != 0:
            typer.echo(
                f"Warning: Post-processing script exited with code {exit_code}",
                err=True,
            )
        elif verbose:
            typer.echo("Post-processing completed successfully")

    except Exception as e:
        typer.echo(f"Warning: Error running post-processing script: {e}", err=True)


def _format_param_value(value: Any) -> str:
    """Format a parameter value for display with consistent significant figures.

    Formats floats to 4 significant figures with trailing zeros for alignment.

    Args:
        value: Parameter value to format

    Returns:
        Formatted string representation
    """
    if isinstance(value, float):
        # Format with 4 significant figures
        formatted = f"{value:.4g}"
        # Pad decimal representations with trailing zeros for alignment
        if '.' in formatted and 'e' not in formatted.lower():
            # Pad to 8 characters with trailing zeros
            formatted = formatted.ljust(8, '0')
        return formatted
    return str(value)


def _format_param_with_padding(key: str, value: Any, widths: dict) -> str:
    """Format a parameter key-value pair with padding for alignment.

    Args:
        key: Parameter name
        value: Parameter value
        widths: Dictionary mapping parameter names to their maximum widths

    Returns:
        Formatted string like "key=value  " with padding
    """
    formatted_value = _format_param_value(value)
    width = widths.get(key, len(formatted_value))
    padded_value = formatted_value.ljust(width)
    return f"{key}={padded_value}"


def _run_experiment_worker(args: Tuple) -> Tuple[int, int, dict]:
    """Worker function for parallel experiment execution.

    Args:
        args: Tuple of (experiment_index, script_path, params, exp_subdir, verbose)

    Returns:
        Tuple of (experiment_index, exit_code, params)
    """
    exp_index, script_path, params, exp_subdir, verbose = args
    exit_code = run_experiment(str(script_path), params, experiment_dir=str(exp_subdir), output_to_console=False, verbose=verbose)
    return (exp_index, exit_code, params)


@app.command()
def run(
    experiment_name: Optional[str] = typer.Argument(None, help="Name of the experiment (optional, will prompt if not provided)"),
    param: Optional[List[str]] = typer.Option(
        None, "--param", "-p", help="Parameter override (key=value)"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed logging and metadata information"),
    parallel: int = typer.Option(1, "--parallel", "-n", help="Number of parallel workers (1 for sequential, 0 for auto-detect leaving 2 cores free)"),
    timing: bool = typer.Option(True, "--timing/--no-timing", help="Show total execution time"),
    show_params: bool = typer.Option(False, "--show-params/--no-show-params", help="Show parameter values when experiments complete (parallel mode)"),
) -> None:
    """Run an experiment by name or interactive selection.

    The experiment script and parameters are specified in experiments/{experiment_name}/config.yaml.
    If experiment_name is provided, runs that experiment directly.
    Otherwise, shows an interactive list of available experiments to choose from.

    Creates a timestamped directory for logs and outputs.
    """
    start_time = time.time()
    try:
        # Find experiments directory
        cwd = Path.cwd()
        experiments_dir = cwd / "experiments"

        if not experiments_dir.exists():
            typer.echo(f"Error: experiments/ directory not found in {cwd}", err=True)
            raise typer.Exit(1)

        # If experiment_name not provided, show interactive selection
        if experiment_name is None:
            import questionary

            available_experiments = get_available_experiments(experiments_dir)

            if not available_experiments:
                typer.echo("Error: No experiments found in experiments/", err=True)
                typer.echo("Create an experiment directory with config.yaml first.", err=True)
                raise typer.Exit(1)

            experiment_name = questionary.select(
                "Select experiment:",
                choices=available_experiments
            ).ask()

            # Handle Ctrl+C / ESC
            if experiment_name is None:
                typer.echo("\nSelection cancelled", err=True)
                raise typer.Exit(1)

        # Find experiment subdirectory
        exp_dir = experiments_dir / experiment_name
        if not exp_dir.exists():
            typer.echo(f"Error: experiments/{experiment_name}/ not found", err=True)
            raise typer.Exit(1)

        # Find config.yaml
        config_file = exp_dir / "config.yaml"
        if not config_file.exists():
            typer.echo(f"Error: {config_file} not found", err=True)
            raise typer.Exit(1)

        # Create timestamped subdirectory for this run
        timestamp = datetime.now().strftime("%Y_%b_%d_%Hh%Mm%Ss")
        run_dir = exp_dir / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)

        # Generate CSV from config.yaml in the run directory
        generated_csv, metadata = yaml_to_csv(str(config_file))
        if not generated_csv:
            typer.echo(f"Error: Could not generate CSV from {config_file}", err=True)
            raise typer.Exit(1)

        # Extract script from metadata
        if "script" not in metadata or not metadata["script"]:
            typer.echo(f"Error: No 'script' field in {config_file}", err=True)
            raise typer.Exit(1)

        script = metadata["script"]

        # Check that script exists
        script_path = cwd / script
        if not script_path.exists():
            typer.echo(f"Error: Script not found: {script}", err=True)
            raise typer.Exit(1)

        # Move generated CSV to run directory
        csv_in_run = run_dir / Path(generated_csv).name
        Path(generated_csv).rename(csv_in_run)

        # Load experiments from CSV
        experiments = load_csv(str(csv_in_run))
        if verbose:
            typer.echo(f"Running {len(experiments)} experiments for '{experiment_name}'")
            typer.echo(f"Results directory: {run_dir}")
            typer.echo("-" * 60)

        # Determine number of workers
        num_workers = parallel
        if num_workers == 0:
            num_workers = max(1, (os.cpu_count() or 1) - 2)

        if verbose:
            typer.echo(f"Using {num_workers} worker(s) for execution")

        failed = 0

        # Prepare experiment arguments
        exp_args = []
        for i, params in enumerate(experiments, 1):
            # Merge CLI parameter overrides
            if param:
                params = merge_params(params, param)

            exp_subdir = run_dir / f"exp_{i:03d}"
            exp_args.append((i, script_path, params, exp_subdir, verbose))

        # Calculate parameter value widths for alignment
        param_widths: dict = {}
        if show_params:
            for params in experiments:
                for key, value in params.items():
                    formatted_value = _format_param_value(value)
                    current_max = param_widths.get(key, 0)
                    param_widths[key] = max(current_max, len(formatted_value))

        if num_workers > 1:
            # Parallel execution
            typer.echo(f"Running {len(experiments)} experiments in parallel ({num_workers} workers)...")

            completed = 0
            total = len(experiments)
            total_width = len(str(total))

            with Pool(num_workers) as pool:
                # Use imap_unordered to get results as they complete
                for exp_index, exit_code, params in pool.imap_unordered(_run_experiment_worker, exp_args):
                    completed += 1

                    # Build status message with proper alignment
                    status = ("succeeded" if exit_code == 0 else "FAILED").ljust(9)
                    msg = f"[{completed:>{total_width}}/{total}] Experiment {exp_index:>{total_width}} completed: {status}"

                    # Optionally show parameters
                    if show_params:
                        params_str = ", ".join(_format_param_with_padding(k, v, param_widths) for k, v in params.items())
                        msg += f" ({params_str})"

                    typer.echo(msg)

                    if exit_code != 0:
                        failed += 1
        else:
            # Sequential execution
            total = len(experiments)
            total_width = len(str(total))

            for exp_index, script_path_arg, params, exp_subdir, verbose_arg in exp_args:
                typer.echo(f"\n[{exp_index:>{total_width}}/{total}] Running experiment...")
                exit_code = run_experiment(str(script_path_arg), params, experiment_dir=str(exp_subdir), verbose=verbose_arg)
                if exit_code != 0:
                    failed += 1

        # Create summary CSV combining parameters and results
        create_results_summary(run_dir, experiments, verbose)

        # Run post-processing script if specified
        if "post_process_script" in metadata:
            post_process_script = metadata["post_process_script"]
            if post_process_script:
                summary_csv_path = run_dir / "summary.csv"
                run_post_processing(str(post_process_script), summary_csv_path, run_dir, verbose)

        if verbose:
            typer.echo("-" * 60)

        # Calculate and display elapsed time
        elapsed_seconds = time.time() - start_time
        if timing:
            minutes, seconds = divmod(elapsed_seconds, 60)
            if minutes > 0:
                time_str = f"{int(minutes)}m {seconds:.1f}s"
            else:
                time_str = f"{seconds:.1f}s"
            typer.echo(f"Completed {len(experiments)} experiments: {len(experiments) - failed} succeeded, {failed} failed (Total time: {time_str})")
        else:
            typer.echo(f"Completed {len(experiments)} experiments: {len(experiments) - failed} succeeded, {failed} failed")

        if verbose:
            typer.echo(f"Results saved to: {run_dir}")
        raise typer.Exit(1 if failed > 0 else 0)

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def hello(name: str) -> None:
    """Say hello to NAME."""
    typer.echo(f"Hello, {name}!")


@app.command()
def status() -> None:
    """Show status of running experiments."""
    typer.echo("No experiments running.")


def version_callback(value: bool) -> None:
    """Display version."""
    if value:
        typer.echo(f"Version: {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None, "--version", callback=version_callback, help="Show version"
    )
) -> None:
    """CLI tool for running, monitoring and logging numerical experiments."""
    pass


if __name__ == "__main__":
    app()
