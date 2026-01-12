"""Main CLI entry point."""

import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional
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


def create_results_summary(run_dir: Path, experiments: List[dict]) -> None:
    """Create a summary CSV combining parameters and results.

    Args:
        run_dir: Path to the run directory
        experiments: List of parameter dictionaries for each experiment
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

        typer.echo(f"Summary saved to: {summary_file}")


def run_post_processing(script_path: str, summary_csv_path: Path, run_dir: Path) -> None:
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
        else:
            typer.echo("Post-processing completed successfully")

    except Exception as e:
        typer.echo(f"Warning: Error running post-processing script: {e}", err=True)


@app.command()
def run(
    experiment_name: Optional[str] = typer.Argument(None, help="Name of the experiment (optional, will prompt if not provided)"),
    param: Optional[List[str]] = typer.Option(
        None, "--param", "-p", help="Parameter override (key=value)"
    ),
) -> None:
    """Run an experiment by name or interactive selection.

    The experiment script and parameters are specified in experiments/{experiment_name}/config.yaml.
    If experiment_name is provided, runs that experiment directly.
    Otherwise, shows an interactive list of available experiments to choose from.

    Creates a timestamped directory for logs and outputs.
    """
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
        typer.echo(f"Running {len(experiments)} experiments for '{experiment_name}'")
        typer.echo(f"Results directory: {run_dir}")
        typer.echo("-" * 60)

        failed = 0
        for i, params in enumerate(experiments, 1):
            typer.echo(f"\n[{i}/{len(experiments)}] Running experiment...")

            # Merge CLI parameter overrides
            if param:
                params = merge_params(params, param)

            # Run the experiment from the run directory
            exp_subdir = run_dir / f"exp_{i:03d}"
            exit_code = run_experiment(str(script_path), params, experiment_dir=str(exp_subdir))
            if exit_code != 0:
                failed += 1

        # Create summary CSV combining parameters and results
        create_results_summary(run_dir, experiments)

        # Run post-processing script if specified
        if "post_process_script" in metadata:
            post_process_script = metadata["post_process_script"]
            if post_process_script:
                summary_csv_path = run_dir / "summary.csv"
                run_post_processing(str(post_process_script), summary_csv_path, run_dir)

        typer.echo("-" * 60)
        typer.echo(f"Completed {len(experiments)} experiments: {len(experiments) - failed} succeeded, {failed} failed")
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
