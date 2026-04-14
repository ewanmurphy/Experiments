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
from experiment.cluster import (
    find_cluster_config,
    find_cluster_configs,
    load_cluster_config,
    merge_experiment_cluster_config,
    RunStateManager,
    generate_slurm_script,
    submit_slurm_job,
    get_job_status,
    cancel_job,
    rsync_to_cluster,
    rsync_from_cluster,
    create_remote_directory,
    SSHError,
    get_cluster_resources,
    format_compact_info,
    format_detailed_info,
    format_verbose_info,
    get_user_limits,
    format_limits_card,
    format_limits_detailed,
    format_limits_compact,
    format_all_limits_analysis,
)

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
        exp_subdir = run_dir / f"exp_{i}"
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
        # Collect all fieldnames from all experiments to handle different result fields
        # Start with original parameter fieldnames, then add any result fields
        fieldnames = list(experiments[0].keys()) if experiments else []
        seen = set(fieldnames)

        # Add any additional result fields that appear in any experiment
        for row in summary:
            for key in row.keys():
                if key not in seen:
                    fieldnames.append(key)
                    seen.add(key)

        with open(summary_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary)

        if verbose:
            typer.echo(f"Summary saved to: {summary_file}")


def run_post_processing(script_path: str, summary_csv_path: Path, run_dir: Path, script_dir: Optional[Path] = None, verbose: bool = False) -> None:
    """Run post-processing script with summary.csv as input.

    The post-processing script is called with the summary.csv path as its first argument.
    The script runs from the run_dir, so any output files are saved there.
    If the script fails or doesn't exist, a warning is logged but execution continues.

    Args:
        script_path: Path to post-processing script (relative to script_dir, or absolute)
        summary_csv_path: Path to the summary.csv file
        run_dir: Run directory where script output will be saved
        script_dir: Directory to resolve relative script paths from (defaults to cwd)
    """
    # Resolve script path
    script_file = Path(script_path)
    if not script_file.is_absolute():
        base_dir = script_dir or Path.cwd()
        script_file = base_dir / script_file

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

        # Check that script exists - look in experiment directory first, then project root
        script_path = exp_dir / script
        if not script_path.exists():
            # Fallback to project root
            script_path = cwd / script

        if not script_path.exists():
            typer.echo(f"Error: Script not found: {script}", err=True)
            typer.echo(f"Checked locations:", err=True)
            typer.echo(f"  1. {exp_dir / script}", err=True)
            typer.echo(f"  2. {cwd / script}", err=True)
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

            exp_subdir = run_dir / f"exp_{i}"
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
                run_post_processing(str(post_process_script), summary_csv_path, run_dir, script_dir=exp_dir, verbose=verbose)

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
def cluster_submit(
    experiment_name: Optional[str] = typer.Argument(None, help="Name of the experiment"),
    cluster_config: str = typer.Option("cluster.yaml", "--cluster-config", help="Path to cluster config (searches: experiments/{name}/cluster.yaml, then ./cluster.yaml)"),
    param: Optional[List[str]] = typer.Option(None, "--param", "-p", help="Parameter override (key=value)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Generate files but don't submit"),
    watch: bool = typer.Option(False, "--watch", help="Monitor job status after submission"),
) -> None:
    """Submit experiments to remote SLURM cluster (non-blocking).

    Searches for cluster config in this order:
    1. experiments/{experiment_name}/cluster.yaml (per-experiment config)
    2. ./cluster.yaml (project root, shared by all experiments)
    3. --cluster-config flag path (explicit override)

    Generates SLURM script, syncs files to cluster, and submits array job.
    Returns immediately with job ID. Use cluster-status to monitor progress.
    """
    try:
        cwd = Path.cwd()
        experiments_dir = cwd / "experiments"

        if not experiments_dir.exists():
            typer.echo(f"Error: experiments/ directory not found in {cwd}", err=True)
            raise typer.Exit(1)

        # Handle interactive selection
        if experiment_name is None:
            import questionary
            available_experiments = get_available_experiments(experiments_dir)
            if not available_experiments:
                typer.echo("Error: No experiments found in experiments/", err=True)
                raise typer.Exit(1)
            experiment_name = questionary.select("Select experiment:", choices=available_experiments).ask()
            if experiment_name is None:
                typer.echo("\nSelection cancelled", err=True)
                raise typer.Exit(1)

        # Find experiment directory
        exp_dir = experiments_dir / experiment_name
        config_file = exp_dir / "config.yaml"
        if not config_file.exists():
            typer.echo(f"Error: {config_file} not found", err=True)
            raise typer.Exit(1)

        # Discover all available cluster configs
        try:
            configs = find_cluster_configs(
                experiment_name,
                cluster_config if cluster_config != "cluster.yaml" else None
            )
        except FileNotFoundError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(1)

        # If multiple configs, prompt user to select one
        if len(configs) > 1:
            import questionary
            choices = [questionary.Choice(name, path) for name, path in configs]
            config_path = questionary.select("Select cluster config:", choices=choices).ask()
            if config_path is None:
                typer.echo("Selection cancelled", err=True)
                raise typer.Exit(1)
        else:
            config_path = configs[0][1]

        # Load selected cluster config
        try:
            cluster_cfg = load_cluster_config(config_path)
        except (FileNotFoundError, ValueError) as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(1)

        # Generate CSV from config
        generated_csv, metadata = yaml_to_csv(str(config_file))
        if not generated_csv:
            typer.echo(f"Error: Could not generate CSV from {config_file}", err=True)
            raise typer.Exit(1)

        if "script" not in metadata:
            typer.echo(f"Error: No 'script' field in {config_file}", err=True)
            raise typer.Exit(1)

        script = metadata["script"]

        # Look for script in experiment directory first, then project root
        script_path = exp_dir / script
        if not script_path.exists():
            # Fallback to project root
            script_path = cwd / script

        if not script_path.exists():
            typer.echo(f"Error: Script not found: {script}", err=True)
            typer.echo(f"Checked locations:", err=True)
            typer.echo(f"  1. {exp_dir / script}", err=True)
            typer.echo(f"  2. {cwd / script}", err=True)
            raise typer.Exit(1)

        # Load experiments from CSV
        csv_path = Path(generated_csv)
        experiments = load_csv(str(csv_path))

        # Create local run directory
        timestamp = datetime.now().strftime("%Y_%b_%d_%Hh%Mm%Ss")
        run_dir = exp_dir / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)

        # Move CSV to run directory
        csv_in_run = run_dir / csv_path.name
        csv_path.rename(csv_in_run)

        if verbose:
            typer.echo(f"Submitting {len(experiments)} experiments to cluster")
            typer.echo(f"Host: {cluster_cfg.ssh.host}")
            typer.echo(f"Partition: {cluster_cfg.slurm.partition}")

        # Generate SLURM script
        # Script will be in parent experiment directory
        # When running from exp_001 subdirectory, need to go up 2 levels (exp_001 -> run_dir -> exp_dir)
        script_basename = Path(script).name
        slurm_script = generate_slurm_script(
            script_path=f"../../{script_basename}",
            num_experiments=len(experiments),
            partition=cluster_cfg.slurm.partition,
            account=cluster_cfg.slurm.account,
            time_limit=cluster_cfg.slurm.time_limit,
            memory=cluster_cfg.slurm.memory,
            cpus=cluster_cfg.slurm.cpus,
            gpus=cluster_cfg.slurm.gpus,
            max_concurrent=cluster_cfg.slurm.max_concurrent,
            modules=cluster_cfg.modules,
            environment=cluster_cfg.environment,
            remote_run_dir=f"{cluster_cfg.ssh.remote_base_dir}/{experiment_name}/{timestamp}",
            experiment_name=experiment_name,
            timestamp=timestamp
        )

        # Write SLURM script locally for reference
        slurm_script_path = run_dir / "cluster_job.sh"
        with open(slurm_script_path, "w") as f:
            f.write(slurm_script)

        if dry_run:
            typer.echo(f"[DRY-RUN] SLURM script generated at {slurm_script_path}")
            typer.echo(f"[DRY-RUN] Would submit {len(experiments)} experiments")
            raise typer.Exit(0)

        # Create remote directory
        remote_run_dir = f"{cluster_cfg.ssh.remote_base_dir}/{experiment_name}/{timestamp}"
        try:
            create_remote_directory(cluster_cfg.ssh.host, cluster_cfg.ssh.user, remote_run_dir, verbose=verbose)
        except SSHError as e:
            typer.echo(f"Error creating remote directory: {e}", err=True)
            raise typer.Exit(1)

        # Sync files to cluster
        if verbose:
            typer.echo("Syncing files to cluster...")
        try:
            # First, sync script to remote experiment directory (parent directory)
            remote_exp_dir = f"{cluster_cfg.ssh.remote_base_dir}/{experiment_name}"
            rsync_to_cluster(
                local_dir=str(exp_dir),
                remote_dir=remote_exp_dir,
                patterns=[script_basename],
                host=cluster_cfg.ssh.host,
                user=cluster_cfg.ssh.user,
                verbose=verbose
            )

            # Then, sync run directory contents
            rsync_to_cluster(
                local_dir=str(run_dir),
                remote_dir=remote_run_dir,
                patterns=["config_generated.csv", "cluster_job.sh"] + cluster_cfg.sync.to_cluster,
                host=cluster_cfg.ssh.host,
                user=cluster_cfg.ssh.user,
                verbose=verbose
            )
        except SSHError as e:
            typer.echo(f"Error syncing files to cluster: {e}", err=True)
            raise typer.Exit(1)

        # Submit SLURM job
        try:
            job_id = submit_slurm_job(slurm_script, remote_run_dir, cluster_cfg.ssh.host, cluster_cfg.ssh.user, verbose=verbose)
        except SSHError as e:
            typer.echo(f"Error submitting SLURM job: {e}", err=True)
            raise typer.Exit(1)

        # Save run metadata
        run_id = f"{experiment_name}_{timestamp}"
        state_manager = RunStateManager()
        from experiment.cluster import RunMetadata, ClusterMetadata
        metadata_obj = RunMetadata(
            run_id=run_id,
            experiment_name=experiment_name,
            timestamp=timestamp,
            local_dir=str(run_dir),
            remote_dir=remote_run_dir,
            cluster=ClusterMetadata(
                host=cluster_cfg.ssh.host,
                user=cluster_cfg.ssh.user,
                slurm_job_id=job_id,
                partition=cluster_cfg.slurm.partition,
                num_experiments=len(experiments)
            ),
            status="submitted",
            submitted_at=datetime.now().isoformat(),
            config_file=str(config_file),
            script=script
        )
        state_manager.save_run(metadata_obj)

        typer.echo(f"Job submitted successfully!")
        typer.echo(f"  Run ID: {run_id}")
        typer.echo(f"  SLURM Job ID: {job_id}")
        typer.echo(f"  Experiments: {len(experiments)}")
        typer.echo()
        typer.echo(f"Monitor status with: experiment cluster-status {run_id}")
        typer.echo(f"Collect results with: experiment cluster-collect {run_id}")

        if watch:
            typer.echo("\nMonitoring job status (Ctrl+C to detach)...")
            try:
                while True:
                    status = get_job_status(job_id, cluster_cfg.ssh.host, cluster_cfg.ssh.user, verbose=False)
                    typer.echo(f"[{status.job_id}] {status.state} - {status.completed_tasks}/{status.total_tasks} tasks")
                    if status.state in ("COMPLETED", "FAILED"):
                        break
                    time.sleep(30)
            except KeyboardInterrupt:
                typer.echo("\nDetached from monitoring")

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def cluster_status(
    run_id: Optional[str] = typer.Argument(None, help="Run ID from cluster-submit"),
    watch: bool = typer.Option(False, "--watch", help="Continuously monitor until completion"),
    interval: int = typer.Option(30, "--interval", help="Polling interval in seconds"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed debugging information"),
) -> None:
    """Check status of submitted cluster jobs."""
    try:
        state_manager = RunStateManager()

        if run_id is None:
            # List all runs sorted by most recent first
            runs = state_manager.list_runs()
            if not runs:
                typer.echo("No submitted cluster jobs found")
                raise typer.Exit(0)

            # Sort by submitted_at timestamp in descending order (most recent first)
            try:
                runs_sorted = sorted(
                    runs,
                    key=lambda r: r.submitted_at if r.submitted_at else "",
                    reverse=True
                )
            except (TypeError, AttributeError):
                # Fallback to original order if sorting fails
                runs_sorted = runs

            # Interactive selection with questionary
            import questionary

            # Format choices with timestamp, status, and experiment name
            choices = []
            for run in runs_sorted:
                # Extract just the time from submitted_at
                submitted_time = run.submitted_at.split("T")[1][:5] if "T" in run.submitted_at else ""
                label = f"{run.run_id:45} [{run.status:10}] {submitted_time}"
                choices.append(questionary.Choice(label, run.run_id))

            selected_id = questionary.select(
                "Select a job to check status:",
                choices=choices
            ).ask()

            if selected_id is None:
                typer.echo("Selection cancelled")
                raise typer.Exit(0)

            run_id = selected_id

        # Get specific run
        try:
            metadata = state_manager.load_run(run_id)
        except FileNotFoundError:
            typer.echo(f"Error: Run not found: {run_id}", err=True)
            typer.echo(f"\nAvailable runs:", err=True)
            for run in state_manager.list_runs():
                typer.echo(f"  {run.run_id}", err=True)
            raise typer.Exit(1)

        if verbose:
            typer.echo(f"[DEBUG] Querying status for run: {run_id}")
            typer.echo(f"[DEBUG] Job ID: {metadata.cluster.slurm_job_id}")
            typer.echo(f"[DEBUG] Host: {metadata.cluster.host}")
            typer.echo(f"[DEBUG] User: {metadata.cluster.user}")

        # Query job status
        try:
            status = get_job_status(
                metadata.cluster.slurm_job_id,
                metadata.cluster.host,
                metadata.cluster.user,
                verbose=verbose
            )
        except SSHError as e:
            typer.echo(f"\nError querying job status from SLURM:", err=True)
            typer.echo(f"  {e}", err=True)
            typer.echo(f"\nThis usually means:", err=True)
            typer.echo(f"  1. SSH connection failed (check ~/.ssh/config and SSH key)", err=True)
            typer.echo(f"  2. Job no longer exists on cluster (too old or already deleted)", err=True)
            typer.echo(f"  3. sacct command not available on cluster", err=True)
            typer.echo(f"\nRun metadata stored locally at: {state_manager.get_run_metadata_path(run_id)}", err=True)
            raise typer.Exit(1)

        # Display status
        typer.echo(f"Run ID: {run_id}")
        typer.echo(f"Experiment: {metadata.experiment_name}")
        typer.echo(f"SLURM Job ID: {metadata.cluster.slurm_job_id}")

        # Check if all tasks are actually done (SLURM may report COMPLETED before all tasks finish)
        all_done = status.total_tasks > 0 and status.completed_tasks == status.total_tasks
        effective_status = "COMPLETED" if all_done else status.state

        typer.echo(f"Status: {effective_status}")
        typer.echo(f"Tasks: {status.completed_tasks}/{status.total_tasks} completed")
        if status.elapsed_time:
            typer.echo(f"Elapsed time: {status.elapsed_time}")
        if status.time_limit:
            typer.echo(f"Time limit: {status.time_limit}")

        if watch:
            typer.echo("\nMonitoring (Ctrl+C to detach)...")
            try:
                while True:
                    status = get_job_status(
                        metadata.cluster.slurm_job_id,
                        metadata.cluster.host,
                        metadata.cluster.user,
                        verbose=verbose
                    )
                    typer.echo(f"[{status.job_id}] {status.state} - {status.completed_tasks}/{status.total_tasks} tasks")

                    # Only consider job done if we have valid task counts AND all tasks are done
                    # Don't exit just because main job state is terminal (array task counts take priority)
                    all_tasks_done = status.total_tasks > 0 and status.completed_tasks == status.total_tasks

                    # Handle explicit job cancellation/timeout even without task counts
                    cancelled_states = ("CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL")
                    explicitly_terminated = status.state in cancelled_states

                    if all_tasks_done or explicitly_terminated:
                        if status.state == "COMPLETED" or all_tasks_done:
                            typer.echo("Job completed!")
                        else:
                            typer.echo(f"Job terminated: {status.state}")
                        state_manager.update_status(run_id, status.state.lower())
                        break
                    time.sleep(interval)
            except KeyboardInterrupt:
                typer.echo("\nDetached from monitoring")

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def cluster_collect(
    run_id: Optional[str] = typer.Argument(None, help="Run ID from cluster-submit (optional, interactive selection if omitted)"),
    force: bool = typer.Option(False, "--force", help="Collect even if jobs not complete"),
    keep_remote: bool = typer.Option(False, "--keep-remote", help="Don't delete remote files"),
    skip_postprocess: bool = typer.Option(False, "--skip-postprocess", help="Skip post-processing"),
) -> None:
    """Collect results from completed cluster job."""
    try:
        state_manager = RunStateManager()

        # Interactive selection if run_id not provided
        if run_id is None:
            runs = state_manager.list_runs()
            if not runs:
                typer.echo("No submitted cluster jobs found")
                raise typer.Exit(0)

            # Sort by submitted_at timestamp in descending order (most recent first)
            try:
                runs_sorted = sorted(
                    runs,
                    key=lambda r: r.submitted_at if r.submitted_at else "",
                    reverse=True
                )
            except (TypeError, AttributeError):
                runs_sorted = runs

            # Interactive selection
            import questionary

            choices = []
            for run in runs_sorted:
                submitted_time = run.submitted_at.split("T")[1][:5] if "T" in run.submitted_at else ""
                label = f"{run.run_id:45} [{run.status:10}] {submitted_time}"
                choices.append(questionary.Choice(label, run.run_id))

            run_id = questionary.select(
                "Select a job to collect results from:",
                choices=choices
            ).ask()

            if run_id is None:
                typer.echo("Selection cancelled")
                raise typer.Exit(0)

        try:
            metadata = state_manager.load_run(run_id)
        except FileNotFoundError:
            typer.echo(f"Error: Run not found: {run_id}", err=True)
            raise typer.Exit(1)

        # Check job completion
        try:
            status = get_job_status(
                metadata.cluster.slurm_job_id,
                metadata.cluster.host,
                metadata.cluster.user
            )
        except SSHError as e:
            typer.echo(f"Error querying job status: {e}", err=True)
            raise typer.Exit(1)

        # Check if all tasks are actually done (not just if SLURM says COMPLETED)
        all_tasks_done = status.total_tasks > 0 and status.completed_tasks == status.total_tasks

        if not force and not all_tasks_done:
            typer.echo(f"Job is still running ({status.completed_tasks}/{status.total_tasks} tasks completed).", err=True)
            typer.echo(f"Use --force to collect anyway.", err=True)
            raise typer.Exit(1)

        # Sync results from cluster
        typer.echo("Collecting results from cluster...")
        try:
            # Load cluster config to get sync patterns
            try:
                config_path = find_cluster_config(metadata.experiment_name)
            except FileNotFoundError:
                # Fallback to default patterns if config not found
                config_path = None

            if config_path:
                cluster_cfg = load_cluster_config(config_path)
                patterns = cluster_cfg.sync.from_cluster
            else:
                # Default patterns - include all experiment directories and SLURM output
                patterns = ["exp_*", "slurm_*.out", "slurm_*.err"]

            rsync_from_cluster(
                remote_dir=metadata.remote_dir,
                local_dir=metadata.local_dir,
                patterns=patterns,
                host=metadata.cluster.host,
                user=metadata.cluster.user,
                verbose=True
            )
        except SSHError as e:
            typer.echo(f"Error syncing results: {e}", err=True)
            raise typer.Exit(1)

        # Generate summary CSV
        experiments = []
        for i in range(1, metadata.cluster.num_experiments + 1):
            exp_dir = Path(metadata.local_dir) / f"exp_{i}"
            if exp_dir.exists():
                # Try to get parameters from original CSV
                csv_file = Path(metadata.local_dir) / "config_generated.csv"
                if csv_file.exists():
                    exps = load_csv(str(csv_file))
                    if i <= len(exps):
                        experiments.append(exps[i-1])
                    else:
                        experiments.append({})
                else:
                    experiments.append({})

        create_results_summary(Path(metadata.local_dir), experiments)

        # Run post-processing if configured
        if not skip_postprocess:
            config_file = Path(metadata.config_file)
            if config_file.exists():
                _, cfg_metadata = yaml_to_csv(str(config_file))
                if "post_process_script" in cfg_metadata:
                    summary_csv = Path(metadata.local_dir) / "summary.csv"
                    run_post_processing(
                        cfg_metadata["post_process_script"],
                        summary_csv,
                        Path(metadata.local_dir),
                        script_dir=config_file.parent
                    )

        state_manager.update_status(run_id, "collected")

        typer.echo(f"Results collected to: {metadata.local_dir}")
        typer.echo(f"Summary saved to: {Path(metadata.local_dir) / 'summary.csv'}")

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def cluster_pull(
    run_id: Optional[str] = typer.Argument(None, help="Run ID from cluster-submit (optional, interactive selection if omitted)"),
    skip_postprocess: bool = typer.Option(False, "--skip-postprocess", help="Skip post-processing"),
    watch: bool = typer.Option(False, "--watch", help="Continuously sync results until completion"),
    interval: int = typer.Option(60, "--interval", help="Polling interval in seconds"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
) -> None:
    """Sync partial results from running cluster job and run post-processing.

    Can be run multiple times to monitor progress without blocking.
    Use --watch to continuously pull results until job completes.
    Unlike cluster-collect, does not finalize the run or delete remote files.
    """
    try:
        state_manager = RunStateManager()

        # Interactive selection if run_id not provided
        if run_id is None:
            runs = state_manager.list_runs()
            if not runs:
                typer.echo("No submitted cluster jobs found")
                raise typer.Exit(0)

            # Sort by submitted_at timestamp in descending order (most recent first)
            try:
                runs_sorted = sorted(
                    runs,
                    key=lambda r: r.submitted_at if r.submitted_at else "",
                    reverse=True
                )
            except (TypeError, AttributeError):
                runs_sorted = runs

            # Interactive selection
            import questionary

            choices = []
            for run in runs_sorted:
                submitted_time = run.submitted_at.split("T")[1][:5] if "T" in run.submitted_at else ""
                label = f"{run.run_id:45} [{run.status:10}] {submitted_time}"
                choices.append(questionary.Choice(label, run.run_id))

            run_id = questionary.select(
                "Select a job to pull results from:",
                choices=choices
            ).ask()

            if run_id is None:
                typer.echo("Selection cancelled")
                raise typer.Exit(0)

        try:
            metadata = state_manager.load_run(run_id)
        except FileNotFoundError:
            typer.echo(f"Error: Run not found: {run_id}", err=True)
            raise typer.Exit(1)

        def pull_once() -> bool:
            """Pull results once. Returns True if all tasks are complete or job terminated according to SLURM."""
            # Show job status and check completion
            all_done = False
            try:
                status = get_job_status(
                    metadata.cluster.slurm_job_id,
                    metadata.cluster.host,
                    metadata.cluster.user,
                    verbose=verbose
                )
                typer.echo(f"[{status.job_id}] {status.state} - {status.completed_tasks}/{status.total_tasks} tasks")
                # Only consider job done if we have valid task counts AND all tasks are done
                # Don't exit just because main job state is terminal (array task counts take priority)
                all_tasks_done = status.total_tasks > 0 and status.completed_tasks == status.total_tasks

                # Handle explicit job cancellation/timeout even without task counts
                cancelled_states = ("CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL")
                explicitly_terminated = status.state in cancelled_states

                all_done = all_tasks_done or explicitly_terminated
            except SSHError:
                # Job may have left SLURM's accounting window, but we can still sync
                typer.echo(f"[Warning] Could not query SLURM status (job may be too old or already removed)")
                # In watch mode, if we can't query SLURM, we can't reliably determine completion
                # so we continue polling (user can Ctrl+C to stop)

            # Sync results from cluster
            try:
                # Load cluster config to get sync patterns
                try:
                    config_path = find_cluster_config(metadata.experiment_name)
                except FileNotFoundError:
                    config_path = None

                if config_path:
                    cluster_cfg = load_cluster_config(config_path)
                    patterns = cluster_cfg.sync.from_cluster
                else:
                    # Default patterns
                    patterns = ["exp_*", "slurm_*.out", "slurm_*.err"]

                rsync_from_cluster(
                    remote_dir=metadata.remote_dir,
                    local_dir=metadata.local_dir,
                    patterns=patterns,
                    host=metadata.cluster.host,
                    user=metadata.cluster.user,
                    verbose=verbose
                )
            except SSHError as e:
                typer.echo(f"Error syncing results: {e}", err=True)
                raise typer.Exit(1)

            # Report local progress (count results.json files modified after job submission)
            local_count = 0
            try:
                submitted_time = datetime.fromisoformat(metadata.submitted_at)
                for i in range(1, metadata.cluster.num_experiments + 1):
                    results_file = Path(metadata.local_dir) / f"exp_{i}" / "results.json"
                    if results_file.exists():
                        # Only count files modified after job submission
                        mtime = datetime.fromtimestamp(results_file.stat().st_mtime)
                        if mtime > submitted_time:
                            local_count += 1
            except (ValueError, OSError):
                # Fallback: if we can't parse timestamps, just count existing files
                for i in range(1, metadata.cluster.num_experiments + 1):
                    results_file = Path(metadata.local_dir) / f"exp_{i}" / "results.json"
                    if results_file.exists():
                        local_count += 1

            typer.echo(f"Local progress: {local_count}/{metadata.cluster.num_experiments} results collected")

            # Generate summary CSV
            experiments = []
            for i in range(1, metadata.cluster.num_experiments + 1):
                exp_dir = Path(metadata.local_dir) / f"exp_{i}"
                if exp_dir.exists():
                    # Try to get parameters from original CSV
                    csv_file = Path(metadata.local_dir) / "config_generated.csv"
                    if csv_file.exists():
                        exps = load_csv(str(csv_file))
                        if i <= len(exps):
                            experiments.append(exps[i-1])
                        else:
                            experiments.append({})
                    else:
                        experiments.append({})

            create_results_summary(Path(metadata.local_dir), experiments, verbose=verbose)

            # Run post-processing if configured
            if not skip_postprocess:
                config_file = Path(metadata.config_file)
                if config_file.exists():
                    _, cfg_metadata = yaml_to_csv(str(config_file))
                    if "post_process_script" in cfg_metadata:
                        summary_csv = Path(metadata.local_dir) / "summary.csv"
                        run_post_processing(
                            cfg_metadata["post_process_script"],
                            summary_csv,
                            Path(metadata.local_dir),
                            script_dir=config_file.parent,
                            verbose=verbose
                        )

            return all_done

        if watch:
            typer.echo("Syncing results (Ctrl+C to stop)...")
            try:
                while True:
                    all_done = pull_once()
                    if all_done:
                        typer.echo("\nAll tasks completed! Run 'experiment cluster-collect' to finalize.")
                        break
                    time.sleep(interval)
            except KeyboardInterrupt:
                typer.echo("\nDetached from syncing")
        else:
            typer.echo("Syncing results from cluster...")
            pull_once()

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def cluster_cancel(
    run_id: Optional[str] = typer.Argument(None, help="Run ID from cluster-submit (optional, interactive selection if omitted)"),
    force: bool = typer.Option(False, "--force", help="Don't ask for confirmation"),
) -> None:
    """Cancel a submitted cluster job."""
    try:
        state_manager = RunStateManager()

        # Interactive selection if run_id not provided
        if run_id is None:
            runs = state_manager.list_runs()
            if not runs:
                typer.echo("No submitted cluster jobs found")
                raise typer.Exit(0)

            # Sort by submitted_at timestamp in descending order (most recent first)
            try:
                runs_sorted = sorted(
                    runs,
                    key=lambda r: r.submitted_at if r.submitted_at else "",
                    reverse=True
                )
            except (TypeError, AttributeError):
                runs_sorted = runs

            # Interactive selection
            import questionary

            choices = []
            for run in runs_sorted:
                submitted_time = run.submitted_at.split("T")[1][:5] if "T" in run.submitted_at else ""
                label = f"{run.run_id:45} [{run.status:10}] {submitted_time}"
                choices.append(questionary.Choice(label, run.run_id))

            run_id = questionary.select(
                "Select a job to cancel:",
                choices=choices
            ).ask()

            if run_id is None:
                typer.echo("Selection cancelled")
                raise typer.Exit(0)

        try:
            metadata = state_manager.load_run(run_id)
        except FileNotFoundError:
            typer.echo(f"Error: Run not found: {run_id}", err=True)
            raise typer.Exit(1)

        if not force:
            import questionary
            confirm = questionary.confirm(f"Cancel job {metadata.cluster.slurm_job_id}?").ask()
            if not confirm:
                typer.echo("Cancelled")
                raise typer.Exit(0)

        try:
            cancel_job(
                metadata.cluster.slurm_job_id,
                metadata.cluster.host,
                metadata.cluster.user
            )
        except SSHError as e:
            typer.echo(f"Error cancelling job: {e}", err=True)
            raise typer.Exit(1)

        state_manager.update_status(run_id, "cancelled")
        typer.echo(f"Job {metadata.cluster.slurm_job_id} cancelled")

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def cluster_list(
    all_runs: bool = typer.Option(False, "--all", help="Include completed runs"),
) -> None:
    """List all submitted cluster jobs (most recent first)."""
    try:
        state_manager = RunStateManager()
        runs = state_manager.list_runs(include_collected=all_runs)

        if not runs:
            typer.echo("No cluster jobs found")
            raise typer.Exit(0)

        # Sort by submitted_at timestamp in descending order (most recent first)
        try:
            runs_sorted = sorted(
                runs,
                key=lambda r: r.submitted_at if r.submitted_at else "",
                reverse=True
            )
        except (TypeError, AttributeError):
            # Fallback to original order if sorting fails
            runs_sorted = runs

        typer.echo("Cluster jobs (most recent first):")
        typer.echo("-" * 100)
        typer.echo(f"{'Run ID':40} {'Status':12} {'Experiment':20} {'Submitted':20}")
        typer.echo("-" * 100)
        for run in runs_sorted:
            submitted = run.submitted_at.split("T")[0] if "T" in run.submitted_at else run.submitted_at
            typer.echo(f"{run.run_id:40} {run.status:12} {run.experiment_name:20} {submitted:20}")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def cluster_info(
    cluster_config: str = typer.Option("cluster.yaml", "--cluster-config", help="Path to cluster config (or config name for interactive selection)"),
    partition: Optional[str] = typer.Option(None, "--partition", "-p", help="Filter by specific partition"),
    detailed: bool = typer.Option(False, "--detailed", "-d", help="Show detailed table view"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show partition configuration details"),
) -> None:
    """Display cluster resource availability and queue status."""
    try:
        # If using default cluster.yaml, check for multiple configs
        if cluster_config == "cluster.yaml":
            try:
                configs = find_cluster_configs("")
            except FileNotFoundError:
                configs = []

            # If multiple configs found, prompt for selection
            if len(configs) > 1:
                import questionary
                choices = [questionary.Choice(name, path) for name, path in configs]
                config_path = questionary.select("Select cluster config:", choices=choices).ask()
                if config_path is None:
                    typer.echo("Selection cancelled", err=True)
                    raise typer.Exit(1)
            elif len(configs) == 1:
                config_path = configs[0][1]
            else:
                config_path = cluster_config
        else:
            config_path = cluster_config

        # Load cluster configuration
        config = load_cluster_config(config_path)
        if config is None:
            typer.echo(f"Error: Could not load cluster config from {config_path}", err=True)
            raise typer.Exit(1)

        # Get cluster resources
        try:
            resources = get_cluster_resources(
                config.ssh.host,
                config.ssh.user,
                partition_filter=partition,
                verbose=verbose
            )
        except SSHError as e:
            typer.echo(f"Error querying cluster resources:", err=True)
            typer.echo(f"  {e}", err=True)
            typer.echo(f"\nTroubleshooting:", err=True)
            typer.echo(f"  1. Check SSH connection: ssh {config.ssh.user}@{config.ssh.host} echo ok", err=True)
            typer.echo(f"  2. Verify SLURM is installed: ssh {config.ssh.user}@{config.ssh.host} sinfo", err=True)
            typer.echo(f"  3. Check ~/.ssh/config and SSH key permissions", err=True)
            raise typer.Exit(1)

        # Check if any partitions were found
        if not resources.partitions:
            if partition:
                typer.echo(f"Error: Partition '{partition}' not found on cluster", err=True)
                typer.echo("\nAvailable partitions:", err=True)
                # Re-query without filter to show available partitions
                try:
                    all_resources = get_cluster_resources(config.ssh.host, config.ssh.user)
                    for p in all_resources.partitions:
                        typer.echo(f"  - {p.name}", err=True)
                except SSHError:
                    pass
            else:
                typer.echo("No partitions found on cluster", err=True)
            raise typer.Exit(1)

        # Format and display output
        if verbose:
            output = format_verbose_info(resources, config.ssh.host, config.ssh.user)
        elif detailed:
            output = format_detailed_info(resources, config.ssh.host, config.ssh.user)
        else:
            output = format_compact_info(resources, config.ssh.host, config.ssh.user)

        typer.echo(output)

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def cluster_limits(
    cluster_config: str = typer.Option("cluster.yaml", "--cluster-config", help="Path to cluster config (or config name for interactive selection)"),
    username: Optional[str] = typer.Option(None, "--user", "-u", help="Username to check (defaults to SSH user)"),
    detailed: bool = typer.Option(False, "--detailed", "-d", help="Show detailed view"),
    all_limits: bool = typer.Option(False, "--all", "-a", help="Show all possible ways you could be limited"),
) -> None:
    """Check your account resource limits on the cluster."""
    try:
        # If using default cluster.yaml, check for multiple configs
        if cluster_config == "cluster.yaml":
            try:
                configs = find_cluster_configs("")
            except FileNotFoundError:
                configs = []

            # If multiple configs found, prompt for selection
            if len(configs) > 1:
                import questionary
                choices = [questionary.Choice(name, path) for name, path in configs]
                config_path = questionary.select("Select cluster config:", choices=choices).ask()
                if config_path is None:
                    typer.echo("Selection cancelled", err=True)
                    raise typer.Exit(1)
            elif len(configs) == 1:
                config_path = configs[0][1]
            else:
                config_path = cluster_config
        else:
            config_path = cluster_config

        # Load cluster configuration
        config = load_cluster_config(config_path)
        if config is None:
            typer.echo(f"Error: Could not load cluster config from {config_path}", err=True)
            raise typer.Exit(1)

        # Determine which user to check
        check_user = username if username else config.ssh.user

        # Get user limits
        try:
            limits = get_user_limits(
                config.ssh.host,
                config.ssh.user,
                check_user
            )
        except SSHError as e:
            typer.echo(f"Error querying account limits:", err=True)
            typer.echo(f"  {e}", err=True)
            typer.echo(f"\nTroubleshooting:", err=True)
            typer.echo(f"  1. Check SSH connection: ssh {config.ssh.user}@{config.ssh.host} echo ok", err=True)
            typer.echo(f"  2. Verify sacctmgr is available: ssh {config.ssh.user}@{config.ssh.host} sacctmgr --version", err=True)
            typer.echo(f"  3. Check if you have account limits configured", err=True)
            raise typer.Exit(1)

        # Format and display output
        if all_limits:
            output = format_all_limits_analysis(limits.account, limits.qos_limits, limits.partition_limits)
        elif detailed:
            output = format_limits_detailed(limits)
        else:
            output = format_limits_compact(limits)

        typer.echo(output)

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


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
