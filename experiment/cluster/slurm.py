"""SLURM job generation, submission, and monitoring."""

import re
import time
from dataclasses import dataclass
from typing import Optional, Dict
from .ssh import execute_remote_command, SSHError


@dataclass
class JobStatus:
    """SLURM job status information."""
    job_id: str
    state: str  # PENDING, RUNNING, COMPLETED, FAILED, etc.
    completed_tasks: int
    total_tasks: int
    elapsed_time: Optional[str] = None
    time_limit: Optional[str] = None


def generate_slurm_script(
    script_path: str,
    num_experiments: int,
    partition: str,
    time_limit: str,
    memory: str,
    cpus: int,
    gpus: int,
    max_concurrent: int,
    modules: list,
    environment: Optional[str],
    remote_run_dir: str,
    experiment_name: str,
    timestamp: str
) -> str:
    """Generate SLURM batch script for array job.

    Args:
        script_path: Name of experiment script (relative path in remote dir)
        num_experiments: Number of experiments (array size)
        partition: SLURM partition/queue
        time_limit: Time limit (HH:MM:SS format)
        memory: Memory per task (e.g., 4G)
        cpus: CPUs per task
        gpus: GPUs per task
        max_concurrent: Max concurrent tasks (0 for unlimited)
        modules: List of modules to load
        environment: Environment activation command
        remote_run_dir: Remote run directory path
        experiment_name: Experiment name
        timestamp: Timestamp string

    Returns:
        SLURM script content as string
    """
    # Build SBATCH directives
    sbatch_lines = [
        f"#SBATCH --job-name={experiment_name}_{timestamp}",
        f"#SBATCH --output={remote_run_dir}/exp_%a/slurm_%A_%a.out",
        f"#SBATCH --error={remote_run_dir}/exp_%a/slurm_%A_%a.err",
    ]

    # Array specification with max concurrent limit
    if max_concurrent > 0:
        sbatch_lines.append(f"#SBATCH --array=1-{num_experiments}%{max_concurrent}")
    else:
        sbatch_lines.append(f"#SBATCH --array=1-{num_experiments}")

    sbatch_lines.extend([
        f"#SBATCH --partition={partition}",
        f"#SBATCH --time={time_limit}",
        f"#SBATCH --mem={memory}",
        f"#SBATCH --cpus-per-task={cpus}",
    ])

    if gpus > 0:
        sbatch_lines.append(f"#SBATCH --gres=gpu:{gpus}")

    # Build script
    script = "#!/bin/bash\n"
    script += "\n".join(sbatch_lines) + "\n"
    script += "\n"

    # Load modules
    if modules:
        for module in modules:
            script += f"module load {module}\n"
        script += "\n"

    # Environment activation
    if environment:
        script += f"{environment}\n"
        script += "\n"

    # Change to run directory
    script += f"cd {remote_run_dir}\n"
    script += "\n"

    # Create experiment subdirectory
    script += "# Create experiment subdirectory\n"
    script += "EXP_NUM=$(printf \"%d\" $SLURM_ARRAY_TASK_ID)\n"
    script += "EXP_DIR=\"exp_$EXP_NUM\"\n"
    script += "mkdir -p $EXP_DIR\n"
    script += "cd $EXP_DIR\n"
    script += "\n"

    # Parse parameters from CSV using an array for proper handling
    script += "# Read parameters from CSV using array task ID\n"
    script += """readarray -t PARAMS < <(awk -F',' -v task=$SLURM_ARRAY_TASK_ID '
NR==1 {
    for(i=1; i<=NF; i++) {
        gsub(/\\r/, "", $i)
        header[i]=$i
    }
    next
}
NR==task+1 {
    for(i=1; i<=NF; i++) {
        gsub(/\\r/, "", $i)
        printf "--%s\\n%s\\n", header[i], $i
    }
}' ../config_generated.csv)
"""
    script += "\n"
    script += "# Debug output\n"
    script += "echo \"Task: $SLURM_ARRAY_TASK_ID\" >&2\n"
    script += "echo \"CSV file:\" >&2\n"
    script += "cat ../config_generated.csv >&2\n"
    script += "echo \"Parameters array elements: ${#PARAMS[@]}\" >&2\n"
    script += "for i in \"${!PARAMS[@]}\"; do echo \"  PARAMS[$i]=${PARAMS[$i]}\" >&2; done\n"
    script += "\n"

    # Run experiment
    # If script_path starts with .., use it as-is (it already has relative path)
    # Otherwise, add ../ to reference parent directory
    # Use array expansion for proper argument handling (unquoted to pass as separate args)
    if script_path.startswith(".."):
        script += f"echo \"Running: python {script_path} ${{PARAMS[@]}}\" >&2\n"
        script += f"python {script_path} ${{PARAMS[@]}}\n"
    else:
        script += f"echo \"Running: python ../{script_path} ${{PARAMS[@]}}\" >&2\n"
        script += f"python ../{script_path} ${{PARAMS[@]}}\n"
    script += "\n"

    # Exit with script's exit code
    script += "exit $?\n"

    return script


def submit_slurm_job(
    slurm_script_content: str,
    remote_dir: str,
    host: str,
    user: str,
    verbose: bool = False
) -> str:
    """Submit SLURM job and return job ID.

    Args:
        slurm_script_content: Content of SLURM script
        remote_dir: Remote directory where script will be written
        host: SSH host
        user: SSH user
        verbose: Print details

    Returns:
        SLURM job ID

    Raises:
        SSHError: If job submission fails
    """
    # Write script to temporary file on remote host
    script_name = "cluster_job.sh"
    script_path = f"{remote_dir}/{script_name}"

    # Escape content for shell
    escaped_content = slurm_script_content.replace("'", "'\\''")

    # Write script to remote host
    write_cmd = f"cat > {script_path} << 'EOF'\n{slurm_script_content}\nEOF"
    return_code, output = execute_remote_command(
        host, user, write_cmd, verbose=verbose
    )
    if return_code != 0:
        raise SSHError(f"Failed to write SLURM script to remote host")

    # Make script executable
    execute_remote_command(
        host, user, f"chmod +x {script_path}", verbose=verbose
    )

    # Submit job
    submit_cmd = f"sbatch {script_path}"
    return_code, output = execute_remote_command(
        host, user, submit_cmd, verbose=verbose
    )

    if return_code != 0:
        raise SSHError(f"sbatch submission failed: {output}")

    # Parse job ID from output (format: "Submitted batch job 12345")
    match = re.search(r"Submitted batch job (\d+)", output)
    if not match:
        raise SSHError(f"Could not parse job ID from sbatch output: {output}")

    job_id = match.group(1)
    return job_id


def get_job_status(
    job_id: str,
    host: str,
    user: str,
    verbose: bool = False
) -> JobStatus:
    """Get SLURM job status.

    Args:
        job_id: SLURM job ID
        host: SSH host
        user: SSH user
        verbose: Print details

    Returns:
        JobStatus object

    Raises:
        SSHError: If status query fails
    """
    # Query job status using sacct
    cmd = f"sacct -j {job_id} --format=JobID,State,Elapsed,Timelimit --noheader --parsable2"
    return_code, output = execute_remote_command(
        host, user, cmd, verbose=verbose
    )

    if return_code != 0:
        raise SSHError(f"Failed to query job status: {output}")

    lines = output.strip().split("\n")
    if not lines:
        raise SSHError(f"No output from sacct for job {job_id}")

    # Parse main job line (first line, without array task ID)
    main_line = None
    for line in lines:
        if line and "|" not in line and "." not in line.split()[0]:
            main_line = line
            break

    if not main_line:
        main_line = lines[0]

    parts = main_line.split("|")
    if len(parts) < 2:
        raise SSHError(f"Could not parse sacct output: {output}")

    job_id_str = parts[0].strip()
    state = parts[1].strip() if len(parts) > 1 else "UNKNOWN"
    elapsed = parts[2].strip() if len(parts) > 2 else None
    timelimit = parts[3].strip() if len(parts) > 3 else None

    # Count completed and total tasks
    # Note: sacct returns 3 entries per array task (main, .batch, .extern)
    # We only count the main task entries (e.g., "4566440_1"), not .batch/.extern
    completed_tasks = 0
    total_tasks = 0
    for line in lines:
        parts = line.split("|")
        job_id_str = parts[0].strip()

        # Only count main array task lines (e.g., "4566440_1")
        # Skip .batch and .extern sub-entries
        if job_id_str.endswith(".batch") or job_id_str.endswith(".extern"):
            continue
        # Count this entry if it has a dot (array task) or check if it's numeric suffix
        if "." in job_id_str or "_" in job_id_str:
            task_state = parts[1].strip() if len(parts) > 1 else ""
            total_tasks += 1
            if task_state in ("COMPLETED", "FAILED"):
                completed_tasks += 1

    return JobStatus(
        job_id=job_id_str,
        state=state,
        completed_tasks=completed_tasks,
        total_tasks=total_tasks,
        elapsed_time=elapsed,
        time_limit=timelimit
    )


def cancel_job(
    job_id: str,
    host: str,
    user: str,
    verbose: bool = False
) -> bool:
    """Cancel SLURM job.

    Args:
        job_id: SLURM job ID
        host: SSH host
        user: SSH user
        verbose: Print details

    Returns:
        True if cancellation successful

    Raises:
        SSHError: If cancellation fails
    """
    cmd = f"scancel {job_id}"
    return_code, output = execute_remote_command(
        host, user, cmd, verbose=verbose
    )

    if return_code != 0:
        raise SSHError(f"Failed to cancel job: {output}")

    return True


def wait_for_completion(
    job_id: str,
    host: str,
    user: str,
    poll_interval: int = 30,
    verbose: bool = False
) -> JobStatus:
    """Poll until job completion.

    Args:
        job_id: SLURM job ID
        host: SSH host
        user: SSH user
        poll_interval: Polling interval in seconds
        verbose: Print status updates

    Returns:
        Final JobStatus

    Raises:
        SSHError: If polling fails
    """
    while True:
        status = get_job_status(job_id, host, user, verbose=verbose)

        if status.state in ("COMPLETED", "FAILED", "CANCELLED"):
            return status

        if verbose:
            print(f"[{status.job_id}] {status.state} - {status.completed_tasks}/{status.total_tasks} tasks completed")

        time.sleep(poll_interval)
