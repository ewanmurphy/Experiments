"""SSH connection and file transfer utilities."""

import subprocess
from pathlib import Path
from typing import Tuple, Optional


class SSHError(Exception):
    """SSH operation error."""
    pass


def _ssh_target(host: str, user: Optional[str] = None) -> str:
    """Build SSH target string, handling optional user.

    Args:
        host: SSH host
        user: SSH user (optional; if None, SSH config alias resolution is used)

    Returns:
        SSH target string: "user@host" if user provided, else "host"
    """
    if user:
        return f"{user}@{host}"
    return host


def test_ssh_connection(host: str, user: Optional[str] = None, timeout: int = 5) -> bool:
    """Test SSH connection to host.

    Args:
        host: SSH host
        user: SSH user (optional; if None, SSH config alias resolution is used)
        timeout: Connection timeout in seconds (default: 5)

    Returns:
        True if connection successful, False otherwise

    Raises:
        SSHError: If connection test fails with error
    """
    try:
        result = subprocess.run(
            ["ssh", "-o", f"ConnectTimeout={timeout}", _ssh_target(host, user), "echo", "ok"],
            capture_output=True,
            text=True,
            timeout=timeout + 2
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception as e:
        raise SSHError(f"SSH connection test failed: {e}")


def execute_remote_command(
    host: str,
    user: Optional[str],
    command: str,
    verbose: bool = False
) -> Tuple[int, str]:
    """Execute command on remote host via SSH.

    Args:
        host: SSH host
        user: SSH user (optional; if None, SSH config alias resolution is used)
        command: Command to execute on remote host
        verbose: Print command and output to stdout

    Returns:
        Tuple of (return_code, stdout)

    Raises:
        SSHError: If SSH execution fails
    """
    ssh_cmd = ["ssh", _ssh_target(host, user), command]

    if verbose:
        print(f"  [SSH] {' '.join(ssh_cmd)}")

    try:
        result = subprocess.run(
            ssh_cmd,
            capture_output=True,
            text=True
        )
        return result.returncode, result.stdout
    except Exception as e:
        raise SSHError(f"SSH command execution failed: {e}")


def create_remote_directory(host: str, user: Optional[str], path: str, verbose: bool = False) -> bool:
    """Create directory on remote host.

    Args:
        host: SSH host
        user: SSH user (optional; if None, SSH config alias resolution is used)
        path: Remote directory path
        verbose: Print details to stdout

    Returns:
        True if successful

    Raises:
        SSHError: If creation fails
    """
    return_code, _ = execute_remote_command(
        host, user, f"mkdir -p {path}", verbose=verbose
    )
    if return_code != 0:
        raise SSHError(f"Failed to create remote directory: {path}")
    return True


def rsync_to_cluster(
    local_dir: str,
    remote_dir: str,
    patterns: list,
    host: str,
    user: Optional[str],
    verbose: bool = False,
    progress: bool = True
) -> bool:
    """Sync files to remote cluster using rsync.

    Args:
        local_dir: Local source directory
        remote_dir: Remote destination directory
        patterns: List of include patterns (rsync --include syntax)
        host: Remote host
        user: Remote user (optional; if None, SSH config alias resolution is used)
        verbose: Print detailed output
        progress: Show progress during transfer

    Returns:
        True if successful

    Raises:
        SSHError: If rsync fails
    """
    local_path = Path(local_dir)
    if not local_path.exists():
        raise SSHError(f"Local directory does not exist: {local_dir}")

    # Build rsync command
    cmd = [
        "rsync",
        "-a",  # Archive mode
        "--delete",  # Delete remote files not in source
    ]

    # Add patterns as include/exclude filters
    for pattern in patterns:
        cmd.extend(["--include", pattern])
    cmd.append("--exclude=*")  # Exclude everything else

    if progress:
        cmd.append("--progress")
    if verbose:
        cmd.append("-v")

    # Source and destination
    cmd.append(f"{local_path}/")
    cmd.append(f"{_ssh_target(host, user)}:{remote_dir}/")

    if verbose:
        print(f"  [RSYNC TO] {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, text=True)
        if result.returncode != 0:
            raise SSHError(f"rsync to cluster failed with code {result.returncode}")
        return True
    except Exception as e:
        raise SSHError(f"rsync to cluster failed: {e}")


def rsync_from_cluster(
    remote_dir: str,
    local_dir: str,
    patterns: list,
    host: str,
    user: Optional[str],
    verbose: bool = False,
    progress: bool = True
) -> bool:
    """Sync files from remote cluster using rsync.

    Args:
        remote_dir: Remote source directory
        local_dir: Local destination directory
        patterns: List of include patterns (rsync --include syntax)
        host: Remote host
        user: Remote user (optional; if None, SSH config alias resolution is used)
        verbose: Print detailed output
        progress: Show progress during transfer

    Returns:
        True if successful

    Raises:
        SSHError: If rsync fails
    """
    local_path = Path(local_dir)
    local_path.mkdir(parents=True, exist_ok=True)

    # Build rsync command
    cmd = [
        "rsync",
        "-a",  # Archive mode
        "-r",  # Recursive
    ]

    # Add patterns as include/exclude filters
    # Important: For directories, we need to include the dir/ AND dir/** to get contents
    for pattern in patterns:
        # If pattern looks like exp_* (a directory pattern), also include contents
        if "*" in pattern and not pattern.endswith(("**", "/", ".out", ".err")):
            cmd.extend(["--include", f"{pattern}/"])
            cmd.extend(["--include", f"{pattern}/**"])
        else:
            cmd.extend(["--include", pattern])
    cmd.append("--exclude=*")  # Exclude everything else

    if progress:
        cmd.append("--progress")
    if verbose:
        cmd.append("-v")

    # Source and destination
    cmd.append(f"{_ssh_target(host, user)}:{remote_dir}/")
    cmd.append(f"{local_path}/")

    if verbose:
        print(f"  [RSYNC FROM] {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, text=True)
        if result.returncode != 0:
            raise SSHError(f"rsync from cluster failed with code {result.returncode}")
        return True
    except Exception as e:
        raise SSHError(f"rsync from cluster failed: {e}")
