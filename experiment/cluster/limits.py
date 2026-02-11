"""Cluster account limits and quotas."""

from dataclasses import dataclass
from typing import List, Optional, Dict
from .ssh import execute_remote_command, SSHError


@dataclass
class AccountLimits:
    """User/account limits on a cluster."""
    username: str
    account: str                       # Account/project name
    max_cpus_per_user: Optional[int]   # Max CPUs user can use at once
    max_cpus_running: Optional[int]    # Max CPUs in running jobs
    max_jobs: Optional[int]            # Max jobs that can run concurrently
    max_jobs_submit: Optional[int]     # Max jobs that can be submitted
    max_node_per_job: Optional[int]    # Max nodes per job
    max_wall_duration: Optional[str]   # Max wall time (HH:MM:SS)
    grp_cpu_limit: Optional[int]       # Group CPU limit
    grp_job_limit: Optional[int]       # Group job limit
    qos_list: List[str]                # List of QoS the user has access to


@dataclass
class QoSLimits:
    """Quality of Service limits."""
    name: str
    max_cpus_per_user: Optional[int]
    max_jobs_per_user: Optional[int]
    max_wall_duration: Optional[str]
    max_nodes_per_job: Optional[int]
    priority: Optional[int]
    description: str


@dataclass
class PartitionLimits:
    """Per-partition job limits."""
    name: str
    max_cpus_per_node: Optional[int]
    max_nodes: Optional[int]
    max_time: Optional[str]            # Default max wall time
    description: str


@dataclass
class UserResourceLimits:
    """Complete resource limits for a user."""
    username: str
    account: Optional[AccountLimits]
    qos_limits: List[QoSLimits]
    partition_limits: List[PartitionLimits]
    timestamp: str


def parse_sacctmgr_user_output(output: str, username: str) -> Optional[AccountLimits]:
    """Parse sacctmgr user account limits.

    Handles multiple output formats depending on SLURM version and configuration.

    Args:
        output: Output from sacctmgr show user command
        username: Username to find in output

    Returns:
        AccountLimits object or None if user not found
    """
    lines = output.strip().split('\n')
    if not lines:
        return None

    # Check if this is pipe-delimited format (contains |)
    if '|' in lines[0]:
        # Parse header
        header = lines[0].split('|')
        header_map = {col.strip(): idx for idx, col in enumerate(header)}

        # Find the user's row
        for line in lines[1:]:
            if not line.strip():
                continue

            parts = [p.strip() for p in line.split('|')]
            if len(parts) > 0 and parts[0] == username:
                try:
                    # Helper to safely get and convert values
                    def get_int(col_name):
                        idx = header_map.get(col_name)
                        if idx is None or idx >= len(parts):
                            return None
                        val = parts[idx]
                        if not val or val == 'None' or val == '':
                            return None
                        try:
                            return int(val)
                        except ValueError:
                            return None

                    def get_str(col_name):
                        idx = header_map.get(col_name)
                        if idx is None or idx >= len(parts):
                            return None
                        val = parts[idx]
                        return val if val and val != 'None' else None

                    # Parse QOS list (comma-separated)
                    qos_str = get_str('QOS')
                    qos_list = [q.strip() for q in qos_str.split(',')] if qos_str else []

                    return AccountLimits(
                        username=username,
                        account=get_str('Account') or 'default',
                        max_cpus_per_user=get_int('MaxCpusPerUser'),
                        max_cpus_running=get_int('MaxRunningCpus'),
                        max_jobs=get_int('MaxJobs'),
                        max_jobs_submit=get_int('MaxSubmitJobs'),
                        max_node_per_job=get_int('MaxNodesPerJob'),
                        max_wall_duration=get_str('MaxWallDurationPerJob') or get_str('MaxWall'),
                        grp_cpu_limit=get_int('GrpCpuLimit'),
                        grp_job_limit=get_int('GrpJobLimit'),
                        qos_list=qos_list
                    )
                except (ValueError, IndexError):
                    continue
    else:
        # Parse default format (space-separated)
        # Look for username in the output
        for line in lines:
            if username in line:
                # Extract account name (second column typically)
                parts = line.split()
                if len(parts) >= 2:
                    return AccountLimits(
                        username=username,
                        account=parts[1] if len(parts) > 1 else 'default',
                        max_cpus_per_user=None,
                        max_cpus_running=None,
                        max_jobs=None,
                        max_jobs_submit=None,
                        max_node_per_job=None,
                        max_wall_duration=None,
                        grp_cpu_limit=None,
                        grp_job_limit=None,
                        qos_list=[]
                    )

    return None


def parse_scontrol_partition_output(output: str) -> Optional[PartitionLimits]:
    """Parse scontrol partition output to extract limits.

    Args:
        output: Output from scontrol show partition command

    Returns:
        PartitionLimits object
    """
    lines = output.strip().split('\n')
    partition_name = None
    max_cpus = None
    max_nodes = None
    max_time = None
    description = ""

    for line in lines:
        line = line.strip()
        if line.startswith('PartitionName='):
            partition_name = line.split('=', 1)[1]
        elif line.startswith('MaxCpusPerNode='):
            try:
                val = line.split('=', 1)[1]
                max_cpus = int(val) if val.isdigit() else None
            except (ValueError, IndexError):
                pass
        elif line.startswith('MaxNodes='):
            try:
                val = line.split('=', 1)[1]
                max_nodes = int(val) if val.isdigit() else None
            except (ValueError, IndexError):
                pass
        elif line.startswith('MaxTime='):
            time_val = line.split('=', 1)[1]
            if time_val != 'UNLIMITED':
                max_time = time_val
        elif line.startswith('PartitionName='):
            # Parse partition name to use as description
            description = partition_name or "Unknown"

    return PartitionLimits(
        name=partition_name or "unknown",
        max_cpus_per_node=max_cpus,
        max_nodes=max_nodes,
        max_time=max_time,
        description=description
    )


def get_user_limits(
    host: str,
    user: str,
    username: str,
    verbose: bool = False
) -> UserResourceLimits:
    """Query user resource limits from cluster.

    Args:
        host: SSH host
        user: SSH user (for authentication)
        username: Username to check limits for
        verbose: Show verbose output

    Returns:
        UserResourceLimits object with account and QoS info

    Raises:
        SSHError: If SSH command fails
    """
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Query user account limits with compatible field names
    # Try multiple format strings as field names vary between SLURM versions
    sacctmgr_cmd = (
        f'sacctmgr show user {username} format='
        'name,account,maxcpusperuser,maxjobs,'
        'maxnodesperjob,maxwall,qos --noheader'
    )

    returncode, sacctmgr_output = execute_remote_command(host, user, sacctmgr_cmd, verbose=verbose)

    if returncode != 0:
        if "not found" in sacctmgr_output or "command not found" in sacctmgr_output:
            raise SSHError("sacctmgr command not found. SLURM may not be installed.")
        # If the command fails, try a simpler format
        sacctmgr_cmd = f'sacctmgr show user {username}'
        returncode, sacctmgr_output = execute_remote_command(host, user, sacctmgr_cmd, verbose=verbose)
        if returncode != 0:
            raise SSHError(f"Failed to query sacctmgr. User '{username}' may not exist or has no limits configured.")

    account_limits = parse_sacctmgr_user_output(sacctmgr_output, username)

    # Query QoS information (if user has access to QoS)
    qos_cmd = 'sacctmgr show qos format=name,maxcpusperuser,maxjobsperuser,maxwall,maxnodes --noheader'
    returncode, qos_output = execute_remote_command(host, user, qos_cmd, verbose=verbose)

    qos_limits = []
    if returncode == 0:
        for line in qos_output.strip().split('\n'):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 5:
                try:
                    qos_limits.append(QoSLimits(
                        name=parts[0],
                        max_cpus_per_user=int(parts[1]) if parts[1].isdigit() else None,
                        max_jobs_per_user=int(parts[2]) if parts[2].isdigit() else None,
                        max_wall_duration=parts[3] if parts[3] != 'UNLIMITED' else None,
                        max_nodes_per_job=int(parts[4]) if parts[4].isdigit() else None,
                        priority=None,
                        description=""
                    ))
                except (ValueError, IndexError):
                    continue

    # Query partition limits (just basic info)
    partition_limits = []

    return UserResourceLimits(
        username=username,
        account=account_limits,
        qos_limits=qos_limits,
        partition_limits=partition_limits,
        timestamp=timestamp
    )


def format_limits_card(limits: UserResourceLimits, width: int = 70) -> List[str]:
    """Format user limits as a card.

    Args:
        limits: UserResourceLimits object
        width: Card width in characters

    Returns:
        List of formatted lines
    """
    lines = []

    # Top border
    lines.append("┌─ " + limits.username + " " + "─" * (width - len(limits.username) - 4) + "┐")

    if limits.account:
        acct = limits.account
        account_str = f"│ Account: {acct.account:<50} │"
        lines.append(account_str.ljust(width + 2) + "│")

        # CPU limits
        if acct.max_cpus_per_user:
            cpu_line = f"│ Max CPUs (concurrent): {acct.max_cpus_per_user:<38} │"
            lines.append(cpu_line.ljust(width + 2) + "│")

        if acct.max_jobs:
            jobs_line = f"│ Max concurrent jobs: {acct.max_jobs:<40} │"
            lines.append(jobs_line.ljust(width + 2) + "│")

        if acct.max_jobs_submit:
            submit_line = f"│ Max jobs to submit: {acct.max_jobs_submit:<41} │"
            lines.append(submit_line.ljust(width + 2) + "│")

        if acct.max_wall_duration:
            time_line = f"│ Max job duration: {acct.max_wall_duration:<44} │"
            lines.append(time_line.ljust(width + 2) + "│")

        if acct.max_node_per_job:
            nodes_line = f"│ Max nodes per job: {acct.max_node_per_job:<43} │"
            lines.append(nodes_line.ljust(width + 2) + "│")

        if acct.qos_list:
            qos_line = f"│ QoS access: {', '.join(acct.qos_list):<50} │"
            lines.append(qos_line.ljust(width + 2) + "│")
    else:
        no_limits_line = "│ No limits found for this user"
        lines.append(no_limits_line.ljust(width + 2) + "│")

    # Bottom border
    lines.append("└" + "─" * width + "┘")

    return lines


def format_limits_detailed(limits: UserResourceLimits) -> str:
    """Format user limits in detailed view.

    Args:
        limits: UserResourceLimits object

    Returns:
        Formatted string for display
    """
    lines = []

    lines.append(f"User: {limits.username}")
    lines.append(f"Queried: {limits.timestamp}")
    lines.append("")

    if limits.account:
        acct = limits.account
        lines.append("ACCOUNT LIMITS")
        lines.append("─" * 60)
        lines.append(f"Account Name: {acct.account}")

        if acct.max_cpus_per_user:
            lines.append(f"Max CPUs per user: {acct.max_cpus_per_user}")
        if acct.max_jobs:
            lines.append(f"Max concurrent jobs: {acct.max_jobs}")
        if acct.max_jobs_submit:
            lines.append(f"Max jobs to submit: {acct.max_jobs_submit}")
        if acct.max_wall_duration:
            lines.append(f"Max job duration: {acct.max_wall_duration}")
        if acct.max_node_per_job:
            lines.append(f"Max nodes per job: {acct.max_node_per_job}")
        if acct.grp_cpu_limit:
            lines.append(f"Group CPU limit: {acct.grp_cpu_limit}")
        if acct.grp_job_limit:
            lines.append(f"Group job limit: {acct.grp_job_limit}")

        if acct.qos_list:
            lines.append(f"Available QoS: {', '.join(acct.qos_list)}")

        lines.append("")

    if limits.qos_limits:
        lines.append("QUALITY OF SERVICE (QoS) LIMITS")
        lines.append("─" * 60)
        for qos in limits.qos_limits:
            lines.append(f"\n{qos.name}:")
            if qos.max_cpus_per_user:
                lines.append(f"  Max CPUs per user: {qos.max_cpus_per_user}")
            if qos.max_jobs_per_user:
                lines.append(f"  Max jobs per user: {qos.max_jobs_per_user}")
            if qos.max_wall_duration:
                lines.append(f"  Max wall duration: {qos.max_wall_duration}")
            if qos.max_nodes_per_job:
                lines.append(f"  Max nodes per job: {qos.max_nodes_per_job}")

    return "\n".join(lines)


def format_limits_compact(limits: UserResourceLimits) -> str:
    """Format user limits in compact card view.

    Args:
        limits: UserResourceLimits object

    Returns:
        Formatted string for display
    """
    lines = []

    card_lines = format_limits_card(limits, width=70)
    lines.extend(card_lines)
    lines.append("")

    # Quick reference for common limits
    if limits.account:
        acct = limits.account
        lines.append("⚠️  QUICK LIMITS REFERENCE:")
        lines.append("─" * 70)

        if acct.max_cpus_per_user:
            lines.append(f"  • Can use up to {acct.max_cpus_per_user} CPUs at once across all jobs")
        if acct.max_jobs:
            lines.append(f"  • Can run up to {acct.max_jobs} jobs at the same time")
        if acct.max_wall_duration:
            lines.append(f"  • Max job duration: {acct.max_wall_duration}")
        if acct.max_node_per_job:
            lines.append(f"  • Can use up to {acct.max_node_per_job} nodes per job")
        if acct.max_jobs_submit:
            lines.append(f"  • Can submit up to {acct.max_jobs_submit} jobs at once")

    return "\n".join(lines)


def format_all_limits_analysis(
    account_limits: Optional[AccountLimits],
    qos_limits: List[QoSLimits],
    partition_limits: List[PartitionLimits]
) -> str:
    """Format comprehensive analysis of all ways user could be limited.

    Shows all constraint layers: user, account, QoS, and partition levels.

    Args:
        account_limits: User account limits
        qos_limits: QoS limits available to user
        partition_limits: Per-partition limits

    Returns:
        Formatted string for display
    """
    lines = []

    lines.append("ALL WAYS YOU COULD BE LIMITED")
    lines.append("=" * 80)
    lines.append("")

    # Layer 1: User-level limits
    lines.append("LAYER 1: USER-LEVEL LIMITS")
    lines.append("─" * 80)
    if account_limits:
        acct = account_limits
        limits_found = False

        if acct.max_cpus_per_user:
            lines.append(f"✓ Max concurrent CPUs: {acct.max_cpus_per_user}")
            limits_found = True
        if acct.max_jobs:
            lines.append(f"✓ Max concurrent jobs: {acct.max_jobs}")
            limits_found = True
        if acct.max_jobs_submit:
            lines.append(f"✓ Max jobs to submit at once: {acct.max_jobs_submit}")
            limits_found = True
        if acct.max_wall_duration:
            lines.append(f"✓ Max wall time per job: {acct.max_wall_duration}")
            limits_found = True
        if acct.max_node_per_job:
            lines.append(f"✓ Max nodes per job: {acct.max_node_per_job}")
            limits_found = True

        if not limits_found:
            lines.append("✗ No user-level limits found (unlimited)")
    else:
        lines.append("✗ No user-level limits found (unlimited)")

    lines.append("")

    # Layer 2: Account/group limits
    lines.append("LAYER 2: ACCOUNT/GROUP-LEVEL LIMITS")
    lines.append("─" * 80)
    if account_limits and (account_limits.grp_cpu_limit or account_limits.grp_job_limit):
        if account_limits.grp_cpu_limit:
            lines.append(f"✓ Account CPU limit: {account_limits.grp_cpu_limit}")
        if account_limits.grp_job_limit:
            lines.append(f"✓ Account job limit: {account_limits.grp_job_limit}")
        lines.append("  (All users in the account share these limits)")
    else:
        lines.append("✗ No account-level limits found (unlimited)")

    lines.append("")

    # Layer 3: QoS limits
    lines.append("LAYER 3: QUALITY OF SERVICE (QoS) LIMITS")
    lines.append("─" * 80)
    if account_limits and account_limits.qos_list:
        lines.append(f"Your QoS options: {', '.join(account_limits.qos_list)}")
        lines.append("")

        if qos_limits:
            for qos in qos_limits:
                lines.append(f"{qos.name}:")
                has_limits = False
                if qos.max_cpus_per_user:
                    lines.append(f"  • Max CPUs: {qos.max_cpus_per_user}")
                    has_limits = True
                if qos.max_jobs_per_user:
                    lines.append(f"  • Max jobs: {qos.max_jobs_per_user}")
                    has_limits = True
                if qos.max_wall_duration:
                    lines.append(f"  • Max wall time: {qos.max_wall_duration}")
                    has_limits = True
                if qos.max_nodes_per_job:
                    lines.append(f"  • Max nodes: {qos.max_nodes_per_job}")
                    has_limits = True
                if not has_limits:
                    lines.append("  • No limits")
        else:
            lines.append("(QoS details not available)")
    else:
        lines.append("✗ No QoS limits (or default QoS has no limits)")

    lines.append("")

    # Layer 4: Partition limits
    lines.append("LAYER 4: PARTITION-SPECIFIC LIMITS")
    lines.append("─" * 80)
    if partition_limits:
        for partition in partition_limits:
            lines.append(f"{partition.name}:")
            has_limits = False
            if partition.max_cpus_per_node:
                lines.append(f"  • Max CPUs per node: {partition.max_cpus_per_node}")
                has_limits = True
            if partition.max_nodes:
                lines.append(f"  • Max nodes: {partition.max_nodes}")
                has_limits = True
            if partition.max_time:
                lines.append(f"  • Max wall time: {partition.max_time}")
                has_limits = True
            if not has_limits:
                lines.append("  • No limits")
    else:
        lines.append("✗ Partition limits not available")

    lines.append("")

    # Summary
    lines.append("KEY POINTS")
    lines.append("─" * 80)
    lines.append("1. LAYERS STACK: If any layer has a limit, that's a constraint")
    lines.append("2. MOST RESTRICTIVE WINS: Lowest limit across all layers applies")
    lines.append("3. NO EXPLICIT LIMITS: May mean cluster uses fair-share scheduling")
    lines.append("4. CHECK WITH ADMIN: Ask your cluster admin for resource policies")
    lines.append("")
    lines.append("EXAMPLE:")
    lines.append("  If you try to run with 256 CPUs but have:")
    lines.append("    - User limit: 128 CPUs → REJECTED (too many)")
    lines.append("    - Partition limit: 64 CPUs per node × 4 nodes = 256 → OK")
    lines.append("  Result: You can use max 128 CPUs (most restrictive)")

    return "\n".join(lines)
