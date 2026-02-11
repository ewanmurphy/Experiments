"""Cluster resource information and queue status."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple, Dict
from .ssh import execute_remote_command, SSHError


@dataclass
class PartitionInfo:
    """Information about a SLURM partition (raw from sinfo)."""
    name: str
    availability: str       # up/down/drain
    time_limit: str         # HH:MM:SS or UNLIMITED
    num_nodes: int
    node_state: str         # idle/allocated/mix/down
    cpus_allocated: int
    cpus_idle: int
    cpus_total: int
    memory_mb: int
    gpus: str               # "gpu:v100:2" or "N/A"
    node_list: str


@dataclass
class HardwareGroup:
    """A group of nodes with identical hardware specs."""
    gpu_type: str           # "v100:2", "h100:4", "N/A"
    cpus_per_node: int
    memory_mb: int
    node_states: Dict[str, Tuple[int, int, int]] = field(default_factory=dict)  # state -> (nodes, alloc_cpus, idle_cpus)


@dataclass
class PartitionGroup:
    """A partition aggregated by hardware specs and node states."""
    name: str
    availability: str
    time_limit: str
    hardware_groups: List[HardwareGroup] = field(default_factory=list)


@dataclass
class QueueInfo:
    """Information about job queue status for a partition."""
    partition: str
    pending_jobs: int
    running_jobs: int
    pending_cpus: int
    running_cpus: int


@dataclass
class ClusterResourceInfo:
    """Complete cluster resource and queue information."""
    partitions: List[PartitionInfo]
    partition_groups: List[PartitionGroup]  # Aggregated view
    queue_stats: List[QueueInfo]
    timestamp: str


def parse_sinfo_output(output: str) -> List[PartitionInfo]:
    """Parse sinfo output into PartitionInfo objects.

    Expected format (pipe-delimited):
    PartitionName|State|TimeLimitDate|Nodes|NodeAIOT|CPUs(A/I/O/T)|Memory|Gres|NodeList

    Args:
        output: Output from sinfo command

    Returns:
        List of PartitionInfo objects
    """
    partitions = []
    lines = output.strip().split('\n')

    for line in lines:
        if not line.strip():
            continue

        parts = line.split('|')
        if len(parts) < 9:
            # Skip malformed lines
            continue

        try:
            name = parts[0].strip()
            availability = parts[1].strip()
            time_limit = parts[2].strip()
            num_nodes = int(parts[3].strip())
            node_state = parts[4].strip()

            # Parse CPU allocation (A/I/O/T format)
            cpu_parts = parts[5].split('/')
            cpus_allocated = int(cpu_parts[0].strip()) if len(cpu_parts) > 0 else 0
            cpus_idle = int(cpu_parts[1].strip()) if len(cpu_parts) > 1 else 0
            cpus_total = int(cpu_parts[3].strip()) if len(cpu_parts) > 3 else cpus_allocated + cpus_idle

            # Parse memory (may have 'M', 'G', 'T' suffix)
            memory_str = parts[6].strip()
            memory_mb = parse_memory_value(memory_str)

            gpus = parts[7].strip() if parts[7].strip() else "N/A"
            node_list = parts[8].strip() if len(parts) > 8 else ""

            partition = PartitionInfo(
                name=name,
                availability=availability,
                time_limit=time_limit,
                num_nodes=num_nodes,
                node_state=node_state,
                cpus_allocated=cpus_allocated,
                cpus_idle=cpus_idle,
                cpus_total=cpus_total,
                memory_mb=memory_mb,
                gpus=gpus,
                node_list=node_list
            )
            partitions.append(partition)
        except (ValueError, IndexError) as e:
            # Skip malformed lines
            continue

    return partitions


def parse_squeue_output(output: str) -> List[Tuple[str, str, str, int, int]]:
    """Parse squeue output to extract job information.

    Expected format (pipe-delimited):
    JobID|Partition|State|NumNodes|NumCPUs

    Args:
        output: Output from squeue command

    Returns:
        List of (partition, state, num_nodes, num_cpus) tuples
    """
    jobs = []
    lines = output.strip().split('\n')

    for line in lines:
        if not line.strip():
            continue

        parts = line.split('|')
        if len(parts) < 5:
            continue

        try:
            job_id = parts[0].strip()
            partition = parts[1].strip()
            state = parts[2].strip()
            num_nodes = int(parts[3].strip())
            num_cpus = int(parts[4].strip())

            jobs.append((partition, state, num_nodes, num_cpus))
        except (ValueError, IndexError):
            continue

    return jobs


def aggregate_queue_stats(jobs: List[Tuple[str, str, int, int]]) -> List[QueueInfo]:
    """Aggregate job information by partition and state.

    Args:
        jobs: List of (partition, state, num_nodes, num_cpus) tuples

    Returns:
        List of QueueInfo objects
    """
    stats_by_partition = {}

    for partition, state, num_nodes, num_cpus in jobs:
        if partition not in stats_by_partition:
            stats_by_partition[partition] = {
                'pending_jobs': 0,
                'running_jobs': 0,
                'pending_cpus': 0,
                'running_cpus': 0
            }

        stats = stats_by_partition[partition]
        if state == "PENDING":
            stats['pending_jobs'] += 1
            stats['pending_cpus'] += num_cpus
        elif state == "RUNNING":
            stats['running_jobs'] += 1
            stats['running_cpus'] += num_cpus

    # Convert to QueueInfo objects
    queue_infos = []
    for partition, stats in sorted(stats_by_partition.items()):
        queue_infos.append(QueueInfo(
            partition=partition,
            pending_jobs=stats['pending_jobs'],
            running_jobs=stats['running_jobs'],
            pending_cpus=stats['pending_cpus'],
            running_cpus=stats['running_cpus']
        ))

    return queue_infos


def aggregate_partitions_by_hardware(partitions: List[PartitionInfo]) -> List[PartitionGroup]:
    """Aggregate partitions by name, hardware type, and node state.

    Groups partitions hierarchically:
    1. By partition name
    2. By hardware (GPU type + CPUs per node + Memory per node)
    3. By node state (idle/allocated/mixed/down)

    Args:
        partitions: List of PartitionInfo from sinfo

    Returns:
        List of PartitionGroup objects (aggregated view)
    """
    # Group by partition name first
    by_partition = {}
    for partition in partitions:
        if partition.name not in by_partition:
            by_partition[partition.name] = {
                'availability': partition.availability,
                'time_limit': partition.time_limit,
                'hardware_specs': {}  # Will group by hardware
            }

        partition_data = by_partition[partition.name]

        # Calculate CPUs per node
        cpus_per_node = partition.cpus_total // partition.num_nodes if partition.num_nodes > 0 else 0

        # Create hardware signature (type + cpus + memory)
        hw_sig = (partition.gpus, cpus_per_node, partition.memory_mb)

        if hw_sig not in partition_data['hardware_specs']:
            partition_data['hardware_specs'][hw_sig] = {
                'gpu_type': partition.gpus,
                'cpus_per_node': cpus_per_node,
                'memory_mb': partition.memory_mb,
                'node_states': {}
            }

        hw_data = partition_data['hardware_specs'][hw_sig]

        # Store node state info
        hw_data['node_states'][partition.node_state] = (
            partition.num_nodes,
            partition.cpus_allocated,
            partition.cpus_idle
        )

    # Convert to PartitionGroup objects
    groups = []
    for partition_name, partition_data in sorted(by_partition.items()):
        hardware_groups = []

        for hw_sig, hw_data in sorted(partition_data['hardware_specs'].items()):
            hw_group = HardwareGroup(
                gpu_type=hw_data['gpu_type'],
                cpus_per_node=hw_data['cpus_per_node'],
                memory_mb=hw_data['memory_mb'],
                node_states=hw_data['node_states']
            )
            hardware_groups.append(hw_group)

        partition_group = PartitionGroup(
            name=partition_name,
            availability=partition_data['availability'],
            time_limit=partition_data['time_limit'],
            hardware_groups=hardware_groups
        )
        groups.append(partition_group)

    return groups


def parse_memory_value(memory_str: str) -> int:
    """Parse memory value with unit suffix to MB.

    Args:
        memory_str: Memory value like "256", "64G", "2T", etc.

    Returns:
        Memory in MB as integer
    """
    memory_str = memory_str.strip().upper()

    # If no suffix, assume MB
    if memory_str.isdigit():
        return int(memory_str)

    # Extract number and unit
    value_str = ''.join(c for c in memory_str if c.isdigit() or c == '.')
    if not value_str:
        return 0

    value = float(value_str)

    # Determine unit
    if 'T' in memory_str:
        return int(value * 1024 * 1024)  # TB to MB
    elif 'G' in memory_str:
        return int(value * 1024)  # GB to MB
    elif 'M' in memory_str:
        return int(value)  # Already in MB
    elif 'K' in memory_str:
        return int(value / 1024)  # KB to MB
    else:
        return int(value)  # Default to MB


def format_memory(memory_mb: int) -> str:
    """Format memory in MB to human-readable string.

    Args:
        memory_mb: Memory in MB

    Returns:
        Formatted memory string (e.g., "256GB")
    """
    if memory_mb >= 1024:
        gb = memory_mb // 1024
        return f"{gb}GB"
    else:
        return f"{memory_mb}MB"


def get_cluster_resources(
    host: str,
    user: str,
    partition_filter: Optional[str] = None,
    verbose: bool = False
) -> ClusterResourceInfo:
    """Query cluster resources and queue status.

    Args:
        host: SSH host
        user: SSH user
        partition_filter: Optional partition name to filter by
        verbose: Show verbose output

    Returns:
        ClusterResourceInfo object with partition and queue data

    Raises:
        SSHError: If SSH command fails
    """
    # Query partition and node information
    sinfo_cmd = 'sinfo --format="%P|%a|%l|%D|%T|%C|%m|%G|%N" --noheader'
    returncode, sinfo_output = execute_remote_command(host, user, sinfo_cmd, verbose=verbose)

    if returncode != 0:
        if "not found" in sinfo_output or "command not found" in sinfo_output:
            raise SSHError("sinfo command not found on cluster. SLURM may not be installed.")
        raise SSHError(f"Failed to query sinfo: {sinfo_output}")

    partitions = parse_sinfo_output(sinfo_output)

    # Filter by partition if specified
    if partition_filter:
        partitions = [p for p in partitions if p.name == partition_filter]

    # Query queue status
    squeue_cmd = 'squeue --format="%i|%P|%T|%D|%C" --noheader --all'
    returncode, squeue_output = execute_remote_command(host, user, squeue_cmd, verbose=verbose)

    if returncode != 0:
        if "not found" in squeue_output or "command not found" in squeue_output:
            raise SSHError("squeue command not found on cluster. SLURM may not be installed.")
        raise SSHError(f"Failed to query squeue: {squeue_output}")

    jobs = parse_squeue_output(squeue_output)
    queue_stats = aggregate_queue_stats(jobs)

    # Filter queue stats to match partitions
    if partition_filter:
        queue_stats = [q for q in queue_stats if q.partition == partition_filter]

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Aggregate partitions by hardware specs and node states
    partition_groups = aggregate_partitions_by_hardware(partitions)

    return ClusterResourceInfo(
        partitions=partitions,
        partition_groups=partition_groups,
        queue_stats=queue_stats,
        timestamp=timestamp
    )


def format_partition_card(
    partition_group: PartitionGroup,
    queue_info: Optional[QueueInfo] = None,
    width: int = 56
) -> List[str]:
    """Format a single partition as a card.

    Args:
        partition_group: PartitionGroup object
        queue_info: QueueInfo for this partition (optional)
        width: Card width in characters

    Returns:
        List of formatted lines for the card
    """
    lines = []

    # Calculate totals
    total_nodes = sum(nodes for hw in partition_group.hardware_groups for nodes, _, _ in hw.node_states.values())
    total_cpus = sum(alloc + idle for hw in partition_group.hardware_groups for _, alloc, idle in hw.node_states.values())
    total_alloc = sum(alloc for hw in partition_group.hardware_groups for _, alloc, _ in hw.node_states.values())
    total_idle = sum(idle for hw in partition_group.hardware_groups for _, _, idle in hw.node_states.values())
    utilization = (total_alloc / total_cpus * 100) if total_cpus > 0 else 0

    pending = queue_info.pending_jobs if queue_info else 0
    running = queue_info.running_jobs if queue_info else 0

    # Top border
    lines.append("┌─ " + partition_group.name + " " + "─" * (width - len(partition_group.name) - 4) + "┐")

    # Header info
    status_line = f"│ Status: {partition_group.availability} ({partition_group.time_limit})"
    lines.append(status_line.ljust(width + 1) + "│")

    resources_line = f"│ Nodes: {total_nodes} | CPUs: {total_cpus} ({total_alloc} alloc, {total_idle} idle, {utilization:.0f}%)"
    lines.append(resources_line.ljust(width + 1) + "│")

    queue_line = f"│ Queue: {pending} pending | {running} running"
    lines.append(queue_line.ljust(width + 1) + "│")

    # Separator
    lines.append("├" + "─" * width + "┤")

    # Hardware breakdown
    for hw_group in partition_group.hardware_groups:
        hw_total_nodes = sum(nodes for nodes, _, _ in hw_group.node_states.values())
        hw_total_alloc = sum(alloc for _, alloc, _ in hw_group.node_states.values())
        hw_total_idle = sum(idle for _, _, idle in hw_group.node_states.values())

        # Format GPU/hardware type name
        if hw_group.gpu_type == "N/A":
            hw_name = "CPU-only"
        else:
            # Extract GPU type (e.g., "gpu:h100:4(S:0-1)" -> "H100 (4-GPU)")
            hw_display = hw_group.gpu_type.replace("gpu:", "").replace("(S:0-1)", "").replace("(S:0)", "").upper()
            hw_name = hw_display

        # Format the line: "H100 (4-GPU)      3 nodes  154 CPU alloc / 230 idle"
        node_word = "node" if hw_total_nodes == 1 else "nodes"
        hw_line = f"│ {hw_name:<20} {hw_total_nodes:2} {node_word}  {hw_total_alloc:3} alloc / {hw_total_idle:3} idle"
        lines.append(hw_line.ljust(width + 1) + "│")

    # Bottom border
    lines.append("└" + "─" * width + "┘")

    return lines


def format_compact_info(info: ClusterResourceInfo, host: str, user: str) -> str:
    """Format cluster info in compact card view.

    Args:
        info: ClusterResourceInfo object
        host: Cluster host
        user: Cluster user

    Returns:
        Formatted string for display
    """
    if not info.partition_groups:
        return "No partitions found"

    lines = []

    # Header with summary
    total_nodes = sum(p.num_nodes for p in info.partitions)
    total_cpus = sum(p.cpus_total for p in info.partitions)
    total_allocated = sum(p.cpus_allocated for p in info.partitions)
    utilization = (total_allocated / total_cpus * 100) if total_cpus > 0 else 0

    header = f"{host} - {len(info.partition_groups)} partitions, {total_nodes} nodes, {total_cpus} CPUs ({utilization:.0f}% utilized)"
    lines.append(header)
    lines.append("")

    # One card per partition group
    for partition_group in info.partition_groups:
        queue_info = next((q for q in info.queue_stats if q.partition == partition_group.name), None)
        card_lines = format_partition_card(partition_group, queue_info, width=56)
        lines.extend(card_lines)
        lines.append("")

    return "\n".join(lines)


def format_detailed_info(info: ClusterResourceInfo, host: str, user: str) -> str:
    """Format cluster info in detailed card format with node state breakdown.

    Args:
        info: ClusterResourceInfo object
        host: Cluster host
        user: Cluster user

    Returns:
        Formatted string for display
    """
    if not info.partition_groups:
        return "No partitions found"

    lines = []

    # Header
    lines.append(f"Cluster: {host} ({user}@{host})")
    lines.append(f"Queried: {info.timestamp}")
    lines.append("")

    # One detailed card per partition
    for partition_group in info.partition_groups:
        queue_info = next((q for q in info.queue_stats if q.partition == partition_group.name), None)
        card_lines = format_partition_card(partition_group, queue_info, width=80)
        lines.extend(card_lines)

        # Add node state breakdown for each hardware type
        for hw_group in partition_group.hardware_groups:
            if len(hw_group.node_states) > 1 or any(state != "idle" for state in hw_group.node_states.keys()):
                # Show breakdown only if there are multiple states
                hw_name = hw_group.gpu_type if hw_group.gpu_type != "N/A" else "CPU-only"
                lines.append(f"  Detailed ({hw_name}):")

                for state, (nodes, alloc_cpus, idle_cpus) in sorted(hw_group.node_states.items()):
                    state_line = f"    {state:12} {nodes:2} nodes  {alloc_cpus:4} CPUs alloc / {idle_cpus:4} idle"
                    lines.append(state_line)

        lines.append("")

    # Resource summary
    total_nodes = sum(p.num_nodes for p in info.partitions)
    total_cpus = sum(p.cpus_total for p in info.partitions)
    total_allocated = sum(p.cpus_allocated for p in info.partitions)
    total_idle = sum(p.cpus_idle for p in info.partitions)
    utilization = (total_allocated / total_cpus * 100) if total_cpus > 0 else 0

    total_pending = sum(q.pending_jobs for q in info.queue_stats)
    total_running = sum(q.running_jobs for q in info.queue_stats)

    lines.append("RESOURCE SUMMARY")
    lines.append("─" * 60)
    lines.append(f"Total Partitions: {len(info.partition_groups)}")
    lines.append(f"Total Nodes: {total_nodes}")
    lines.append(f"Total CPUs: {total_cpus} ({total_allocated} allocated, {total_idle} idle, {utilization:.1f}% utilization)")
    lines.append(f"Queued Jobs: {total_pending} pending, {total_running} running")

    return "\n".join(lines)


def format_verbose_info(info: ClusterResourceInfo, host: str, user: str) -> str:
    """Format cluster info with verbose partition and hardware details.

    Args:
        info: ClusterResourceInfo object
        host: Cluster host
        user: Cluster user

    Returns:
        Formatted string for display
    """
    if not info.partition_groups:
        return "No partitions found"

    lines = []

    # Header
    lines.append(f"Cluster: {host} ({user}@{host})")
    lines.append(f"Queried: {info.timestamp}")
    lines.append("")

    # One detailed card per partition with full hardware breakdown
    for partition_group in info.partition_groups:
        queue_info = next((q for q in info.queue_stats if q.partition == partition_group.name), None)
        card_lines = format_partition_card(partition_group, queue_info, width=80)
        lines.extend(card_lines)

        # Add comprehensive hardware and node state breakdown
        for hw_group in partition_group.hardware_groups:
            memory_str = format_memory(hw_group.memory_mb)
            hw_display = hw_group.gpu_type if hw_group.gpu_type != "N/A" else "CPU-only"

            lines.append(f"  Hardware: {hw_display}")
            lines.append(f"    CPUs per node: {hw_group.cpus_per_node}")
            lines.append(f"    Memory per node: {memory_str}")

            for state, (nodes, alloc_cpus, idle_cpus) in sorted(hw_group.node_states.items()):
                util = (alloc_cpus / (alloc_cpus + idle_cpus) * 100) if (alloc_cpus + idle_cpus) > 0 else 0
                state_line = f"    {state.upper():12} - {nodes:2} nodes ({alloc_cpus:4} CPU alloc, {idle_cpus:4} idle, {util:.0f}% util)"
                lines.append(state_line)

        lines.append("")

    # Resource summary
    total_nodes = sum(p.num_nodes for p in info.partitions)
    total_cpus = sum(p.cpus_total for p in info.partitions)
    total_allocated = sum(p.cpus_allocated for p in info.partitions)
    total_idle = sum(p.cpus_idle for p in info.partitions)
    utilization = (total_allocated / total_cpus * 100) if total_cpus > 0 else 0

    total_pending = sum(q.pending_jobs for q in info.queue_stats)
    total_running = sum(q.running_jobs for q in info.queue_stats)

    lines.append("RESOURCE SUMMARY")
    lines.append("─" * 60)
    lines.append(f"Total Partitions: {len(info.partition_groups)}")
    lines.append(f"Total Nodes: {total_nodes}")
    lines.append(f"Total CPUs: {total_cpus} ({total_allocated} allocated, {total_idle} idle, {utilization:.1f}% utilization)")
    lines.append(f"Queued Jobs: {total_pending} pending, {total_running} running")

    return "\n".join(lines)
