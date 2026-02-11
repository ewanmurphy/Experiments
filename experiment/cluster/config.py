"""Cluster configuration loading and merging."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any
import yaml


@dataclass
class SSHConfig:
    """SSH connection configuration."""
    host: str
    user: str
    remote_base_dir: str


@dataclass
class SlurmConfig:
    """SLURM resource allocation configuration."""
    partition: str = "standard"
    time_limit: str = "01:00:00"
    memory: str = "4G"
    cpus: int = 1
    gpus: int = 0
    max_concurrent: int = 0  # 0 means no limit


@dataclass
class SyncConfig:
    """File synchronization patterns."""
    to_cluster: List[str] = field(default_factory=lambda: [
        "*.py", "*.yaml", "*.txt", "data/**", "src/**"
    ])
    from_cluster: List[str] = field(default_factory=lambda: [
        "exp_*/results.json", "exp_*/logs/**", "slurm_*.out", "slurm_*.err"
    ])


@dataclass
class ClusterConfig:
    """Complete cluster configuration."""
    ssh: SSHConfig
    slurm: SlurmConfig = field(default_factory=SlurmConfig)
    modules: List[str] = field(default_factory=list)
    environment: Optional[str] = None
    sync: SyncConfig = field(default_factory=SyncConfig)


def find_cluster_config(
    experiment_name: str,
    explicit_config_path: Optional[str] = None
) -> str:
    """Find cluster configuration file.

    Searches in this order:
    1. Explicit path (if provided via --cluster-config)
    2. Per-experiment config: experiments/{name}/cluster.yaml
    3. Project root config: ./cluster.yaml

    Args:
        experiment_name: Name of experiment
        explicit_config_path: Explicit path provided by user

    Returns:
        Path to cluster config file

    Raises:
        FileNotFoundError: If no config file found
    """
    # 1. Check explicit path first
    if explicit_config_path:
        path = Path(explicit_config_path)
        if path.exists():
            return str(path)
        raise FileNotFoundError(f"Cluster config not found at: {explicit_config_path}")

    # 2. Check per-experiment config
    cwd = Path.cwd()
    exp_cluster_config = cwd / "experiments" / experiment_name / "cluster.yaml"
    if exp_cluster_config.exists():
        return str(exp_cluster_config)

    # 3. Check project root config
    root_cluster_config = cwd / "cluster.yaml"
    if root_cluster_config.exists():
        return str(root_cluster_config)

    # Not found
    raise FileNotFoundError(
        f"No cluster config found for '{experiment_name}'.\n"
        f"Please create one of:\n"
        f"  - experiments/{experiment_name}/cluster.yaml (per-experiment)\n"
        f"  - cluster.yaml (project root)"
    )


def load_cluster_config(config_path: str) -> ClusterConfig:
    """Load cluster configuration from YAML file.

    Args:
        config_path: Path to cluster.yaml file

    Returns:
        ClusterConfig object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If required fields are missing
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Cluster config not found: {config_path}")

    with open(path) as f:
        data = yaml.safe_load(f) or {}

    # Validate required SSH fields
    ssh_data = data.get("ssh", {})
    if not ssh_data.get("host"):
        raise ValueError("cluster.yaml: ssh.host is required")
    if not ssh_data.get("user"):
        raise ValueError("cluster.yaml: ssh.user is required")
    if not ssh_data.get("remote_base_dir"):
        raise ValueError("cluster.yaml: ssh.remote_base_dir is required")

    ssh = SSHConfig(
        host=ssh_data["host"],
        user=ssh_data["user"],
        remote_base_dir=ssh_data["remote_base_dir"]
    )

    # Parse SLURM config with defaults
    slurm_data = data.get("slurm", {})
    slurm = SlurmConfig(
        partition=slurm_data.get("partition", "standard"),
        time_limit=slurm_data.get("time_limit", "01:00:00"),
        memory=slurm_data.get("memory", "4G"),
        cpus=slurm_data.get("cpus", 1),
        gpus=slurm_data.get("gpus", 0),
        max_concurrent=slurm_data.get("max_concurrent", 0)
    )

    # Parse optional fields
    modules = data.get("modules", [])
    environment = data.get("environment")

    # Parse sync patterns
    sync_data = data.get("sync", {})
    sync = SyncConfig(
        to_cluster=sync_data.get("to_cluster", SyncConfig().to_cluster),
        from_cluster=sync_data.get("from_cluster", SyncConfig().from_cluster)
    )

    return ClusterConfig(
        ssh=ssh,
        slurm=slurm,
        modules=modules,
        environment=environment,
        sync=sync
    )


def merge_experiment_cluster_config(
    base: ClusterConfig,
    experiment_override: Optional[Dict[str, Any]]
) -> ClusterConfig:
    """Merge experiment-specific cluster config with base config.

    Args:
        base: Base cluster configuration from cluster.yaml
        experiment_override: Optional 'cluster' section from experiment config.yaml

    Returns:
        Merged ClusterConfig
    """
    if not experiment_override:
        return base

    # Create a copy of base config
    merged = ClusterConfig(
        ssh=base.ssh,  # SSH never overridden per-experiment
        slurm=SlurmConfig(
            partition=experiment_override.get("partition", base.slurm.partition),
            time_limit=experiment_override.get("time_limit", base.slurm.time_limit),
            memory=experiment_override.get("memory", base.slurm.memory),
            cpus=experiment_override.get("cpus", base.slurm.cpus),
            gpus=experiment_override.get("gpus", base.slurm.gpus),
            max_concurrent=experiment_override.get("max_concurrent", base.slurm.max_concurrent)
        ),
        modules=base.modules.copy(),
        environment=experiment_override.get("environment", base.environment),
        sync=SyncConfig(
            to_cluster=base.sync.to_cluster.copy(),
            from_cluster=base.sync.from_cluster.copy()
        )
    )

    # Append experiment-specific modules
    if "modules" in experiment_override:
        merged.modules.extend(experiment_override["modules"])

    # Append experiment-specific sync patterns
    if "sync" in experiment_override:
        sync_override = experiment_override["sync"]
        if "to_cluster" in sync_override:
            merged.sync.to_cluster.extend(sync_override["to_cluster"])
        if "from_cluster" in sync_override:
            merged.sync.from_cluster.extend(sync_override["from_cluster"])

    return merged
