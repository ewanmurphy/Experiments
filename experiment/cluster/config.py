"""Cluster configuration loading and merging."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import yaml


@dataclass
class SSHConfig:
    """SSH connection configuration."""
    host: str
    remote_base_dir: str
    user: Optional[str] = None  # None uses SSH config alias resolution


@dataclass
class SlurmConfig:
    """SLURM resource allocation configuration."""
    partition: Optional[str] = None  # None uses cluster default
    account: Optional[str] = None  # None uses user's default account
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


def _config_display_name(path: Path) -> str:
    """Extract display name from cluster config filename.

    Args:
        path: Path to cluster config file

    Returns:
        Display name: "default" for cluster.yaml, or the part between "cluster_" and ".yaml"
    """
    filename = path.name
    if filename == "cluster.yaml":
        return "default"
    # Extract name from cluster_[name].yaml
    if filename.startswith("cluster_") and filename.endswith(".yaml"):
        return filename[8:-5]  # Remove "cluster_" prefix and ".yaml" suffix
    return filename


def find_cluster_configs(
    experiment_name: str,
    explicit_config_path: Optional[str] = None
) -> List[Tuple[str, str]]:
    """Find all available cluster configurations.

    Returns list of (display_name, path) tuples for all available cluster configs.

    Search order (all matches returned, per-experiment before root):
    - experiments/{name}/cluster.yaml       → name "default"
    - experiments/{name}/cluster_*.yaml     → name extracted from filename
    - ./cluster.yaml                        → name "default"
    - ./cluster_*.yaml                      → name extracted from filename

    With explicit_config_path: returns only that file (or raises FileNotFoundError).

    Args:
        experiment_name: Name of experiment
        explicit_config_path: Explicit path provided by user

    Returns:
        List of (display_name, path) tuples, sorted by location and name

    Raises:
        FileNotFoundError: If explicit_config_path provided but doesn't exist
    """
    # Handle explicit path first
    if explicit_config_path:
        path = Path(explicit_config_path)
        if path.exists():
            display_name = _config_display_name(path)
            return [(display_name, str(path))]
        raise FileNotFoundError(f"Cluster config not found at: {explicit_config_path}")

    cwd = Path.cwd()
    configs: List[Tuple[str, str]] = []

    # Search per-experiment configs first
    exp_cluster_dir = cwd / "experiments" / experiment_name
    if exp_cluster_dir.exists():
        # Look for cluster.yaml
        cluster_yaml = exp_cluster_dir / "cluster.yaml"
        if cluster_yaml.exists():
            configs.append(("default", str(cluster_yaml)))

        # Look for cluster_*.yaml files
        for config_file in sorted(exp_cluster_dir.glob("cluster_*.yaml")):
            display_name = _config_display_name(config_file)
            configs.append((display_name, str(config_file)))

    # Search project root configs
    # Look for cluster.yaml
    root_cluster_yaml = cwd / "cluster.yaml"
    if root_cluster_yaml.exists():
        configs.append(("default", str(root_cluster_yaml)))

    # Look for cluster_*.yaml files
    for config_file in sorted(cwd.glob("cluster_*.yaml")):
        display_name = _config_display_name(config_file)
        configs.append((display_name, str(config_file)))

    # Remove duplicates while preserving order (per-experiment takes precedence)
    seen = set()
    unique_configs = []
    for display_name, path in configs:
        if display_name not in seen:
            unique_configs.append((display_name, path))
            seen.add(display_name)

    return unique_configs


def find_cluster_config(
    experiment_name: str,
    explicit_config_path: Optional[str] = None
) -> str:
    """Find a single cluster configuration file.

    Searches in this order:
    1. Explicit path (if provided via --cluster-config)
    2. Per-experiment configs: experiments/{name}/cluster.yaml and cluster_*.yaml
    3. Project root configs: ./cluster.yaml and cluster_*.yaml

    When multiple configs are found, raises FileNotFoundError with a list of
    available configs (to force explicit selection).

    Args:
        experiment_name: Name of experiment
        explicit_config_path: Explicit path provided by user

    Returns:
        Path to cluster config file (when exactly one found)

    Raises:
        FileNotFoundError: If no config found, or if multiple configs found
                         (includes list of available configs in message)
    """
    try:
        configs = find_cluster_configs(experiment_name, explicit_config_path)
    except FileNotFoundError:
        raise

    # No configs found
    if not configs:
        raise FileNotFoundError(
            f"No cluster config found for '{experiment_name}'.\n"
            f"Please create one of:\n"
            f"  - experiments/{experiment_name}/cluster.yaml (per-experiment)\n"
            f"  - cluster.yaml (project root)"
        )

    # One config found - return it
    if len(configs) == 1:
        return configs[0][1]

    # Multiple configs found - list them and ask for explicit selection
    config_list = "\n".join(
        f"  - {display_name}: {path}" for display_name, path in configs
    )
    raise FileNotFoundError(
        f"Multiple cluster configs found for '{experiment_name}'.\n"
        f"Available configs:\n"
        f"{config_list}\n"
        f"Use --cluster-config to specify one explicitly."
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
    if not ssh_data.get("remote_base_dir"):
        raise ValueError("cluster.yaml: ssh.remote_base_dir is required")

    ssh = SSHConfig(
        host=ssh_data["host"],
        remote_base_dir=ssh_data["remote_base_dir"],
        user=ssh_data.get("user")  # Optional: None uses SSH config alias resolution
    )

    # Parse SLURM config with defaults
    slurm_data = data.get("slurm", {})
    slurm = SlurmConfig(
        partition=slurm_data.get("partition"),  # None uses cluster default
        account=slurm_data.get("account"),  # None uses user's default account
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
            account=experiment_override.get("account", base.slurm.account),
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
