"""Configuration loader for experiments."""

import csv
import itertools
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# Reserved field names that are metadata, not experiment parameters
RESERVED_FIELDS = {"post_process_script", "script"}


def extract_metadata(yaml_path: str) -> Dict[str, Any]:
    """Extract metadata fields from YAML config.

    Metadata fields are reserved fields like 'post_process_script' that are
    not part of the experiment parameter grid.

    Args:
        yaml_path: Path to YAML file

    Returns:
        Dictionary of metadata fields and their values
    """
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")

    with open(path) as f:
        data = yaml.safe_load(f) or {}

    metadata = {}
    for field in RESERVED_FIELDS:
        if field in data:
            metadata[field] = data[field]

    return metadata


def yaml_to_csv(config_path: str) -> Tuple[Optional[str], Dict[str, Any]]:
    """Convert YAML config to CSV file. All YAML files are treated as parameter specs.

    Args:
        config_path: Path to YAML file

    Returns:
        Tuple of (path to generated CSV file or None, metadata dictionary)

    Raises:
        FileNotFoundError: If config file does not exist
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Only convert YAML files
    if path.suffix not in (".yaml", ".yml"):
        return None, {}

    # Extract metadata
    metadata = extract_metadata(str(path))

    # Generate CSV from YAML
    output_csv = path.parent / f"{path.stem}_generated.csv"
    generate_csv_from_yaml(str(path), str(output_csv))
    return str(output_csv), metadata


def load_csv(csv_path: str) -> List[Dict[str, Any]]:
    """Load experiments from CSV file.

    Each row represents one experiment with columns as parameter names.

    Args:
        csv_path: Path to CSV file

    Returns:
        List of dictionaries, one per row

    Raises:
        FileNotFoundError: If CSV file does not exist
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    experiments = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric strings to appropriate types
            converted_row = {}
            for key, value in row.items():
                if value is None or value == "":
                    converted_row[key] = value
                else:
                    # Try to convert to int
                    try:
                        converted_row[key] = int(value)
                    except ValueError:
                        # Try to convert to float
                        try:
                            converted_row[key] = float(value)
                        except ValueError:
                            # Keep as string
                            converted_row[key] = value
            experiments.append(converted_row)

    return experiments


def generate_csv_from_yaml(yaml_path: str, output_csv: str) -> int:
    """Generate CSV file from YAML with parameter ranges.

    Each parameter value can be a scalar or list. Generates a CSV with
    all combinations of parameters (Cartesian product).

    Skips reserved fields like 'post_process_script' which are metadata, not parameters.

    Args:
        yaml_path: Path to YAML file with parameter ranges
        output_csv: Path to output CSV file

    Returns:
        Number of experiment combinations generated

    Raises:
        FileNotFoundError: If YAML file does not exist
    """
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")

    # Load YAML
    with open(path) as f:
        params = yaml.safe_load(f) or {}

    if not params:
        raise ValueError("YAML file is empty")

    # Convert scalar values to lists, excluding reserved fields
    param_lists: Dict[str, List[Any]] = {}
    for key, value in params.items():
        if key in RESERVED_FIELDS:
            continue
        if isinstance(value, list):
            param_lists[key] = value
        else:
            param_lists[key] = [value]

    if not param_lists:
        raise ValueError("YAML file contains only metadata fields, no parameters")

    # Generate all combinations
    keys = list(param_lists.keys())
    values = [param_lists[k] for k in keys]
    combinations = list(itertools.product(*values))

    # Write to CSV
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for combo in combinations:
            row = dict(zip(keys, combo))
            writer.writerow(row)

    return len(combinations)


def merge_params(config: Dict[str, Any], cli_params: Dict[str, str]) -> Dict[str, Any]:
    """Merge config file parameters with CLI overrides.

    CLI parameters override config file parameters.

    Args:
        config: Parameters from config file
        cli_params: Parameters from command line (key=value format)

    Returns:
        Merged parameters dictionary
    """
    merged = config.copy()

    for param in cli_params:
        if "=" not in param:
            raise ValueError(f"Invalid parameter format: {param}. Use key=value")
        key, value = param.split("=", 1)
        merged[key] = value

    return merged
