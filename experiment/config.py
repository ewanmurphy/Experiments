"""Configuration loader for experiments."""

import csv
import itertools
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# Reserved field names that are metadata, not experiment parameters
RESERVED_FIELDS = {"post_process_script", "script"}


def is_range_spec(value: Any) -> bool:
    """Check if value is a range specification dict.

    A range spec must be a dict with 'start', 'end', and 'divisions' keys.

    Args:
        value: Value to check

    Returns:
        True if value is a valid range specification dict
    """
    if not isinstance(value, dict):
        return False
    return "start" in value and "end" in value and "divisions" in value


def expand_range(start: float, end: float, divisions: int) -> List[Any]:
    """Generate evenly-spaced points from start to end (inclusive).

    Generates a list of evenly-spaced numeric values. If all generated values
    are whole numbers (within floating point precision), returns integers.
    Otherwise returns floats.

    Args:
        start: Starting value
        end: Ending value (inclusive)
        divisions: Number of points to generate

    Returns:
        List of values (int or float), as integers if all values are whole numbers

    Raises:
        ValueError: If divisions < 1
    """
    if divisions < 1:
        raise ValueError(f"divisions must be >= 1, got {divisions}")

    if divisions == 1:
        return [start]

    # Generate evenly-spaced values
    step = (end - start) / (divisions - 1)
    values = [start + i * step for i in range(divisions)]

    # Convert to integers if all values are whole numbers (within floating point tolerance)
    if all(abs(v - round(v)) < 1e-9 for v in values):
        return [int(round(v)) for v in values]

    return values


def expand_parameter_value(value: Any) -> List[Any]:
    """Expand a parameter value to a list.

    Handles:
    - Range specs: {start: a, end: b, divisions: n} → expanded list of evenly-spaced values
    - Lists: returned as-is
    - Scalars: wrapped in single-item list

    Args:
        value: Parameter value to expand

    Returns:
        List of parameter values

    Raises:
        ValueError: If range specification is invalid
    """
    if is_range_spec(value):
        # Validate and extract range parameters
        try:
            start = float(value["start"])
            end = float(value["end"])
            divisions = int(value["divisions"])
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid range specification: {e}")

        return expand_range(start, end, divisions)
    elif isinstance(value, list):
        return value
    else:
        return [value]


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
            # Convert strings to appropriate types
            converted_row = {}
            for key, value in row.items():
                if value is None or value == "":
                    converted_row[key] = value
                else:
                    # Try to convert to boolean
                    if value.lower() in ("true", "false", "yes", "no", "on", "off"):
                        converted_row[key] = value.lower() in ("true", "yes", "on")
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

    # Expand parameter values (ranges, lists, scalars) and exclude reserved fields
    param_lists: Dict[str, List[Any]] = {}
    for key, value in params.items():
        if key in RESERVED_FIELDS:
            continue
        param_lists[key] = expand_parameter_value(value)

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
