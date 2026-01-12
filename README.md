# Experiments

A CLI tool for running, monitoring, and logging computational experiments with support for parameter ranges, parallel execution, and real-time progress tracking.

## Overview

This project provides utilities and infrastructure for:
- **Running experiments**: Execute Python scripts with various parameter combinations
- **Parameter ranges**: Specify ranges with simple syntax instead of listing all values
- **Parallel execution**: Run multiple experiments concurrently with automatic worker management
- **Real-time progress**: Monitor job completion as they finish during parallel execution
- **Logging**: Comprehensive logs and metadata for each experiment run
- **Results aggregation**: Automatic CSV summary combining parameters and results
- **Post-processing**: Optional scripts to analyze experiment results

## Features

### Range Parameter Specification
Define parameter ranges without listing all values:

```yaml
script: experiment.py
learning_rate: {start: 0.001, end: 0.1, divisions: 5}  # Generates [0.001, 0.03, 0.05, 0.07, 0.1]
batch_size: [32, 64, 128]
model: ["small", "large"]
```

### Real-Time Progress Display
During parallel execution, see experiments complete in real-time:

```
Running 10 experiments in parallel (4 workers)...
[ 1/10] Experiment  3 completed: succeeded (model=small, learning_rate=0.001000)
[ 2/10] Experiment  1 completed: succeeded (model=small, learning_rate=0.030000)
[ 3/10] Experiment  5 completed: FAILED    (model=large, learning_rate=0.070000)
```

### Aligned Output
All output columns automatically align for easy scanning:
- Completion counter and experiment numbers
- Status column (succeeded/FAILED)
- Parameter values

## Installation

```bash
pip install -e .
```

### Dependencies
- `typer>=0.9.0` - CLI framework
- `pyyaml>=6.0` - YAML config parsing
- `questionary>=2.0.0` - Interactive prompts

## Getting Started

### 1. Create an experiment configuration

Create `experiments/my_experiment/config.yaml`:

```yaml
script: my_script.py
learning_rate: {start: 0.0001, end: 0.1, divisions: 5}
batch_size: [16, 32, 64]
epochs: 100
post_process_script: analyze_results.py  # Optional
```

### 2. Create your experiment script

Create `my_script.py`:

```python
import json
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--epochs', type=int, required=True)

    args = parser.parse_args()

    # Run your experiment
    accuracy = train_model(args.learning_rate, args.batch_size, args.epochs)

    # Save results
    results = {'accuracy': accuracy}
    with open('results.json', 'w') as f:
        json.dump(results, f)

if __name__ == '__main__':
    main()
```

### 3. Run experiments

```bash
# Interactive selection
experiment run

# Run specific experiment
experiment run my_experiment

# Show parameters with real-time progress
experiment run my_experiment --show-params

# Use parallel workers
experiment run my_experiment --parallel 4

# Override parameters
experiment run my_experiment --param learning_rate=0.05
```

## Usage

```
experiment run [EXPERIMENT_NAME] [OPTIONS]

Options:
  --param TEXT                    Parameter override (key=value)
  --parallel INTEGER              Number of parallel workers (1 for sequential, 0 for auto-detect)
  --show-params                   Show parameter values when experiments complete
  --verbose                       Show detailed logging
  --timing / --no-timing          Show total execution time
```

## Project Structure

```
experiments/
├── my_experiment/
│   ├── config.yaml              # Experiment configuration
│   └── YYYY_Mon_DD_HHhMMmSSs/   # Timestamped run directories
│       ├── exp_001/
│       │   ├── logs/            # Experiment logs
│       │   └── results.json      # Experiment results
│       └── summary.csv          # Combined parameters and results
experiment/
├── config.py                    # Configuration loading and CSV generation
├── runner.py                    # Experiment execution
├── logger.py                    # Logging and metadata tracking
└── main.py                      # CLI entry point
tests/
└── test_config.py               # Configuration tests
```

## Results

Each experiment run creates:
- **Timestamped directory**: `experiments/{name}/YYYY_Mon_DD_HHhMMmSSs/`
- **Experiment subdirectories**: `exp_001`, `exp_002`, etc.
- **Logs**: `logs/{script_name}_{timestamp}.log`
- **Metadata**: `logs/{script_name}_{timestamp}.json`
- **Results**: `results.json` (written by your script)
- **Summary**: `summary.csv` combining all parameters and results

## Examples

### Example 1: Simple parameter sweep
```yaml
script: train.py
learning_rate: {start: 0.001, end: 0.1, divisions: 5}
dropout: [0.1, 0.3, 0.5]
```
Generates 5 × 3 = 15 experiments

### Example 2: With post-processing
```yaml
script: train.py
epochs: {start: 10, end: 100, divisions: 5}
model: ["resnet18", "resnet50"]
post_process_script: plot_results.py
```

The `plot_results.py` receives `summary.csv` as an argument and runs from the experiment directory.

## Testing

Run the test suite:
```bash
pytest tests/ -v
```

## License

MIT
