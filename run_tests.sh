#!/bin/bash
# Test runner for cluster integration tests

set -e

echo "================================"
echo "Cluster Integration Test Suite"
echo "================================"
echo

# Check if pytest is available
if ! command -v pytest &> /dev/null; then
    echo "Installing test dependencies..."
    pip install pytest pytest-mock pyyaml typer questionary
fi

echo "Running unit tests..."
echo "================================"

# Run unit tests
pytest tests/test_cluster_config.py -v
echo

pytest tests/test_cluster_state.py -v
echo

pytest tests/test_cluster_slurm.py -v
echo

echo "Running integration tests with mocks..."
echo "================================"

# Run integration tests (mocked, no real cluster needed)
pytest tests/test_cluster_integration.py -v
echo

echo "================================"
echo "Test Summary"
echo "================================"

# Run all tests with summary
pytest tests/test_cluster_*.py --tb=short -q

echo
echo "✓ All tests passed!"
echo
echo "Next steps:"
echo "1. Review TESTING_GUIDE.md for end-to-end testing"
echo "2. Create cluster.yaml with your SLURM cluster details"
echo "3. Run 'experiment cluster-submit --dry-run' to test"
echo "4. Submit a real job with 'experiment cluster-submit'"
