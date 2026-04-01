"""Tests for experiment configuration module."""

import csv
import tempfile
from pathlib import Path

import pytest
import yaml

from experiment.config import (
    expand_parameter_value,
    expand_range,
    generate_csv_from_yaml,
    is_range_spec,
    load_csv,
)


class TestIsRangeSpec:
    """Tests for is_range_spec() function."""

    def test_valid_range_spec(self):
        """Should return True for dict with start, end, divisions."""
        spec = {"start": 0, "end": 100, "divisions": 5}
        assert is_range_spec(spec)

    def test_missing_start(self):
        """Should return False if start is missing."""
        spec = {"end": 100, "divisions": 5}
        assert not is_range_spec(spec)

    def test_missing_end(self):
        """Should return False if end is missing."""
        spec = {"start": 0, "divisions": 5}
        assert not is_range_spec(spec)

    def test_missing_divisions(self):
        """Should return False if divisions is missing."""
        spec = {"start": 0, "end": 100}
        assert not is_range_spec(spec)

    def test_non_dict_value(self):
        """Should return False for non-dict values."""
        assert not is_range_spec([0, 100, 5])
        assert not is_range_spec("0:100:5")
        assert not is_range_spec(100)
        assert not is_range_spec(None)

    def test_extra_keys_allowed(self):
        """Should still return True if dict has extra keys."""
        spec = {"start": 0, "end": 100, "divisions": 5, "extra": "ignored"}
        assert is_range_spec(spec)


class TestExpandRange:
    """Tests for expand_range() function."""

    def test_basic_integer_range(self):
        """Should generate evenly-spaced integer values."""
        result = expand_range(0, 100, 5)
        assert result == [0, 25, 50, 75, 100]

    def test_basic_float_range(self):
        """Should generate evenly-spaced float values when needed."""
        result = expand_range(0, 1, 3)
        assert result == [0.0, 0.5, 1.0]

    def test_float_result_from_integers(self):
        """Should return floats when division results in non-integers."""
        result = expand_range(0, 100, 7)
        assert len(result) == 7
        assert result[0] == 0
        assert result[-1] == 100
        assert isinstance(result[1], float)
        # Check values are approximately correct
        assert abs(result[1] - 16.666666666666668) < 1e-9

    def test_edge_case_divisions_1(self):
        """divisions=1 should return only the start value."""
        result = expand_range(10, 100, 1)
        assert result == [10]

    def test_edge_case_divisions_2(self):
        """divisions=2 should return start and end."""
        result = expand_range(0, 100, 2)
        assert result == [0, 100]

    def test_negative_range(self):
        """Should handle negative values correctly."""
        result = expand_range(-10, 10, 5)
        assert result == [-10, -5, 0, 5, 10]

    def test_descending_range(self):
        """Should handle descending ranges (end < start)."""
        result = expand_range(100, 0, 3)
        assert result == [100, 50, 0]

    def test_single_value_range(self):
        """Should handle range where start == end."""
        result = expand_range(42, 42, 5)
        assert result == [42, 42, 42, 42, 42]

    def test_negative_divisions_raises_error(self):
        """Should raise ValueError for divisions < 1."""
        with pytest.raises(ValueError, match="divisions must be >= 1"):
            expand_range(0, 100, 0)
        with pytest.raises(ValueError, match="divisions must be >= 1"):
            expand_range(0, 100, -5)

    def test_large_range(self):
        """Should handle large ranges with many divisions."""
        result = expand_range(0, 1000, 101)
        assert len(result) == 101
        assert result[0] == 0
        assert result[-1] == 1000
        assert result[50] == 500  # Middle value should be midpoint

    def test_log_scale_basic(self):
        """Should generate logarithmically-spaced values."""
        result = expand_range(1, 1000, 4, scale="log")
        assert len(result) == 4
        assert abs(result[0] - 1.0) < 1e-9
        assert abs(result[-1] - 1000.0) < 1e-9
        # For log scale 1 to 1000 with 4 points: 1, 10, 100, 1000
        assert abs(result[1] - 10.0) < 1e-9
        assert abs(result[2] - 100.0) < 1e-9

    def test_log_scale_various_ranges(self):
        """Should handle different log ranges."""
        result = expand_range(10, 10000, 3, scale="log")
        assert len(result) == 3
        assert abs(result[0] - 10.0) < 1e-9
        assert abs(result[-1] - 10000.0) < 1e-9
        # Geometric mean: sqrt(10 * 10000) ≈ 316.23
        assert abs(result[1] - 316.22776601683796) < 1e-6

    def test_log_scale_invalid_scale_raises_error(self):
        """Should raise ValueError for invalid scale."""
        with pytest.raises(ValueError, match="scale must be 'linear' or 'log'"):
            expand_range(1, 100, 5, scale="exponential")

    def test_log_scale_negative_start_raises_error(self):
        """Should raise ValueError for negative start with log scale."""
        with pytest.raises(ValueError, match="start and end must be positive"):
            expand_range(-10, 100, 5, scale="log")

    def test_log_scale_negative_end_raises_error(self):
        """Should raise ValueError for negative end with log scale."""
        with pytest.raises(ValueError, match="start and end must be positive"):
            expand_range(1, -100, 5, scale="log")

    def test_log_scale_zero_start_raises_error(self):
        """Should raise ValueError for zero start with log scale."""
        with pytest.raises(ValueError, match="start and end must be positive"):
            expand_range(0, 100, 5, scale="log")

    def test_log_scale_single_division(self):
        """Should handle single division with log scale."""
        result = expand_range(10, 10, 1, scale="log")
        assert result == [10]

    def test_linear_scale_explicit(self):
        """Should handle explicit linear scale (backwards compatible)."""
        result_linear = expand_range(10, 100, 5, scale="linear")
        result_default = expand_range(10, 100, 5)
        assert result_linear == result_default


class TestExpandParameterValue:
    """Tests for expand_parameter_value() function."""

    def test_expand_range_spec(self):
        """Should expand range specification to list."""
        spec = {"start": 0, "end": 100, "divisions": 5}
        result = expand_parameter_value(spec)
        assert result == [0, 25, 50, 75, 100]

    def test_pass_through_list(self):
        """Should pass through lists unchanged."""
        lst = ["a", "b", "c"]
        result = expand_parameter_value(lst)
        assert result == lst

    def test_wrap_scalar_in_list(self):
        """Should wrap scalar values in a list."""
        assert expand_parameter_value(42) == [42]
        assert expand_parameter_value("hello") == ["hello"]
        assert expand_parameter_value(3.14) == [3.14]

    def test_empty_list_pass_through(self):
        """Should pass through empty lists."""
        result = expand_parameter_value([])
        assert result == []

    def test_incomplete_spec_treated_as_dict(self):
        """Should treat incomplete range specs (missing keys) as regular dicts."""
        # A dict without all three required keys should be wrapped in a list, not treated as a range
        spec = {"start": 0, "end": 100}  # Missing divisions - NOT a range spec
        result = expand_parameter_value(spec)
        # Should be wrapped as a single-item list containing the dict
        assert result == [spec]

    def test_invalid_range_spec_non_numeric_start(self):
        """Should raise ValueError if start is not numeric."""
        spec = {"start": "zero", "end": 100, "divisions": 5}
        with pytest.raises(ValueError, match="Invalid range specification"):
            expand_parameter_value(spec)

    def test_invalid_range_spec_non_numeric_end(self):
        """Should raise ValueError if end is not numeric."""
        spec = {"start": 0, "end": "hundred", "divisions": 5}
        with pytest.raises(ValueError, match="Invalid range specification"):
            expand_parameter_value(spec)

    def test_invalid_range_spec_non_integer_divisions(self):
        """Should raise ValueError if divisions is not an integer."""
        spec = {"start": 0, "end": 100, "divisions": "five"}
        with pytest.raises(ValueError, match="Invalid range specification"):
            expand_parameter_value(spec)

    def test_expand_range_spec_with_log_scale(self):
        """Should expand range spec with logarithmic scale."""
        spec = {"start": 1, "end": 1000, "divisions": 4, "scale": "log"}
        result = expand_parameter_value(spec)
        assert len(result) == 4
        assert abs(result[0] - 1.0) < 1e-9
        assert abs(result[-1] - 1000.0) < 1e-9
        assert abs(result[1] - 10.0) < 1e-9
        assert abs(result[2] - 100.0) < 1e-9

    def test_expand_range_spec_with_linear_scale_explicit(self):
        """Should expand range spec with explicit linear scale."""
        spec_explicit = {"start": 0, "end": 100, "divisions": 5, "scale": "linear"}
        spec_implicit = {"start": 0, "end": 100, "divisions": 5}
        result_explicit = expand_parameter_value(spec_explicit)
        result_implicit = expand_parameter_value(spec_implicit)
        assert result_explicit == result_implicit == [0, 25, 50, 75, 100]


class TestGenerateCsvFromYaml:
    """Tests for generate_csv_from_yaml() function."""

    def test_basic_range_generation(self):
        """Should generate CSV from YAML with range specification."""
        yaml_content = """
script: test.py
param1: {start: 0, end: 100, divisions: 5}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "config.yaml"
            yaml_path.write_text(yaml_content)

            csv_path = Path(tmpdir) / "output.csv"
            count = generate_csv_from_yaml(str(yaml_path), str(csv_path))

            assert count == 5
            rows = load_csv(str(csv_path))
            assert len(rows) == 5
            assert rows[0]["param1"] == 0
            assert rows[2]["param1"] == 50
            assert rows[4]["param1"] == 100

    def test_mixed_range_and_list(self):
        """Should generate cartesian product of range and list."""
        yaml_content = """
script: test.py
param1: {start: 0, end: 100, divisions: 3}
param2: [a, b]
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "config.yaml"
            yaml_path.write_text(yaml_content)

            csv_path = Path(tmpdir) / "output.csv"
            count = generate_csv_from_yaml(str(yaml_path), str(csv_path))

            assert count == 6  # 3 × 2
            rows = load_csv(str(csv_path))
            assert len(rows) == 6

            # Check combinations exist
            param1_values = [row["param1"] for row in rows]
            param2_values = [row["param2"] for row in rows]
            assert set(param1_values) == {0, 50, 100}
            assert set(param2_values) == {"a", "b"}

    def test_multiple_ranges(self):
        """Should generate cartesian product of multiple ranges."""
        yaml_content = """
script: test.py
param1: {start: 0, end: 10, divisions: 3}
param2: {start: 0, end: 100, divisions: 2}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "config.yaml"
            yaml_path.write_text(yaml_content)

            csv_path = Path(tmpdir) / "output.csv"
            count = generate_csv_from_yaml(str(yaml_path), str(csv_path))

            assert count == 6  # 3 × 2
            rows = load_csv(str(csv_path))
            assert len(rows) == 6

    def test_reserved_fields_not_in_parameters(self):
        """Should exclude reserved fields from parameter grid."""
        yaml_content = """
script: test.py
post_process_script: process.py
param1: [a, b]
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "config.yaml"
            yaml_path.write_text(yaml_content)

            csv_path = Path(tmpdir) / "output.csv"
            count = generate_csv_from_yaml(str(yaml_path), str(csv_path))

            assert count == 2  # Only param1, not script or post_process_script
            rows = load_csv(str(csv_path))
            assert "script" not in rows[0]
            assert "post_process_script" not in rows[0]
            assert "param1" in rows[0]

    def test_scalar_parameter(self):
        """Should handle scalar parameters."""
        yaml_content = """
script: test.py
param1: 42
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "config.yaml"
            yaml_path.write_text(yaml_content)

            csv_path = Path(tmpdir) / "output.csv"
            count = generate_csv_from_yaml(str(yaml_path), str(csv_path))

            assert count == 1
            rows = load_csv(str(csv_path))
            assert rows[0]["param1"] == 42

    def test_complex_cartesian_product(self):
        """Should generate correct cartesian product."""
        yaml_content = """
script: test.py
param1: {start: 0, end: 10, divisions: 2}
param2: [x, y, z]
param3: 42
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "config.yaml"
            yaml_path.write_text(yaml_content)

            csv_path = Path(tmpdir) / "output.csv"
            count = generate_csv_from_yaml(str(yaml_path), str(csv_path))

            # 2 (param1) × 3 (param2) × 1 (param3)
            assert count == 6
            rows = load_csv(str(csv_path))
            assert len(rows) == 6

            # All param3 values should be 42
            assert all(row["param3"] == 42 for row in rows)

    def test_float_range_in_csv(self):
        """Should correctly handle float ranges in generated CSV."""
        yaml_content = """
script: test.py
param1: {start: 0, end: 1, divisions: 3}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "config.yaml"
            yaml_path.write_text(yaml_content)

            csv_path = Path(tmpdir) / "output.csv"
            count = generate_csv_from_yaml(str(yaml_path), str(csv_path))

            assert count == 3
            rows = load_csv(str(csv_path))
            assert rows[0]["param1"] == 0.0
            assert rows[1]["param1"] == 0.5
            assert rows[2]["param1"] == 1.0

    def test_csv_file_created_with_proper_headers(self):
        """Should create CSV with proper headers."""
        yaml_content = """
script: test.py
param_a: {start: 0, end: 10, divisions: 2}
param_b: [x, y]
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "config.yaml"
            yaml_path.write_text(yaml_content)

            csv_path = Path(tmpdir) / "output.csv"
            generate_csv_from_yaml(str(yaml_path), str(csv_path))

            # Read CSV and verify headers
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames
                assert set(headers) == {"param_a", "param_b"}


class TestIntegration:
    """Integration tests with actual YAML and CSV files."""

    def test_example_config_format(self):
        """Should handle realistic example configuration."""
        yaml_content = """
script: test_script.py
learning_rate: {start: 0.001, end: 0.1, divisions: 4}
batch_size: [16, 32, 64]
dropout: 0.5
epochs: 100
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "config.yaml"
            yaml_path.write_text(yaml_content)

            csv_path = Path(tmpdir) / "output.csv"
            count = generate_csv_from_yaml(str(yaml_path), str(csv_path))

            # 4 learning rates × 3 batch sizes = 12 combinations
            assert count == 12

            rows = load_csv(str(csv_path))
            assert len(rows) == 12

            # Verify all combinations exist
            learning_rates = [0.001, 0.034, 0.067, 0.1]  # start=0.001, end=0.1, divisions=4
            for row in rows:
                assert row["dropout"] == 0.5
                assert row["epochs"] == 100
                # Use approximate comparison for float values
                assert any(abs(row["learning_rate"] - lr) < 1e-3 for lr in learning_rates)
                assert row["batch_size"] in [16, 32, 64]

    def test_backward_compatibility_with_old_format(self):
        """Should still work with old list-only format."""
        yaml_content = """
script: test.py
param1: [a, b, c]
param2: [1, 2]
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "config.yaml"
            yaml_path.write_text(yaml_content)

            csv_path = Path(tmpdir) / "output.csv"
            count = generate_csv_from_yaml(str(yaml_path), str(csv_path))

            assert count == 6  # 3 × 2
            rows = load_csv(str(csv_path))
            assert len(rows) == 6


class TestBooleanConversion:
    """Tests for boolean type conversion in load_csv."""

    def test_boolean_true_variants(self):
        """Should convert various true representations to True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test.csv"
            with open(csv_path, "w", newline="") as f:
                f.write("enabled\n")
                f.write("True\n")
                f.write("true\n")
                f.write("yes\n")
                f.write("on\n")

            rows = load_csv(str(csv_path))
            assert len(rows) == 4
            assert all(row["enabled"] is True for row in rows)

    def test_boolean_false_variants(self):
        """Should convert various false representations to False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test.csv"
            with open(csv_path, "w", newline="") as f:
                f.write("enabled\n")
                f.write("False\n")
                f.write("false\n")
                f.write("no\n")
                f.write("off\n")

            rows = load_csv(str(csv_path))
            assert len(rows) == 4
            assert all(row["enabled"] is False for row in rows)

    def test_boolean_with_other_types(self):
        """Should convert mixed types correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test.csv"
            with open(csv_path, "w", newline="") as f:
                f.write("enabled,count,name,rate\n")
                f.write("true,5,test,0.5\n")
                f.write("false,10,model,0.1\n")

            rows = load_csv(str(csv_path))
            assert len(rows) == 2

            # First row
            assert rows[0]["enabled"] is True
            assert rows[0]["count"] == 5
            assert rows[0]["name"] == "test"
            assert rows[0]["rate"] == 0.5

            # Second row
            assert rows[1]["enabled"] is False
            assert rows[1]["count"] == 10
            assert rows[1]["name"] == "model"
            assert rows[1]["rate"] == 0.1

    def test_boolean_case_insensitive(self):
        """Should handle mixed case boolean values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test.csv"
            with open(csv_path, "w", newline="") as f:
                f.write("flag\n")
                f.write("True\n")
                f.write("FALSE\n")
                f.write("Yes\n")
                f.write("NO\n")

            rows = load_csv(str(csv_path))
            assert rows[0]["flag"] is True
            assert rows[1]["flag"] is False
            assert rows[2]["flag"] is True
            assert rows[3]["flag"] is False

    def test_non_boolean_strings_not_converted(self):
        """Should not convert strings that look similar to booleans."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test.csv"
            with open(csv_path, "w", newline="") as f:
                f.write("value\n")
                f.write("truee\n")
                f.write("falsee\n")

            rows = load_csv(str(csv_path))
            assert rows[0]["value"] == "truee"
            assert rows[1]["value"] == "falsee"

    def test_empty_values_unchanged(self):
        """Should keep empty values unchanged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test.csv"
            with open(csv_path, "w", newline="") as f:
                f.write("flag,count\n")
                f.write("true,\n")
                f.write(",5\n")

            rows = load_csv(str(csv_path))
            assert rows[0]["flag"] is True
            assert rows[0]["count"] == ""
            assert rows[1]["flag"] == ""
            assert rows[1]["count"] == 5
