import numpy as np
import pytest
from unittest.mock import patch

from open_water_thermal_analyst.compliance import evaluate_compliance
from open_water_thermal_analyst.config import ComplianceThresholds


class TestComplianceEvaluation:
    """Test compliance evaluation functions."""

    def test_evaluate_compliance_normal_case(self):
        """Test compliance evaluation with normal temperature data."""
        # Create test temperature data
        temp_data = np.array([
            [25.0, 26.0, 27.0],
            [28.0, 29.0, 30.0],
            [31.0, 32.0, 33.0]
        ])

        thresholds = ComplianceThresholds(
            warning_celsius=28.0,
            critical_celsius=32.0,
            rolling_minutes=60
        )

        results = evaluate_compliance(temp_data, thresholds)

        # Check basic structure
        assert "stats" in results
        assert "thresholds" in results
        assert "warning_exceedances" in results
        assert "critical_exceedances" in results
        assert "warning_mask" in results
        assert "critical_mask" in results

        # Check statistics
        stats = results["stats"]
        assert stats["min"] == 25.0
        assert stats["max"] == 33.0
        assert stats["count"] == 9
        assert abs(stats["mean"] - 29.0) < 0.1

        # Check exceedances
        assert results["warning_exceedances"] == 6  # Values >= 28
        assert results["critical_exceedances"] == 2  # Values >= 32

        # Check masks
        assert results["warning_mask"].shape == temp_data.shape
        assert results["critical_mask"].shape == temp_data.shape

    def test_evaluate_compliance_no_exceedances(self):
        """Test compliance evaluation with no threshold exceedances."""
        temp_data = np.array([
            [20.0, 21.0, 22.0],
            [23.0, 24.0, 25.0],
            [26.0, 27.0, 28.0]
        ])

        thresholds = ComplianceThresholds(
            warning_celsius=30.0,
            critical_celsius=35.0,
            rolling_minutes=60
        )

        results = evaluate_compliance(temp_data, thresholds)

        assert results["warning_exceedances"] == 0
        assert results["critical_exceedances"] == 0
        assert np.sum(results["warning_mask"]) == 0
        assert np.sum(results["critical_mask"]) == 0

    def test_evaluate_compliance_all_exceedances(self):
        """Test compliance evaluation with all values exceeding thresholds."""
        temp_data = np.full((3, 3), 35.0)  # All values at 35°C

        thresholds = ComplianceThresholds(
            warning_celsius=30.0,
            critical_celsius=32.0,
            rolling_minutes=60
        )

        results = evaluate_compliance(temp_data, thresholds)

        assert results["warning_exceedances"] == 9  # All values
        assert results["critical_exceedances"] == 9  # All values
        assert np.all(results["warning_mask"])
        assert np.all(results["critical_mask"])

    def test_evaluate_compliance_with_nans(self):
        """Test compliance evaluation with NaN values."""
        temp_data = np.array([
            [25.0, np.nan, 27.0],
            [28.0, 29.0, np.nan],
            [31.0, 32.0, 33.0]
        ])

        thresholds = ComplianceThresholds(
            warning_celsius=28.0,
            critical_celsius=32.0,
            rolling_minutes=60
        )

        results = evaluate_compliance(temp_data, thresholds)

        # Should only count valid values
        assert results["stats"]["count"] == 7  # 9 total - 2 NaN

        # Statistics should ignore NaN values
        assert results["stats"]["min"] == 25.0
        assert results["stats"]["max"] == 33.0

        # Masks should be False for NaN positions
        assert not results["warning_mask"][0, 1]  # NaN position
        assert not results["warning_mask"][1, 2]  # NaN position

    def test_evaluate_compliance_empty_array(self):
        """Test compliance evaluation with empty array."""
        temp_data = np.array([])

        thresholds = ComplianceThresholds(
            warning_celsius=28.0,
            critical_celsius=32.0,
            rolling_minutes=60
        )

        results = evaluate_compliance(temp_data, thresholds)

        assert results["stats"]["count"] == 0
        assert np.isnan(results["stats"]["min"])
        assert np.isnan(results["stats"]["mean"])
        assert np.isnan(results["stats"]["max"])
        assert results["warning_exceedances"] == 0
        assert results["critical_exceedances"] == 0

    def test_evaluate_compliance_single_value(self):
        """Test compliance evaluation with single temperature value."""
        temp_data = np.array([[25.0]])

        thresholds = ComplianceThresholds(
            warning_celsius=28.0,
            critical_celsius=32.0,
            rolling_minutes=60
        )

        results = evaluate_compliance(temp_data, thresholds)

        assert results["stats"]["count"] == 1
        assert results["stats"]["min"] == 25.0
        assert results["stats"]["max"] == 25.0
        assert results["stats"]["mean"] == 25.0
        assert results["warning_exceedances"] == 0
        assert results["critical_exceedances"] == 0

    def test_thresholds_serialization(self):
        """Test that thresholds are properly serialized in results."""
        temp_data = np.array([[25.0, 30.0, 35.0]])

        thresholds = ComplianceThresholds(
            warning_celsius=28.0,
            critical_celsius=32.0,
            rolling_minutes=60
        )

        results = evaluate_compliance(temp_data, thresholds)

        serialized_thresholds = results["thresholds"]
        assert serialized_thresholds["warning_celsius"] == 28.0
        assert serialized_thresholds["critical_celsius"] == 32.0
        assert serialized_thresholds["rolling_minutes"] == 60
