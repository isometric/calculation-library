# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

import numpy as np
import pytest

from ..weathering_signal import (
    SignificanceTestResult,
    check_weathering_significance,
    infer_post_application_concentrations,
)


def test_infer_post_application_applies_mixing_formula() -> None:
    """C_post = (C_bl + m * C_feed) / (1 + m) where m = app_rate / (BD * depth * 100)."""
    baseline_concentrations_mg_kg = np.array([100.0, 120.0, 110.0])
    feedstock_concentration_mg_kg = 5000.0
    application_rate_kg_ha = 15000.0
    bulk_density_kg_m3 = 1000.0
    depth_cm = 30.0

    result = infer_post_application_concentrations(
        baseline_concentrations_mg_kg=baseline_concentrations_mg_kg,
        feedstock_concentration_mg_kg=feedstock_concentration_mg_kg,
        application_rate_kg_ha=application_rate_kg_ha,
        bulk_density_kg_m3=bulk_density_kg_m3,
        depth_cm=depth_cm,
    )

    m = application_rate_kg_ha / (bulk_density_kg_m3 * depth_cm * 100)
    expected = (baseline_concentrations_mg_kg + m * feedstock_concentration_mg_kg) / (1 + m)
    np.testing.assert_allclose(result, expected)


def test_infer_post_application_zero_app_rate_returns_baseline() -> None:
    baseline_concentrations_mg_kg = np.array([100.0, 120.0])
    result = infer_post_application_concentrations(
        baseline_concentrations_mg_kg=baseline_concentrations_mg_kg,
        feedstock_concentration_mg_kg=5000.0,
        application_rate_kg_ha=0.0,
        bulk_density_kg_m3=1000.0,
        depth_cm=30.0,
    )
    np.testing.assert_allclose(result, baseline_concentrations_mg_kg)


def test_significance_detects_clear_decrease() -> None:
    """Large decrease from post-application to end-of-rp should be significant."""
    rng = np.random.default_rng(42)
    post_application_mg_kg = rng.normal(loc=150.0, scale=10.0, size=50)
    end_of_reporting_period_mg_kg = rng.normal(loc=100.0, scale=10.0, size=50)

    result = check_weathering_significance(
        post_application_concentrations_mg_kg=post_application_mg_kg,
        end_of_reporting_period_concentrations_mg_kg=end_of_reporting_period_mg_kg,
    )

    assert result == SignificanceTestResult(
        test_name="welch_t_test",
        statistic=result.statistic,
        p_value=result.p_value,
        significant=True,
        significance_level=0.05,
        n_post_application=50,
        n_end_of_reporting_period=50,
    )
    assert result.p_value < 0.05
    assert result.statistic > 0


def test_significance_no_difference_is_not_significant() -> None:
    """Identical distributions should not show significance."""
    rng = np.random.default_rng(42)
    values_mg_kg = rng.normal(loc=100.0, scale=10.0, size=50)

    result = check_weathering_significance(
        post_application_concentrations_mg_kg=values_mg_kg,
        end_of_reporting_period_concentrations_mg_kg=values_mg_kg,
    )

    assert result == SignificanceTestResult(
        test_name=result.test_name,
        statistic=result.statistic,
        p_value=result.p_value,
        significant=False,
        significance_level=0.05,
        n_post_application=50,
        n_end_of_reporting_period=50,
    )
    assert result.p_value >= 0.05


def test_significance_uses_mann_whitney_for_non_normal() -> None:
    """Skewed distributions should trigger Mann-Whitney U instead of Welch's t-test."""
    rng = np.random.default_rng(42)
    post_application_mg_kg = rng.exponential(scale=50.0, size=100) + 100
    end_of_reporting_period_mg_kg = rng.exponential(scale=50.0, size=100) + 50

    result = check_weathering_significance(
        post_application_concentrations_mg_kg=post_application_mg_kg,
        end_of_reporting_period_concentrations_mg_kg=end_of_reporting_period_mg_kg,
    )

    assert result.test_name == "mann_whitney_u"


def test_significance_uses_welch_for_normal() -> None:
    """Normal distributions should use Welch's t-test."""
    rng = np.random.default_rng(42)
    post_application_mg_kg = rng.normal(loc=150.0, scale=10.0, size=100)
    end_of_reporting_period_mg_kg = rng.normal(loc=100.0, scale=10.0, size=100)

    result = check_weathering_significance(
        post_application_concentrations_mg_kg=post_application_mg_kg,
        end_of_reporting_period_concentrations_mg_kg=end_of_reporting_period_mg_kg,
    )

    assert result.test_name == "welch_t_test"


def test_significance_custom_significance_level() -> None:
    rng = np.random.default_rng(42)
    post_application_mg_kg = rng.normal(loc=105.0, scale=10.0, size=30)
    end_of_reporting_period_mg_kg = rng.normal(loc=100.0, scale=10.0, size=30)

    result_loose = check_weathering_significance(
        post_application_concentrations_mg_kg=post_application_mg_kg,
        end_of_reporting_period_concentrations_mg_kg=end_of_reporting_period_mg_kg,
        significance_level=0.5,
    )
    result_strict = check_weathering_significance(
        post_application_concentrations_mg_kg=post_application_mg_kg,
        end_of_reporting_period_concentrations_mg_kg=end_of_reporting_period_mg_kg,
        significance_level=0.001,
    )

    assert result_loose.significance_level == pytest.approx(0.5)
    assert result_strict.significance_level == pytest.approx(0.001)
    assert result_loose.significant
    assert not result_strict.significant
