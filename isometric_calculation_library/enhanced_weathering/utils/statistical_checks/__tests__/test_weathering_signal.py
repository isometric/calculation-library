# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

import numpy as np
import pandas as pd
import pytest

from ..weathering_signal import (
    SignificanceTestResult,
    check_weathering_significance,
    infer_post_application_concentrations,
    run_significance_tests,
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


# ---------------------------------------------------------------------------
# run_significance_tests
# ---------------------------------------------------------------------------


def test_run_significance_tests_returns_one_row_per_element() -> None:
    """Result DataFrame should have one row per element with p-values in [0, 1]."""
    rng = np.random.default_rng(42)
    n = 30

    treat_bl = pd.DataFrame({
        "mass_fraction_ca": rng.normal(200, 10, size=n),
        "mass_fraction_mg": rng.normal(150, 8, size=n),
    })
    treat_rp = pd.DataFrame({
        "mass_fraction_ca": rng.normal(180, 10, size=n),
        "mass_fraction_mg": rng.normal(135, 8, size=n),
    })
    feedstock = pd.DataFrame({
        "mass_fraction_ca": rng.normal(50_000, 2000, size=10),
        "mass_fraction_mg": rng.normal(30_000, 1500, size=10),
    })

    result = run_significance_tests(
        treatment_baseline=treat_bl,
        treatment_reporting_period=treat_rp,
        feedstock_samples=feedstock,
        bulk_density_kg_m3=1200.0,
        application_rate_kg_ha=15_000.0,
        elements=["Ca", "Mg"],
        sampling_depth_cm=30.0,
    )

    assert set(result["cation"]) == {"Ca", "Mg"}
    assert len(result) == 2
    assert all(result["p_value"].between(0, 1))
