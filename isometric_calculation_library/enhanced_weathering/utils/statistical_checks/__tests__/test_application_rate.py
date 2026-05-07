# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

import numpy as np

from ..application_rate import build_application_rate_check


def test_build_application_rate_check_columns() -> None:
    """DataFrame has all expected columns."""
    boot = np.random.default_rng(42).normal(15_000, 500, size=10_000)
    result = build_application_rate_check(
        soil_based_application_rate_bootstrap_replicates_kg_ha=boot,
        known_application_rate_kg_ha=15_000.0,
    )

    expected_cols = {
        "plot_type",
        "known_app_rate_t_ha",
        "soil_based_app_rate_mean_t_ha",
        "soil_based_app_rate_std_t_ha",
        "soil_based_app_rate_p5_t_ha",
        "soil_based_app_rate_p16_t_ha",
        "soil_based_app_rate_p84_t_ha",
        "soil_based_app_rate_p95_t_ha",
        "known_within_2std",
        "deviation_in_std",
    }
    assert set(result.columns) == expected_cols
    assert len(result) == 1


def test_build_application_rate_check_within_2std() -> None:
    """When actual rate matches bootstrap mean, it should be within 2 std."""
    boot = np.random.default_rng(42).normal(15_000, 500, size=10_000)
    result = build_application_rate_check(
        soil_based_application_rate_bootstrap_replicates_kg_ha=boot,
        known_application_rate_kg_ha=15_000.0,
    )

    assert result["known_within_2std"].iloc[0]


def test_build_application_rate_check_outside_2std() -> None:
    """When actual rate is far from bootstrap mean, it should be outside 2 std."""
    boot = np.full(10_000, 15_000.0) + np.random.default_rng(42).normal(0, 100, size=10_000)
    result = build_application_rate_check(
        soil_based_application_rate_bootstrap_replicates_kg_ha=boot,
        known_application_rate_kg_ha=50_000.0,
    )

    assert not result["known_within_2std"].iloc[0]


def test_build_application_rate_check_zero_std() -> None:
    """When bootstrap has zero variance, deviation_in_std should be inf."""
    boot = np.full(100, 15_000.0)
    result = build_application_rate_check(
        soil_based_application_rate_bootstrap_replicates_kg_ha=boot,
        known_application_rate_kg_ha=50_000.0,
    )

    assert result["deviation_in_std"].iloc[0] == float("inf")
    assert not result["known_within_2std"].iloc[0]
