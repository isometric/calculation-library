# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

import re

import numpy as np
import pytest

from isometric_calculation_library.enhanced_weathering.utils.statistical_checks.bulk_density_change import (
    BulkDensityChangeStatus,
    BulkDensityIncreaseTest,
    check_bulk_density_increase,
    pool_bulk_density_by_group,
    resolve_bulk_density_groups,
)


def _test_with_status(status: BulkDensityChangeStatus) -> BulkDensityIncreaseTest:
    return BulkDensityIncreaseTest(
        status=status,
        test_name=None,
        statistic=None,
        p_value=None,
        n_earlier=0,
        n_later=0,
        mean_earlier_kg_m3=None,
        mean_later_kg_m3=None,
        significance_level=0.05,
    )


def test_check_bulk_density_increase_detects_a_clear_increase() -> None:
    rng = np.random.default_rng(42)
    result = check_bulk_density_increase(
        rng.normal(1_300, 20, size=30),
        rng.normal(1_400, 20, size=30),
    )

    assert result.status == "significant_increase"
    assert result.test_name == "welch_t_test"
    assert (result.n_earlier, result.n_later) == (30, 30)


def test_check_bulk_density_increase_ignores_a_decrease() -> None:
    """One-tailed: a clear decrease is not a significant increase."""
    rng = np.random.default_rng(42)
    result = check_bulk_density_increase(
        rng.normal(1_400, 20, size=30),
        rng.normal(1_300, 20, size=30),
    )

    assert result.status == "no_significant_increase"


def test_check_bulk_density_increase_is_not_testable_without_enough_measurements() -> None:
    result = check_bulk_density_increase(
        np.array([], dtype=float),
        np.array([1_400.0, 1_420.0, 1_450.0]),
    )

    assert result == BulkDensityIncreaseTest(
        status="not_testable",
        test_name=None,
        statistic=None,
        p_value=None,
        n_earlier=0,
        n_later=3,
        mean_earlier_kg_m3=None,
        mean_later_kg_m3=1_423.3333333333333,
        significance_level=0.05,
    )


def test_check_bulk_density_increase_rejects_missing_values() -> None:
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Bulk density measurements must be finite. Drop missing values before testing, "
            "so the counts reported with the test are the measurements it used.",
        ),
    ):
        check_bulk_density_increase(
            np.array([1_300.0, np.nan, 1_310.0]),
            np.array([1_400.0, 1_420.0, 1_450.0]),
        )


@pytest.mark.parametrize(
    ("baseline_to_post_application", "post_application_to_reporting_period", "expected"),
    [
        (
            "no_significant_increase",
            "no_significant_increase",
            {
                "baseline": "baseline+post_application+reporting_period",
                "post_application": "baseline+post_application+reporting_period",
                "reporting_period": "baseline+post_application+reporting_period",
            },
        ),
        (
            "significant_increase",
            "no_significant_increase",
            {
                "baseline": "baseline",
                "post_application": "post_application+reporting_period",
                "reporting_period": "post_application+reporting_period",
            },
        ),
        (
            "no_significant_increase",
            "significant_increase",
            {
                "baseline": "baseline+post_application",
                "post_application": "baseline+post_application",
                "reporting_period": "reporting_period",
            },
        ),
        (
            "significant_increase",
            "significant_increase",
            {
                "baseline": "baseline",
                "post_application": "post_application",
                "reporting_period": "reporting_period",
            },
        ),
        (
            "not_testable",
            "not_testable",
            {
                "baseline": "baseline+post_application+reporting_period",
                "post_application": "baseline+post_application+reporting_period",
                "reporting_period": "baseline+post_application+reporting_period",
            },
        ),
    ],
)
def test_resolve_bulk_density_groups(
    baseline_to_post_application: BulkDensityChangeStatus,
    post_application_to_reporting_period: BulkDensityChangeStatus,
    expected: dict[str, str],
) -> None:
    groups = resolve_bulk_density_groups(
        _test_with_status(baseline_to_post_application),
        _test_with_status(post_application_to_reporting_period),
    )

    assert groups == expected


def test_pool_bulk_density_by_group_pools_and_sorts_each_group() -> None:
    pooled = pool_bulk_density_by_group(
        {
            "baseline": np.array([1_310.0, 1_300.0]),
            "post_application": np.array([1_420.0]),
            "reporting_period": np.array([1_410.0, 1_400.0]),
        },
        {
            "baseline": "baseline",
            "post_application": "post_application+reporting_period",
            "reporting_period": "post_application+reporting_period",
        },
    )

    assert {label: values.tolist() for label, values in pooled.items()} == {
        "baseline": [1_300.0, 1_310.0],
        "post_application+reporting_period": [1_400.0, 1_410.0, 1_420.0],
    }


def test_pool_bulk_density_by_group_rejects_a_group_without_measurements() -> None:
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Bulk density group(s) ['baseline'] have no measurements, so their density is "
            "undefined.",
        ),
    ):
        pool_bulk_density_by_group(
            {
                "baseline": np.array([], dtype=float),
                "post_application": np.array([1_420.0]),
                "reporting_period": np.array([1_400.0]),
            },
            {
                "baseline": "baseline",
                "post_application": "post_application+reporting_period",
                "reporting_period": "post_application+reporting_period",
            },
        )
