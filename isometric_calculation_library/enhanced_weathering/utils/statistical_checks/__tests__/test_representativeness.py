# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

import numpy as np
import pytest

from isometric_calculation_library.enhanced_weathering.utils.statistical_checks.representativeness import (
    RepresentativenessTestResult,
    check_representativeness,
)


def test_representativeness_identical_distributions_not_significant() -> None:
    """Same distribution drawn twice should not be significantly different."""
    rng = np.random.default_rng(42)
    values = rng.normal(loc=100.0, scale=10.0, size=50)

    result = check_representativeness(group_a=values, group_b=values)

    assert result == RepresentativenessTestResult(
        test_name=result.test_name,  # test_name selection is tested separately
        statistic=result.statistic,
        p_value=result.p_value,
        significant=False,
        significance_level=0.05,
        n_group_a=50,
        n_group_b=50,
    )
    assert result.statistic == pytest.approx(0.0, abs=1e-10)
    assert result.p_value >= 0.05


def test_representativeness_detects_different_means() -> None:
    """Groups with clearly different means should be flagged as significant."""
    rng = np.random.default_rng(42)
    group_a = rng.normal(loc=100.0, scale=5.0, size=100)
    group_b = rng.normal(loc=130.0, scale=5.0, size=100)

    result = check_representativeness(group_a=group_a, group_b=group_b)

    assert result == RepresentativenessTestResult(
        test_name="welch_t_test",
        statistic=result.statistic,
        p_value=result.p_value,
        significant=True,
        significance_level=0.05,
        n_group_a=100,
        n_group_b=100,
    )
    assert result.p_value < 0.05


def test_representativeness_uses_welch_for_normal() -> None:
    rng = np.random.default_rng(42)
    group_a = rng.normal(loc=100.0, scale=10.0, size=100)
    group_b = rng.normal(loc=100.0, scale=10.0, size=100)

    result = check_representativeness(group_a=group_a, group_b=group_b)

    assert result.test_name == "welch_t_test"


def test_representativeness_uses_mann_whitney_for_non_normal() -> None:
    rng = np.random.default_rng(42)
    group_a = rng.exponential(scale=50.0, size=100) + 100
    group_b = rng.exponential(scale=50.0, size=100) + 100

    result = check_representativeness(group_a=group_a, group_b=group_b)

    assert result.test_name == "mann_whitney_u"


def test_representativeness_two_tailed_symmetric() -> None:
    """Swapping groups should give the same p-value (two-tailed test is symmetric)."""
    rng = np.random.default_rng(42)
    group_a = rng.normal(loc=100.0, scale=10.0, size=50)
    group_b = rng.normal(loc=110.0, scale=10.0, size=50)

    result_ab = check_representativeness(group_a=group_a, group_b=group_b)
    result_ba = check_representativeness(group_a=group_b, group_b=group_a)

    assert result_ab.p_value == pytest.approx(result_ba.p_value)


def test_representativeness_raises_on_too_few_samples() -> None:
    with pytest.raises(ValueError, match="at least 2 samples per group"):
        check_representativeness(
            group_a=np.array([1.0]),
            group_b=np.array([1.0, 2.0, 3.0]),
        )
