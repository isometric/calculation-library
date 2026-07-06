# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

import numpy as np
import pytest

from isometric_calculation_library.enhanced_weathering.utils.statistical_checks.multiple_testing import (
    benjamini_hochberg,
    permutation_test_median_difference,
)

# -- benjamini_hochberg --------------------------------------------------------


def test_bh_single_p_value_unchanged() -> None:
    result = benjamini_hochberg(np.array([0.03]))
    assert result[0] == pytest.approx(0.03)


def test_bh_empty_returns_empty() -> None:
    result = benjamini_hochberg(np.array([]))
    assert len(result) == 0


def test_bh_monotonicity() -> None:
    """Adjusted p-values should be non-decreasing when sorted by raw p."""
    p_values = np.array([0.001, 0.01, 0.03, 0.04, 0.1])
    adjusted = benjamini_hochberg(p_values)

    sorted_idx = np.argsort(p_values)
    adj_sorted = adjusted[sorted_idx]
    assert all(adj_sorted[i] <= adj_sorted[i + 1] for i in range(len(adj_sorted) - 1))


def test_bh_clipped_to_one() -> None:
    adjusted = benjamini_hochberg(np.array([0.8, 0.9, 0.95]))
    assert all(a <= 1.0 for a in adjusted)


# -- permutation_test_median_difference ----------------------------------------


def test_permutation_identical_groups_not_significant() -> None:
    rng = np.random.default_rng(42)
    values = rng.normal(100, 10, size=50)
    result = permutation_test_median_difference(
        group_a=values,
        group_b=values.copy(),
        rng=rng,
        n_permutations=999,
    )
    assert result.observed_difference == pytest.approx(0.0)
    assert result.p_value > 0.05


def test_permutation_large_shift_is_significant() -> None:
    rng = np.random.default_rng(42)
    group_a = rng.normal(120, 5, size=60)
    group_b = rng.normal(100, 5, size=60)
    result = permutation_test_median_difference(
        group_a=group_a,
        group_b=group_b,
        rng=rng,
        n_permutations=999,
    )
    assert result.observed_difference > 0
    assert result.p_value < 0.05


def test_permutation_one_sided_greater() -> None:
    rng = np.random.default_rng(0)
    group_a = rng.normal(120, 5, size=60)
    group_b = rng.normal(100, 5, size=60)
    greater = permutation_test_median_difference(
        group_a=group_a,
        group_b=group_b,
        rng=rng,
        n_permutations=999,
        alternative="greater",
    )
    less = permutation_test_median_difference(
        group_a=group_a,
        group_b=group_b,
        rng=rng,
        n_permutations=999,
        alternative="less",
    )
    assert greater.p_value < 0.05
    assert less.p_value > 0.95


def test_permutation_raises_on_empty_group() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        permutation_test_median_difference(
            group_a=np.array([]),
            group_b=np.array([1.0, 2.0]),
            rng=np.random.default_rng(0),
        )
