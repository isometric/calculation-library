# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

"""Multiple-testing correction and permutation-test utilities.

Used to control the false discovery rate when testing many hypotheses at once
(Benjamini-Hochberg), and to test for a difference in medians between two
samples without assuming a distribution (permutation test).
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from isometric_calculation_library.enhanced_weathering.utils.types import Np1DArray


def benjamini_hochberg(p_values: Np1DArray[np.floating]) -> Np1DArray[np.floating]:
    """Apply Benjamini-Hochberg FDR correction to an array of p-values.

    Returns adjusted p-values in the same order as the input, each clipped to
    [0, 1]. Monotonicity is enforced by stepping down from the largest rank.
    """
    n_tests = len(p_values)
    if n_tests == 0:
        return p_values.copy()

    sorted_indices = np.argsort(p_values)
    bh_adjusted = np.empty(n_tests)

    for rank_idx, orig_idx in enumerate(sorted_indices):
        rank = rank_idx + 1
        bh_adjusted[orig_idx] = p_values[orig_idx] * n_tests / rank

    # Enforce monotonicity (step down from the largest rank).
    for i in range(n_tests - 2, -1, -1):
        idx = sorted_indices[i]
        idx_next = sorted_indices[i + 1]
        bh_adjusted[idx] = min(bh_adjusted[idx], bh_adjusted[idx_next])

    return np.clip(bh_adjusted, 0.0, 1.0)


@dataclass(frozen=True)
class PermutationTestResult:
    """Result of a two-sample permutation test on the difference in medians."""

    observed_difference: float
    """median(group_a) - median(group_b)."""

    p_value: float
    """Permutation p-value for the chosen alternative (with a +1 correction)."""

    n_permutations: int
    """Number of label permutations used."""


def permutation_test_median_difference(
    *,
    group_a: Np1DArray[np.floating],
    group_b: Np1DArray[np.floating],
    rng: np.random.Generator,
    n_permutations: int = 1_999,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
) -> PermutationTestResult:
    """Two-sample permutation test for a difference in medians.

    The observed statistic is ``median(group_a) - median(group_b)``. Labels are
    shuffled across the pooled values ``n_permutations`` times and the statistic
    recomputed; the p-value is the fraction of permuted statistics at least as
    extreme as the observed one, with a +1 correction in numerator and
    denominator for an unbiased, conservative estimate.

    Args:
        group_a: First sample.
        group_b: Second sample.
        rng: Random number generator (required for reproducibility).
        n_permutations: Number of label permutations (protocol minimum 1000).
        alternative: ``"two-sided"`` tests |difference|; ``"greater"`` tests
            group_a median > group_b median; ``"less"`` the reverse.
    """
    if len(group_a) == 0 or len(group_b) == 0:
        raise ValueError(
            f"Permutation test requires non-empty groups (got {len(group_a)} and {len(group_b)}).",
        )

    observed = float(np.median(group_a) - np.median(group_b))
    pooled = np.concatenate([group_a, group_b])
    n_a = len(group_a)

    perm_diffs = np.empty(n_permutations)
    for i in range(n_permutations):
        shuffled = rng.permutation(pooled)
        perm_diffs[i] = np.median(shuffled[:n_a]) - np.median(shuffled[n_a:])

    match alternative:
        case "two-sided":
            n_extreme = int(np.sum(np.abs(perm_diffs) >= abs(observed)))
        case "greater":
            n_extreme = int(np.sum(perm_diffs >= observed))
        case "less":
            n_extreme = int(np.sum(perm_diffs <= observed))

    p_value = (n_extreme + 1) / (n_permutations + 1)

    return PermutationTestResult(
        observed_difference=observed,
        p_value=p_value,
        n_permutations=n_permutations,
    )
