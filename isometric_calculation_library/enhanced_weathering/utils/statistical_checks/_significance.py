# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

"""Shared paired and unpaired significance tests.

Each test picks a parametric or rank-based implementation depending on whether the data
is normally distributed. That normality check is an implementation detail of choosing a
valid test, not the question being answered.
"""

from typing import Literal, NamedTuple

import numpy as np
from scipy import stats

from isometric_calculation_library.utils.types import Np1DArray


def check_normality(
    samples: Np1DArray[np.floating],
    significance_level: float = 0.05,
) -> bool:
    """Check normality using Shapiro-Wilk test.

    Returns True if the null hypothesis of normality is not rejected.
    Requires at least 3 samples; returns False otherwise.
    """
    if len(samples) < 3:
        return False
    _, p_value = stats.shapiro(samples)
    return float(p_value) >= significance_level


class PairedSignificanceTest(NamedTuple):
    """Outcome of a paired significance test."""

    test_name: Literal["paired_t_test", "wilcoxon_signed_rank"]
    """``paired_t_test`` when the paired differences are normal, else ``wilcoxon_signed_rank``."""

    differences_are_normal: bool
    """Whether the paired differences passed the Shapiro-Wilk normality check."""

    statistic: float
    p_value: float


def run_paired_significance_test(
    first: Np1DArray[np.floating],
    second: Np1DArray[np.floating],
    *,
    alternative: Literal["two-sided", "less", "greater"],
    significance_level: float = 0.05,
) -> PairedSignificanceTest:
    """Test whether paired samples differ significantly.

    A Shapiro-Wilk check on the paired differences (``first - second``) selects a paired
    t-test when they are normal, otherwise a Wilcoxon signed-rank test. Both arrays must
    be equal length and already free of NaNs. ``alternative`` is passed straight through
    to the underlying test (``"greater"`` tests ``first > second``).
    """
    differences = first - second
    differences_are_normal = check_normality(differences, significance_level=significance_level)
    if differences_are_normal:
        result = stats.ttest_rel(first, second, alternative=alternative)
        test_name: Literal["paired_t_test", "wilcoxon_signed_rank"] = "paired_t_test"
    else:
        result = stats.wilcoxon(first, second, alternative=alternative)
        test_name = "wilcoxon_signed_rank"
    return PairedSignificanceTest(
        test_name=test_name,
        differences_are_normal=differences_are_normal,
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
    )


class UnpairedSignificanceTest(NamedTuple):
    """Outcome of an unpaired (two-sample) significance test."""

    test_name: Literal["welch_t_test", "mann_whitney_u"]
    """``welch_t_test`` when both samples are normal, else ``mann_whitney_u``."""

    both_distributions_normal: bool
    """Whether both samples passed the Shapiro-Wilk normality check."""

    statistic: float
    p_value: float


def run_unpaired_significance_test(
    first: Np1DArray[np.floating],
    second: Np1DArray[np.floating],
    *,
    alternative: Literal["two-sided", "less", "greater"],
    significance_level: float = 0.05,
) -> UnpairedSignificanceTest:
    """Test whether two independent samples differ significantly.

    A Shapiro-Wilk check on each sample selects Welch's t-test (unequal variances) when both
    are normal, otherwise a Mann-Whitney U test. Samples may differ in length and must
    already be free of NaNs. ``alternative`` is passed straight through (``"greater"`` tests
    ``first > second``).
    """
    both_distributions_normal = check_normality(
        first,
        significance_level=significance_level,
    ) and check_normality(second, significance_level=significance_level)
    if both_distributions_normal:
        result = stats.ttest_ind(first, second, equal_var=False, alternative=alternative)
        test_name: Literal["welch_t_test", "mann_whitney_u"] = "welch_t_test"
    else:
        result = stats.mannwhitneyu(first, second, alternative=alternative)
        test_name = "mann_whitney_u"
    return UnpairedSignificanceTest(
        test_name=test_name,
        both_distributions_normal=both_distributions_normal,
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
    )
