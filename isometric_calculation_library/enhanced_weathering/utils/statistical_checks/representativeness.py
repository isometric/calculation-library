# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

"""Two-tailed representativeness test for comparing treatment vs deployment areas."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import stats

from isometric_calculation_library.enhanced_weathering.utils.types import Np1DArray

from ._normality import check_normality


@dataclass(frozen=True)
class RepresentativenessTestResult:
    """Result of a two-tailed representativeness test between treatment and deployment groups."""

    test_name: Literal["welch_t_test", "mann_whitney_u"]
    """Name of the statistical test used."""

    statistic: float
    """Test statistic value."""

    p_value: float
    """Two-tailed p-value."""

    significant: bool
    """Whether the difference is significant. Note: False means failure to detect a difference, not equivalence."""

    significance_level: float
    """Significance level used."""

    n_group_a: int
    """Number of samples in group A."""

    n_group_b: int
    """Number of samples in group B."""


def check_representativeness(
    *,
    group_a: Np1DArray[np.floating],
    group_b: Np1DArray[np.floating],
    significance_level: float = 0.05,
) -> RepresentativenessTestResult:
    """Two-tailed test for whether two groups have statistically different distributions.

    H0: The groups have the same distribution. H1: The groups differ.
    Uses Welch's t-test if both samples pass Shapiro-Wilk normality, otherwise Mann-Whitney U.

    Note: A non-significant result indicates failure to detect a difference, not proof of equivalence.

    Args:
        group_a: Values for group A (e.g. deployment).
        group_b: Values for group B (e.g. treatment).
        significance_level: Significance level (default 0.05 per protocol).
    """
    if len(group_a) < 2 or len(group_b) < 2:
        raise ValueError(
            "Representativeness test requires at least 2 samples per group "
            f"(got {len(group_a)} and {len(group_b)}); a smaller group yields a "
            "meaningless or NaN p-value that would silently read as non-significant.",
        )

    both_normal = check_normality(group_a) and check_normality(group_b)

    if both_normal:
        result = stats.ttest_ind(
            group_a,
            group_b,
            equal_var=False,
            alternative="two-sided",
        )
        test_name = "welch_t_test"
    else:
        result = stats.mannwhitneyu(
            group_a,
            group_b,
            alternative="two-sided",
        )
        test_name = "mann_whitney_u"

    return RepresentativenessTestResult(
        test_name=test_name,
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        significant=float(result.pvalue) < significance_level,
        significance_level=significance_level,
        n_group_a=len(group_a),
        n_group_b=len(group_b),
    )
