# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

"""One-tailed significance test for enhanced weathering signal."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

from ..types import Np1DArray, mass_fraction_column_name
from ._normality import check_normality


@dataclass(frozen=True)
class SignificanceTestResult:
    """Result of a one-tailed significance test for weathering signal."""

    test_name: Literal["welch_t_test", "mann_whitney_u"]
    """Name of the statistical test used."""

    statistic: float
    """Test statistic value."""

    p_value: float
    """One-tailed p-value."""

    significant: bool
    """Whether the result is significant at the given significance level."""

    significance_level: float
    """Significance level used."""

    n_post_application: int
    """Number of (inferred) post-application samples."""

    n_end_of_reporting_period: int
    """Number of end-of-reporting-period samples."""


def infer_post_application_concentrations(
    *,
    baseline_concentrations_mg_kg: Np1DArray[np.floating],
    feedstock_concentration_mg_kg: float,
    application_rate_kg_ha: float,
    bulk_density_kg_m3: float,
    depth_cm: float,
) -> Np1DArray[np.floating]:
    """Infer per-sample post-application cation concentrations by applying the mixing formula to each baseline sample.

    Args:
        baseline_concentrations_mg_kg: Baseline cation concentrations per sample.
        feedstock_concentration_mg_kg: Mean feedstock cation concentration.
        application_rate_kg_ha: Known feedstock application rate in kg/ha.
        bulk_density_kg_m3: Mean soil bulk density in kg/m3.
        depth_cm: Sampling depth in cm.
    """
    soil_mass_kg_ha = bulk_density_kg_m3 * depth_cm * 100
    mass_ratio = application_rate_kg_ha / soil_mass_kg_ha

    return (baseline_concentrations_mg_kg + mass_ratio * feedstock_concentration_mg_kg) / (
        1 + mass_ratio
    )


def check_weathering_significance(
    *,
    post_application_concentrations_mg_kg: Np1DArray[np.floating],
    end_of_reporting_period_concentrations_mg_kg: Np1DArray[np.floating],
    significance_level: float = 0.05,
) -> SignificanceTestResult:
    """Test for a statistically significant decrease in cation concentration between post-application and end of reporting period.

    H0: C_post <= C_rp (no weathering). H1: C_post > C_rp (weathering occurred).
    Uses Welch's t-test if both samples pass Shapiro-Wilk normality, otherwise Mann-Whitney U.

    Args:
        post_application_concentrations_mg_kg: Inferred or measured post-application cation concentrations.
        end_of_reporting_period_concentrations_mg_kg: Measured end-of-reporting-period cation concentrations.
        significance_level: Significance level (default 0.05 per protocol).
    """
    both_distributions_normal = check_normality(
        post_application_concentrations_mg_kg,
    ) and check_normality(end_of_reporting_period_concentrations_mg_kg)

    if both_distributions_normal:
        result = stats.ttest_ind(
            post_application_concentrations_mg_kg,
            end_of_reporting_period_concentrations_mg_kg,
            equal_var=False,
            alternative="greater",
        )
        test_name = "welch_t_test"
    else:
        result = stats.mannwhitneyu(
            post_application_concentrations_mg_kg,
            end_of_reporting_period_concentrations_mg_kg,
            alternative="greater",
        )
        test_name = "mann_whitney_u"

    p_value = float(result.pvalue)

    return SignificanceTestResult(
        test_name=test_name,
        statistic=float(result.statistic),
        p_value=p_value,
        significant=p_value < significance_level,
        significance_level=significance_level,
        n_post_application=len(post_application_concentrations_mg_kg),
        n_end_of_reporting_period=len(end_of_reporting_period_concentrations_mg_kg),
    )


def run_significance_tests(
    *,
    treatment_baseline: pd.DataFrame,
    treatment_reporting_period: pd.DataFrame,
    feedstock_samples: pd.DataFrame,
    bulk_density_kg_m3: float,
    application_rate_kg_ha: float,
    elements: Sequence[str],
    sampling_depth_cm: float,
) -> pd.DataFrame:
    """Run per-element weathering significance tests and return as a DataFrame.

    For each element, infers post-application concentrations from the baseline
    using the mixing formula, then tests whether the end-of-reporting-period
    concentrations are significantly lower (indicating weathering).

    Args:
        treatment_baseline: Baseline treatment soil samples.
        treatment_reporting_period: End-of-reporting-period treatment soil samples.
        feedstock_samples: Feedstock geochemistry samples.
        bulk_density_kg_m3: Mean bulk density in kg/m3.
        application_rate_kg_ha: Known feedstock application rate in kg/ha.
        elements: Element names to test (e.g. ``["Ca", "Mg"]``).
        sampling_depth_cm: Sampling depth in cm.
    """
    results = dict[str, SignificanceTestResult]()
    for element in elements:
        col = mass_fraction_column_name(element)
        feedstock_mean = float(feedstock_samples[col].dropna().mean())

        post_app = infer_post_application_concentrations(
            baseline_concentrations_mg_kg=treatment_baseline[col].dropna().to_numpy(),
            feedstock_concentration_mg_kg=feedstock_mean,
            application_rate_kg_ha=application_rate_kg_ha,
            bulk_density_kg_m3=bulk_density_kg_m3,
            depth_cm=sampling_depth_cm,
        )

        results[element] = check_weathering_significance(
            post_application_concentrations_mg_kg=post_app,
            end_of_reporting_period_concentrations_mg_kg=(
                treatment_reporting_period[col].dropna().to_numpy()
            ),
        )

    return pd.DataFrame([
        {
            "cation": element,
            "test_name": sig_result.test_name,
            "statistic": sig_result.statistic,
            "p_value": sig_result.p_value,
            "significant": sig_result.significant,
            "n_post_application": sig_result.n_post_application,
            "n_end_of_reporting_period": sig_result.n_end_of_reporting_period,
        }
        for element, sig_result in results.items()
    ])
