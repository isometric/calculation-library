# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

"""One-tailed significance tests for enhanced weathering signal.

Provides both unpaired (independent-sample) and paired (matched-location)
variants.  The paired variant should be preferred when baseline and
reporting-period samples can be matched by spatial location, because it
controls for between-location variance and has more power to detect small
weathering signals.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from isometric_calculation_library.enhanced_weathering.utils.conversions import (
    compute_feedstock_soil_mass_ratio,
)
from isometric_calculation_library.enhanced_weathering.utils.types import (
    mass_fraction_column_name,
)
from isometric_calculation_library.utils.elements import ElementSymbol
from isometric_calculation_library.utils.types import Np1DArray

from ._significance import run_paired_significance_test, run_unpaired_significance_test


@dataclass(frozen=True)
class SignificanceTestResult:
    """Result of a one-tailed significance test for weathering signal."""

    test_name: Literal["welch_t_test", "mann_whitney_u", "paired_t_test", "wilcoxon_signed_rank"]
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
    application_rate_kg_ha: float | Np1DArray[np.floating],
    bulk_density_kg_m3: float | Np1DArray[np.floating],
    depth_cm: float,
) -> Np1DArray[np.floating]:
    """Infer per-sample post-application cation concentrations by applying the mixing formula to each baseline sample.

    Args:
        baseline_concentrations_mg_kg: Baseline cation concentrations per sample.
        feedstock_concentration_mg_kg: Mean feedstock cation concentration.
        application_rate_kg_ha: Known feedstock application rate in kg/ha. Pass a bootstrap
            distribution when the rate is itself uncertain (e.g. inverted from BLP
            enrichment); the mixing formula is then applied at each rate replicate and the
            returned array is the pooled set of inferred concentrations, so the rate's
            uncertainty is carried into the significance test rather than discarded. A
            scalar returns one concentration per baseline sample.
        bulk_density_kg_m3: Soil bulk density in kg/m3. Like the rate, this may be a
            bootstrap distribution when several bulk densities were measured, in which case
            its spread reaches the test too; the two must then have equal length.
        depth_cm: Sampling depth in cm.
    """
    mass_ratio = compute_feedstock_soil_mass_ratio(
        application_rate_kg_ha=application_rate_kg_ha,
        soil_bulk_density_kg_m3=bulk_density_kg_m3,
        depth_cm=depth_cm,
    )

    # Outer product over (mass-ratio replicate, baseline sample), then flattened: the test
    # compares populations, so the pooled set carries both sources of spread. A scalar rate
    # and bulk density give a single replicate, hence one concentration per baseline sample.
    finite_mass_ratio = mass_ratio[np.isfinite(mass_ratio)]
    if len(finite_mass_ratio) == 0:
        raise ValueError(
            "No finite rock-to-soil mass ratio could be formed from application_rate_kg_ha "
            "and bulk_density_kg_m3, so no post-application concentration can be inferred.",
        )
    ratio_column = finite_mass_ratio[:, np.newaxis]
    inferred = (
        baseline_concentrations_mg_kg[np.newaxis, :] + ratio_column * feedstock_concentration_mg_kg
    ) / (1 + ratio_column)
    return inferred.reshape(-1)


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
    if (
        len(post_application_concentrations_mg_kg) < 2
        or len(end_of_reporting_period_concentrations_mg_kg) < 2
    ):
        raise ValueError(
            "Weathering significance test requires at least 2 samples per group "
            f"(got {len(post_application_concentrations_mg_kg)} and "
            f"{len(end_of_reporting_period_concentrations_mg_kg)}); a smaller group "
            "yields a meaningless or NaN p-value that would silently read as non-significant.",
        )

    # One-sided: H1 is weathering, i.e. post-application concentration > end-of-period.
    test = run_unpaired_significance_test(
        post_application_concentrations_mg_kg,
        end_of_reporting_period_concentrations_mg_kg,
        alternative="greater",
        significance_level=significance_level,
    )

    return SignificanceTestResult(
        test_name=test.test_name,
        statistic=test.statistic,
        p_value=test.p_value,
        significant=test.p_value < significance_level,
        significance_level=significance_level,
        n_post_application=len(post_application_concentrations_mg_kg),
        n_end_of_reporting_period=len(end_of_reporting_period_concentrations_mg_kg),
    )


def check_weathering_significance_paired(
    *,
    post_application_concentrations_mg_kg: Np1DArray[np.floating],
    end_of_reporting_period_concentrations_mg_kg: Np1DArray[np.floating],
    significance_level: float = 0.05,
) -> SignificanceTestResult:
    """Paired test for a statistically significant decrease in cation concentration.

    Like :func:`check_weathering_significance`, but for **matched** samples
    (same spatial location at two time points).  This controls for
    between-location variance, giving more statistical power when the
    weathering signal is small relative to spatial variability.

    H0: median(C_post - C_rp) <= 0 (no weathering).
    H1: median(C_post - C_rp) > 0 (weathering occurred).

    Uses a paired t-test if the *differences* pass Shapiro-Wilk normality,
    otherwise Wilcoxon signed-rank.

    Args:
        post_application_concentrations_mg_kg: Post-application concentrations,
            one per matched location.
        end_of_reporting_period_concentrations_mg_kg: End-of-reporting-period
            concentrations, same order/length as post-application.
        significance_level: Significance level (default 0.05 per protocol).
    """
    if len(post_application_concentrations_mg_kg) != len(
        end_of_reporting_period_concentrations_mg_kg,
    ):
        msg = (
            "Paired test requires equal-length arrays "
            f"(got {len(post_application_concentrations_mg_kg)} vs "
            f"{len(end_of_reporting_period_concentrations_mg_kg)})"
        )
        raise ValueError(msg)

    if len(post_application_concentrations_mg_kg) < 2:
        raise ValueError(
            "Paired weathering test requires at least 2 matched samples, "
            f"got {len(post_application_concentrations_mg_kg)}.",
        )

    # One-sided: H1 is weathering, i.e. post-application concentration > end-of-period.
    test = run_paired_significance_test(
        post_application_concentrations_mg_kg,
        end_of_reporting_period_concentrations_mg_kg,
        alternative="greater",
        significance_level=significance_level,
    )
    n = len(post_application_concentrations_mg_kg)

    return SignificanceTestResult(
        test_name=test.test_name,
        statistic=test.statistic,
        p_value=test.p_value,
        significant=test.p_value < significance_level,
        significance_level=significance_level,
        n_post_application=n,
        n_end_of_reporting_period=n,
    )


def run_significance_tests(
    *,
    treatment_baseline: pd.DataFrame,
    treatment_reporting_period: pd.DataFrame,
    feedstock_samples: pd.DataFrame,
    bulk_density_kg_m3: float,
    application_rate_kg_ha: float,
    elements: Sequence[ElementSymbol],
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
    results = dict[ElementSymbol, SignificanceTestResult]()
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
