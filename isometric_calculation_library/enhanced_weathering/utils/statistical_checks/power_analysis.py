# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

"""Power analysis for enhanced weathering sampling design (Protocol Eq. 22-23).

Determines whether the sampling design has enough independent observations
to detect the expected enrichment signal given observed variability and
deployment-derived application rate.

When spatial autocorrelation is significant, n_eff < n_actual is used
for the pass condition, since correlated observations provide less
independent information than their count suggests.

Eq. 22 - Expected concentration change after rock application:
    r = R / (BD * D * 10000)           # rock-to-soil mass ratio (kg rock / kg soil)
    delta = r * (C_F - C_BL) / (1 + r) # net enrichment (mg/kg)
where R is application rate (kg/ha), BD bulk density (kg/m³), D sampling depth (m),
C_F feedstock concentration (mg/kg), C_BL mean baseline soil concentration (mg/kg).
The (1 + r) denominator accounts for soil mass dilution by the added rock.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from isometric_calculation_library.enhanced_weathering.utils.conversions import (
    compute_feedstock_soil_mass_ratio,
)
from isometric_calculation_library.utils.types import Np1DArray

_Z_ALPHA = float(norm.ppf(1 - 0.05 / 2))  # 1.96 for two-sided alpha = 0.05
_Z_BETA = float(norm.ppf(0.80))  # 0.84 for power = 0.80


@dataclass(frozen=True)
class PowerAnalysisResult:
    """Power analysis result for one element."""

    element: str
    delta_mg_kg: float
    """Expected concentration change (Eq. 22): r*(C_F - C_BL)/(1+r) where r = R/(BD*D*10000).

    The median when the application rate is a distribution; the single value when scalar.
    """

    delta_mg_kg_p16: float
    """16th percentile of the expected change. Equal to ``delta_mg_kg`` for a scalar rate."""

    delta_mg_kg_p84: float
    """84th percentile of the expected change. Equal to ``delta_mg_kg`` for a scalar rate."""

    sigma_baseline: float
    """Standard deviation of baseline concentrations."""

    sigma_reporting_period: float
    """Standard deviation of reporting period concentrations."""

    n_required: float
    """Minimum samples required (Eq. 23)."""

    n_actual: int
    """Actual number of paired locations."""

    n_eff: float
    """Effective sample size (from Moran's I, or n_actual if no autocorrelation)."""

    passes: bool
    """True if n_eff >= n_required."""


def compute_power_analysis(
    paired: pd.DataFrame,
    feedstock_concentrations: Mapping[str, float],
    effective_application_rate_kg_ha: float | Np1DArray[np.floating],
    n_eff: float,
    bulk_density_kg_m3: float | Np1DArray[np.floating],
    sampling_depth_cm: float,
    elements: Sequence[str],
) -> list[PowerAnalysisResult]:
    """Compute power analysis per element (Protocol Eq. 22-23).

    Eq. 22 - Expected concentration change after rock application:
        r = R / (BD * D * 10000)           # rock-to-soil mass ratio (dimensionless)
        delta = r * (C_F - C_BL) / (1 + r) # net enrichment (mg/kg)
    where R is the application rate (kg/ha), BD bulk density (kg/m³),
    D the sampling depth (m), C_F feedstock concentration (mg/kg),
    C_BL mean baseline soil concentration (mg/kg).

    Eq. 23 - Minimum samples required (unequal variances, perfectly paired design, k=1):
        n_req = (z_alpha + z_beta)^2 * (sigma_bl^2 + sigma_rp^2) / delta^2

    Pass condition: n_eff >= n_req.

    Args:
        paired: DataFrame with ``baseline_mass_fraction_<element>`` and
            ``reporting_period_mass_fraction_<element>`` columns (one row per paired location).
        feedstock_concentrations: Mean feedstock concentration per element (mg/kg).
            Keys should match elements (e.g. {"Ti": 17000, "Ca": 68000, "Mg": 29000}).
        effective_application_rate_kg_ha: Deployment-derived application rate. Pass a
            bootstrap distribution when the rate is itself uncertain (e.g. inverted from
            BLP enrichment); Eq. 22 is then evaluated per replicate and the reported
            ``delta_mg_kg`` is the median of the resulting enrichment distribution, with
            its 16th-84th percentile range carried alongside. A scalar is treated as a
            distribution of one, so scalar callers are unaffected.
        n_eff: Effective sample size from spatial autocorrelation test.
        bulk_density_kg_m3: Soil bulk density for mass calculation. Like the rate, this may
            be a bootstrap distribution when several bulk densities were measured, in which
            case its spread widens the expected enrichment too; the two must then have equal
            length.
        sampling_depth_cm: Depth of the sampling layer.
        elements: Elements to analyse (e.g. ["Ti", "Ca", "Mg", "Na", "K"]).
    """
    n_actual = len(paired)

    missing_keys = [e for e in elements if e not in feedstock_concentrations]
    if missing_keys:
        raise ValueError(
            f"Elements {missing_keys!r} not found in feedstock_concentrations "
            f"(available: {list(feedstock_concentrations)!r}).",
        )
    expected_cols = [
        c
        for e in elements
        for c in (
            f"baseline_mass_fraction_{e.lower()}",
            f"reporting_period_mass_fraction_{e.lower()}",
        )
    ]
    missing_cols = [c for c in expected_cols if c not in paired.columns]
    if missing_cols:
        raise ValueError(
            f"Expected columns {missing_cols!r} not found in paired "
            f"(available: {list(paired.columns)!r}).",
        )

    # r = rock-to-soil mass ratio (Eq. 22). Kept as an array so an uncertain rate and an
    # uncertain bulk density both propagate through the enrichment, rather than collapsing
    # to a point estimate.
    r = compute_feedstock_soil_mass_ratio(
        application_rate_kg_ha=effective_application_rate_kg_ha,
        soil_bulk_density_kg_m3=bulk_density_kg_m3,
        depth_cm=sampling_depth_cm,
    )
    if np.all(np.isnan(r)):
        raise ValueError(
            "No finite rock-to-soil mass ratio could be formed from "
            "effective_application_rate_kg_ha and bulk_density_kg_m3, so no expected "
            "enrichment can be computed. Supply values with at least one finite pairing.",
        )

    results = list[PowerAnalysisResult]()

    for element in elements:
        col = f"mass_fraction_{element.lower()}"
        baseline_col = f"baseline_{col}"
        reporting_period_col = f"reporting_period_{col}"

        baseline_values = paired[baseline_col].dropna()
        reporting_period_values = paired[reporting_period_col].dropna()
        if len(baseline_values) < 2 or len(reporting_period_values) < 2:
            raise ValueError(
                f"Power analysis for element {element!r} requires at least 2 non-null "
                f"baseline and reporting-period values (got {len(baseline_values)} and "
                f"{len(reporting_period_values)}); the standard deviation is otherwise "
                "undefined and would silently yield NaN.",
            )

        feedstock_concentration = feedstock_concentrations[element]
        mean_baseline = float(baseline_values.mean())

        # Evaluated per replicate, then summarised, so an uncertain rate widens the
        # expected enrichment instead of being averaged away before Eq. 23.
        delta_distribution = r * (feedstock_concentration - mean_baseline) / (1 + r)
        delta_mg_kg = float(np.nanpercentile(delta_distribution, 50))
        delta_p16 = float(np.nanpercentile(delta_distribution, 16))
        delta_p84 = float(np.nanpercentile(delta_distribution, 84))

        sigma_baseline = float(baseline_values.std())
        sigma_reporting_period = float(reporting_period_values.std())

        if delta_mg_kg > 0:
            numerator = (_Z_ALPHA + _Z_BETA) ** 2 * (sigma_baseline**2 + sigma_reporting_period**2)
            n_required = numerator / (delta_mg_kg**2)
        else:
            n_required = float("inf")

        results.append(
            PowerAnalysisResult(
                element=element,
                delta_mg_kg=delta_mg_kg,
                delta_mg_kg_p16=delta_p16,
                delta_mg_kg_p84=delta_p84,
                sigma_baseline=sigma_baseline,
                sigma_reporting_period=sigma_reporting_period,
                n_required=n_required,
                n_actual=n_actual,
                n_eff=n_eff,
                passes=n_eff >= n_required,
            ),
        )

    return results
