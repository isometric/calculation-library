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

import pandas as pd
from scipy.stats import norm

_Z_ALPHA = float(norm.ppf(1 - 0.05 / 2))  # 1.96 for two-sided alpha = 0.05
_Z_BETA = float(norm.ppf(0.80))  # 0.84 for power = 0.80


@dataclass(frozen=True)
class PowerAnalysisResult:
    """Power analysis result for one element."""

    element: str
    delta_mg_kg: float
    """Expected concentration change (Eq. 22): r*(C_F - C_BL)/(1+r) where r = R/(BD*D*10000)."""

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
    effective_application_rate_kg_ha: float,
    n_eff: float,
    bulk_density_kg_m3: float,
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
        paired: DataFrame with ``bl_mass_fraction_<element>`` and
            ``rp_mass_fraction_<element>`` columns (one row per paired location).
        feedstock_concentrations: Mean feedstock concentration per element (mg/kg).
            Keys should match elements (e.g. {"Ti": 17000, "Ca": 68000, "Mg": 29000}).
        effective_application_rate_kg_ha: Deployment-derived application rate.
        n_eff: Effective sample size from spatial autocorrelation test.
        bulk_density_kg_m3: Soil bulk density for mass calculation.
        sampling_depth_cm: Depth of the sampling layer.
        elements: Elements to analyse (e.g. ["Ti", "Ca", "Mg"]).
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
        for c in (f"bl_mass_fraction_{e.lower()}", f"rp_mass_fraction_{e.lower()}")
    ]
    missing_cols = [c for c in expected_cols if c not in paired.columns]
    if missing_cols:
        raise ValueError(
            f"Expected columns {missing_cols!r} not found in paired "
            f"(available: {list(paired.columns)!r}).",
        )

    soil_mass_kg_ha = bulk_density_kg_m3 * (sampling_depth_cm / 100) * 1e4
    # r = rock-to-soil mass ratio (Eq. 22)
    r = effective_application_rate_kg_ha / soil_mass_kg_ha

    results = list[PowerAnalysisResult]()

    for element in elements:
        col = f"mass_fraction_{element.lower()}"
        baseline_col = f"bl_{col}"
        reporting_period_col = f"rp_{col}"

        feedstock_concentration = feedstock_concentrations[element]
        mean_baseline = float(paired[baseline_col].mean())

        delta_mg_kg = r * (feedstock_concentration - mean_baseline) / (1 + r)

        sigma_baseline = float(paired[baseline_col].dropna().std())
        sigma_reporting_period = float(paired[reporting_period_col].dropna().std())

        if delta_mg_kg > 0:
            numerator = (_Z_ALPHA + _Z_BETA) ** 2 * (sigma_baseline**2 + sigma_reporting_period**2)
            n_required = numerator / (delta_mg_kg**2)
        else:
            n_required = float("inf")

        results.append(
            PowerAnalysisResult(
                element=element,
                delta_mg_kg=delta_mg_kg,
                sigma_baseline=sigma_baseline,
                sigma_reporting_period=sigma_reporting_period,
                n_required=n_required,
                n_actual=n_actual,
                n_eff=n_eff,
                passes=n_eff >= n_required,
            ),
        )

    return results
