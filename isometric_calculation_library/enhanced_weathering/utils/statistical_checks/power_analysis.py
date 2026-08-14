# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

"""Power analysis for enhanced weathering sampling design (Protocol Eq. 22-23).

Determines whether the sampling design has enough independent observations
to detect the expected enrichment signal given observed variability and
deployment-derived application rate.

When spatial autocorrelation is significant, n_eff is below the raw sample count and is
what the pass condition uses, since correlated observations provide less independent
information than their count suggests.

Eq. 22 - Expected concentration change after rock application (shared by both variants
below):
    r = R / (BD * D * 10000)           # rock-to-soil mass ratio (kg rock / kg soil)
    delta = r * (C_F - C_BL) / (1 + r) # net enrichment (mg/kg)
where R is application rate (kg/ha), BD bulk density (kg/m³), D sampling depth (m),
C_F feedstock concentration (mg/kg), C_BL mean baseline soil concentration (mg/kg).
The (1 + r) denominator accounts for soil mass dilution by the added rock.

Eq. 23 - Minimum samples required - differs between the two variants only in which
variance term stands in for the sampling noise:

- ``compute_power_analysis_unpaired`` treats baseline and reporting-period as
  independent samples: ``sigma_bl^2 + sigma_rp^2 / k``, where ``k = n_rp / n_bl`` is the
  observed allocation ratio between the two events. (The protocol writes this ratio ``r``;
  it is renamed here to keep it apart from Eq. 22's rock-to-soil mass ratio.) This is the
  right choice when the two samples cannot be matched by location. It takes the two
  sampling events as separate frames, which need not be the same size — the protocol
  allows a different number of samples per event, and the ``/ k`` term is what lets the
  extra samples on one side count. ``n_required`` is then a count of *baseline* samples,
  the reporting-period event being assumed to scale with it at the same ratio.
- ``compute_power_analysis_paired`` uses the variance of the within-pair difference
  (reporting-period minus baseline at the *same* location): ``sigma_diff^2``. Since
  ``sigma_diff^2 = sigma_bl^2 + sigma_rp^2 - 2*rho*sigma_bl*sigma_rp``, this is smaller
  whenever a location's baseline and reporting-period concentrations move together
  (rho > 0) — the common case, since a location's mineralogy is persistent — so a
  genuinely paired design needs fewer samples than the unpaired formula would imply.
  It takes one row per matched location, e.g. from ``pair_locations``.

Both variants require their inputs to already be complete (no nulls) for every requested
element, and raise otherwise rather than silently dropping rows: which samples back a
variance estimate is exactly what these counts mean, so the caller — not this function —
decides how to handle missing data.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, NamedTuple

import numpy as np
import pandas as pd
from scipy.stats import norm

from isometric_calculation_library.enhanced_weathering.utils.conversions import (
    compute_feedstock_soil_mass_ratio,
)
from isometric_calculation_library.enhanced_weathering.utils.data_cleaning import (
    check_measured_values,
)
from isometric_calculation_library.enhanced_weathering.utils.pairing import (
    paired_column_names,
    require_complete_pairs,
)
from isometric_calculation_library.enhanced_weathering.utils.types import mass_fraction_column_name
from isometric_calculation_library.utils.elements import ElementSymbol
from isometric_calculation_library.utils.types import Np1DArray

_Z_ALPHA = float(norm.ppf(1 - 0.05 / 2))  # 1.96 for two-sided alpha = 0.05
_Z_BETA = float(norm.ppf(0.80))  # 0.84 for power = 0.80

type _VarianceKind = Literal["paired", "unpaired"]


class _ElementValues(NamedTuple):
    """One element's measurements from each sampling event.

    The two are row-aligned by location for a paired design and unrelated otherwise, which
    is why only the paired variant may difference them.
    """

    baseline: Np1DArray[np.floating]
    reporting_period: Np1DArray[np.floating]


@dataclass(frozen=True)
class PowerAnalysisResult:
    """Power analysis result for one element.

    Shared by both :func:`compute_power_analysis_paired` and
    :func:`compute_power_analysis_unpaired`, which differ only in which of ``sigma_diff``
    or (``sigma_baseline``, ``sigma_reporting_period``) feeds ``n_required``. The paired
    variant populates all three, so a reviewer can see what the unpaired formula would
    have required on the same data; the unpaired variant has no matching to difference
    across and leaves ``sigma_diff`` unset.
    """

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
    """Standard deviation of baseline concentrations. Feeds ``n_required`` when unpaired."""

    sigma_reporting_period: float
    """Standard deviation of reporting-period concentrations. Feeds ``n_required`` when unpaired."""

    sigma_diff: float | None
    """Standard deviation of the within-pair difference (reporting-period minus baseline).

    Feeds ``n_required`` when paired, and is ``None`` for an unpaired design, since there
    is no location matching.
    """

    n_required: float
    """Minimum samples required (Eq. 23).

    Matched locations for a paired design. For an unpaired one it counts *baseline*
    samples, the reporting-period event being assumed to scale with it at the observed
    allocation ratio — the two coincide whenever the events are the same size.
    """

    n_baseline: int
    """Baseline samples backing the variance estimate."""

    n_reporting_period: int
    """Reporting-period samples backing the variance estimate.

    Equal to ``n_baseline`` for a paired design, where both count the matched locations.
    """

    n_eff: float
    """Effective sample size (from Moran's I, or the raw count if no autocorrelation)."""

    passes: bool
    """True if n_eff >= n_required."""


def compute_power_analysis_paired(
    *,
    paired: pd.DataFrame,
    feedstock_concentrations: Mapping[str, float],
    effective_application_rate_kg_ha: float | Np1DArray[np.floating],
    n_eff: float,
    bulk_density_kg_m3: float | Np1DArray[np.floating],
    sampling_depth_cm: float,
    elements: Sequence[ElementSymbol],
) -> list[PowerAnalysisResult]:
    """Compute power analysis per element for a paired (matched-location) design.

    Eq. 23 (paired form) - Minimum samples required:
        n_req = (z_alpha + z_beta)^2 * sigma_diff^2 / delta^2
    where sigma_diff is the standard deviation of the within-pair difference
    (reporting-period minus baseline) at matched locations. Pass condition: n_eff >= n_req.

    ``paired`` must already have no null values in the baseline/reporting-period columns
    for every requested element — this function raises rather than dropping incomplete
    rows, since silently changing which locations back the variance estimate would
    misrepresent the design. Clean the input (e.g. a row-wise dropna on the relevant
    columns) before calling.

    Args:
        paired: DataFrame with ``baseline_mass_fraction_<element>`` and
            ``reporting_period_mass_fraction_<element>`` columns (one row per matched
            location, e.g. from ``pair_locations``), with no null values in those columns.
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
    require_complete_pairs(
        paired,
        paired_column_names([mass_fraction_column_name(e) for e in elements]),
    )
    if len(paired) < 2:
        raise ValueError(
            f"Power analysis requires at least 2 paired locations, got {len(paired)}. "
            "The standard deviation of the difference is otherwise undefined.",
        )
    values_by_element = {
        element: _ElementValues(
            baseline=paired[f"baseline_{mass_fraction_column_name(element)}"].to_numpy(dtype=float),
            reporting_period=paired[
                f"reporting_period_{mass_fraction_column_name(element)}"
            ].to_numpy(
                dtype=float,
            ),
        )
        for element in elements
    }
    return _compute_power_analysis(
        values_by_element=values_by_element,
        feedstock_concentrations=feedstock_concentrations,
        effective_application_rate_kg_ha=effective_application_rate_kg_ha,
        n_eff=n_eff,
        bulk_density_kg_m3=bulk_density_kg_m3,
        sampling_depth_cm=sampling_depth_cm,
        elements=elements,
        n_baseline=len(paired),
        n_reporting_period=len(paired),
        variance_kind="paired",
    )


def compute_power_analysis_unpaired(
    *,
    baseline_samples: pd.DataFrame,
    reporting_period_samples: pd.DataFrame,
    feedstock_concentrations: Mapping[str, float],
    effective_application_rate_kg_ha: float | Np1DArray[np.floating],
    n_eff: float,
    bulk_density_kg_m3: float | Np1DArray[np.floating],
    sampling_depth_cm: float,
    elements: Sequence[ElementSymbol],
) -> list[PowerAnalysisResult]:
    """Compute power analysis per element, treating baseline and reporting-period as independent.

    Eq. 23 (unpaired form) - Minimum baseline samples required, at unequal variances and
    the allocation ratio the two events were actually sampled at:
        k = n_rp / n_bl
        n_req = (z_alpha + z_beta)^2 * (sigma_bl^2 + sigma_rp^2 / k) / delta^2
    Pass condition: n_eff >= n_req. Equal event sizes give k = 1 and the familiar
    ``sigma_bl^2 + sigma_rp^2``; sampling the reporting period harder (k > 1) shrinks that
    event's contribution to the noise, which is the point of allowing unequal counts.

    Use :func:`compute_power_analysis_paired` instead whenever baseline and
    reporting-period samples can be matched to the same location: this formula ignores
    any correlation between the two, which overstates the noise (and so the required
    sample count) whenever that correlation is positive — the common case, since a
    location's mineralogy tends to persist across sampling events.

    The two events are read from separate frames and may hold different numbers of
    samples, since nothing in the unpaired variance relates a baseline sample to a
    reporting-period one. Both must already have no null values in every requested
    element's column — this function raises rather than dropping incomplete rows.

    Args:
        baseline_samples: Pre-application soil samples, with a
            ``mass_fraction_<element>`` column per requested element and no nulls in them.
        reporting_period_samples: End-of-reporting-period soil samples, same columns. Need
            not have the same number of rows as ``baseline_samples``, or describe the same
            locations.
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
    values_by_element = {
        element: _ElementValues(
            baseline=check_measured_values(
                baseline_samples,
                mass_fraction_column_name(element),
                source="baseline sample",
            ),
            reporting_period=check_measured_values(
                reporting_period_samples,
                mass_fraction_column_name(element),
                source="reporting-period sample",
            ),
        )
        for element in elements
    }
    too_small = {
        event: n
        for event, n in (
            ("baseline", len(baseline_samples)),
            ("reporting-period", len(reporting_period_samples)),
        )
        if n < 2
    }
    if len(too_small) > 0:
        raise ValueError(
            f"Power analysis requires at least 2 samples per sampling event, got "
            f"{too_small!r}. A standard deviation is otherwise undefined.",
        )
    return _compute_power_analysis(
        values_by_element=values_by_element,
        feedstock_concentrations=feedstock_concentrations,
        effective_application_rate_kg_ha=effective_application_rate_kg_ha,
        n_eff=n_eff,
        bulk_density_kg_m3=bulk_density_kg_m3,
        sampling_depth_cm=sampling_depth_cm,
        elements=elements,
        n_baseline=len(baseline_samples),
        n_reporting_period=len(reporting_period_samples),
        variance_kind="unpaired",
    )


def _compute_power_analysis(
    *,
    values_by_element: Mapping[str, _ElementValues],
    feedstock_concentrations: Mapping[str, float],
    effective_application_rate_kg_ha: float | Np1DArray[np.floating],
    n_eff: float,
    bulk_density_kg_m3: float | Np1DArray[np.floating],
    sampling_depth_cm: float,
    elements: Sequence[ElementSymbol],
    n_baseline: int,
    n_reporting_period: int,
    variance_kind: _VarianceKind,
) -> list[PowerAnalysisResult]:
    """Shared implementation for the paired and unpaired variants.

    Everything downstream of reading the measurements is identical between the two except
    which variance term feeds ``n_required`` (Eq. 23). Each variant reads and validates its
    own inputs, because a matched frame and two independent events have nothing in common
    to validate: one checks that no pair is half-measured, the other that neither event is.
    """
    missing_keys = [e for e in elements if e not in feedstock_concentrations]
    if missing_keys:
        raise ValueError(
            f"Elements {missing_keys!r} not found in feedstock_concentrations "
            f"(available: {list(feedstock_concentrations)!r}).",
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
        baseline_values = values_by_element[element].baseline
        reporting_period_values = values_by_element[element].reporting_period

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

        match variance_kind:
            case "paired":
                sigma_diff = float((reporting_period_values - baseline_values).std())
                variance_term = sigma_diff**2
            case "unpaired":
                sigma_diff = None
                allocation_ratio = n_reporting_period / n_baseline
                variance_term = sigma_baseline**2 + sigma_reporting_period**2 / allocation_ratio

        if delta_mg_kg > 0:
            n_required = (_Z_ALPHA + _Z_BETA) ** 2 * variance_term / (delta_mg_kg**2)
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
                sigma_diff=sigma_diff,
                n_required=n_required,
                n_baseline=n_baseline,
                n_reporting_period=n_reporting_period,
                n_eff=n_eff,
                passes=n_eff >= n_required,
            ),
        )

    return results
