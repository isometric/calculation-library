# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

from typing import Literal

import numpy as np

from isometric_calculation_library.enhanced_weathering.utils.conversions import (
    compute_soil_mass_kg_ha,
)
from isometric_calculation_library.utils.types import Np1DArray

type ImmobileTracer = Literal["Zr", "Ti", "Cu"]
"""Tracer elements accepted for the feedstock-soil mass balance.

Zirconium and titanium are the conservative choices: both sit in resistant phases and are
effectively immobile over a reporting period, so an enrichment traces feedstock mass.

Copper is admitted because deployments characterise it, but it is weakly mobile — it sorbs
to organic matter, is taken up by plants, and is applied in some fungicides. A model using
it should read the tracer resolvability output before trusting the result, and treat a
soil-derived application rate well above the operational one as a sign the tracer is not
behaving conservatively."""


def compute_mass_ratio_from_immobile_tracer(
    *,
    feedstock_tracer_mg_kg: float | Np1DArray[np.floating],
    soil_baseline_tracer_mg_kg: Np1DArray[np.floating],
    soil_end_of_reporting_period_tracer_mg_kg: Np1DArray[np.floating],
) -> Np1DArray[np.floating]:
    """Compute feedstock-to-soil mass ratio from immobile tracer mass balance.

    m = m_f / m_s = (T_rp - T_bl) / (T_feed - T_rp)

    Infinite values (from a zero denominator, i.e. feedstock tracer equal to the
    soil reporting-period tracer) are replaced with NaN, matching
    `compute_fraction_dissolved`. Note this only catches an exactly-zero
    denominator, not merely implausible (e.g. negative) finite ratios.
    """
    mass_ratio = (soil_end_of_reporting_period_tracer_mg_kg - soil_baseline_tracer_mg_kg) / (
        feedstock_tracer_mg_kg - soil_end_of_reporting_period_tracer_mg_kg
    )
    return np.where(np.isinf(mass_ratio), np.nan, mass_ratio)


def compute_post_application_concentration(
    *,
    feedstock_soil_mass_ratio: Np1DArray[np.floating],
    soil_baseline_mg_kg: Np1DArray[np.floating],
    feedstock_mg_kg: float | Np1DArray[np.floating],
) -> Np1DArray[np.floating]:
    """Compute post-application cation concentration from soil-feedstock mixing.

    C_post = (C_bl + m * C_feed) / (1 + m)
    """
    return (soil_baseline_mg_kg + feedstock_soil_mass_ratio * feedstock_mg_kg) / (
        1 + feedstock_soil_mass_ratio
    )


def compute_fraction_dissolved(
    *,
    feedstock_soil_mass_ratio: Np1DArray[np.floating],
    post_application_concentration_mg_kg: Np1DArray[np.floating],
    soil_end_of_reporting_period_mg_kg: Np1DArray[np.floating],
    feedstock_mg_kg: float | Np1DArray[np.floating],
    control_correction_delta_mg_kg: float | Np1DArray[np.floating] = 0.0,
    control_correction_ratio: float | Np1DArray[np.floating] = 1.0,
) -> Np1DArray[np.floating]:
    """Compute fraction of feedstock cation dissolved using immobile tracer method.

    f_d = ((1 + m) / m) * (C_post * cc - C_rp - delta) / C_feed

    ``control_correction_delta_mg_kg`` is the additive background loss measured in the
    control plots, signed so that a positive delta means the control lost cations and
    the correction therefore reduces the fraction dissolved. Pass the distribution from
    ``apply_control_correction_delta_paired`` / ``_unpaired``.

    ``control_correction_ratio`` is the older multiplicative form and is deprecated. Prefer
    the delta.

    Infinite values (from m = 0 or C_feed = 0) are replaced with NaN.
    """
    fraction_dissolved = ((1 + feedstock_soil_mass_ratio) / feedstock_soil_mass_ratio) * (
        (
            post_application_concentration_mg_kg * control_correction_ratio
            - soil_end_of_reporting_period_mg_kg
            - control_correction_delta_mg_kg
        )
        / feedstock_mg_kg
    )

    return np.where(np.isinf(fraction_dissolved), np.nan, fraction_dissolved)


def compute_application_rate_from_tracer(
    *,
    feedstock_soil_mass_ratio: Np1DArray[np.floating],
    soil_bulk_density_kg_m3: Np1DArray[np.floating],
    depth_cm: float,
) -> Np1DArray[np.floating]:
    """Estimate feedstock application rate from tracer-derived mass ratio.

    app_rate_kg_ha = m * BD * depth_cm * 100
    """
    soil_mass_kg_per_ha = compute_soil_mass_kg_ha(
        soil_bulk_density_kg_m3=soil_bulk_density_kg_m3,
        depth_cm=depth_cm,
    )
    return feedstock_soil_mass_ratio * soil_mass_kg_per_ha
