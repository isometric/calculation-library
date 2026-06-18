# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

from typing import Literal

import numpy as np

from isometric_calculation_library.enhanced_weathering.utils.types import Np1DArray

type ImmobileTracer = Literal["Zr", "Ti"]
"""Immobile tracer elements used for feedstock-soil mass balance."""


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
    control_correction_ratio: float | Np1DArray[np.floating] = 1.0,
) -> Np1DArray[np.floating]:
    """Compute fraction of feedstock cation dissolved using immobile tracer method.

    f_d = ((1 + m) / m) * (C_post * cc - C_rp) / C_feed

    Infinite values (from m = 0 or C_feed = 0) are replaced with NaN.
    """
    corrected_post_application = post_application_concentration_mg_kg * control_correction_ratio

    fraction_dissolved = ((1 + feedstock_soil_mass_ratio) / feedstock_soil_mass_ratio) * (
        (corrected_post_application - soil_end_of_reporting_period_mg_kg) / feedstock_mg_kg
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
    soil_mass_kg_per_ha = soil_bulk_density_kg_m3 * depth_cm * 100
    return feedstock_soil_mass_ratio * soil_mass_kg_per_ha
