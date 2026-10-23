# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

import operator
from functools import reduce
from typing import NamedTuple

import numpy as np

from isometric_calculation_library.enhanced_weathering.utils.conversions import (
    Cation,
    convert_cation_kg_to_co2_kg,
    convert_mg_kg_to_kg_ha,
)
from isometric_calculation_library.utils.types import Np1DArray


def compute_cation_stock_kg_ha(
    concentration_mg_kg: Np1DArray[np.floating],
    bulk_density_kg_m3: Np1DArray[np.floating],
    depth_cm: float,
) -> Np1DArray[np.floating]:
    """Convert soil cation concentration to stock per hectare.

    Args:
        concentration_mg_kg: Cation concentration in soil (mg/kg).
        bulk_density_kg_m3: Soil bulk density (kg/m³).
        depth_cm: Sampling depth in cm.

    Returns:
        Cation stock in kg/ha.
    """
    return convert_mg_kg_to_kg_ha(
        soil_mass_fraction_mg_kg=concentration_mg_kg,
        soil_bulk_density_kg_m3=bulk_density_kg_m3,
        depth_cm=depth_cm,
    )


def compute_depth_weighted_concentration_kg_ha(
    concentration_mg_kg: Np1DArray[np.floating],
    depth_cm: Np1DArray[np.floating],
) -> Np1DArray[np.floating]:
    """Scale per-sample concentrations by their individual sampling depths.

    Returns per-sample depth-weighted values in kg/ha (excluding bulk density),
    suitable for passing to a bootstrap resampler. Multiplying the bootstrapped
    mean by a bootstrapped bulk density then gives the cation stock.

    Using per-sample depths avoids the bias introduced by computing
    mean(concentration) * mean(depth) when depth and concentration vary jointly
    across locations.

    Args:
        concentration_mg_kg: Per-sample cation concentration (mg/kg).
        depth_cm: Per-sample sampling depth (cm), one value per sample.

    Returns:
        Per-sample depth-weighted concentration (kg/ha / (kg/m³)), i.e. the
        stock divided by bulk density. Multiply by a bootstrapped bulk density
        to obtain stock in kg/ha.
    """
    volume_m3_per_ha = 100.0 * depth_cm
    return concentration_mg_kg * volume_m3_per_ha / 1e6


def compute_feedstock_cation_kg_ha(
    feedstock_amount_kg_ha: float | Np1DArray[np.floating],
    feedstock_cation_mg_kg: float | Np1DArray[np.floating],
) -> float | Np1DArray[np.floating]:
    """Calculate cations added from feedstock application.

    Args:
        feedstock_amount_kg_ha: Feedstock applied (kg/ha). Can be a known rate
            (total cation method) or estimated from tracer (tracer method).
        feedstock_cation_mg_kg: Cation concentration in feedstock (mg/kg).

    Returns:
        Cations added from feedstock in kg/ha.
    """
    feedstock_cation_kg_kg = feedstock_cation_mg_kg / 1e6
    return feedstock_amount_kg_ha * feedstock_cation_kg_kg


def compute_cdr_from_stocks(
    baseline_stock_kg_ha: Np1DArray[np.floating],
    end_of_reporting_period_stock_kg_ha: Np1DArray[np.floating],
    feedstock_cation_kg_ha: float | Np1DArray[np.floating],
    control_dissolved_kg_ha: Np1DArray[np.floating],
) -> tuple[Np1DArray[np.floating], Np1DArray[np.floating]]:
    """Compute CDR from pre-calculated cation stocks.

    The core CDR calculation:
        post_application = baseline + feedstock
        delta_raw = post_application - end_of_reporting_period  (before control correction)
        cdr = delta_raw - control_dissolved  (after control correction)
    """
    post_application_kg_ha = baseline_stock_kg_ha + feedstock_cation_kg_ha
    delta_raw = post_application_kg_ha - end_of_reporting_period_stock_kg_ha
    cdr = delta_raw - control_dissolved_kg_ha
    return delta_raw, cdr


def compute_cdr_density(
    bulk_density_baseline_kg_m3: Np1DArray[np.floating],
    bulk_density_end_of_reporting_period_kg_m3: Np1DArray[np.floating],
    depth_cm: float,
    feedstock_amount_kg_ha: float | Np1DArray[np.floating],
    feedstock_cation_mg_kg: float,
    soil_cation_baseline_mg_kg: Np1DArray[np.floating],
    soil_cation_end_of_reporting_period_mg_kg: Np1DArray[np.floating],
    control_dissolved_kg_ha: Np1DArray[np.floating],
) -> tuple[Np1DArray[np.floating], Np1DArray[np.floating]]:
    """Compute CDR density from concentrations and feedstock amount.

    This is the main CDR calculation function that works for both:
    - **Total Cation method**: feedstock_amount_kg_ha is the known application rate
    - **Tracer method**: feedstock_amount_kg_ha is estimated from tracer mass balance

    CDR = (baseline_stock + feedstock_cations) - end_of_reporting_period_stock - control_dissolved
    """
    baseline_stock = compute_cation_stock_kg_ha(
        soil_cation_baseline_mg_kg,
        bulk_density_baseline_kg_m3,
        depth_cm,
    )

    end_stock = compute_cation_stock_kg_ha(
        soil_cation_end_of_reporting_period_mg_kg,
        bulk_density_end_of_reporting_period_kg_m3,
        depth_cm,
    )

    feedstock_cation = compute_feedstock_cation_kg_ha(
        feedstock_amount_kg_ha,
        feedstock_cation_mg_kg,
    )

    return compute_cdr_from_stocks(
        baseline_stock,
        end_stock,
        feedstock_cation,
        control_dissolved_kg_ha,
    )


def compute_control_dissolved_kg_ha(
    control_baseline_mg_kg: Np1DArray[np.floating],
    control_end_of_reporting_period_mg_kg: Np1DArray[np.floating],
    bulk_density_baseline_kg_m3: Np1DArray[np.floating],
    bulk_density_end_of_reporting_period_kg_m3: Np1DArray[np.floating],
    depth_cm: float,
) -> Np1DArray[np.floating]:
    """Calculate cations that dissolved/left the control area.

    Positive values indicate cations exported (left the soil).
    Negative values indicate cations accumulated.
    """
    baseline_stock = compute_cation_stock_kg_ha(
        control_baseline_mg_kg,
        bulk_density_baseline_kg_m3,
        depth_cm,
    )
    end_stock = compute_cation_stock_kg_ha(
        control_end_of_reporting_period_mg_kg,
        bulk_density_end_of_reporting_period_kg_m3,
        depth_cm,
    )
    return baseline_stock - end_stock


def convert_cdr_to_co2(
    cdr_cation_kg_ha: Np1DArray[np.floating],
    cation: Cation,
    area_hectares: float,
) -> tuple[Np1DArray[np.floating], Np1DArray[np.floating], Np1DArray[np.floating]]:
    """Convert cation CDR to CO2 equivalent and scale by area.

    Uses stoichiometry: one mole of CO2 per unit of ionic charge, so 2 moles per mole of
    a divalent cation (Ca²⁺, Mg²⁺) and 1 per mole of a monovalent one (Na⁺, K⁺).

    Args:
        cdr_cation_kg_ha: CDR in kg of cation per hectare.
        cation: Cation element symbol ("Ca", "Mg", "Na", or "K").
        area_hectares: Treatment area size in hectares.

    Returns:
        Tuple of (co2_kg_ha, total_co2_kg, total_co2_tonnes).
    """
    co2_kg_ha = convert_cation_kg_to_co2_kg(cation_kg=cdr_cation_kg_ha, cation=cation)
    total_co2_kg = co2_kg_ha * area_hectares
    total_co2_tonnes = total_co2_kg / 1000
    return co2_kg_ha, total_co2_kg, total_co2_tonnes


def compute_weathered_fraction_standard_tca(
    *,
    baseline_post_application_concentration: Np1DArray[np.floating],
    resampling_concentration: Np1DArray[np.floating],
    baseline_concentration: Np1DArray[np.floating],
    cation_dissolved_kg_ha: Np1DArray[np.floating],
) -> Np1DArray[np.floating]:
    """Compute weathered fraction using measured post-application baseline samples.

    Weathered fraction represents how much of the applied feedstock cation has
    dissolved during the reporting period, accounting for background weathering
    via a control correction term.

    Formula:
        Fw = (BLP - R1 - control) / (BLP - BL)

    Where BLP is measured post-application baseline, R1 is end-of-reporting-period,
    BL is pre-application baseline, and control is background cation dissolution.

    Locations where the denominator (BLP - BL) is zero — meaning no feedstock
    signal is detectable — produce NaN. These NaN values propagate through
    subsequent aggregation functions (e.g. nanmean, nanpercentile) without
    contaminating valid bootstrap samples.

    Args:
        baseline_post_application_concentration: Measured post-application baseline
            cation concentration (kg/ha), bootstrapped array.
        resampling_concentration: End-of-reporting-period cation concentration
            (kg/ha), bootstrapped array.
        baseline_concentration: Pre-application baseline cation concentration
            (kg/ha), bootstrapped array.
        cation_dissolved_kg_ha: Background cation dissolution from control plots
            (kg/ha), bootstrapped array.
    """
    denominator = baseline_post_application_concentration - baseline_concentration
    numerator = (
        baseline_post_application_concentration - resampling_concentration - cation_dissolved_kg_ha
    )
    valid = np.abs(denominator) > 0
    result = np.full_like(numerator, np.nan)
    result[valid] = numerator[valid] / denominator[valid]
    return result


class WeatheredFractionResult(NamedTuple):
    """Result from compute_weathered_fraction."""

    weathered_fraction: Np1DArray[np.floating]
    """CDR as proportion of theoretical maximum CO2."""

    theoretical_potential_tco2: float | Np1DArray[np.floating]
    """Maximum CO2 if all applied cations dissolved completely.

    A number when every feedstock concentration was a number, otherwise one value per replicate.
    """


def compute_weathered_fraction(
    *,
    cdr_tco2: Np1DArray[np.floating],
    feedstock_amount_kg_ha: float,
    area_hectares: float,
    feedstock_ca_mg_kg: float | Np1DArray[np.floating] | None = None,
    feedstock_mg_mg_kg: float | Np1DArray[np.floating] | None = None,
    feedstock_na_mg_kg: float | Np1DArray[np.floating] | None = None,
) -> WeatheredFractionResult:
    """Compute weathered fraction and theoretical potential CO2.

    The theoretical potential is the CO2 the applied feedstock would capture if all of the
    cations passed dissolved. Pass the concentrations of exactly the cations whose CO2 is in
    ``cdr_tco2``, meaning the credited ones: a cation left out of the CDR, for example because it
    failed the power analysis, must be left out here too, and a credited cation must be passed,
    or the weathered fraction is wrong.

    Each feedstock concentration can be a number or a bootstrap array with one value per
    replicate. Pass the arrays the CDR was calculated from, so numerator and denominator use
    the same feedstock in each replicate.
    """
    concentrations_by_cation: tuple[tuple[Cation, float | Np1DArray[np.floating] | None], ...] = (
        ("Ca", feedstock_ca_mg_kg),
        ("Mg", feedstock_mg_mg_kg),
        ("Na", feedstock_na_mg_kg),
    )
    potential_co2_kg_ha_by_cation = [
        convert_cation_kg_to_co2_kg(
            cation_kg=np.asarray(
                compute_feedstock_cation_kg_ha(feedstock_amount_kg_ha, concentration),
                dtype=float,
            ),
            cation=cation,
        )
        for cation, concentration in concentrations_by_cation
        if concentration is not None
    ]
    if len(potential_co2_kg_ha_by_cation) == 0:
        raise ValueError(
            "Pass the feedstock concentration of at least one credited cation (Ca, Mg or Na).",
        )
    potential_co2_kg_ha = reduce(operator.add, potential_co2_kg_ha_by_cation)
    potential_tco2_array = np.asarray(potential_co2_kg_ha * area_hectares / 1000, dtype=float)

    theoretical_potential_tco2: float | Np1DArray[np.floating]
    if potential_tco2_array.ndim == 0:
        theoretical_potential_tco2 = float(potential_tco2_array)
        if theoretical_potential_tco2 <= 0:
            raise ValueError(
                "theoretical_potential_tco2 must be positive to compute a weathered "
                f"fraction, got {theoretical_potential_tco2}. This happens when "
                "feedstock_amount_kg_ha, the feedstock cation concentrations, or "
                "area_hectares is zero — check the feedstock and area inputs.",
            )
    else:
        theoretical_potential_tco2 = potential_tco2_array
        n_non_positive = int(np.count_nonzero(~(potential_tco2_array > 0)))
        if n_non_positive > 0:
            raise ValueError(
                "theoretical_potential_tco2 must be positive to compute a weathered "
                f"fraction, but {n_non_positive} of {len(potential_tco2_array)} replicates are "
                "not. This happens when feedstock_amount_kg_ha, the feedstock cation "
                "concentrations, or area_hectares is zero — check the feedstock and area inputs.",
            )
    return WeatheredFractionResult(
        weathered_fraction=cdr_tco2 / theoretical_potential_tco2,
        theoretical_potential_tco2=theoretical_potential_tco2,
    )
