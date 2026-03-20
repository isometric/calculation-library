# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

from typing import NamedTuple

import numpy as np

from isometric_calculation_library.enhanced_weathering.utils.conversions import (
    Cation,
    convert_cation_kg_to_co2_kg,
    convert_mg_kg_to_kg_ha,
)
from isometric_calculation_library.enhanced_weathering.utils.types import Np1DArray


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

    Uses stoichiometry: 2 moles CO2 per mole of divalent cation (Ca²⁺, Mg²⁺).

    Args:
        cdr_cation_kg_ha: CDR in kg of cation per hectare.
        cation: Cation element symbol ("Ca" or "Mg").
        area_hectares: Treatment area size in hectares.

    Returns:
        Tuple of (co2_kg_ha, total_co2_kg, total_co2_tonnes).
    """
    co2_kg_ha = convert_cation_kg_to_co2_kg(cation_kg=cdr_cation_kg_ha, cation=cation)
    total_co2_kg = co2_kg_ha * area_hectares
    total_co2_tonnes = total_co2_kg / 1000
    return co2_kg_ha, total_co2_kg, total_co2_tonnes


class WeatheredFractionResult(NamedTuple):
    """Result from compute_weathered_fraction."""

    weathered_fraction: Np1DArray[np.floating]
    """CDR as proportion of theoretical maximum CO2."""

    theoretical_potential_tco2: float
    """Maximum CO2 if all applied cations dissolved completely."""


def compute_weathered_fraction(
    *,
    cdr_tco2: Np1DArray[np.floating],
    feedstock_amount_kg_ha: float,
    feedstock_ca_mg_kg: float,
    feedstock_mg_mg_kg: float,
    area_hectares: float,
) -> WeatheredFractionResult:
    """Compute weathered fraction and theoretical potential CO2."""
    ca_kg_ha = compute_feedstock_cation_kg_ha(feedstock_amount_kg_ha, feedstock_ca_mg_kg)
    mg_kg_ha = compute_feedstock_cation_kg_ha(feedstock_amount_kg_ha, feedstock_mg_mg_kg)
    pot_co2_kg_ha = float(
        convert_cation_kg_to_co2_kg(cation_kg=np.array([ca_kg_ha]), cation="Ca")[0]
        + convert_cation_kg_to_co2_kg(cation_kg=np.array([mg_kg_ha]), cation="Mg")[0],
    )
    theoretical_potential_tco2 = (pot_co2_kg_ha * area_hectares) / 1000
    return WeatheredFractionResult(
        weathered_fraction=cdr_tco2 / theoretical_potential_tco2,
        theoretical_potential_tco2=theoretical_potential_tco2,
    )
