# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

from typing import Literal

import numpy as np

from isometric_calculation_library.enhanced_weathering.utils.types import Np1DArray

Cation = Literal["Ca", "Mg"]

MOLAR_MASS_CO2 = 44.01
"""Molar mass of CO2 in g/mol."""


def _cation_to_molar_mass_g_per_mol(cation: Cation) -> float:
    """Get molar mass for a given cation."""
    match cation:
        case "Ca":
            return 40.08
        case "Mg":
            return 24.3


def _cation_to_charge(cation: Cation) -> int:
    """Get ionic charge for a given cation."""
    match cation:
        case "Ca":
            return 2
        case "Mg":
            return 2


def convert_mg_kg_to_kg_ha(
    *,
    soil_mass_fraction_mg_kg: Np1DArray[np.floating],
    soil_bulk_density_kg_m3: Np1DArray[np.floating],
    depth_cm: float,
) -> Np1DArray[np.floating]:
    """Convert soil mass fraction from mg/kg to kg/ha."""
    volume_m3_per_ha = 100 * depth_cm
    soil_mass_kg_per_ha = volume_m3_per_ha * soil_bulk_density_kg_m3
    return soil_mass_fraction_mg_kg * soil_mass_kg_per_ha / 1e6


def convert_kg_ha_to_mg_kg(
    *,
    mass_per_area_kg_ha: Np1DArray[np.floating],
    soil_bulk_density_kg_m3: Np1DArray[np.floating],
    depth_cm: float,
) -> Np1DArray[np.floating]:
    """Convert mass per area from kg/ha to soil mass fraction mg/kg."""
    volume_m3_per_ha = 100 * depth_cm
    soil_mass_kg_per_ha = volume_m3_per_ha * soil_bulk_density_kg_m3
    return mass_per_area_kg_ha * 1e6 / soil_mass_kg_per_ha


def convert_cation_kg_to_co2_kg(
    *,
    cation_kg: Np1DArray[np.floating],
    cation: Cation,
) -> Np1DArray[np.floating]:
    """Convert cation mass to CO2 mass equivalent.

    For divalent cations (Ca²⁺, Mg²⁺), each mole of cation dissolved
    corresponds to one mole of CO2 captured per unit charge via carbonate
    weathering.
    """
    molar_mass = _cation_to_molar_mass_g_per_mol(cation)
    charge = _cation_to_charge(cation)
    return cation_kg * charge * MOLAR_MASS_CO2 / molar_mass
