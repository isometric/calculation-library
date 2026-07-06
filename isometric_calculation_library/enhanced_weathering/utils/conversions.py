# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

from typing import Literal

import numpy as np

from isometric_calculation_library.enhanced_weathering.utils.types import Np1DArray
from isometric_calculation_library.utils.elements import atomic_weight

Cation = Literal["Ca", "Mg"]

MOLAR_MASS_CO2 = atomic_weight("C") + 2 * atomic_weight("O")
"""Molar mass of CO2 in g/mol."""


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

    Each mole of cation captures one mole of CO2 per unit of ionic charge,
    so a divalent cation (Ca²⁺, Mg²⁺) corresponds to 2 moles of CO2.
    """
    molar_mass = atomic_weight(cation)
    charge = _cation_to_charge(cation)
    return cation_kg * charge * MOLAR_MASS_CO2 / molar_mass


def convert_cation_kg_to_charge_equivalents(
    *,
    cation_kg: Np1DArray[np.floating],
    cation: Cation,
) -> Np1DArray[np.floating]:
    """Convert cation mass (kg) to charge equivalents (mol of charge).

    equivalents = (mass / molar_mass) * charge, i.e. moles of the cation times
    its ionic valence. Expresses alkalinity contributions on a common charge
    basis across cations.
    """
    molar_mass_kg_per_mol = atomic_weight(cation) / 1000
    charge = _cation_to_charge(cation)
    return cation_kg / molar_mass_kg_per_mol * charge
