# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

from typing import Literal

import numpy as np

from isometric_calculation_library.utils.elements import atomic_weight
from isometric_calculation_library.utils.types import Np1DArray

Cation = Literal["Ca", "Mg", "Na", "K"]

MOLAR_MASS_CO2 = atomic_weight("C") + 2 * atomic_weight("O")
"""Molar mass of CO2 in g/mol."""


def _cation_to_charge(cation: Cation) -> int:
    """Get ionic charge for a given cation.

    Calcium and magnesium are divalent; sodium and potassium are monovalent, so they
    capture half as much CO2 per mole.
    """
    match cation:
        case "Ca" | "Mg":
            return 2
        case "Na" | "K":
            return 1


def compute_soil_mass_kg_ha(
    *,
    soil_bulk_density_kg_m3: float | Np1DArray[np.floating],
    depth_cm: float,
) -> float | Np1DArray[np.floating]:
    """Mass of soil in one hectare to a given depth (kg/ha).

    A hectare is 10,000 m2, so a layer ``depth_cm`` deep holds ``100 * depth_cm`` cubic
    metres per hectare.
    """
    volume_m3_per_ha = 100 * depth_cm
    return volume_m3_per_ha * soil_bulk_density_kg_m3


def compute_feedstock_soil_mass_ratio(
    *,
    application_rate_kg_ha: float | Np1DArray[np.floating],
    soil_bulk_density_kg_m3: float | Np1DArray[np.floating],
    depth_cm: float,
) -> Np1DArray[np.floating]:
    """Rock-to-soil mass ratio ``r = R / (BD * D * 10000)`` (Eq. 22), as replicates.

    Inputs can be bootstrap distributions, allowing an uncertain application rate and
    multi-location bulk density to propagate instead of collapsing to point estimates
    beforehand. These distributions must be the same length because they are paired
    replicate-for-replicate rather than crossed, ensuring the rate and soil mass describe the
    same draw within each replicate.
    """
    rate = np.atleast_1d(np.asarray(application_rate_kg_ha, dtype=float))
    soil_mass_kg_ha = np.atleast_1d(
        np.asarray(
            compute_soil_mass_kg_ha(
                soil_bulk_density_kg_m3=soil_bulk_density_kg_m3,
                depth_cm=depth_cm,
            ),
            dtype=float,
        ),
    )
    if len(rate) > 1 and len(soil_mass_kg_ha) > 1 and len(rate) != len(soil_mass_kg_ha):
        raise ValueError(
            f"application_rate_kg_ha has {len(rate)} replicates and "
            f"soil_bulk_density_kg_m3 has {len(soil_mass_kg_ha)}. Bootstrap both over the "
            "same number of runs so each replicate pairs a rate with the soil mass drawn "
            "alongside it, or pass one of them as a scalar.",
        )
    return rate / soil_mass_kg_ha


def convert_to_equivalent_soil_mass(
    *,
    mass_fraction_mg_kg: Np1DArray[np.floating],
    measured_bulk_density_kg_m3: float | Np1DArray[np.floating],
    reference_bulk_density_kg_m3: float | Np1DArray[np.floating],
) -> Np1DArray[np.floating]:
    """Restate a mass fraction on the soil mass of a different sampling event.

    A mass fraction is per kilogram of soil, but a soil sample is taken to a fixed depth and
    so covers a fixed *volume*. When bulk density differs between two sampling events, each
    event's kilogram represents a different depth of profile, and their mass fractions are not
    directly comparable: an unchanged mass of an element per unit volume reports as a different
    mg/kg purely because the mass of soil in the sampled volume moved.

    Scaling by ``measured / reference`` puts the measured event onto the reference event's soil
    mass. The reference is normally the baseline, that being the basis a feedstock-to-soil mass
    balance is defined against. Both densities accept bootstrap replicates, so the correction
    can carry its own sampling error rather than being applied as an exactly-known ratio.

    This matters more than the size of the density change suggests, because an immobile-tracer
    difference ``T_rp - T_bl`` is a small difference of large numbers: a density change of a
    few percent is a much larger relative change in that difference, and flows straight into
    the derived mixing ratio.

    Feedstock composition is not a candidate for this conversion. It describes a material
    rather than a field measurement over a sampled volume, so it has no soil mass basis.
    """
    return mass_fraction_mg_kg * (
        np.asarray(measured_bulk_density_kg_m3, dtype=float)
        / np.asarray(reference_bulk_density_kg_m3, dtype=float)
    )


def compute_residual_equivalent_soil_mass_ratio(
    *,
    measured_bulk_density_replicates_kg_m3: Np1DArray[np.floating],
    reference_bulk_density_replicates_kg_m3: Np1DArray[np.floating],
) -> Np1DArray[np.floating]:
    """Per-replicate correction to apply on top of an already point-corrected mass fraction.

    ``convert_to_equivalent_soil_mass`` is normally applied once to the per-sample values, at
    the two events' mean densities, so that every consumer working from measured values -
    spatial autocorrelation, power analysis, significance testing - sees the same corrected
    difference. Bootstrap replicates drawn from those corrected values therefore already carry
    the point correction.

    Multiplying such replicates by this factor swaps that fixed correction for a resampled one,
    without applying the conversion twice::

        (rho_measured_boot / rho_reference_boot) / (mean(rho_measured) / mean(rho_reference))

    Use it when the two events' densities are close enough that the difference between them is
    within sampling error: treating the ratio as exactly known would then hand its full effect
    to the result on evidence that does not support it.
    """
    measured = np.asarray(measured_bulk_density_replicates_kg_m3, dtype=float)
    reference = np.asarray(reference_bulk_density_replicates_kg_m3, dtype=float)
    if len(measured) != len(reference):
        raise ValueError(
            f"measured_bulk_density_replicates_kg_m3 has {len(measured)} replicates and "
            f"reference_bulk_density_replicates_kg_m3 has {len(reference)}. Bootstrap both "
            "over the same number of runs so each replicate pairs the two densities drawn "
            "alongside each other.",
        )
    if len(measured) == 0:
        raise ValueError(
            "Bulk density replicate arrays are empty; the residual ratio is undefined without "
            "a bootstrap to take the mean of.",
        )
    point_ratio = float(np.mean(measured)) / float(np.mean(reference))
    return (measured / reference) / point_ratio


def convert_mg_kg_to_kg_ha(
    *,
    soil_mass_fraction_mg_kg: Np1DArray[np.floating],
    soil_bulk_density_kg_m3: Np1DArray[np.floating],
    depth_cm: float,
) -> Np1DArray[np.floating]:
    """Convert soil mass fraction from mg/kg to kg/ha."""
    soil_mass_kg_per_ha = compute_soil_mass_kg_ha(
        soil_bulk_density_kg_m3=soil_bulk_density_kg_m3,
        depth_cm=depth_cm,
    )
    return soil_mass_fraction_mg_kg * soil_mass_kg_per_ha / 1e6


def convert_kg_ha_to_mg_kg(
    *,
    mass_per_area_kg_ha: Np1DArray[np.floating],
    soil_bulk_density_kg_m3: Np1DArray[np.floating],
    depth_cm: float,
) -> Np1DArray[np.floating]:
    """Convert mass per area from kg/ha to soil mass fraction mg/kg."""
    soil_mass_kg_per_ha = compute_soil_mass_kg_ha(
        soil_bulk_density_kg_m3=soil_bulk_density_kg_m3,
        depth_cm=depth_cm,
    )
    return mass_per_area_kg_ha * 1e6 / soil_mass_kg_per_ha


def convert_cation_kg_to_co2_kg(
    *,
    cation_kg: Np1DArray[np.floating],
    cation: Cation,
) -> Np1DArray[np.floating]:
    """Convert cation mass to CO2 mass equivalent.

    Each mole of cation captures one mole of CO2 per unit of ionic charge, so a divalent
    cation (Ca²⁺, Mg²⁺) corresponds to 2 moles of CO2 and a monovalent one (Na⁺, K⁺) to 1.
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
