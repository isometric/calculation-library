# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

import numpy as np
import pytest

from isometric_calculation_library.enhanced_weathering.utils.conversions import (
    convert_cation_kg_to_co2_kg,
    convert_kg_ha_to_mg_kg,
    convert_mg_kg_to_kg_ha,
)


def test_mg_kg_to_kg_ha_single_value() -> None:
    """1 mg/kg at BD=1000 kg/m3 and depth=30cm => 1 * (100*30) * 1000 / 1e6 = 3 kg/ha."""
    result = convert_mg_kg_to_kg_ha(
        soil_mass_fraction_mg_kg=np.array([1.0]),
        soil_bulk_density_kg_m3=np.array([1000.0]),
        depth_cm=30.0,
    )
    assert result == pytest.approx([3.0])


def test_kg_ha_to_mg_kg_single_value() -> None:
    """Inverse of mg_kg_to_kg_ha: 3 kg/ha => 1 mg/kg."""
    result = convert_kg_ha_to_mg_kg(
        mass_per_area_kg_ha=np.array([3.0]),
        soil_bulk_density_kg_m3=np.array([1000.0]),
        depth_cm=30.0,
    )
    assert result == pytest.approx([1.0])


def test_mg_kg_and_kg_ha_are_inverses() -> None:
    rng = np.random.default_rng(42)
    concentrations = rng.uniform(10, 500, size=20)
    bulk_densities = rng.uniform(800, 1200, size=20)
    depth = 25.0

    kg_ha = convert_mg_kg_to_kg_ha(
        soil_mass_fraction_mg_kg=concentrations,
        soil_bulk_density_kg_m3=bulk_densities,
        depth_cm=depth,
    )
    roundtrip = convert_kg_ha_to_mg_kg(
        mass_per_area_kg_ha=kg_ha,
        soil_bulk_density_kg_m3=bulk_densities,
        depth_cm=depth,
    )
    np.testing.assert_allclose(roundtrip, concentrations)


def test_convert_ca_kg_to_co2_kg() -> None:
    """1 kg Ca => charge * M_CO2 / M_Ca kg CO2 = 2 * 44.00955 / 40.078."""
    result = convert_cation_kg_to_co2_kg(
        cation_kg=np.array([1.0]),
        cation="Ca",
    )
    expected = 2 * 44.00955 / 40.078
    assert result == pytest.approx([expected])


def test_convert_mg_kg_to_co2_kg() -> None:
    """1 kg Mg => charge * M_CO2 / M_Mg kg CO2 = 2 * 44.00955 / 24.3051."""
    result = convert_cation_kg_to_co2_kg(
        cation_kg=np.array([1.0]),
        cation="Mg",
    )
    expected = 2 * 44.00955 / 24.3051
    assert result == pytest.approx([expected])


def test_mg_kg_to_kg_ha_vector() -> None:
    """Different concentrations and bulk densities produce element-wise results."""
    result = convert_mg_kg_to_kg_ha(
        soil_mass_fraction_mg_kg=np.array([1.0, 2.0, 3.0]),
        soil_bulk_density_kg_m3=np.array([1000.0, 1200.0, 800.0]),
        depth_cm=30.0,
    )
    expected = np.array([
        1.0 * (100 * 30) * 1000.0 / 1e6,
        2.0 * (100 * 30) * 1200.0 / 1e6,
        3.0 * (100 * 30) * 800.0 / 1e6,
    ])
    np.testing.assert_allclose(result, expected)


def test_kg_ha_to_mg_kg_vector() -> None:
    """Different masses and bulk densities produce element-wise results."""
    result = convert_kg_ha_to_mg_kg(
        mass_per_area_kg_ha=np.array([3.0, 7.2, 1.92]),
        soil_bulk_density_kg_m3=np.array([1000.0, 1200.0, 800.0]),
        depth_cm=30.0,
    )
    expected = np.array([
        3.0 * 1e6 / (100 * 30 * 1000.0),
        7.2 * 1e6 / (100 * 30 * 1200.0),
        1.92 * 1e6 / (100 * 30 * 800.0),
    ])
    np.testing.assert_allclose(result, expected)


def test_convert_cation_kg_to_co2_kg_vector() -> None:
    """Multiple cation masses produce element-wise CO2 equivalents."""
    result = convert_cation_kg_to_co2_kg(
        cation_kg=np.array([1.0, 2.0, 5.0]),
        cation="Ca",
    )
    expected = np.array([1.0, 2.0, 5.0]) * 2 * 44.00955 / 40.078
    np.testing.assert_allclose(result, expected)


def test_co2_conversion_scales_linearly() -> None:
    single = convert_cation_kg_to_co2_kg(cation_kg=np.array([1.0]), cation="Ca")
    double = convert_cation_kg_to_co2_kg(cation_kg=np.array([2.0]), cation="Ca")
    assert double == pytest.approx(single * 2)
