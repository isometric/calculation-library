# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

import numpy as np
import pytest

from isometric_calculation_library.enhanced_weathering.utils.conversions import (
    compute_residual_equivalent_soil_mass_ratio,
    convert_cation_kg_to_charge_equivalents,
    convert_cation_kg_to_co2_kg,
    convert_kg_ha_to_mg_kg,
    convert_mg_kg_to_kg_ha,
    convert_to_equivalent_soil_mass,
)
from isometric_calculation_library.utils.elements import atomic_weight


def test_cation_kg_to_charge_equivalents_ca() -> None:
    """1 kg Ca = (1000 g / molar_mass) * charge 2 equivalents."""
    result = convert_cation_kg_to_charge_equivalents(cation_kg=np.array([1.0]), cation="Ca")
    expected = 1.0 / (atomic_weight("Ca") / 1000) * 2
    assert result[0] == pytest.approx(expected)


def test_cation_kg_to_charge_equivalents_scales_linearly() -> None:
    result = convert_cation_kg_to_charge_equivalents(
        cation_kg=np.array([1.0, 2.0, 4.0]),
        cation="Mg",
    )
    np.testing.assert_allclose(result / result[0], [1.0, 2.0, 4.0])


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


def test_equivalent_soil_mass_is_identity_when_densities_match() -> None:
    """Equal densities mean the two events already share a soil mass basis."""
    values = np.array([100.0, 120.0, 140.0])
    result = convert_to_equivalent_soil_mass(
        mass_fraction_mg_kg=values,
        measured_bulk_density_kg_m3=1500.0,
        reference_bulk_density_kg_m3=1500.0,
    )
    np.testing.assert_allclose(result, values)


def test_equivalent_soil_mass_scales_down_for_looser_soil() -> None:
    """A less dense measured event holds less soil per unit volume, so mg/kg reads high."""
    result = convert_to_equivalent_soil_mass(
        mass_fraction_mg_kg=np.array([100.0]),
        measured_bulk_density_kg_m3=1470.0,
        reference_bulk_density_kg_m3=1500.0,
    )
    np.testing.assert_allclose(result, np.array([98.0]))


def test_equivalent_soil_mass_accepts_density_replicates() -> None:
    """Bootstrap densities propagate element-wise, one ratio per replicate."""
    result = convert_to_equivalent_soil_mass(
        mass_fraction_mg_kg=np.array([100.0, 100.0]),
        measured_bulk_density_kg_m3=np.array([1470.0, 1530.0]),
        reference_bulk_density_kg_m3=np.array([1500.0, 1500.0]),
    )
    np.testing.assert_allclose(result, np.array([98.0, 102.0]))


def test_equivalent_soil_mass_amplifies_a_small_difference_of_large_numbers() -> None:
    """A 2% density change is a much larger relative change in a tracer difference.

    This is why the correction matters: the mass balance uses ``T_rp - T_bl``, not the
    concentrations themselves.
    """
    baseline_tracer = 9.35
    reporting_tracer = 16.89
    corrected = float(
        convert_to_equivalent_soil_mass(
            mass_fraction_mg_kg=np.array([reporting_tracer]),
            measured_bulk_density_kg_m3=1474.0,
            reference_bulk_density_kg_m3=1511.0,
        )[0],
    )

    uncorrected_difference = reporting_tracer - baseline_tracer
    corrected_difference = corrected - baseline_tracer
    # ~2.4% off the concentration, but ~5.5% off the difference the mass balance inverts.
    assert corrected / reporting_tracer == pytest.approx(0.9755, abs=1e-3)
    assert corrected_difference / uncorrected_difference == pytest.approx(0.945, abs=1e-2)


def test_residual_ratio_has_unit_mean_ratio() -> None:
    """The residual removes the point correction, leaving only replicate-to-replicate spread.

    Applying it to values already corrected at the mean densities therefore neither shifts
    nor doubles the correction; it only reintroduces its uncertainty.
    """
    rng = np.random.default_rng(42)
    measured = rng.normal(1474.0, 14.0, size=10_000)
    reference = rng.normal(1511.0, 16.0, size=10_000)

    residual = compute_residual_equivalent_soil_mass_ratio(
        measured_bulk_density_replicates_kg_m3=measured,
        reference_bulk_density_replicates_kg_m3=reference,
    )

    assert float(np.mean(residual)) == pytest.approx(1.0, abs=1e-3)
    assert float(np.std(residual)) > 0.0


def test_residual_ratio_is_all_ones_for_constant_densities() -> None:
    """With no bootstrap spread there is no uncertainty left to reintroduce."""
    residual = compute_residual_equivalent_soil_mass_ratio(
        measured_bulk_density_replicates_kg_m3=np.full(100, 1474.0),
        reference_bulk_density_replicates_kg_m3=np.full(100, 1511.0),
    )
    np.testing.assert_allclose(residual, np.ones(100))


def test_residual_ratio_raises_on_mismatched_replicate_counts() -> None:
    """Unequal runs cannot be paired replicate-for-replicate."""
    with pytest.raises(ValueError, match="Bootstrap both"):
        compute_residual_equivalent_soil_mass_ratio(
            measured_bulk_density_replicates_kg_m3=np.full(10, 1474.0),
            reference_bulk_density_replicates_kg_m3=np.full(11, 1511.0),
        )


def test_residual_ratio_raises_on_empty_replicates() -> None:
    """An empty bootstrap has no mean to normalise against."""
    with pytest.raises(ValueError, match="empty"):
        compute_residual_equivalent_soil_mass_ratio(
            measured_bulk_density_replicates_kg_m3=np.array([]),
            reference_bulk_density_replicates_kg_m3=np.array([]),
        )
