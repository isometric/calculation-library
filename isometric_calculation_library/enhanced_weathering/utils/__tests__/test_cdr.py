# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

import numpy as np
import pytest

from isometric_calculation_library.enhanced_weathering.utils.cdr import (
    compute_depth_weighted_concentration_kg_ha,
    compute_weathered_fraction_standard_tca,
)


def test_compute_depth_weighted_concentration_kg_ha_uniform_depth() -> None:
    """With uniform depth, result equals concentration * volume / 1e6."""
    conc = np.array([1000.0, 2000.0, 3000.0])
    depth = np.array([20.0, 20.0, 20.0])
    result = compute_depth_weighted_concentration_kg_ha(conc, depth)
    # volume_m3_per_ha = 100 * 20 = 2000
    expected = conc * 2000.0 / 1e6
    np.testing.assert_allclose(result, expected)


def test_compute_depth_weighted_concentration_kg_ha_mixed_depths() -> None:
    """Samples at different depths produce different scaled values."""
    conc = np.array([1000.0, 1000.0])
    depth = np.array([15.0, 20.0])
    result = compute_depth_weighted_concentration_kg_ha(conc, depth)
    # 15 cm sample should give a smaller value than 20 cm
    assert result[0] < result[1]
    np.testing.assert_allclose(result[0], 1000.0 * 1500.0 / 1e6)
    np.testing.assert_allclose(result[1], 1000.0 * 2000.0 / 1e6)


def test_compute_depth_weighted_mean_vs_naive_mean() -> None:
    """Mean of depth-weighted values differs from naive mean(conc) * mean(depth).

    This confirms the function corrects the Jensen's inequality bias when depth
    and concentration are correlated across locations.
    """
    # High concentration at shallow depth, low concentration at deep depth
    conc = np.array([3000.0, 1000.0])
    depth = np.array([15.0, 20.0])

    depth_weighted_mean = float(np.mean(compute_depth_weighted_concentration_kg_ha(conc, depth)))
    naive_mean = float(np.mean(conc)) * float(np.mean(depth)) * 100.0 / 1e6

    assert depth_weighted_mean != pytest.approx(naive_mean)


def test_compute_depth_weighted_concentration_kg_ha_multiplied_by_bd_gives_stock() -> None:
    """Multiplying the mean depth-weighted value by BD recovers the correct stock."""
    conc = np.array([1000.0, 1000.0])
    depth = np.array([20.0, 20.0])
    bd = 1500.0  # kg/m³

    dw = compute_depth_weighted_concentration_kg_ha(conc, depth)
    stock = float(np.mean(dw)) * bd
    # Expected: 1000 mg/kg * (100*20) m³/ha * 1500 kg/m³ / 1e6 = 3000 kg/ha
    np.testing.assert_allclose(stock, 3000.0)


def test_weathered_fraction_standard_tca_complete_weathering_returns_one() -> None:
    """When all feedstock cation has dissolved (R1 = BL, no control), Fw = 1."""
    blp = np.array([200.0, 200.0, 200.0])
    bl = np.array([100.0, 100.0, 100.0])
    r1 = np.array([100.0, 100.0, 100.0])
    control = np.array([0.0, 0.0, 0.0])

    result = compute_weathered_fraction_standard_tca(
        baseline_post_application_concentration=blp,
        resampling_concentration=r1,
        baseline_concentration=bl,
        cation_dissolved_kg_ha=control,
    )
    np.testing.assert_allclose(result, 1.0)


def test_weathered_fraction_standard_tca_no_weathering_returns_zero() -> None:
    """When nothing weathered (R1 = BLP, no control), Fw = 0."""
    blp = np.array([200.0, 200.0])
    bl = np.array([100.0, 100.0])
    r1 = np.array([200.0, 200.0])
    control = np.array([0.0, 0.0])

    result = compute_weathered_fraction_standard_tca(
        baseline_post_application_concentration=blp,
        resampling_concentration=r1,
        baseline_concentration=bl,
        cation_dissolved_kg_ha=control,
    )
    np.testing.assert_allclose(result, 0.0)


def test_weathered_fraction_standard_tca_with_control_correction() -> None:
    """Control correction reduces the apparent weathering signal."""
    blp = np.array([200.0])
    bl = np.array([100.0])
    r1 = np.array([150.0])
    control = np.array([10.0])

    result = compute_weathered_fraction_standard_tca(
        baseline_post_application_concentration=blp,
        resampling_concentration=r1,
        baseline_concentration=bl,
        cation_dissolved_kg_ha=control,
    )
    # Fw = (200 - 150 - 10) / (200 - 100) = 40/100 = 0.4
    np.testing.assert_allclose(result, 0.4)


def test_weathered_fraction_standard_tca_zero_denominator_returns_nan() -> None:
    """When BLP equals BL (no feedstock signal), result is NaN."""
    blp = np.array([100.0, 200.0])
    bl = np.array([100.0, 100.0])
    r1 = np.array([90.0, 150.0])
    control = np.array([0.0, 0.0])

    result = compute_weathered_fraction_standard_tca(
        baseline_post_application_concentration=blp,
        resampling_concentration=r1,
        baseline_concentration=bl,
        cation_dissolved_kg_ha=control,
    )
    assert np.isnan(result[0])
    np.testing.assert_allclose(result[1], 0.5)


def test_weathered_fraction_standard_tca_negative_denominator_computes() -> None:
    """When BLP < BL (soil heterogeneity), the guard does not trigger — abs(denom) > 0."""
    blp = np.array([95.0])
    bl = np.array([100.0])
    r1 = np.array([90.0])
    control = np.array([0.0])

    result = compute_weathered_fraction_standard_tca(
        baseline_post_application_concentration=blp,
        resampling_concentration=r1,
        baseline_concentration=bl,
        cation_dissolved_kg_ha=control,
    )
    # denominator = 95 - 100 = -5, abs(-5) > 0 so it computes: (95-90-0)/(-5) = -1.0
    np.testing.assert_allclose(result, -1.0)


def test_weathered_fraction_standard_tca_nan_values_ignored_by_nanmean() -> None:
    """NaN from guarded locations does not contaminate bootstrap aggregation."""
    blp = np.array([100.0, 200.0, 200.0])
    bl = np.array([100.0, 100.0, 100.0])
    r1 = np.array([80.0, 100.0, 100.0])
    control = np.array([0.0, 0.0, 0.0])

    result = compute_weathered_fraction_standard_tca(
        baseline_post_application_concentration=blp,
        resampling_concentration=r1,
        baseline_concentration=bl,
        cation_dissolved_kg_ha=control,
    )
    # First element: NaN (denominator = 0). Others: (200-100-0)/(200-100) = 1.0
    assert np.isnan(result[0])
    mean = float(np.nanmean(result))
    np.testing.assert_allclose(mean, 1.0)
