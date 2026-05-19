# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

import numpy as np
import pytest

from isometric_calculation_library.enhanced_weathering.utils.cdr import (
    compute_depth_weighted_concentration_kg_ha,
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
