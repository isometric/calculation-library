# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

import numpy as np
import pandas as pd
import pytest

from isometric_calculation_library.enhanced_weathering.utils.control_correction import (
    bootstrap_control_correction_ratios,
    compute_control_correction_ratio,
)
from isometric_calculation_library.enhanced_weathering.utils.resampling import (
    generate_bootstrap_location_indices,
)


def test_control_correction_ratio() -> None:
    """cc = C_rp_ctrl / C_bl_ctrl."""
    result = compute_control_correction_ratio(
        control_baseline_mg_kg=np.array([100.0, 200.0]),
        control_end_of_reporting_period_mg_kg=np.array([90.0, 210.0]),
    )
    np.testing.assert_allclose(result, [0.9, 1.05])


def test_bootstrap_ratios_stable_control() -> None:
    """When control concentrations are stable, p50 ratios should be close to 1.0."""
    rng = np.random.default_rng(42)
    n_locations = 20
    ctrl_paired = pd.DataFrame({
        "bl_mass_fraction_ca": rng.normal(200, 5, size=n_locations),
        "bl_mass_fraction_mg": rng.normal(150, 4, size=n_locations),
        "rp_mass_fraction_ca": rng.normal(200, 5, size=n_locations),
        "rp_mass_fraction_mg": rng.normal(150, 4, size=n_locations),
    })
    indices = generate_bootstrap_location_indices(rng, n_locations, 10_000)

    ratios = bootstrap_control_correction_ratios(
        ctrl_paired=ctrl_paired,
        resampled_control_locations=indices,
        elements=["Ca", "Mg"],
    )

    assert set(ratios.keys()) == {"Ca", "Mg"}
    ca_p50 = float(np.percentile(ratios["Ca"], 50))
    mg_p50 = float(np.percentile(ratios["Mg"], 50))
    assert ca_p50 == pytest.approx(1.0, abs=0.05)
    assert mg_p50 == pytest.approx(1.0, abs=0.05)


def test_bootstrap_ratios_shifted_control() -> None:
    """When control RP is higher than baseline, p50 ratio should be > 1."""
    rng = np.random.default_rng(42)
    n_locations = 30
    ctrl_paired = pd.DataFrame({
        "bl_mass_fraction_ca": np.full(n_locations, 200.0),
        "rp_mass_fraction_ca": np.full(n_locations, 220.0),
    })
    indices = generate_bootstrap_location_indices(rng, n_locations, 10_000)

    ratios = bootstrap_control_correction_ratios(
        ctrl_paired=ctrl_paired,
        resampled_control_locations=indices,
        elements=["Ca"],
    )

    ca_p50 = float(np.percentile(ratios["Ca"], 50))
    assert ca_p50 == pytest.approx(1.1, abs=0.01)


def test_bootstrap_ratios_returns_distributions() -> None:
    """Each element's value should be a full bootstrap distribution, not a scalar."""
    rng = np.random.default_rng(42)
    n_locations = 20
    n_runs = 5_000
    ctrl_paired = pd.DataFrame({
        "bl_mass_fraction_ca": rng.normal(200, 5, size=n_locations),
        "rp_mass_fraction_ca": rng.normal(200, 5, size=n_locations),
    })
    indices = generate_bootstrap_location_indices(rng, n_locations, n_runs)

    ratios = bootstrap_control_correction_ratios(
        ctrl_paired=ctrl_paired,
        resampled_control_locations=indices,
        elements=["Ca"],
    )

    assert len(ratios["Ca"]) == n_runs
