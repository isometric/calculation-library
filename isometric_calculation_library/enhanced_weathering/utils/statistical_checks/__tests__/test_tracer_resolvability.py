# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

import numpy as np
import pandas as pd
import pytest

from isometric_calculation_library.enhanced_weathering.utils.statistical_checks.tracer_resolvability import (
    build_tracer_resolvability_df,
    calculate_tracer_resolvability,
)


def test_build_tracer_resolvability_df_structure() -> None:
    """DataFrame has expected columns and positive resolvability."""
    rng = np.random.default_rng(42)
    baseline_samples = pd.DataFrame({
        "mass_fraction_ti": rng.normal(100, 10, size=20),
    })
    feedstock_tracer = rng.normal(5000, 200, size=10)
    bulk_density = rng.normal(1200, 50, size=15)

    result = build_tracer_resolvability_df(
        baseline_samples=baseline_samples,
        feedstock_tracer_values=feedstock_tracer,
        bulk_density_values=bulk_density,
        area_ha=10.0,
        application_rate_kg_ha=15_000.0,
        tracer="Ti",
        sampling_depth_cm=30.0,
    )

    assert len(result) == 1
    assert result["plot_type"].iloc[0] == "treatment"
    assert result["resolvability_index"].iloc[0] > 0
    assert "soil_mass_kg" in result.columns
    assert "feedstock_mass_kg" in result.columns


def test_calculate_tracer_resolvability_raises_on_zero_noise() -> None:
    """Single-sample inputs give zero standard errors, zeroing the noise term;
    this must raise rather than silently divide to inf."""
    with pytest.raises(ValueError, match="noise"):
        calculate_tracer_resolvability(
            soil_mass_kg=1e6,
            feedstock_mass_kg=1e4,
            feedstock_tracer_mg_kg=np.array([5000.0]),
            baseline_treatment_tracer_mg_kg=np.array([100.0]),
        )


def test_calculate_tracer_resolvability_raises_on_zero_mass() -> None:
    """Zero feedstock and soil mass (e.g. an absent plot type defaulting its area
    to zero) makes the mixing fraction 0/0 = NaN; this must raise rather than let
    the NaN slip past the noise guard (NaN <= 0 is False) and return a NaN index."""
    with pytest.raises(ValueError, match="positive total mass"):
        calculate_tracer_resolvability(
            soil_mass_kg=0.0,
            feedstock_mass_kg=0.0,
            feedstock_tracer_mg_kg=np.array([5000.0, 5100.0]),
            baseline_treatment_tracer_mg_kg=np.array([100.0, 110.0]),
        )


def test_build_tracer_resolvability_df_accepts_copper() -> None:
    """Copper is an accepted tracer, read from its own mass fraction column.

    The index is computed from the arrays alone, so it matches what titanium would give
    for the same numbers — the element only selects the column.
    """
    rng = np.random.default_rng(42)
    copper_samples = pd.DataFrame({"mass_fraction_cu": rng.normal(100, 10, size=20)})
    titanium_samples = copper_samples.rename(columns={"mass_fraction_cu": "mass_fraction_ti"})
    feedstock_tracer = rng.normal(5000, 200, size=10)
    bulk_density = rng.normal(1200, 50, size=15)

    copper = build_tracer_resolvability_df(
        baseline_samples=copper_samples,
        feedstock_tracer_values=feedstock_tracer,
        bulk_density_values=bulk_density,
        area_ha=10.0,
        application_rate_kg_ha=15_000.0,
        tracer="Cu",
        sampling_depth_cm=30.0,
    )
    titanium = build_tracer_resolvability_df(
        baseline_samples=titanium_samples,
        feedstock_tracer_values=feedstock_tracer,
        bulk_density_values=bulk_density,
        area_ha=10.0,
        application_rate_kg_ha=15_000.0,
        tracer="Ti",
        sampling_depth_cm=30.0,
    )

    assert copper["resolvability_index"].iloc[0] == titanium["resolvability_index"].iloc[0]
    assert copper["n_baseline_samples"].iloc[0] == 20
