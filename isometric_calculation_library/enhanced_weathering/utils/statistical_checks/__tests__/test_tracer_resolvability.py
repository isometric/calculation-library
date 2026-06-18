# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

import numpy as np
import pandas as pd
import pytest

from ..tracer_resolvability import (
    build_tracer_resolvability_df,
    calculate_tracer_resolvability,
)


def test_build_tracer_resolvability_df_structure() -> None:
    """DataFrame has expected columns and positive resolvability."""
    rng = np.random.default_rng(42)
    bl_samples = pd.DataFrame({
        "mass_fraction_ti": rng.normal(100, 10, size=20),
    })
    feedstock_tracer = rng.normal(5000, 200, size=10)
    bulk_density = rng.normal(1200, 50, size=15)

    result = build_tracer_resolvability_df(
        baseline_samples=bl_samples,
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
