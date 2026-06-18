# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

import numpy as np
import pandas as pd
import pytest

from ..power_analysis import PowerAnalysisResult, compute_power_analysis


def _make_paired_df(
    n: int,
    rng: np.random.Generator,
    baseline_ti_std: float = 1000.0,
    reporting_period_ti_std: float = 1200.0,
) -> pd.DataFrame:
    """Create a paired DataFrame with bl_/rp_ columns for Ti, Ca, Mg."""
    return pd.DataFrame({
        "bl_mass_fraction_ti": rng.normal(5000, baseline_ti_std, n),
        "rp_mass_fraction_ti": rng.normal(5500, reporting_period_ti_std, n),
        "bl_mass_fraction_ca": rng.normal(15000, 2000, n),
        "rp_mass_fraction_ca": rng.normal(18000, 5000, n),
        "bl_mass_fraction_mg": rng.normal(8000, 1400, n),
        "rp_mass_fraction_mg": rng.normal(9500, 2500, n),
    })


def test_returns_correct_structure() -> None:
    rng = np.random.default_rng(42)
    paired = _make_paired_df(100, rng)

    results = compute_power_analysis(
        paired=paired,
        feedstock_concentrations={"Ti": 17000.0, "Ca": 68000.0, "Mg": 29000.0},
        effective_application_rate_kg_ha=5000.0,
        n_eff=100.0,
        bulk_density_kg_m3=1580.0,
        sampling_depth_cm=7.5,
        elements=["Ti", "Ca", "Mg"],
    )

    assert len(results) == 3
    assert all(isinstance(r, PowerAnalysisResult) for r in results)
    assert results[0].element == "Ti"
    assert results[1].element == "Ca"
    assert results[2].element == "Mg"


def test_delta_computation() -> None:
    """delta = r * (C_F - C_BL) / (1 + r), r = R / (BD * D * 10000)."""
    rng = np.random.default_rng(0)
    paired = _make_paired_df(50, rng)
    mean_baseline_ti = float(paired["bl_mass_fraction_ti"].mean())

    results = compute_power_analysis(
        paired=paired,
        feedstock_concentrations={"Ti": 10000.0},
        effective_application_rate_kg_ha=5000.0,
        n_eff=50.0,
        bulk_density_kg_m3=1580.0,
        sampling_depth_cm=7.5,
        elements=["Ti"],
    )

    # soil_mass = 1580 * 0.075 * 10000 = 1,185,000 kg/ha
    # r = 5000 / 1,185,000
    # delta = r * (10000 - mean_baseline) / (1 + r)
    expected_soil_mass = 1580.0 * 0.075 * 10000
    r = 5000.0 / expected_soil_mass
    expected_delta = r * (10000.0 - mean_baseline_ti) / (1 + r)
    assert results[0].delta_mg_kg == pytest.approx(expected_delta, rel=1e-6)


def test_high_variability_increases_n_required() -> None:
    """Higher standard deviation should require more samples."""
    rng = np.random.default_rng(42)

    # Low variability
    paired_low = pd.DataFrame({
        "bl_mass_fraction_ti": rng.normal(5000, 200, 100),
        "rp_mass_fraction_ti": rng.normal(5500, 200, 100),
    })
    # High variability
    paired_high = pd.DataFrame({
        "bl_mass_fraction_ti": rng.normal(5000, 2000, 100),
        "rp_mass_fraction_ti": rng.normal(5500, 2000, 100),
    })

    result_low = compute_power_analysis(
        paired=paired_low,
        feedstock_concentrations={"Ti": 17000.0},
        effective_application_rate_kg_ha=5000.0,
        n_eff=100.0,
        bulk_density_kg_m3=1580.0,
        sampling_depth_cm=7.5,
        elements=["Ti"],
    )
    result_high = compute_power_analysis(
        paired=paired_high,
        feedstock_concentrations={"Ti": 17000.0},
        effective_application_rate_kg_ha=5000.0,
        n_eff=100.0,
        bulk_density_kg_m3=1580.0,
        sampling_depth_cm=7.5,
        elements=["Ti"],
    )

    assert result_high[0].n_required > result_low[0].n_required


def test_passes_when_n_eff_sufficient() -> None:
    """With enough effective samples and realistic app rate, power passes."""
    rng = np.random.default_rng(42)
    paired = _make_paired_df(200, rng, baseline_ti_std=1000, reporting_period_ti_std=1200)

    # Effective app rate = total_tonnes * 1000 / area_ha (e.g. 8160*1000/115 ≈ 70,000)
    results = compute_power_analysis(
        paired=paired,
        feedstock_concentrations={"Ti": 17000.0, "Ca": 68000.0, "Mg": 29000.0},
        effective_application_rate_kg_ha=70_000.0,
        n_eff=200.0,
        bulk_density_kg_m3=1580.0,
        sampling_depth_cm=7.5,
        elements=["Ti", "Ca", "Mg"],
    )

    assert all(r.passes for r in results)


def test_fails_when_n_eff_too_low() -> None:
    """With very few effective samples and high variability, power fails."""
    rng = np.random.default_rng(42)
    paired = _make_paired_df(200, rng, baseline_ti_std=5000, reporting_period_ti_std=5000)

    # Even with realistic app rate, very high variability + n_eff=5 -> fail
    results = compute_power_analysis(
        paired=paired,
        feedstock_concentrations={"Ti": 17000.0},
        effective_application_rate_kg_ha=70_000.0,
        n_eff=5.0,  # Very low effective n
        bulk_density_kg_m3=1580.0,
        sampling_depth_cm=7.5,
        elements=["Ti"],
    )

    assert not results[0].passes


def test_n_eff_determines_pass_not_n_actual() -> None:
    """Pass condition uses n_eff, not n_actual."""
    rng = np.random.default_rng(42)
    # Moderate variability with realistic app rate → n_required ~ 25
    paired = _make_paired_df(200, rng, baseline_ti_std=1000, reporting_period_ti_std=1200)

    # n_actual=200 but n_eff=3 → should fail (3 < ~25)
    result_low_neff = compute_power_analysis(
        paired=paired,
        feedstock_concentrations={"Ti": 17000.0},
        effective_application_rate_kg_ha=70_000.0,
        n_eff=3.0,
        bulk_density_kg_m3=1580.0,
        sampling_depth_cm=7.5,
        elements=["Ti"],
    )
    # n_actual=200 and n_eff=200 → should pass (200 > ~25)
    result_full_neff = compute_power_analysis(
        paired=paired,
        feedstock_concentrations={"Ti": 17000.0},
        effective_application_rate_kg_ha=70_000.0,
        n_eff=200.0,
        bulk_density_kg_m3=1580.0,
        sampling_depth_cm=7.5,
        elements=["Ti"],
    )

    assert result_low_neff[0].n_actual == 200
    assert not result_low_neff[0].passes
    assert result_full_neff[0].passes


def test_zero_delta_returns_inf_n_required() -> None:
    """Zero feedstock concentration gives infinite n_required."""
    rng = np.random.default_rng(42)
    paired = _make_paired_df(50, rng)

    results = compute_power_analysis(
        paired=paired,
        feedstock_concentrations={"Ti": 0.0},
        effective_application_rate_kg_ha=5000.0,
        n_eff=50.0,
        bulk_density_kg_m3=1580.0,
        sampling_depth_cm=7.5,
        elements=["Ti"],
    )

    assert results[0].n_required == float("inf")
    assert not results[0].passes


def test_power_analysis_raises_on_too_few_samples() -> None:
    rng = np.random.default_rng(0)
    paired = _make_paired_df(1, rng)
    with pytest.raises(ValueError, match="at least 2 non-null"):
        compute_power_analysis(
            paired=paired,
            feedstock_concentrations={"Ti": 17000.0},
            effective_application_rate_kg_ha=5000.0,
            n_eff=1.0,
            bulk_density_kg_m3=1580.0,
            sampling_depth_cm=7.5,
            elements=["Ti"],
        )
