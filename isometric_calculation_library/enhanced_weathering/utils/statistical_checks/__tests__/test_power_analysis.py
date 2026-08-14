# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from isometric_calculation_library.enhanced_weathering.utils.statistical_checks.power_analysis import (
    PowerAnalysisResult,
    compute_power_analysis_paired,
    compute_power_analysis_unpaired,
)


def _make_paired_df(
    n: int,
    rng: np.random.Generator,
    baseline_ti_std: float = 1000.0,
    reporting_period_ti_std: float = 1200.0,
) -> pd.DataFrame:
    """Create a paired DataFrame with baseline_/reporting_period_ columns for Ti, Ca, Mg."""
    return pd.DataFrame({
        "baseline_mass_fraction_ti": rng.normal(5000, baseline_ti_std, n),
        "reporting_period_mass_fraction_ti": rng.normal(5500, reporting_period_ti_std, n),
        "baseline_mass_fraction_ca": rng.normal(15000, 2000, n),
        "reporting_period_mass_fraction_ca": rng.normal(18000, 5000, n),
        "baseline_mass_fraction_mg": rng.normal(8000, 1400, n),
        "reporting_period_mass_fraction_mg": rng.normal(9500, 2500, n),
    })


def _baseline_of(paired: pd.DataFrame) -> pd.DataFrame:
    """The baseline event of a paired frame, as the unpaired variant takes it."""
    prefix = "baseline_"
    columns = [c for c in paired.columns if c.startswith(prefix)]
    return paired[columns].rename(columns={c: c.removeprefix(prefix) for c in columns})


def _reporting_period_of(paired: pd.DataFrame) -> pd.DataFrame:
    """The reporting-period event of a paired frame, as the unpaired variant takes it."""
    prefix = "reporting_period_"
    columns = [c for c in paired.columns if c.startswith(prefix)]
    return paired[columns].rename(columns={c: c.removeprefix(prefix) for c in columns})


def test_returns_correct_structure() -> None:
    rng = np.random.default_rng(42)
    paired = _make_paired_df(100, rng)

    results = compute_power_analysis_unpaired(
        baseline_samples=_baseline_of(paired),
        reporting_period_samples=_reporting_period_of(paired),
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
    mean_baseline_ti = float(paired["baseline_mass_fraction_ti"].mean())

    results = compute_power_analysis_unpaired(
        baseline_samples=_baseline_of(paired),
        reporting_period_samples=_reporting_period_of(paired),
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
        "baseline_mass_fraction_ti": rng.normal(5000, 200, 100),
        "reporting_period_mass_fraction_ti": rng.normal(5500, 200, 100),
    })
    # High variability
    paired_high = pd.DataFrame({
        "baseline_mass_fraction_ti": rng.normal(5000, 2000, 100),
        "reporting_period_mass_fraction_ti": rng.normal(5500, 2000, 100),
    })

    result_low = compute_power_analysis_unpaired(
        baseline_samples=_baseline_of(paired_low),
        reporting_period_samples=_reporting_period_of(paired_low),
        feedstock_concentrations={"Ti": 17000.0},
        effective_application_rate_kg_ha=5000.0,
        n_eff=100.0,
        bulk_density_kg_m3=1580.0,
        sampling_depth_cm=7.5,
        elements=["Ti"],
    )
    result_high = compute_power_analysis_unpaired(
        baseline_samples=_baseline_of(paired_high),
        reporting_period_samples=_reporting_period_of(paired_high),
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
    results = compute_power_analysis_unpaired(
        baseline_samples=_baseline_of(paired),
        reporting_period_samples=_reporting_period_of(paired),
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
    results = compute_power_analysis_unpaired(
        baseline_samples=_baseline_of(paired),
        reporting_period_samples=_reporting_period_of(paired),
        feedstock_concentrations={"Ti": 17000.0},
        effective_application_rate_kg_ha=70_000.0,
        n_eff=5.0,  # Very low effective n
        bulk_density_kg_m3=1580.0,
        sampling_depth_cm=7.5,
        elements=["Ti"],
    )

    assert not results[0].passes


def test_n_eff_determines_pass_not_the_raw_sample_count() -> None:
    """Pass condition uses n_eff, not the number of samples taken."""
    rng = np.random.default_rng(42)
    # Moderate variability with realistic app rate → n_required ~ 25
    paired = _make_paired_df(200, rng, baseline_ti_std=1000, reporting_period_ti_std=1200)

    # 200 samples per event but n_eff=3 → should fail (3 < ~25)
    result_low_neff = compute_power_analysis_unpaired(
        baseline_samples=_baseline_of(paired),
        reporting_period_samples=_reporting_period_of(paired),
        feedstock_concentrations={"Ti": 17000.0},
        effective_application_rate_kg_ha=70_000.0,
        n_eff=3.0,
        bulk_density_kg_m3=1580.0,
        sampling_depth_cm=7.5,
        elements=["Ti"],
    )
    # 200 samples per event and n_eff=200 → should pass (200 > ~25)
    result_full_neff = compute_power_analysis_unpaired(
        baseline_samples=_baseline_of(paired),
        reporting_period_samples=_reporting_period_of(paired),
        feedstock_concentrations={"Ti": 17000.0},
        effective_application_rate_kg_ha=70_000.0,
        n_eff=200.0,
        bulk_density_kg_m3=1580.0,
        sampling_depth_cm=7.5,
        elements=["Ti"],
    )

    assert result_low_neff[0].n_baseline == 200
    assert result_low_neff[0].n_reporting_period == 200
    assert not result_low_neff[0].passes
    assert result_full_neff[0].passes


def test_zero_delta_returns_inf_n_required() -> None:
    """Zero feedstock concentration gives infinite n_required."""
    rng = np.random.default_rng(42)
    paired = _make_paired_df(50, rng)

    results = compute_power_analysis_unpaired(
        baseline_samples=_baseline_of(paired),
        reporting_period_samples=_reporting_period_of(paired),
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
    with pytest.raises(ValueError, match="at least 2 samples per sampling event"):
        compute_power_analysis_unpaired(
            baseline_samples=_baseline_of(paired),
            reporting_period_samples=_reporting_period_of(paired),
            feedstock_concentrations={"Ti": 17000.0},
            effective_application_rate_kg_ha=5000.0,
            n_eff=1.0,
            bulk_density_kg_m3=1580.0,
            sampling_depth_cm=7.5,
            elements=["Ti"],
        )


def test_distributed_bulk_density_widens_the_expected_enrichment() -> None:
    """Bulk density measured at several locations carries its spread into Eq. 22."""
    rng = np.random.default_rng(42)
    paired = _make_paired_df(100, rng)
    bulk_densities = np.random.default_rng(7).normal(1580.0, 150.0, 2_000)

    distributed = compute_power_analysis_unpaired(
        baseline_samples=_baseline_of(paired),
        reporting_period_samples=_reporting_period_of(paired),
        feedstock_concentrations={"Ti": 17000.0},
        effective_application_rate_kg_ha=5000.0,
        n_eff=100.0,
        bulk_density_kg_m3=bulk_densities,
        sampling_depth_cm=7.5,
        elements=["Ti"],
    )
    fixed = compute_power_analysis_unpaired(
        baseline_samples=_baseline_of(paired),
        reporting_period_samples=_reporting_period_of(paired),
        feedstock_concentrations={"Ti": 17000.0},
        effective_application_rate_kg_ha=5000.0,
        n_eff=100.0,
        bulk_density_kg_m3=float(np.mean(bulk_densities)),
        sampling_depth_cm=7.5,
        elements=["Ti"],
    )

    assert distributed[0].delta_mg_kg_p84 > distributed[0].delta_mg_kg_p16
    assert fixed[0].delta_mg_kg_p84 == pytest.approx(fixed[0].delta_mg_kg_p16)
    # The median tracks the fixed case; only the spread is new.
    assert distributed[0].delta_mg_kg == pytest.approx(fixed[0].delta_mg_kg, rel=0.02)


def test_power_analysis_rejects_mismatched_rate_and_bulk_density_replicates() -> None:
    """Unequal distributions cannot be paired, and must not silently broadcast."""
    rng = np.random.default_rng(42)
    paired = _make_paired_df(50, rng)

    with pytest.raises(ValueError, match="same number of runs"):
        compute_power_analysis_unpaired(
            baseline_samples=_baseline_of(paired),
            reporting_period_samples=_reporting_period_of(paired),
            feedstock_concentrations={"Ti": 17000.0},
            effective_application_rate_kg_ha=np.array([4000.0, 5000.0, 6000.0]),
            n_eff=50.0,
            bulk_density_kg_m3=np.array([1500.0, 1600.0]),
            sampling_depth_cm=7.5,
            elements=["Ti"],
        )


def test_unpaired_raises_on_null_baseline_value() -> None:
    """Nulls are rejected, not silently dropped: the caller must pre-clean the input."""
    rng = np.random.default_rng(0)
    paired = _make_paired_df(10, rng)
    paired.loc[3, "baseline_mass_fraction_ti"] = np.nan
    with pytest.raises(ValueError, match="null 'mass_fraction_ti'"):
        compute_power_analysis_unpaired(
            baseline_samples=_baseline_of(paired),
            reporting_period_samples=_reporting_period_of(paired),
            feedstock_concentrations={"Ti": 17000.0},
            effective_application_rate_kg_ha=5000.0,
            n_eff=10.0,
            bulk_density_kg_m3=1580.0,
            sampling_depth_cm=7.5,
            elements=["Ti"],
        )


def test_unpaired_leaves_sigma_diff_unset_but_reports_the_marginal_sigmas() -> None:
    """Without matching, a difference would subtract unrelated locations, so none is reported."""
    rng = np.random.default_rng(42)
    paired = _make_paired_df(100, rng)
    baseline = _baseline_of(paired)
    reporting_period = _reporting_period_of(paired)

    results = compute_power_analysis_unpaired(
        baseline_samples=baseline,
        reporting_period_samples=reporting_period,
        feedstock_concentrations={"Ti": 17000.0},
        effective_application_rate_kg_ha=5000.0,
        n_eff=100.0,
        bulk_density_kg_m3=1580.0,
        sampling_depth_cm=7.5,
        elements=["Ti"],
    )

    assert results[0].sigma_diff is None
    assert results[0].sigma_baseline == pytest.approx(
        float(np.std(baseline["mass_fraction_ti"].to_numpy())),
    )
    assert results[0].sigma_reporting_period == pytest.approx(
        float(np.std(reporting_period["mass_fraction_ti"].to_numpy())),
    )


def test_unpaired_accepts_events_of_different_sizes() -> None:
    """The protocol allows a different number of samples per event, so nothing may require parity.

    The two sigmas must come from their own event alone — asserting them against the
    separately-sized inputs catches any attempt to align the events row-wise.
    """
    rng = np.random.default_rng(42)
    baseline = pd.DataFrame({"mass_fraction_ti": rng.normal(5000, 900, 31)})
    reporting_period = pd.DataFrame({"mass_fraction_ti": rng.normal(5500, 1300, 17)})

    results = compute_power_analysis_unpaired(
        baseline_samples=baseline,
        reporting_period_samples=reporting_period,
        feedstock_concentrations={"Ti": 17000.0},
        effective_application_rate_kg_ha=5000.0,
        n_eff=17.0,
        bulk_density_kg_m3=1580.0,
        sampling_depth_cm=7.5,
        elements=["Ti"],
    )

    assert results[0].n_baseline == 31
    assert results[0].n_reporting_period == 17
    assert results[0].sigma_baseline == pytest.approx(
        float(np.std(baseline["mass_fraction_ti"].to_numpy())),
    )
    assert results[0].sigma_reporting_period == pytest.approx(
        float(np.std(reporting_period["mass_fraction_ti"].to_numpy())),
    )


def test_unpaired_scales_the_reporting_period_variance_by_the_allocation_ratio() -> None:
    """Sampling the reporting period harder lowers n_required, by the protocol's factor.

    Tiling the same reporting-period values leaves sigma_rp untouched and the baseline
    event — so delta and sigma_bl — identical, which isolates the allocation ratio as the
    only thing that moved. Summing the two marginal variances would leave n_required flat.
    """
    rng = np.random.default_rng(42)
    baseline = pd.DataFrame({"mass_fraction_ti": rng.normal(5000, 900, 31)})
    reporting_period_values = rng.normal(5500, 1300, 17)

    sparse = compute_power_analysis_unpaired(
        baseline_samples=baseline,
        reporting_period_samples=pd.DataFrame({"mass_fraction_ti": reporting_period_values}),
        feedstock_concentrations={"Ti": 17000.0},
        effective_application_rate_kg_ha=5000.0,
        n_eff=17.0,
        bulk_density_kg_m3=1580.0,
        sampling_depth_cm=7.5,
        elements=["Ti"],
    )[0]
    dense = compute_power_analysis_unpaired(
        baseline_samples=baseline,
        reporting_period_samples=pd.DataFrame({
            "mass_fraction_ti": np.tile(reporting_period_values, 3),
        }),
        feedstock_concentrations={"Ti": 17000.0},
        effective_application_rate_kg_ha=5000.0,
        n_eff=17.0,
        bulk_density_kg_m3=1580.0,
        sampling_depth_cm=7.5,
        elements=["Ti"],
    )[0]

    assert dense.n_reporting_period == 51
    assert dense.sigma_reporting_period == pytest.approx(sparse.sigma_reporting_period)
    assert dense.delta_mg_kg == pytest.approx(sparse.delta_mg_kg)

    expected_ratio = (sparse.sigma_baseline**2 + sparse.sigma_reporting_period**2 / (17 / 31)) / (
        dense.sigma_baseline**2 + dense.sigma_reporting_period**2 / (51 / 31)
    )
    assert sparse.n_required / dense.n_required == pytest.approx(expected_ratio)
    assert sparse.n_required > dense.n_required


def test_unpaired_equal_event_sizes_reduce_to_the_summed_marginal_variances() -> None:
    """At k = 1 the allocation ratio drops out, so equal-sized designs keep the familiar form."""
    rng = np.random.default_rng(42)
    paired = _make_paired_df(100, rng)
    baseline = _baseline_of(paired)
    reporting_period = _reporting_period_of(paired)

    result = compute_power_analysis_unpaired(
        baseline_samples=baseline,
        reporting_period_samples=reporting_period,
        feedstock_concentrations={"Ti": 17000.0},
        effective_application_rate_kg_ha=5000.0,
        n_eff=100.0,
        bulk_density_kg_m3=1580.0,
        sampling_depth_cm=7.5,
        elements=["Ti"],
    )[0]

    z_sum = float(norm.ppf(0.975)) + float(norm.ppf(0.80))
    expected = (
        z_sum**2
        * (result.sigma_baseline**2 + result.sigma_reporting_period**2)
        / result.delta_mg_kg**2
    )
    assert result.n_required == pytest.approx(expected)


def test_unpaired_raises_when_one_event_is_too_small() -> None:
    """A single sample has no standard deviation, and the message must name which event."""
    rng = np.random.default_rng(0)
    baseline = pd.DataFrame({"mass_fraction_ti": rng.normal(5000, 900, 20)})
    reporting_period = pd.DataFrame({"mass_fraction_ti": [5500.0]})

    with pytest.raises(ValueError, match="'reporting-period': 1"):
        compute_power_analysis_unpaired(
            baseline_samples=baseline,
            reporting_period_samples=reporting_period,
            feedstock_concentrations={"Ti": 17000.0},
            effective_application_rate_kg_ha=5000.0,
            n_eff=1.0,
            bulk_density_kg_m3=1580.0,
            sampling_depth_cm=7.5,
            elements=["Ti"],
        )


def test_scalar_rate_leaves_delta_percentiles_equal_to_the_median() -> None:
    """A scalar rate is a distribution of one, so the spread collapses."""
    rng = np.random.default_rng(42)
    paired = _make_paired_df(100, rng)

    results = compute_power_analysis_unpaired(
        baseline_samples=_baseline_of(paired),
        reporting_period_samples=_reporting_period_of(paired),
        feedstock_concentrations={"Ti": 17000.0},
        effective_application_rate_kg_ha=5000.0,
        n_eff=100.0,
        bulk_density_kg_m3=1580.0,
        sampling_depth_cm=7.5,
        elements=["Ti"],
    )

    assert results[0].delta_mg_kg_p16 == pytest.approx(results[0].delta_mg_kg)
    assert results[0].delta_mg_kg_p84 == pytest.approx(results[0].delta_mg_kg)


def test_distributed_rate_matches_its_own_median_as_a_scalar() -> None:
    """The reported delta is the median enrichment, so a symmetric rate agrees with its p50."""
    rng = np.random.default_rng(42)
    paired = _make_paired_df(100, rng)

    scalar = compute_power_analysis_unpaired(
        baseline_samples=_baseline_of(paired),
        reporting_period_samples=_reporting_period_of(paired),
        feedstock_concentrations={"Ti": 17000.0},
        effective_application_rate_kg_ha=5000.0,
        n_eff=100.0,
        bulk_density_kg_m3=1580.0,
        sampling_depth_cm=7.5,
        elements=["Ti"],
    )
    distributed = compute_power_analysis_unpaired(
        baseline_samples=_baseline_of(paired),
        reporting_period_samples=_reporting_period_of(paired),
        feedstock_concentrations={"Ti": 17000.0},
        effective_application_rate_kg_ha=rng.normal(5000.0, 250.0, size=20_000),
        n_eff=100.0,
        bulk_density_kg_m3=1580.0,
        sampling_depth_cm=7.5,
        elements=["Ti"],
    )

    assert distributed[0].delta_mg_kg == pytest.approx(scalar[0].delta_mg_kg, rel=1e-2)
    # The rate is uncertain, so the expected enrichment now has real spread.
    assert distributed[0].delta_mg_kg_p16 < distributed[0].delta_mg_kg
    assert distributed[0].delta_mg_kg_p84 > distributed[0].delta_mg_kg


def test_distributed_rate_ignores_nan_replicates() -> None:
    """Non-physical replicates must not drag the median enrichment down."""
    rng = np.random.default_rng(42)
    paired = _make_paired_df(100, rng)
    rate = np.concatenate([np.full(5_000, 5000.0), np.full(5_000, np.nan)])

    results = compute_power_analysis_unpaired(
        baseline_samples=_baseline_of(paired),
        reporting_period_samples=_reporting_period_of(paired),
        feedstock_concentrations={"Ti": 17000.0},
        effective_application_rate_kg_ha=rate,
        n_eff=100.0,
        bulk_density_kg_m3=1580.0,
        sampling_depth_cm=7.5,
        elements=["Ti"],
    )
    scalar = compute_power_analysis_unpaired(
        baseline_samples=_baseline_of(paired),
        reporting_period_samples=_reporting_period_of(paired),
        feedstock_concentrations={"Ti": 17000.0},
        effective_application_rate_kg_ha=5000.0,
        n_eff=100.0,
        bulk_density_kg_m3=1580.0,
        sampling_depth_cm=7.5,
        elements=["Ti"],
    )

    assert results[0].delta_mg_kg == pytest.approx(scalar[0].delta_mg_kg)


def test_all_nan_rate_raises() -> None:
    rng = np.random.default_rng(42)
    paired = _make_paired_df(100, rng)
    with pytest.raises(ValueError, match="No finite rock-to-soil mass ratio"):
        compute_power_analysis_unpaired(
            baseline_samples=_baseline_of(paired),
            reporting_period_samples=_reporting_period_of(paired),
            feedstock_concentrations={"Ti": 17000.0},
            effective_application_rate_kg_ha=np.full(100, np.nan),
            n_eff=100.0,
            bulk_density_kg_m3=1580.0,
            sampling_depth_cm=7.5,
            elements=["Ti"],
        )


def _make_correlated_paired_df(
    n: int,
    rng: np.random.Generator,
    rho: float,
    sigma_bl: float = 1000.0,
    sigma_rp: float = 1000.0,
    mean_bl: float = 5000.0,
    mean_rp: float = 5500.0,
) -> pd.DataFrame:
    """Paired Ti values drawn from a bivariate normal with a known correlation.

    Lets a test assert on the *sign* of pairing's effect from a controlled rho, rather
    than an incidental one.
    """
    cov = [[sigma_bl**2, rho * sigma_bl * sigma_rp], [rho * sigma_bl * sigma_rp, sigma_rp**2]]
    samples = rng.multivariate_normal([mean_bl, mean_rp], cov, size=n)
    return pd.DataFrame({
        "baseline_mass_fraction_ti": samples[:, 0],
        "reporting_period_mass_fraction_ti": samples[:, 1],
    })


def test_paired_returns_correct_structure() -> None:
    rng = np.random.default_rng(42)
    paired = _make_paired_df(100, rng)

    results = compute_power_analysis_paired(
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
    assert [r.element for r in results] == ["Ti", "Ca", "Mg"]


def test_paired_delta_computation_matches_the_unpaired_formula() -> None:
    """Eq. 22 (the expected signal) doesn't depend on pairing, only Eq. 23's noise term does."""
    rng = np.random.default_rng(0)
    paired = _make_paired_df(50, rng)

    paired_result = compute_power_analysis_paired(
        paired=paired,
        feedstock_concentrations={"Ti": 10000.0},
        effective_application_rate_kg_ha=5000.0,
        n_eff=50.0,
        bulk_density_kg_m3=1580.0,
        sampling_depth_cm=7.5,
        elements=["Ti"],
    )
    unpaired_result = compute_power_analysis_unpaired(
        baseline_samples=_baseline_of(paired),
        reporting_period_samples=_reporting_period_of(paired),
        feedstock_concentrations={"Ti": 10000.0},
        effective_application_rate_kg_ha=5000.0,
        n_eff=50.0,
        bulk_density_kg_m3=1580.0,
        sampling_depth_cm=7.5,
        elements=["Ti"],
    )

    assert paired_result[0].delta_mg_kg == pytest.approx(unpaired_result[0].delta_mg_kg)


def test_paired_sigma_diff_matches_the_row_wise_difference() -> None:
    rng = np.random.default_rng(0)
    paired = _make_paired_df(50, rng)

    results = compute_power_analysis_paired(
        paired=paired,
        feedstock_concentrations={"Ti": 17000.0},
        effective_application_rate_kg_ha=5000.0,
        n_eff=50.0,
        bulk_density_kg_m3=1580.0,
        sampling_depth_cm=7.5,
        elements=["Ti"],
    )

    diff = paired["reporting_period_mass_fraction_ti"] - paired["baseline_mass_fraction_ti"]
    expected = float(np.std(diff.to_numpy()))
    assert results[0].sigma_diff == pytest.approx(expected)


def test_paired_requires_fewer_samples_than_unpaired_when_positively_correlated() -> None:
    """The point of pairing: locations whose baseline and reporting-period levels move
    together need fewer samples than treating them as independent would suggest.
    """
    rng = np.random.default_rng(42)
    paired = _make_correlated_paired_df(500, rng, rho=0.8)
    feedstock_concentrations = {"Ti": 17000.0}

    paired_result = compute_power_analysis_paired(
        paired=paired,
        feedstock_concentrations=feedstock_concentrations,
        effective_application_rate_kg_ha=5000.0,
        n_eff=500.0,
        bulk_density_kg_m3=1580.0,
        sampling_depth_cm=7.5,
        elements=["Ti"],
    )
    unpaired_result = compute_power_analysis_unpaired(
        baseline_samples=_baseline_of(paired),
        reporting_period_samples=_reporting_period_of(paired),
        feedstock_concentrations=feedstock_concentrations,
        effective_application_rate_kg_ha=5000.0,
        n_eff=500.0,
        bulk_density_kg_m3=1580.0,
        sampling_depth_cm=7.5,
        elements=["Ti"],
    )

    assert paired_result[0].n_required < unpaired_result[0].n_required


def test_paired_requires_more_samples_than_unpaired_when_negatively_correlated() -> None:
    """Pairing reflects the true variance structure, not a blanket discount."""
    rng = np.random.default_rng(42)
    paired = _make_correlated_paired_df(500, rng, rho=-0.8)
    feedstock_concentrations = {"Ti": 17000.0}

    paired_result = compute_power_analysis_paired(
        paired=paired,
        feedstock_concentrations=feedstock_concentrations,
        effective_application_rate_kg_ha=5000.0,
        n_eff=500.0,
        bulk_density_kg_m3=1580.0,
        sampling_depth_cm=7.5,
        elements=["Ti"],
    )
    unpaired_result = compute_power_analysis_unpaired(
        baseline_samples=_baseline_of(paired),
        reporting_period_samples=_reporting_period_of(paired),
        feedstock_concentrations=feedstock_concentrations,
        effective_application_rate_kg_ha=5000.0,
        n_eff=500.0,
        bulk_density_kg_m3=1580.0,
        sampling_depth_cm=7.5,
        elements=["Ti"],
    )

    assert paired_result[0].n_required > unpaired_result[0].n_required


def test_paired_raises_on_null_reporting_period_value() -> None:
    rng = np.random.default_rng(0)
    paired = _make_paired_df(10, rng)
    paired.loc[7, "reporting_period_mass_fraction_ti"] = np.nan
    with pytest.raises(ValueError, match="null measurements"):
        compute_power_analysis_paired(
            paired=paired,
            feedstock_concentrations={"Ti": 17000.0},
            effective_application_rate_kg_ha=5000.0,
            n_eff=10.0,
            bulk_density_kg_m3=1580.0,
            sampling_depth_cm=7.5,
            elements=["Ti"],
        )


def test_paired_raises_on_too_few_samples() -> None:
    rng = np.random.default_rng(0)
    paired = _make_paired_df(1, rng)
    with pytest.raises(ValueError, match="at least 2 paired locations"):
        compute_power_analysis_paired(
            paired=paired,
            feedstock_concentrations={"Ti": 17000.0},
            effective_application_rate_kg_ha=5000.0,
            n_eff=1.0,
            bulk_density_kg_m3=1580.0,
            sampling_depth_cm=7.5,
            elements=["Ti"],
        )


def test_paired_distributed_rate_matches_its_own_median_as_a_scalar() -> None:
    """Eq. 22's rate-distribution handling is shared code, exercised here too."""
    rng = np.random.default_rng(42)
    paired = _make_paired_df(100, rng)

    scalar = compute_power_analysis_paired(
        paired=paired,
        feedstock_concentrations={"Ti": 17000.0},
        effective_application_rate_kg_ha=5000.0,
        n_eff=100.0,
        bulk_density_kg_m3=1580.0,
        sampling_depth_cm=7.5,
        elements=["Ti"],
    )
    distributed = compute_power_analysis_paired(
        paired=paired,
        feedstock_concentrations={"Ti": 17000.0},
        effective_application_rate_kg_ha=rng.normal(5000.0, 250.0, size=20_000),
        n_eff=100.0,
        bulk_density_kg_m3=1580.0,
        sampling_depth_cm=7.5,
        elements=["Ti"],
    )

    assert distributed[0].delta_mg_kg == pytest.approx(scalar[0].delta_mg_kg, rel=1e-2)
    assert distributed[0].delta_mg_kg_p16 < distributed[0].delta_mg_kg
    assert distributed[0].delta_mg_kg_p84 > distributed[0].delta_mg_kg
