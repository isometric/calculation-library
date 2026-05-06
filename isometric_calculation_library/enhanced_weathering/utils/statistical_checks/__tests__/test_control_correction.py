# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

import numpy as np
import pandas as pd
import pytest

from isometric_calculation_library.enhanced_weathering.utils.control_correction import (
    apply_control_correction_delta_paired,
    apply_control_correction_delta_unpaired,
    apply_control_correction_paired,
    apply_control_correction_unpaired,
    bootstrap_control_correction_ratios,
    check_background_weathering_significance,
    check_background_weathering_significance_paired,
    check_background_weathering_significance_unpaired,
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


# -- test_background_weathering_significance_paired ---------------------------


def test_background_weathering_not_significant_for_stable_controls() -> None:
    """Stable control concentrations → not significant."""
    rng = np.random.default_rng(42)
    n = 30
    ctrl_paired = pd.DataFrame({
        "bl_mass_fraction_ca": rng.normal(200, 5, size=n),
        "rp_mass_fraction_ca": rng.normal(200, 5, size=n),
        "bl_mass_fraction_mg": rng.normal(150, 4, size=n),
        "rp_mass_fraction_mg": rng.normal(150, 4, size=n),
        "measurement_location_reference_id": [f"loc_{i}" for i in range(n)],
    })

    results = check_background_weathering_significance_paired(
        ctrl_paired=ctrl_paired,
        elements=["Ca", "Mg"],
    )

    assert len(results) == 2
    for r in results:
        assert r.n_baseline_samples == n
        assert r.n_reporting_period_samples == n
        assert r.paired is True
        assert r.is_significant is False


def test_background_weathering_significant_for_shifted_controls() -> None:
    """Large systematic shift in control RP → significant."""
    n = 50
    bl = np.full(n, 200.0)
    rp = np.full(n, 220.0)  # 10% increase
    ctrl_paired = pd.DataFrame({
        "bl_mass_fraction_ca": bl,
        "rp_mass_fraction_ca": rp,
    })

    results = check_background_weathering_significance_paired(
        ctrl_paired=ctrl_paired,
        elements=["Ca"],
    )

    ca_result = results[0]
    assert ca_result.is_significant is True
    assert ca_result.p_value < 0.05
    assert ca_result.mean_reporting_period == pytest.approx(220.0, abs=0.01)
    assert ca_result.mean_baseline == pytest.approx(200.0, abs=0.01)


def test_background_weathering_too_few_pairs_raises() -> None:
    """Fewer than 3 paired locations → ValueError."""
    ctrl_paired = pd.DataFrame({
        "bl_mass_fraction_ca": [100.0, 200.0],
        "rp_mass_fraction_ca": [110.0, 190.0],
    })

    with pytest.raises(ValueError, match="at least 3 are required"):
        check_background_weathering_significance_paired(ctrl_paired=ctrl_paired, elements=["Ca"])


def test_background_weathering_missing_columns_raises() -> None:
    """Missing columns → ValueError with a clear message."""
    ctrl_paired = pd.DataFrame({
        "bl_mass_fraction_ca": [100.0, 200.0, 300.0],
    })

    with pytest.raises(ValueError, match="rp_mass_fraction_ca"):
        check_background_weathering_significance_paired(ctrl_paired=ctrl_paired, elements=["Ca"])


def test_background_weathering_nan_handling() -> None:
    """NaN values are excluded from pairing."""
    ctrl_paired = pd.DataFrame({
        "bl_mass_fraction_ca": [100.0, 200.0, np.nan, 400.0, 500.0],
        "rp_mass_fraction_ca": [100.0, np.nan, 300.0, 400.0, 500.0],
    })

    results = check_background_weathering_significance_paired(
        ctrl_paired=ctrl_paired,
        elements=["Ca"],
    )

    # Only 3 valid pairs (indices 0, 3, 4)
    assert results[0].n_baseline_samples == 3
    assert results[0].n_reporting_period_samples == 3


def test_check_background_weathering_significance_alias() -> None:
    """Old name is still importable and works identically."""
    n = 30
    rng = np.random.default_rng(0)
    ctrl_paired = pd.DataFrame({
        "bl_mass_fraction_ca": rng.normal(200, 5, n),
        "rp_mass_fraction_ca": rng.normal(200, 5, n),
    })
    r1 = check_background_weathering_significance(ctrl_paired=ctrl_paired, elements=["Ca"])
    r2 = check_background_weathering_significance_paired(ctrl_paired=ctrl_paired, elements=["Ca"])
    assert r1 == r2


# -- test_background_weathering_significance_unpaired -------------------------


def test_unpaired_not_significant_for_similar_populations() -> None:
    """Two populations from the same distribution → not significant, delta = 0.0."""
    rng = np.random.default_rng(42)
    n = 50
    control_rp = pd.DataFrame({"mass_fraction_ca": rng.normal(200, 10, n)})
    control_bl = pd.DataFrame({"mass_fraction_ca": rng.normal(200, 10, n)})

    results = check_background_weathering_significance_unpaired(
        control_reporting_period_samples=control_rp,
        control_baseline_samples=control_bl,
        elements=["Ca"],
    )

    assert results[0].is_significant is False
    assert results[0].n_reporting_period_samples == n
    assert results[0].n_baseline_samples == n
    assert results[0].paired is False


def test_unpaired_significant_for_shifted_populations() -> None:
    """Control EOY clearly lower than baseline → significant."""
    n = 60
    control_rp = pd.DataFrame({"mass_fraction_ca": np.full(n, 180.0)})
    control_bl = pd.DataFrame({"mass_fraction_ca": np.full(n, 200.0)})

    results = check_background_weathering_significance_unpaired(
        control_reporting_period_samples=control_rp,
        control_baseline_samples=control_bl,
        elements=["Ca"],
    )

    r = results[0]
    assert r.is_significant is True
    assert r.p_value < 0.05
    assert r.mean_reporting_period == pytest.approx(180.0)
    assert r.mean_baseline == pytest.approx(200.0)


def test_unpaired_not_significant_for_upward_shift() -> None:
    """Control EOY higher than baseline → not significant (one-sided, depletion only)."""
    n = 60
    control_rp = pd.DataFrame({"mass_fraction_ca": np.full(n, 220.0)})
    control_bl = pd.DataFrame({"mass_fraction_ca": np.full(n, 200.0)})

    results = check_background_weathering_significance_unpaired(
        control_reporting_period_samples=control_rp,
        control_baseline_samples=control_bl,
        elements=["Ca"],
    )

    r = results[0]
    assert r.is_significant is False


def test_unpaired_too_few_control_samples_raises() -> None:
    """Fewer than 3 control samples → ValueError."""
    control_rp = pd.DataFrame({"mass_fraction_ca": [100.0, 200.0]})
    control_bl = pd.DataFrame({"mass_fraction_ca": np.full(20, 200.0)})

    with pytest.raises(ValueError, match="2 valid control reporting period"):
        check_background_weathering_significance_unpaired(
            control_reporting_period_samples=control_rp,
            control_baseline_samples=control_bl,
            elements=["Ca"],
        )


def test_unpaired_too_few_baseline_samples_raises() -> None:
    """Fewer than 3 baseline samples → ValueError."""
    control_rp = pd.DataFrame({"mass_fraction_ca": np.full(20, 180.0)})
    control_bl = pd.DataFrame({"mass_fraction_ca": [200.0, 210.0]})

    with pytest.raises(ValueError, match="2 valid control baseline"):
        check_background_weathering_significance_unpaired(
            control_reporting_period_samples=control_rp,
            control_baseline_samples=control_bl,
            elements=["Ca"],
        )


def test_unpaired_missing_column_raises() -> None:
    """Missing column in either DataFrame → ValueError."""
    control_rp = pd.DataFrame({"mass_fraction_mg": np.full(10, 150.0)})
    control_bl = pd.DataFrame({"mass_fraction_ca": np.full(10, 200.0)})

    with pytest.raises(ValueError, match="mass_fraction_ca"):
        check_background_weathering_significance_unpaired(
            control_reporting_period_samples=control_rp,
            control_baseline_samples=control_bl,
            elements=["Ca"],
        )


def test_unpaired_nan_handling() -> None:
    """NaN values are excluded before testing."""
    control_rp = pd.DataFrame({
        "mass_fraction_ca": [np.nan, 180.0, 180.0, 180.0, 180.0],
    })
    control_bl = pd.DataFrame({
        "mass_fraction_ca": [200.0, 200.0, np.nan, 200.0, 200.0],
    })

    results = check_background_weathering_significance_unpaired(
        control_reporting_period_samples=control_rp,
        control_baseline_samples=control_bl,
        elements=["Ca"],
    )

    assert results[0].n_reporting_period_samples == 4
    assert results[0].n_baseline_samples == 4


# -- apply_control_correction_delta_paired -----------------------------------------


def test_apply_control_correction_paired_not_significant_returns_zeros() -> None:
    """Not significant → delta distribution is all 0.0."""
    rng = np.random.default_rng(42)
    n = 30
    ctrl_paired = pd.DataFrame({
        "bl_mass_fraction_ca": rng.normal(200, 5, n),
        "rp_mass_fraction_ca": rng.normal(200, 5, n),
    })

    results = apply_control_correction_delta_paired(
        ctrl_paired=ctrl_paired,
        elements=["Ca"],
        rng=rng,
        n_runs=1_000,
    )

    r = results[0]
    assert r.is_significant is False
    assert r.cc_delta_point == pytest.approx(0.0)
    np.testing.assert_array_equal(r.cc_delta_distribution, np.zeros(1_000))


def test_apply_control_correction_paired_significant_returns_distribution() -> None:
    """Significant → delta distribution centred near rp_mean - bl_mean."""
    rng = np.random.default_rng(42)
    n = 50
    ctrl_paired = pd.DataFrame({
        "bl_mass_fraction_ca": rng.normal(200, 5, n),
        "rp_mass_fraction_ca": rng.normal(220, 5, n),
    })

    results = apply_control_correction_delta_paired(
        ctrl_paired=ctrl_paired,
        elements=["Ca"],
        rng=rng,
        n_runs=5_000,
    )

    r = results[0]
    assert r.is_significant is True
    assert len(r.cc_delta_distribution) == 5_000
    assert float(np.std(r.cc_delta_distribution)) > 0.0
    assert float(np.median(r.cc_delta_distribution)) == pytest.approx(20.0, abs=2.0)


def test_apply_control_correction_paired_floor_at_zero() -> None:
    """floor_at_zero=True (default) prevents negative deltas in bootstrap."""
    rng = np.random.default_rng(42)
    n = 50
    ctrl_paired = pd.DataFrame({
        "bl_mass_fraction_ca": rng.normal(220, 5, n),
        "rp_mass_fraction_ca": rng.normal(200, 5, n),  # rp < bl → raw delta would be negative
    })

    results = apply_control_correction_delta_paired(
        ctrl_paired=ctrl_paired,
        elements=["Ca"],
        rng=rng,
        n_runs=5_000,
        floor_at_zero=True,
    )

    r = results[0]
    if r.is_significant:
        assert float(r.cc_delta_distribution.min()) >= 0.0


def test_apply_control_correction_paired_no_floor() -> None:
    """floor_at_zero=False allows negative deltas in bootstrap."""
    rng = np.random.default_rng(42)
    n = 50
    ctrl_paired = pd.DataFrame({
        "bl_mass_fraction_ca": rng.normal(220, 5, n),
        "rp_mass_fraction_ca": rng.normal(200, 5, n),
    })

    results = apply_control_correction_delta_paired(
        ctrl_paired=ctrl_paired,
        elements=["Ca"],
        rng=rng,
        n_runs=5_000,
        floor_at_zero=False,
    )

    r = results[0]
    assert r.is_significant is True
    # With no floor and rp < bl, delta = rp - bl is negative
    assert float(r.cc_delta_distribution.max()) < 0.0


def test_apply_control_correction_paired_alias() -> None:
    """Old name apply_control_correction_paired is still callable."""
    rng = np.random.default_rng(42)
    n = 30
    ctrl_paired = pd.DataFrame({
        "bl_mass_fraction_ca": rng.normal(200, 5, n),
        "rp_mass_fraction_ca": rng.normal(200, 5, n),
    })
    results = apply_control_correction_paired(
        ctrl_paired=ctrl_paired,
        elements=["Ca"],
        rng=rng,
        n_runs=100,
    )
    assert len(results) == 1


# -- apply_control_correction_delta_unpaired -----------------------------------


def test_apply_control_correction_unpaired_not_significant_returns_zeros() -> None:
    """Not significant → delta distribution is all 0.0."""
    rng = np.random.default_rng(42)
    n = 50
    control_rp = pd.DataFrame({"mass_fraction_ca": rng.normal(200, 10, n)})
    control_bl = pd.DataFrame({"mass_fraction_ca": rng.normal(200, 10, n)})

    results = apply_control_correction_delta_unpaired(
        control_reporting_period_samples=control_rp,
        control_baseline_samples=control_bl,
        elements=["Ca"],
        rng=rng,
        n_runs=1_000,
    )

    r = results[0]
    assert r.is_significant is False
    assert r.cc_delta_point == pytest.approx(0.0)
    np.testing.assert_array_equal(r.cc_delta_distribution, np.zeros(1_000))


def test_apply_control_correction_unpaired_significant_returns_distribution() -> None:
    """Significant → delta distribution centred near baseline_mean - rp_mean."""
    rng = np.random.default_rng(42)
    n = 60
    control_rp = pd.DataFrame({"mass_fraction_ca": rng.normal(180, 5, n)})
    control_bl = pd.DataFrame({"mass_fraction_ca": rng.normal(200, 5, n)})

    results = apply_control_correction_delta_unpaired(
        control_reporting_period_samples=control_rp,
        control_baseline_samples=control_bl,
        elements=["Ca"],
        rng=rng,
        n_runs=5_000,
    )

    r = results[0]
    assert r.is_significant is True
    assert len(r.cc_delta_distribution) == 5_000
    assert float(np.std(r.cc_delta_distribution)) > 0.0
    assert float(np.median(r.cc_delta_distribution)) == pytest.approx(20.0, abs=2.0)
    assert r.n_control_reporting_period_samples == n
    assert r.n_control_baseline_samples == n


def test_apply_control_correction_unpaired_floor_at_zero() -> None:
    """floor_at_zero=True (default) prevents negative deltas in bootstrap."""
    rng = np.random.default_rng(42)
    n = 60
    control_rp = pd.DataFrame({"mass_fraction_ca": rng.normal(195, 20, n)})
    control_bl = pd.DataFrame({"mass_fraction_ca": rng.normal(200, 20, n)})

    results = apply_control_correction_delta_unpaired(
        control_reporting_period_samples=control_rp,
        control_baseline_samples=control_bl,
        elements=["Ca"],
        rng=rng,
        n_runs=5_000,
        floor_at_zero=True,
    )

    r = results[0]
    if r.is_significant:
        assert float(r.cc_delta_distribution.min()) >= 0.0


def test_apply_control_correction_unpaired_no_floor() -> None:
    """floor_at_zero=False allows negative deltas in bootstrap."""
    rng = np.random.default_rng(42)
    n = 60
    control_rp = pd.DataFrame({"mass_fraction_ca": rng.normal(180, 5, n)})
    control_bl = pd.DataFrame({"mass_fraction_ca": rng.normal(200, 5, n)})

    results = apply_control_correction_delta_unpaired(
        control_reporting_period_samples=control_rp,
        control_baseline_samples=control_bl,
        elements=["Ca"],
        rng=rng,
        n_runs=5_000,
        floor_at_zero=False,
    )

    r = results[0]
    assert r.is_significant is True
    # With no floor, distribution may contain negative values
    assert float(r.cc_delta_distribution.min()) < float(r.cc_delta_distribution.max())


def test_apply_control_correction_unpaired_alias() -> None:
    """Old name apply_control_correction_unpaired is still callable."""
    rng = np.random.default_rng(42)
    n = 50
    control_rp = pd.DataFrame({"mass_fraction_ca": rng.normal(200, 10, n)})
    control_bl = pd.DataFrame({"mass_fraction_ca": rng.normal(200, 10, n)})
    results = apply_control_correction_unpaired(
        control_reporting_period_samples=control_rp,
        reference_baseline_samples=control_bl,
        elements=["Ca"],
        rng=rng,
        n_runs=100,
    )
    assert len(results) == 1
