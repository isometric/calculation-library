# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

import numpy as np
import pandas as pd
import pytest

from isometric_calculation_library.geospatial.spatial_autocorrelation import (
    MoransIResult,
    NeffResult,
    _build_inverse_distance_weights,  # pyright: ignore[reportPrivateUsage]
    _compute_morans_i,  # pyright: ignore[reportPrivateUsage]
    compute_morans_i_permutation_test,
    compute_neff_from_morans_i,
)


def _make_paired_df(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Create a paired DataFrame with lat/lon and difference columns."""
    return pd.DataFrame({
        "latitude": rng.uniform(22.0, 23.0, n),
        "longitude": rng.uniform(79.0, 82.0, n),
        "diff_ti": rng.normal(500, 200, n),
        "diff_ca": rng.normal(100, 300, n),
        "diff_mg": rng.normal(-50, 150, n),
    })


def _make_spatially_autocorrelated_df(n_side: int = 10) -> pd.DataFrame:
    """Create data with strong spatial gradient (value = latitude)."""
    lats = np.linspace(22.0, 23.0, n_side)
    lons = np.linspace(79.0, 80.0, n_side)
    lat_grid, lon_grid = np.meshgrid(lats, lons)
    lat_flat = lat_grid.ravel()
    lon_flat = lon_grid.ravel()
    # Value tracks latitude → strong spatial autocorrelation
    return pd.DataFrame({
        "latitude": lat_flat,
        "longitude": lon_flat,
        "diff_ti": lat_flat * 1000,
    })


# -- _build_inverse_distance_weights ------------------------------------------


def test_weights_are_row_normalised() -> None:
    coords = np.array([[22.0, 79.0], [22.1, 79.1], [22.2, 79.2], [22.3, 79.3]])
    w = _build_inverse_distance_weights(coords)

    row_sums = w.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)


def test_weights_diagonal_is_zero() -> None:
    coords = np.array([[22.0, 79.0], [22.1, 79.1], [22.2, 79.2]])
    w = _build_inverse_distance_weights(coords)

    np.testing.assert_array_equal(np.diag(w), 0.0)


def test_weights_shape_matches_input() -> None:
    coords = np.array([[22.0, 79.0], [22.5, 80.0], [23.0, 81.0], [22.3, 79.5]])
    w = _build_inverse_distance_weights(coords)

    assert w.shape == (4, 4)


# -- _compute_morans_i --------------------------------------------------------


def test_morans_i_zero_for_constant_values() -> None:
    coords = np.array([[22.0, 79.0], [22.1, 79.1], [22.2, 79.2]])
    w = _build_inverse_distance_weights(coords)
    values = np.ones(3)

    assert _compute_morans_i(values, w) == pytest.approx(0.0)


def test_morans_i_positive_for_spatial_gradient() -> None:
    """Values correlated with position should give positive I."""
    coords = np.array([
        [22.0, 79.0],
        [22.1, 79.0],
        [22.2, 79.0],
        [22.3, 79.0],
        [22.4, 79.0],
        [22.5, 79.0],
    ])
    w = _build_inverse_distance_weights(coords)
    values = coords[:, 0]  # value = latitude

    i_val = _compute_morans_i(values, w)
    assert i_val > 0


# -- compute_morans_i_permutation_test ----------------------------------------


def test_permutation_test_returns_correct_structure() -> None:
    rng = np.random.default_rng(42)
    paired = _make_paired_df(30, rng)

    results = compute_morans_i_permutation_test(
        paired,
        ["diff_ti", "diff_ca"],
        n_permutations=99,
        rng=rng,
    )

    assert len(results) == 2
    assert all(isinstance(r, MoransIResult) for r in results)
    assert results[0].variable == "diff_ti"
    assert results[1].variable == "diff_ca"


def test_permutation_test_p_values_bounded() -> None:
    rng = np.random.default_rng(0)
    paired = _make_paired_df(20, rng)

    results = compute_morans_i_permutation_test(
        paired,
        ["diff_ti"],
        n_permutations=99,
        rng=rng,
    )

    assert 0.0 < results[0].p_value <= 1.0


def test_permutation_test_detects_strong_autocorrelation() -> None:
    """Spatially autocorrelated data should be flagged as significant."""
    paired = _make_spatially_autocorrelated_df(10)

    results = compute_morans_i_permutation_test(
        paired,
        ["diff_ti"],
        n_permutations=499,
        rng=np.random.default_rng(42),
    )

    assert results[0].observed_i > 0.3
    assert results[0].p_value < 0.05
    assert results[0].significant is True


def test_permutation_test_random_data_not_significant() -> None:
    """Randomly assigned values should not show significant autocorrelation."""
    rng = np.random.default_rng(123)
    paired = _make_paired_df(50, rng)

    results = compute_morans_i_permutation_test(
        paired,
        ["diff_ti"],
        n_permutations=499,
        rng=rng,
    )

    assert results[0].significant is False


# -- compute_neff_from_morans_i ------------------------------------------------


def test_neff_no_autocorrelation_returns_n() -> None:
    """Without significant autocorrelation, n_eff should equal n."""
    rng = np.random.default_rng(42)
    paired = _make_paired_df(50, rng)

    result = compute_neff_from_morans_i(
        paired,
        ["diff_ti", "diff_ca"],
        n_permutations=99,
        rng=rng,
    )

    assert isinstance(result, NeffResult)
    assert result.n == 50
    assert result.n_eff == pytest.approx(50.0)
    assert result.n_eff_int == 50


def test_neff_with_autocorrelation_reduces_n() -> None:
    """Strong spatial autocorrelation should reduce n_eff below n."""
    paired = _make_spatially_autocorrelated_df(10)

    result = compute_neff_from_morans_i(
        paired,
        ["diff_ti"],
        n_permutations=499,
        rng=np.random.default_rng(42),
    )

    assert result.n == 100
    assert result.n_eff < 100
    assert result.n_eff_int < 100
    assert result.n_eff_int >= 2


def test_neff_minimum_is_2() -> None:
    """n_eff_int should never drop below 2."""
    # Create extremely autocorrelated data (I close to 1)
    paired = pd.DataFrame({
        "latitude": [22.0, 22.001, 22.002, 22.003],
        "longitude": [79.0, 79.001, 79.002, 79.003],
        "diff_ti": [1.0, 1.001, 1.002, 1.003],
    })

    result = compute_neff_from_morans_i(
        paired,
        ["diff_ti"],
        n_permutations=199,
        rng=np.random.default_rng(0),
    )

    assert result.n_eff_int >= 2


def test_neff_per_variable_details() -> None:
    rng = np.random.default_rng(42)
    paired = _make_paired_df(30, rng)

    result = compute_neff_from_morans_i(
        paired,
        ["diff_ti", "diff_ca", "diff_mg"],
        n_permutations=99,
        rng=rng,
    )

    assert set(result.per_variable.keys()) == {"diff_ti", "diff_ca", "diff_mg"}
    for details in result.per_variable.values():
        assert "I_obs" in details
        assert "z_score" in details
        assert "p_adj" in details
        assert "significant" in details
        assert "d_eff" in details
        assert "n_eff" in details


def test_neff_takes_minimum_across_variables() -> None:
    """n_eff should be the most conservative (minimum) across all variables."""
    # Mix one autocorrelated variable with random ones
    n_side = 10
    lats = np.linspace(22.0, 23.0, n_side)
    lons = np.linspace(79.0, 80.0, n_side)
    lat_grid, lon_grid = np.meshgrid(lats, lons)

    rng = np.random.default_rng(42)
    paired = pd.DataFrame({
        "latitude": lat_grid.ravel(),
        "longitude": lon_grid.ravel(),
        "diff_autocorr": lat_grid.ravel() * 1000,  # strong autocorrelation
        "diff_random": rng.normal(0, 100, n_side * n_side),  # no autocorrelation
    })

    result = compute_neff_from_morans_i(
        paired,
        ["diff_autocorr", "diff_random"],
        n_permutations=499,
        rng=rng,
    )

    # The autocorrelated variable should drive n_eff down
    autocorr_neff = result.per_variable["diff_autocorr"]["n_eff"]
    assert result.n_eff == autocorr_neff
    assert result.n_eff < result.n
