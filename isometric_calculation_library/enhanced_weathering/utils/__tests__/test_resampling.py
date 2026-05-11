# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

import numpy as np
import pytest

from ..resampling import (
    compute_resampled_means_from_indices,
    generate_bootstrap_location_indices,
    summarize_distributions,
)


def test_summarize_distributions_columns_and_rows() -> None:
    """Returns one row per distribution with expected summary columns."""
    rng = np.random.default_rng(42)
    distributions = {
        "dist_a": rng.normal(10, 2, size=1000),
        "dist_b": rng.normal(50, 5, size=1000),
    }
    result = summarize_distributions(distributions)

    assert len(result) == 2
    expected_cols = {
        "distribution_name",
        "mean",
        "std",
        "p5",
        "p16",
        "p30",
        "p40",
        "median",
        "p84",
        "p95",
    }
    assert set(result.columns) == expected_cols
    assert list(result["distribution_name"]) == ["dist_a", "dist_b"]


# -- compute_resampled_means_from_indices (noise) ------------------------------


def test_resampled_means_no_noise_is_deterministic() -> None:
    """Without noise, result depends only on bootstrap rng, not noise_rng."""
    values = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    rng = np.random.default_rng(0)
    indices = generate_bootstrap_location_indices(rng, len(values), n_runs=500)

    result_a = compute_resampled_means_from_indices(values, indices)
    result_b = compute_resampled_means_from_indices(
        values,
        indices,
        noise_rng=None,
        noise_fraction=0.1,
    )
    np.testing.assert_array_equal(result_a, result_b)


def test_resampled_means_noise_is_reproducible() -> None:
    """Same noise_rng seed produces identical results."""
    values = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    rng = np.random.default_rng(0)
    indices = generate_bootstrap_location_indices(rng, len(values), n_runs=500)

    result_a = compute_resampled_means_from_indices(
        values,
        indices,
        noise_rng=np.random.default_rng(99),
        noise_fraction=0.1,
    )
    result_b = compute_resampled_means_from_indices(
        values,
        indices,
        noise_rng=np.random.default_rng(99),
        noise_fraction=0.1,
    )
    np.testing.assert_array_equal(result_a, result_b)


def test_resampled_means_noise_differs_from_no_noise() -> None:
    """With noise, bootstrap means differ from the noiseless case."""
    values = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    rng = np.random.default_rng(0)
    indices = generate_bootstrap_location_indices(rng, len(values), n_runs=500)

    no_noise = compute_resampled_means_from_indices(values, indices)
    with_noise = compute_resampled_means_from_indices(
        values,
        indices,
        noise_rng=np.random.default_rng(99),
        noise_fraction=0.1,
    )
    assert not np.allclose(no_noise, with_noise)


def test_resampled_means_noise_widens_distribution() -> None:
    """Noise increases the std of the bootstrap distribution."""
    rng = np.random.default_rng(0)
    values = np.arange(1.0, 101.0)
    indices = generate_bootstrap_location_indices(rng, len(values), n_runs=10_000)

    no_noise = compute_resampled_means_from_indices(values, indices)
    with_noise = compute_resampled_means_from_indices(
        values,
        indices,
        noise_rng=np.random.default_rng(1),
        noise_fraction=0.1,
    )
    assert float(np.std(with_noise)) > float(np.std(no_noise))


def test_resampled_means_noise_proportional_to_magnitude() -> None:
    """Larger values produce larger absolute noise (proportional noise model)."""
    rng_lo = np.random.default_rng(0)
    rng_hi = np.random.default_rng(0)
    values_lo = np.ones(50)
    values_hi = np.ones(50) * 1000.0
    indices_lo = generate_bootstrap_location_indices(rng_lo, 50, n_runs=5_000)
    indices_hi = generate_bootstrap_location_indices(rng_hi, 50, n_runs=5_000)

    std_lo = float(
        np.std(
            compute_resampled_means_from_indices(
                values_lo,
                indices_lo,
                noise_rng=np.random.default_rng(7),
                noise_fraction=0.1,
            ),
        ),
    )
    std_hi = float(
        np.std(
            compute_resampled_means_from_indices(
                values_hi,
                indices_hi,
                noise_rng=np.random.default_rng(7),
                noise_fraction=0.1,
            ),
        ),
    )
    assert pytest.approx(std_hi / std_lo, rel=0.1) == 1000.0


# -- summarize_distributions ---------------------------------------------------


def test_summarize_distributions_values_are_consistent() -> None:
    """Percentiles are ordered: p5 < p16 < median < p84 < p95."""
    rng = np.random.default_rng(99)
    distributions = {"x": rng.normal(0, 1, size=10_000)}
    result = summarize_distributions(distributions)

    row = result.iloc[0]
    assert (
        row["p5"] < row["p16"] < row["p30"] < row["p40"] < row["median"] < row["p84"] < row["p95"]
    )
