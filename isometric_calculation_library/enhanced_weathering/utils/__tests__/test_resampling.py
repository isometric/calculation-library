# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

import numpy as np
import pytest

from isometric_calculation_library.enhanced_weathering.utils.resampling import (
    compute_resampled_means_from_indices,
    generate_bootstrap_location_indices,
    resample_columns_together,
    resample_events_together,
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


def test_resample_events_together_shares_one_draw_across_columns() -> None:
    """Within a replicate every column comes from the same locations.

    This is the reason the multi-column form exists: resampling each column separately
    would let a replicate mix locations, so anything combining columns within a replicate
    would be summing values that do not describe the same soil.
    """
    # Two columns that are exact multiples of each other, so a shared draw keeps the ratio
    # exact in every replicate while independent draws would break it.
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    means = resample_events_together(
        np.random.default_rng(0),
        baseline_values={"a": values, "b": values * 10},
        reporting_period_values={"a": values, "b": values * 10},
        n_runs=500,
        paired=True,
    )

    assert np.allclose(means.baseline["b"], means.baseline["a"] * 10)
    assert np.allclose(means.reporting_period["b"], means.reporting_period["a"] * 10)


def test_column_unpacks_both_events_baseline_first() -> None:
    """The single-column accessor returns the same arrays as indexing both mappings."""
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    means = resample_events_together(
        np.random.default_rng(0),
        baseline_values={"a": values},
        reporting_period_values={"a": values * 10},
        n_runs=100,
        paired=True,
    )

    baseline, reporting_period = means.column("a")

    assert baseline is means.baseline["a"]
    assert reporting_period is means.reporting_period["a"]


def test_resample_events_together_paired_shares_the_draw_between_events() -> None:
    """A paired design draws once for both events, so a perfect correlation survives."""
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    paired = resample_events_together(
        np.random.default_rng(0),
        baseline_values={"a": values},
        reporting_period_values={"a": values},
        n_runs=500,
        paired=True,
    )
    unpaired = resample_events_together(
        np.random.default_rng(0),
        baseline_values={"a": values},
        reporting_period_values={"a": values},
        n_runs=500,
        paired=False,
    )

    assert np.allclose(paired.baseline["a"], paired.reporting_period["a"])
    assert not np.allclose(unpaired.baseline["a"], unpaired.reporting_period["a"])


def test_resample_events_together_unpaired_allows_different_event_sizes() -> None:
    means = resample_events_together(
        np.random.default_rng(0),
        baseline_values={"a": np.arange(10, dtype=float)},
        reporting_period_values={"a": np.arange(4, dtype=float)},
        n_runs=50,
        paired=False,
    )

    assert len(means.baseline["a"]) == 50
    assert len(means.reporting_period["a"]) == 50


def test_resample_events_together_paired_rejects_different_event_sizes() -> None:
    """Unequal lengths mean the rows are not matched, so the pairing claim is false."""
    with pytest.raises(ValueError, match="one row per matched location"):
        resample_events_together(
            np.random.default_rng(0),
            baseline_values={"a": np.arange(10, dtype=float)},
            reporting_period_values={"a": np.arange(4, dtype=float)},
            n_runs=50,
            paired=True,
        )


def test_resample_events_together_rejects_misaligned_columns_within_an_event() -> None:
    with pytest.raises(ValueError, match="Columns within the baseline event must be aligned"):
        resample_events_together(
            np.random.default_rng(0),
            baseline_values={"a": np.arange(10, dtype=float), "b": np.arange(4, dtype=float)},
            reporting_period_values={
                "a": np.arange(10, dtype=float),
                "b": np.arange(10, dtype=float),
            },
            n_runs=50,
            paired=False,
        )


def test_resample_events_together_rejects_mismatched_column_sets() -> None:
    with pytest.raises(ValueError, match="same columns"):
        resample_events_together(
            np.random.default_rng(0),
            baseline_values={"a": np.arange(10, dtype=float)},
            reporting_period_values={"b": np.arange(10, dtype=float)},
            n_runs=50,
            paired=False,
        )


def test_resample_columns_together_shares_one_draw_across_columns() -> None:
    """Columns measured on the same physical samples share a draw, so ratios survive.

    Feedstock cations rely on this: Ca and Mg come off the same rock samples, so pooling
    them per replicate is only meaningful if the replicate drew one set of samples.
    """
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    means = resample_columns_together(
        np.random.default_rng(0),
        values={"ca": values, "mg": values * 10},
        n_runs=500,
        event="feedstock samples",
    )

    assert np.allclose(means["mg"], means["ca"] * 10)


def test_resample_columns_together_rejects_misaligned_columns() -> None:
    """The caller names the event, so the message points at which one is misaligned."""
    with pytest.raises(ValueError, match="Columns within the feedstock samples must be aligned"):
        resample_columns_together(
            np.random.default_rng(0),
            values={"ca": np.arange(5, dtype=float), "mg": np.arange(3, dtype=float)},
            n_runs=10,
            event="feedstock samples",
        )
