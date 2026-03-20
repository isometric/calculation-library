# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

from collections.abc import Sequence

import numpy as np
import pandas as pd

from isometric_calculation_library.enhanced_weathering.utils.types import Np1DArray


def resample_mean(
    rng: np.random.Generator,
    values: Np1DArray[np.floating],
    n_runs: int,
) -> Np1DArray[np.floating]:
    """Bootstrap resample an array and compute mean for each iteration."""
    n_samples = len(values)
    resampled = rng.choice(values, size=(n_runs, n_samples), replace=True)
    return np.nanmean(resampled, axis=1)


def resample_dataframe_unpaired(
    rng: np.random.Generator,
    baseline_df: pd.DataFrame,
    end_of_reporting_period_df: pd.DataFrame,
    column: str,
    n_runs: int,
) -> tuple[Np1DArray[np.floating], Np1DArray[np.floating]]:
    """Bootstrap resample a column from baseline and end-of-reporting-period DataFrames independently.

    Returns:
        Tuple of (baseline_means, end_of_reporting_period_means) each of length n_runs.
    """
    baseline_values = baseline_df[column].to_numpy()
    end_values = end_of_reporting_period_df[column].to_numpy()
    baseline_means = resample_mean(rng, baseline_values, n_runs)
    end_means = resample_mean(rng, end_values, n_runs)
    return baseline_means, end_means


def resample_dataframe_paired(
    rng: np.random.Generator,
    paired_df: pd.DataFrame,
    baseline_column: str,
    end_of_reporting_period_column: str,
    n_runs: int,
) -> tuple[Np1DArray[np.floating], Np1DArray[np.floating]]:
    """Bootstrap resample paired baseline/end-of-reporting-period columns from a single DataFrame.

    Use this with data prepared by `pairing.prepare_paired_data()`.

    Returns:
        Tuple of (baseline_means, end_of_reporting_period_means) each of length n_runs.
    """
    baseline_values = paired_df[baseline_column].to_numpy()
    end_values = paired_df[end_of_reporting_period_column].to_numpy()

    if len(baseline_values) != len(end_values):
        msg = f"Paired arrays must have same length: {len(baseline_values)} != {len(end_values)}"
        raise ValueError(msg)

    n_pairs = len(baseline_values)
    indices = rng.integers(0, n_pairs, size=(n_runs, n_pairs))
    baseline_means = np.nanmean(baseline_values[indices], axis=1)
    end_means = np.nanmean(end_values[indices], axis=1)
    return baseline_means, end_means


def bootstrap_bulk_density_unpaired(
    rng: np.random.Generator,
    bulk_density_values: Np1DArray[np.floating],
    n_runs: int,
) -> tuple[Np1DArray[np.floating], Np1DArray[np.floating]]:
    """Bootstrap resample bulk density for unpaired baseline and end-of-reporting-period.

    Resamples the same bulk density pool independently for both periods,
    since bulk density measurements are typically not time-period specific.

    Returns:
        Tuple of (baseline_bd, end_of_reporting_period_bd) bootstrap distributions.
    """
    baseline = resample_mean(rng, bulk_density_values, n_runs)
    end = resample_mean(rng, bulk_density_values, n_runs)
    return baseline, end


def bootstrap_bulk_density_paired(
    rng: np.random.Generator,
    bulk_density_values: Np1DArray[np.floating],
    n_runs: int,
) -> Np1DArray[np.floating]:
    """Bootstrap resample bulk density for paired sampling.

    In paired sampling, we use a single bulk density distribution
    (same value for baseline and end-of-reporting-period in each iteration).
    """
    return resample_mean(rng, bulk_density_values, n_runs)


def generate_bootstrap_location_indices(
    rng: np.random.Generator,
    n_locations: int,
    n_runs: int,
) -> np.ndarray:
    """Generate bootstrap resampling indices for location-level resampling.

    These indices can be reused across multiple variables measured at the same
    locations (e.g., different cation concentrations) to ensure consistent
    resampling and preserve cross-variable correlations.

    Args:
        rng: NumPy random generator.
        n_locations: Number of locations to resample from.
        n_runs: Number of bootstrap iterations.

    Returns:
        Integer array of shape (n_runs, n_locations) with values in
        [0, n_locations).
    """
    return rng.integers(0, n_locations, size=(n_runs, n_locations))


def compute_resampled_means_from_indices(
    values: Np1DArray[np.floating],
    indices: np.ndarray,
) -> Np1DArray[np.floating]:
    """Compute bootstrap means using pre-generated resampling indices.

    Use with indices from ``generate_bootstrap_location_indices`` to apply
    consistent resampling across multiple variables at the same locations.

    Args:
        values: Array of per-location values to resample.
        indices: Bootstrap location indices of shape (n_runs, n_locations).
    """
    return np.mean(values[indices], axis=1)


def resample_by_group(
    rng: np.random.Generator,
    location_data: pd.DataFrame,
    n_runs: int,
    group_labels: Sequence[int],
) -> tuple[Np1DArray[np.floating], Np1DArray[np.floating]]:
    """Bootstrap resample by group, preserving baseline/end-of-reporting-period correlation.

    Each iteration resamples groups with replacement. Locations sharing a
    label are resampled together. When each location has its own label,
    this is paired resampling. When multiple locations share a label
    (e.g. spatial blocks), this captures autocorrelation at a coarser scale.

    Args:
        rng: NumPy random generator.
        location_data: DataFrame with columns "baseline_mean" and
            "end_of_reporting_period_mean", one row per location.
            Produced by ``pairing.build_location_level_data``.
        n_runs: Number of bootstrap iterations.
        group_labels: Integer label per location assigning it to a group.
            Length must equal len(location_data).

    Returns:
        Tuple of (baseline_means, end_of_reporting_period_means) each of length n_runs.
    """
    baseline_values = location_data["baseline_mean"].to_numpy()
    end_values = location_data["end_of_reporting_period_mean"].to_numpy()

    labels = np.asarray(group_labels)
    unique_groups = np.unique(labels)
    n_groups = len(unique_groups)

    group_indices = [np.where(labels == g)[0] for g in unique_groups]

    resampled_groups = rng.integers(0, n_groups, size=(n_runs, n_groups))

    baseline_means = np.empty(n_runs)
    end_means = np.empty(n_runs)

    for i in range(n_runs):
        loc_idx = np.concatenate([group_indices[g] for g in resampled_groups[i]])
        baseline_means[i] = np.mean(baseline_values[loc_idx])
        end_means[i] = np.mean(end_values[loc_idx])

    return baseline_means, end_means
