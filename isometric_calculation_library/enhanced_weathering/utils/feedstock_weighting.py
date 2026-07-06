# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

"""Feedstock composition weighting for multi-batch enhanced weathering projects.

When multiple feedstock batches (crushers) supply different plots in a project,
the effective feedstock composition for CDR quantification is a weighted average
across batches. The weight for each batch reflects its contribution to the
sampled area, measured by the fraction of soil sampling locations that received
rock from that batch.
"""

from collections.abc import Mapping, Sequence
from typing import NamedTuple

import numpy as np
import pandas as pd

from isometric_calculation_library.utils.types import Np1DArray


def compute_weighted_feedstock_composition(
    feedstock_samples: pd.DataFrame,
    batch_weights: Mapping[str, float],
    value_columns: Sequence[str],
    batch_column: str = "feedstock_batch_id",
) -> pd.DataFrame:
    """Compute weighted average feedstock composition across batches.

    For each batch, computes the mean composition from its samples, then
    takes the weighted average across batches. Returns a single-row
    DataFrame with the weighted composition.

    Args:
        feedstock_samples: DataFrame with feedstock composition data.
            Must contain ``batch_column`` and all ``value_columns``.
        batch_weights: Mapping from batch identifier to its weight.
            Weights are normalised internally so they don't need to sum to 1.
        value_columns: Columns to compute weighted averages for
            (e.g. mass fraction columns for Ca, Mg, Ti).
        batch_column: Column identifying the feedstock batch.
    """
    weight_sum = sum(batch_weights.values())
    if weight_sum <= 0:
        raise ValueError(
            f"batch_weights must sum to a positive value, got {weight_sum}.",
        )
    batch_means = feedstock_samples.groupby(batch_column)[list(value_columns)].mean()
    batch_mean_dicts = {col: batch_means[col].to_dict() for col in value_columns}

    # Normalise weights over the batches actually present in feedstock_samples.
    present_batches = set(batch_means.index)
    effective_weights = {k: v for k, v in batch_weights.items() if k in present_batches}
    effective_weight_sum = sum(effective_weights.values())
    if effective_weight_sum <= 0:
        raise ValueError(
            "None of the weighted batches are present in feedstock_samples "
            f"(batch_weights keys: {list(batch_weights)!r}, "
            f"present batches: {sorted(map(str, present_batches))!r}).",
        )
    effective_weights = {k: v / effective_weight_sum for k, v in effective_weights.items()}

    weighted_values = dict[str, float]()
    for col in value_columns:
        col_means = batch_mean_dicts[col]
        total = sum(
            weight * float(col_means[batch_id]) for batch_id, weight in effective_weights.items()
        )
        weighted_values[col] = total

    return pd.DataFrame([weighted_values])


def compute_plot_coverage_weights(
    sample_locations: pd.DataFrame,
    batch_column: str = "feedstock_batch_id",
    location_column: str = "measurement_location_reference_id",
) -> dict[str, float]:
    """Compute feedstock batch weights from plot coverage of soil sampling locations.

    The weight for each batch is the number of unique soil sampling
    locations that received rock from that batch, divided by the total
    number of locations. This assumes ``sample_locations`` has already
    been joined with batch assignment information.

    Args:
        sample_locations: DataFrame with at least ``batch_column`` and
            ``location_column``. Each row represents a soil sample that
            has been linked to the feedstock batch applied at its location.
        batch_column: Column identifying the feedstock batch.
        location_column: Column identifying the measurement location.
    """
    locations_per_batch = sample_locations.groupby(batch_column)[location_column].nunique()
    return {str(k): float(v) for k, v in locations_per_batch.items()}


class BootstrapWeightedFeedstockResult(NamedTuple):
    """Result from bootstrap_weighted_feedstock."""

    point_estimate: pd.DataFrame
    """Single-row DataFrame with weighted mean composition (one value per column)."""
    bootstrap_distributions: dict[str, Np1DArray[np.floating]]
    """Mapping from column name to bootstrap distribution array (n_runs,)."""


def bootstrap_weighted_feedstock(
    feedstock_samples: pd.DataFrame,
    batch_weights: Mapping[str, float],
    value_columns: Sequence[str],
    batch_column: str,
    rng: np.random.Generator,
    n_runs: int,
    *,
    noise_rng: np.random.Generator | None = None,
    noise_fractions: Mapping[str, float] | None = None,
) -> BootstrapWeightedFeedstockResult:
    """Compute weighted feedstock composition with bootstrap uncertainty.

    For each bootstrap iteration, resamples feedstock samples within each
    batch (with replacement), computes per-batch means, then applies the
    fixed weights. This captures measurement uncertainty within batches
    while preserving the plot-based weighting structure.

    A single call with multiple ``value_columns`` shares bootstrap indices
    across all columns, preserving within-replicate correlation between
    co-measured elements (e.g. Ti, Ca, Mg from the same ICP run).

    Args:
        feedstock_samples: DataFrame with feedstock composition data.
            Must contain ``batch_column`` and all ``value_columns``.
        batch_weights: Mapping from batch identifier to its weight.
            Weights are normalised internally so they don't need to sum to 1.
        value_columns: Columns to bootstrap (e.g. mass fraction columns).
        batch_column: Column identifying the feedstock batch.
        rng: Random number generator for bootstrap.
        n_runs: Number of bootstrap iterations.
        noise_rng: If provided, adds independent Gaussian noise to each
            resampled value before averaging. Noise std = ``noise_fractions[col]
            * |sample|`` per column. Ignored when ``noise_fractions`` is None.
        noise_fractions: Per-column relative noise levels (e.g. ``{"Ti": 0.031,
            "Ca": 0.028}``). Columns absent from this mapping receive no noise.
            Ignored when ``noise_rng`` is None.
    """
    weight_sum = sum(batch_weights.values())
    if weight_sum <= 0:
        raise ValueError(
            f"batch_weights must sum to a positive value, got {weight_sum}.",
        )
    normalised_weights = {k: v / weight_sum for k, v in batch_weights.items()}
    cols = list(value_columns)
    n_cols = len(cols)

    batch_groups = dict[str, np.ndarray]()
    batch_means_arr = dict[str, np.ndarray]()

    effective_weight_sum = 0.0
    for batch_id, weight in normalised_weights.items():
        # dropna() drops a row if ANY column is NaN. This is intentional: feedstock
        # composition per batch should be complete; sparse tracers (e.g. Ti) should
        # be flagged as missing data before reaching this step.
        batch_data = feedstock_samples[feedstock_samples[batch_column] == batch_id][cols].dropna()
        if len(batch_data) > 0:
            batch_groups[batch_id] = batch_data.to_numpy()
            batch_means_arr[batch_id] = batch_data.mean().to_numpy()
            effective_weight_sum += weight

    if effective_weight_sum <= 0:
        raise ValueError(
            "None of the weighted feedstock batches have complete (non-NaN) samples, "
            "so a weighted composition cannot be computed. "
            f"batch_weights keys: {list(batch_weights)!r}.",
        )
    normalised_weights = {
        k: v / effective_weight_sum for k, v in normalised_weights.items() if k in batch_groups
    }

    # Point estimate: weighted mean of batch means
    point_values = np.zeros(n_cols)
    for batch_id, mean_vals in batch_means_arr.items():
        point_values += mean_vals * normalised_weights[batch_id]

    point_estimate = pd.DataFrame([dict(zip(cols, point_values, strict=True))])

    # Bootstrap: resample within each batch, then apply weights (vectorised)
    boot_weighted = np.zeros((n_runs, n_cols))

    run_idx = np.arange(n_runs)[:, np.newaxis]  # (n_runs, 1) for advanced indexing

    for batch_id, samples in batch_groups.items():
        n_samples = len(samples)
        indices = rng.integers(0, n_samples, size=(n_runs, n_samples))
        if noise_rng is not None and noise_fractions is not None:
            # Build a per-replicate noisy pool (n_runs, n_samples, n_cols) so that
            # each physical sample has one noise draw per replicate — drawing it
            # multiple times in the same replicate yields the same noisy value.
            noisy_pool = np.empty((n_runs, n_samples, n_cols))
            noisy_pool[:] = samples  # broadcast (n_samples, n_cols) → (n_runs, ...)
            for col_idx, col in enumerate(cols):
                frac = noise_fractions.get(col, 0.0)
                if frac > 0.0:
                    noisy_pool[:, :, col_idx] += noise_rng.normal(
                        scale=np.abs(samples[:, col_idx]) * frac,
                        size=(n_runs, n_samples),
                    )
            selected = noisy_pool[run_idx, indices]  # (n_runs, draw_size, n_cols)
        else:
            selected = samples[indices]  # (n_runs, n_samples, n_cols)
        resampled_means = selected.mean(axis=1)
        boot_weighted += resampled_means * normalised_weights[batch_id]

    bootstrap_distributions: dict[str, Np1DArray[np.floating]] = {
        col: boot_weighted[:, i] for i, col in enumerate(cols)
    }

    return BootstrapWeightedFeedstockResult(
        point_estimate=point_estimate,
        bootstrap_distributions=bootstrap_distributions,
    )
