# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

from collections.abc import Mapping, Sequence
from typing import NamedTuple

import numpy as np
import pandas as pd

from isometric_calculation_library.utils.types import Np1DArray


def resample_mean(
    rng: np.random.Generator,
    values: Np1DArray[np.floating],
    n_runs: int,
) -> Np1DArray[np.floating]:
    """Bootstrap resample an array and compute mean for each iteration."""
    n_samples = len(values)
    resampled = rng.choice(values.astype(np.float32), size=(n_runs, n_samples), replace=True)
    return np.nanmean(resampled, axis=1)


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
    *,
    resample_size: int | None = None,
) -> np.ndarray:
    """Generate bootstrap resampling indices for location-level resampling.

    These indices can be reused across multiple variables measured at the same
    locations (e.g., different cation concentrations) to ensure consistent
    resampling and preserve cross-variable correlations.

    Args:
        rng: NumPy random generator.
        n_locations: Number of locations to resample from (population size).
        n_runs: Number of bootstrap iterations.
        resample_size: Number of locations to draw per replicate. Defaults to
            ``n_locations`` (standard bootstrap). Set to ``n_eff`` when spatial
            autocorrelation reduces effective sample size.

    Returns:
        Integer array of shape (n_runs, draw_size) with values in
        [0, n_locations), where draw_size is ``resample_size`` or
        ``n_locations``.
    """
    draw_size = resample_size if resample_size is not None else n_locations
    return rng.integers(0, n_locations, size=(n_runs, draw_size))


def compute_resampled_means_from_indices(
    values: Np1DArray[np.floating],
    indices: np.ndarray,
    *,
    noise_rng: np.random.Generator | None = None,
    noise_fraction: float = 0.0,
) -> Np1DArray[np.floating]:
    """Compute bootstrap means using pre-generated resampling indices.

    Use with indices from ``generate_bootstrap_location_indices`` to apply
    consistent resampling across multiple variables at the same locations.

    Args:
        values: Array of per-location values to resample.
        indices: Bootstrap location indices of shape (n_runs, n_locations).
        noise_rng: If provided, pre-perturbs each physical sample once per
            replicate before resampling. A sample drawn multiple times in the
            same replicate receives the same noise draw, because noise is tied
            to the sample rather than to each draw event.
        noise_fraction: Relative noise level (e.g. 0.1 for 10%). Ignored when
            ``noise_rng`` is None.
    """
    if noise_rng is not None and noise_fraction > 0.0:
        n_runs = indices.shape[0]
        n_pool = len(values)
        # Pre-perturb each physical sample once per replicate (n_runs, n_pool).
        # Indices then draw from this noisy pool — duplicated draws within a
        # replicate pick the same pre-perturbed value.
        noisy_pool = values + noise_rng.normal(
            scale=np.abs(values) * noise_fraction,
            size=(n_runs, n_pool),
        )
        selected = noisy_pool[np.arange(n_runs)[:, np.newaxis], indices]
    else:
        selected = values[indices]
    return np.mean(selected, axis=1)


def summarize_distributions(
    distributions: Mapping[str, Np1DArray[np.floating]],
) -> pd.DataFrame:
    """Summarize bootstrap distributions with standard percentile statistics.

    Args:
        distributions: Mapping from distribution name to bootstrap samples.
    """
    return pd.DataFrame([
        {
            "distribution_name": name,
            "mean": float(np.nanmean(values)),
            "std": float(np.nanstd(values)),
            "p5": float(np.nanpercentile(values, 5)),
            "p16": float(np.nanpercentile(values, 16)),
            "p30": float(np.nanpercentile(values, 30)),
            "p40": float(np.nanpercentile(values, 40)),
            "median": float(np.nanmedian(values)),
            "p84": float(np.nanpercentile(values, 84)),
            "p95": float(np.nanpercentile(values, 95)),
        }
        for name, values in distributions.items()
    ])


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


class SamplingEventMeans(NamedTuple):
    """Bootstrapped replicate means for two sampling events, keyed by value column."""

    baseline: Mapping[str, Np1DArray[np.floating]]
    reporting_period: Mapping[str, Np1DArray[np.floating]]

    def column(self, name: str) -> tuple[Np1DArray[np.floating], Np1DArray[np.floating]]:
        """Both events' replicate means for one column, baseline first.

        Unpacks the single-column case, which would otherwise have to index both mappings
        by the same key.
        """
        return self.baseline[name], self.reporting_period[name]


def _aligned_length(values: Mapping[str, Np1DArray[np.floating]], event: str) -> int:
    """Length shared by every column of one event, raising if they disagree."""
    if len(values) == 0:
        raise ValueError(f"At least one value column is required to resample the {event}.")
    lengths = {len(v) for v in values.values()}
    if len(lengths) > 1:
        raise ValueError(
            f"Columns within the {event} must be aligned, got lengths {sorted(lengths)!r}.",
        )
    return next(iter(lengths))


def _means_by_column(
    values: Mapping[str, Np1DArray[np.floating]],
    indices: np.ndarray,
) -> dict[str, Np1DArray[np.floating]]:
    """Apply one already-drawn index set to every column of an event."""
    return {
        col: compute_resampled_means_from_indices(column_values, indices)
        for col, column_values in values.items()
    }


def resample_columns_together(
    rng: np.random.Generator,
    *,
    values: Mapping[str, Np1DArray[np.floating]],
    n_runs: int,
    event: str,
) -> dict[str, Np1DArray[np.floating]]:
    """Bootstrap the mean of one sampling event across several value columns at once.

    One index set is drawn and applied to every column, so the values a replicate combines
    come from the same draw of samples. Resampling each column separately would let a
    replicate mix samples, so anything combining columns afterwards — pooling cations onto a
    charge-equivalent basis, say — would be summing values measured on different rock or soil.

    Args:
        rng: Random number generator.
        values: Per-sample values keyed by column. All arrays must be the same length and
            aligned, element i of each referring to the same physical sample.
        n_runs: Bootstrap replicates.
        event: What the samples are, for the error messages (e.g. ``"baseline event"``).
    """
    indices = generate_bootstrap_location_indices(rng, _aligned_length(values, event), n_runs)
    return _means_by_column(values, indices)


def resample_events_together(
    rng: np.random.Generator,
    *,
    baseline_values: Mapping[str, Np1DArray[np.floating]],
    reporting_period_values: Mapping[str, Np1DArray[np.floating]],
    n_runs: int,
    paired: bool,
) -> SamplingEventMeans:
    """Bootstrap the mean of two sampling events across one or more value columns at once.

    One index set is shared across all columns of an event, so a replicate is a coherent draw
    of samples. Drawing per column instead would let a replicate mix samples, and a caller
    that then combines columns within a replicate — pooling cations onto a charge-equivalent
    basis, say — would be summing values that do not describe the same soil.

    A single column is the one-entry case; use ``SamplingEventMeans.column`` to unpack it.

    ``paired`` decides whether the two *events* also share that index set: True when the rows
    are matched by location (e.g. from ``pair_locations``), so the spatial variance the two
    events share cancels; False when they are independent, in which case they may also have
    different lengths.

    Args:
        rng: Random number generator.
        baseline_values: Per-location values keyed by column. All arrays must be the same
            length and aligned, element i of each referring to the same location.
        reporting_period_values: The same columns for the reporting-period event. Must be the
            same length as ``baseline_values`` when ``paired``.
        n_runs: Bootstrap replicates.
        paired: Whether the two events are matched by location.
    """
    if set(baseline_values) != set(reporting_period_values):
        raise ValueError(
            f"Both events must carry the same columns, got {sorted(baseline_values)!r} and "
            f"{sorted(reporting_period_values)!r}.",
        )
    n_baseline = _aligned_length(baseline_values, "baseline event")
    n_reporting_period = _aligned_length(reporting_period_values, "reporting-period event")
    if paired and n_baseline != n_reporting_period:
        raise ValueError(
            f"A paired design needs one row per matched location in both events, got "
            f"{n_baseline} baseline and {n_reporting_period} reporting-period rows.",
        )

    baseline_indices = generate_bootstrap_location_indices(rng, n_baseline, n_runs)
    reporting_period_indices = (
        baseline_indices
        if paired
        else generate_bootstrap_location_indices(rng, n_reporting_period, n_runs)
    )
    return SamplingEventMeans(
        baseline=_means_by_column(baseline_values, baseline_indices),
        reporting_period=_means_by_column(reporting_period_values, reporting_period_indices),
    )
