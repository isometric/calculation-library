# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

"""Location-level pairing of baseline and reporting-period soil samples."""

from collections.abc import Sequence
from typing import NamedTuple

import pandas as pd


class PairingResult(NamedTuple):
    """Result of pairing baseline and reporting-period samples by location."""

    paired: pd.DataFrame
    """DataFrame with one row per paired location.

    Columns are ``{location_col}``, ``bl_{col}`` and ``rp_{col}`` for each
    value column, where ``bl_`` and ``rp_`` are averages per location.
    """

    n_baseline_only: int
    """Number of locations present only in baseline."""

    n_reporting_period_only: int
    """Number of locations present only in reporting period."""


def pair_locations(
    baseline_samples: pd.DataFrame,
    reporting_period_samples: pd.DataFrame,
    value_columns: Sequence[str],
    location_col: str = "measurement_location_reference_id",
) -> PairingResult:
    """Build per-location paired data for multiple columns.

    Averages each value column per location, then inner-joins on location
    so only locations present in both periods are kept.

    Args:
        baseline_samples: Baseline soil samples with at least ``location_col``
            and all ``value_columns``.
        reporting_period_samples: End-of-reporting-period soil samples with the same columns.
        value_columns: Columns to average and pair (e.g. mass fraction columns).
        location_col: Column identifying measurement locations.
    """
    bl_agg = baseline_samples.groupby(location_col)[list(value_columns)].mean().reset_index()
    rp_agg = (
        reporting_period_samples.groupby(location_col)[list(value_columns)].mean().reset_index()
    )

    bl_renamed = bl_agg.rename(columns={col: f"bl_{col}" for col in value_columns})
    rp_renamed = rp_agg.rename(columns={col: f"rp_{col}" for col in value_columns})

    paired = bl_renamed.merge(rp_renamed, on=location_col, how="inner")

    return PairingResult(
        paired=paired,
        n_baseline_only=len(bl_agg) - len(paired),
        n_reporting_period_only=len(rp_agg) - len(paired),
    )
