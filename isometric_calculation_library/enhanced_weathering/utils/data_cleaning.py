# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

"""Data cleaning utilities for enhanced weathering quantification."""

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import NamedTuple, override

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ProcessingStep:
    """Statistics tracking for a single data processing step."""

    step_name: str
    rows_before: int
    rows_after: int
    values_affected: int = 0
    details: Mapping[str, int] = field(default_factory=dict)

    @property
    def rows_removed(self) -> int:
        return self.rows_before - self.rows_after

    @override
    def __str__(self) -> str:
        msg = f"{self.step_name}: {self.rows_before} -> {self.rows_after} rows"
        if self.rows_removed > 0:
            msg += f" ({self.rows_removed} removed)"
        if self.values_affected > 0:
            msg += f", {self.values_affected} values affected"
        if len(self.details) > 0:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            msg += f" [{details_str}]"
        return msg


@dataclass
class ProcessingReport:
    """Full report of all data processing steps."""

    steps: list[ProcessingStep] = field(default_factory=list)

    def add(self, stats: ProcessingStep) -> None:
        self.steps.append(stats)
        logger.info(str(stats))

    def summary(self) -> str:
        lines = ["Data Processing Report", "=" * 50]
        lines.extend(str(step) for step in self.steps)
        lines.append("=" * 50)
        if len(self.steps) > 0:
            total_removed = self.steps[0].rows_before - self.steps[-1].rows_after
            lines.append(
                f"Total: {self.steps[0].rows_before} -> {self.steps[-1].rows_after} rows "
                f"({total_removed} removed, "
                f"{total_removed / self.steps[0].rows_before * 100:.1f}%)",
            )
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "step": s.step_name,
                "rows_before": s.rows_before,
                "rows_after": s.rows_after,
                "rows_removed": s.rows_removed,
                "values_affected": s.values_affected,
            }
            for s in self.steps
        ])


class ZeroFilterResult(NamedTuple):
    """Result from zero_filter."""

    samples: pd.DataFrame
    n_samples_flagged: int
    """Number of individual samples with a zero value."""
    n_samples_dropped: int
    """Total samples dropped (all samples at affected locations)."""


class WinsoriseResult(NamedTuple):
    """Result from winsorise."""

    samples: pd.DataFrame
    n_values_clipped: int
    """Number of individual values clipped."""


def zero_filter(
    samples: pd.DataFrame,
    columns: Sequence[str],
    location_col: str,
    paired: bool = True,
) -> ZeroFilterResult:
    """Remove samples with zero values in any of the specified columns.

    Args:
        samples: DataFrame with sample data.
        columns: Columns to check for zero values.
        location_col: Column identifying sampling locations.
        paired: If True, drop all samples at locations where any sample has
            a zero value. If False, drop only the individual samples with
            zero values.
    """
    zero_mask = pd.Series(False, index=samples.index)
    for col in columns:
        zero_mask |= samples[col] == 0

    n_flagged = int(zero_mask.sum())

    if paired:
        flagged_locs = samples.loc[zero_mask, location_col].unique()
        drop_mask = samples[location_col].isin(flagged_locs)
    else:
        drop_mask = zero_mask

    n_dropped = int(drop_mask.sum())

    return ZeroFilterResult(
        samples=samples[~drop_mask].copy(),
        n_samples_flagged=n_flagged,
        n_samples_dropped=n_dropped,
    )


def winsorise(
    samples: pd.DataFrame,
    columns: Sequence[str],
    group_columns: Sequence[str],
    n_std: float = 3.0,
    min_group_size: int = 5,
) -> WinsoriseResult:
    """Clip values to mean +/- n_std standard deviations per group.

    Args:
        samples: DataFrame with sample data.
        columns: Numeric columns to winsorise.
        group_columns: Columns defining groups for per-group statistics.
        n_std: Number of standard deviations for clipping bounds.
        min_group_size: Minimum group size to apply clipping.
    """
    samples = samples.copy()
    values_clipped = 0
    group_cols_list = list(group_columns)

    for col in columns:
        group_means = samples.groupby(group_cols_list)[col].transform("mean")
        group_stds = samples.groupby(group_cols_list)[col].transform("std")
        group_counts = samples.groupby(group_cols_list)[col].transform("count")

        can_clip = (
            samples[col].notna()
            & group_stds.notna()
            & (group_stds != 0)
            & (group_counts >= min_group_size)
        )

        lower = group_means - n_std * group_stds
        upper = group_means + n_std * group_stds
        clipped = samples[col].clip(lower=lower, upper=upper)

        changed = can_clip & (clipped != samples[col])
        values_clipped += int(changed.sum())
        samples.loc[can_clip, col] = clipped[can_clip]

    return WinsoriseResult(samples=samples, n_values_clipped=values_clipped)
