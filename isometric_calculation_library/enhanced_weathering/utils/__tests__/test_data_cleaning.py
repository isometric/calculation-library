# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

import numpy as np
import pandas as pd
import pytest

from isometric_calculation_library.enhanced_weathering.utils.data_cleaning import (
    ProcessingReport,
    ProcessingStep,
    iterative_sigma_clip,
    null_filter,
    winsorise,
    zero_filter,
)


def _make_df(values: list[float], group: str = "g") -> pd.DataFrame:
    return pd.DataFrame({"value": values, "group": group})


def _make_sample_df(
    *,
    values: list[float | None],
    location_ids: list[str],
) -> pd.DataFrame:
    return pd.DataFrame({"measurement": values, "location_id": location_ids})


# ---------------------------------------------------------------------------
# zero_filter
# ---------------------------------------------------------------------------


def test_zero_filter_drops_samples_with_zero_values() -> None:
    """Samples whose measurement is exactly zero are removed."""
    df = _make_sample_df(values=[1.0, 0.0, 2.0], location_ids=["a", "b", "c"])
    result = zero_filter(df, columns=["measurement"], location_col="location_id", paired=False)
    assert result.n_samples_flagged == 1
    assert result.n_samples_dropped == 1
    assert list(result.samples["measurement"]) == [1.0, 2.0]


def test_zero_filter_paired_drops_all_samples_at_zero_location() -> None:
    """When paired=True, all samples at a location with any zero are dropped."""
    df = _make_sample_df(
        values=[1.0, 0.0, 2.0, 3.0],
        location_ids=["a", "a", "b", "c"],
    )
    result = zero_filter(df, columns=["measurement"], location_col="location_id", paired=True)
    assert result.n_samples_flagged == 1
    assert result.n_samples_dropped == 2
    assert set(result.samples["location_id"]) == {"b", "c"}


def test_zero_filter_does_not_drop_nan_values() -> None:
    """NaN is not equal to zero, so NaN values pass through zero_filter unchanged."""
    df = _make_sample_df(values=[1.0, float("nan"), 2.0], location_ids=["a", "b", "c"])
    result = zero_filter(df, columns=["measurement"], location_col="location_id", paired=False)
    assert result.n_samples_flagged == 0
    assert result.n_samples_dropped == 0
    assert len(result.samples) == 3


def test_zero_filter_no_zeros_returns_all_samples() -> None:
    df = _make_sample_df(values=[1.0, 2.0, 3.0], location_ids=["a", "b", "c"])
    result = zero_filter(df, columns=["measurement"], location_col="location_id", paired=False)
    assert result.n_samples_flagged == 0
    assert result.n_samples_dropped == 0
    assert len(result.samples) == 3


# ---------------------------------------------------------------------------
# null_filter
# ---------------------------------------------------------------------------


def test_null_filter_drops_samples_with_nan_values() -> None:
    """Samples whose measurement is NaN are removed."""
    df = _make_sample_df(values=[1.0, float("nan"), 2.0], location_ids=["a", "b", "c"])
    result = null_filter(df, columns=["measurement"], location_col="location_id", paired=False)
    assert result.n_samples_flagged == 1
    assert result.n_samples_dropped == 1
    assert list(result.samples["measurement"]) == [1.0, 2.0]


def test_null_filter_paired_drops_all_samples_at_nan_location() -> None:
    """When paired=True, all samples at a location with any NaN are dropped."""
    df = _make_sample_df(
        values=[1.0, float("nan"), 2.0, 3.0],
        location_ids=["a", "a", "b", "c"],
    )
    result = null_filter(df, columns=["measurement"], location_col="location_id", paired=True)
    assert result.n_samples_flagged == 1
    assert result.n_samples_dropped == 2
    assert set(result.samples["location_id"]) == {"b", "c"}


def test_null_filter_does_not_drop_zero_values() -> None:
    """Zero is not NaN, so zero values pass through null_filter unchanged."""
    df = _make_sample_df(values=[1.0, 0.0, 2.0], location_ids=["a", "b", "c"])
    result = null_filter(df, columns=["measurement"], location_col="location_id", paired=False)
    assert result.n_samples_flagged == 0
    assert result.n_samples_dropped == 0
    assert len(result.samples) == 3


def test_null_filter_no_nulls_returns_all_samples() -> None:
    df = _make_sample_df(values=[1.0, 2.0, 3.0], location_ids=["a", "b", "c"])
    result = null_filter(df, columns=["measurement"], location_col="location_id", paired=False)
    assert result.n_samples_flagged == 0
    assert result.n_samples_dropped == 0
    assert len(result.samples) == 3


def test_null_filter_multiple_columns_flags_any_nan() -> None:
    """A row is flagged if any of the checked columns contains NaN."""
    df = pd.DataFrame({
        "ca": [1.0, float("nan"), 3.0],
        "mg": [1.0, 2.0, float("nan")],
        "location_id": ["a", "b", "c"],
    })
    result = null_filter(df, columns=["ca", "mg"], location_col="location_id", paired=False)
    assert result.n_samples_flagged == 2
    assert result.n_samples_dropped == 2
    assert list(result.samples["ca"]) == [1.0]


# Tight cluster of 10 values around 1.0 (std ~0.05) plus one extreme outlier.
# With std ~0.05, a value of 100 is >> 3 sigma above the mean.
_CLEAN = [1.00, 1.02, 0.98, 1.01, 0.99, 1.03, 0.97, 1.00, 1.02, 0.98]
_OUTLIER = 100.0
_EXTREME_OUTLIER = 10_000.0


# ---------------------------------------------------------------------------
# winsorise
# ---------------------------------------------------------------------------


def test_winsorise_clips_outlier() -> None:
    """Values beyond n_std are clipped to the group bound."""
    df = _make_df([*_CLEAN, _OUTLIER])
    result = winsorise(df, columns=["value"], group_columns=["group"], n_std=3.0)
    assert result.n_values_clipped == 1
    assert result.samples["value"].max() < _OUTLIER


def test_winsorise_no_clip_when_no_outliers() -> None:
    df = _make_df(_CLEAN)
    result = winsorise(df, columns=["value"], group_columns=["group"], n_std=3.0)
    assert result.n_values_clipped == 0


def test_winsorise_skips_small_groups() -> None:
    """Groups smaller than min_group_size are left untouched."""
    df = _make_df([1.0, _OUTLIER], group="g")
    result = winsorise(df, columns=["value"], group_columns=["group"], n_std=3.0, min_group_size=5)
    assert result.n_values_clipped == 0
    assert result.samples["value"].max() == _OUTLIER


# ---------------------------------------------------------------------------
# iterative_sigma_clip
# ---------------------------------------------------------------------------


def test_iterative_sigma_clip_clips_outlier() -> None:
    """A single extreme outlier is clipped."""
    df = _make_df([*_CLEAN, _OUTLIER])
    result = iterative_sigma_clip(df, columns=["value"], group_columns=["group"], n_std=3.0)
    assert result.n_values_clipped == 1
    assert result.samples["value"].max() < _OUTLIER


def test_iterative_sigma_clip_tighter_than_winsorise_with_extreme_outlier() -> None:
    """Iterative clipping anchors bounds to the clean distribution, so the
    replacement value is lower than standard winsorisation when a strong outlier
    inflates the group std."""
    df = _make_df([*_CLEAN, _EXTREME_OUTLIER])

    std_result = winsorise(df, columns=["value"], group_columns=["group"], n_std=3.0)
    iter_result = iterative_sigma_clip(df, columns=["value"], group_columns=["group"], n_std=3.0)

    # Both clip the outlier
    assert std_result.n_values_clipped == 1
    assert iter_result.n_values_clipped == 1

    # Iterative bound is tighter because the outlier is excluded from stat computation
    assert iter_result.samples["value"].max() < std_result.samples["value"].max()


def test_iterative_sigma_clip_no_clip_when_no_outliers() -> None:
    df = _make_df(_CLEAN)
    result = iterative_sigma_clip(df, columns=["value"], group_columns=["group"], n_std=3.0)
    assert result.n_values_clipped == 0


def test_iterative_sigma_clip_skips_small_groups() -> None:
    df = _make_df([1.0, _OUTLIER])
    result = iterative_sigma_clip(
        df,
        columns=["value"],
        group_columns=["group"],
        n_std=3.0,
        min_group_size=5,
    )
    assert result.n_values_clipped == 0
    assert result.samples["value"].max() == _OUTLIER


def test_iterative_sigma_clip_multiple_outliers() -> None:
    """Two one-sided outliers of different magnitudes are both clipped via iteration.

    The extreme outlier (5000) inflates std enough that the moderate one (500) is
    within 3-sigma of the contaminated distribution in iteration 1. Once 5000 is
    removed, the moderate outlier becomes visible in iteration 2.
    """
    df = _make_df([*_CLEAN, 500.0, 5000.0])
    result = iterative_sigma_clip(df, columns=["value"], group_columns=["group"], n_std=3.0)
    assert result.n_values_clipped == 2


def test_iterative_sigma_clip_respects_max_iterations() -> None:
    """All three clear outliers (5, 6, 7) are clipped within max_iterations passes."""
    rng = np.random.default_rng(0)
    normal_vals = rng.normal(1.0, 0.1, size=20).tolist()
    df = _make_df([*normal_vals, 5.0, 6.0, 7.0])
    result = iterative_sigma_clip(
        df,
        columns=["value"],
        group_columns=["group"],
        n_std=2.0,
        max_iterations=3,
    )
    assert result.samples["value"].max() < 5.0


def test_iterative_sigma_clip_preserves_non_outlier_values() -> None:
    """Non-outlier values are unchanged after clipping."""
    df = _make_df([*_CLEAN, _OUTLIER])
    result = iterative_sigma_clip(df, columns=["value"], group_columns=["group"], n_std=3.0)
    result_clean = result.samples["value"].iloc[: len(_CLEAN)].tolist()
    assert result_clean == pytest.approx(_CLEAN)


# ---------------------------------------------------------------------------
# ProcessingReport
# ---------------------------------------------------------------------------


def test_processing_report_summary_handles_zero_initial_rows() -> None:
    """summary() must not raise (ZeroDivisionError) when the first step starts from
    zero rows; the removed percentage is reported as 'n/a'."""
    report = ProcessingReport()
    report.add(ProcessingStep("empty step", rows_before=0, rows_after=0))
    summary = report.summary()
    assert "n/a" in summary
