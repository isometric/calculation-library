# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

import numpy as np
import pandas as pd
import pytest

from ..data_cleaning import iterative_sigma_clip, winsorise


def _make_df(values: list[float], group: str = "g") -> pd.DataFrame:
    return pd.DataFrame({"value": values, "group": group})


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
