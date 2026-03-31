# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

import numpy as np

from ..resampling import summarize_distributions


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
        "median",
        "p84",
        "p95",
    }
    assert set(result.columns) == expected_cols
    assert list(result["distribution_name"]) == ["dist_a", "dist_b"]


def test_summarize_distributions_values_are_consistent() -> None:
    """Percentiles are ordered: p5 < p16 < median < p84 < p95."""
    rng = np.random.default_rng(99)
    distributions = {"x": rng.normal(0, 1, size=10_000)}
    result = summarize_distributions(distributions)

    row = result.iloc[0]
    assert row["p5"] < row["p16"] < row["median"] < row["p84"] < row["p95"]
