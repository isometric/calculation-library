# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

import pandas as pd
import pytest

from isometric_calculation_library.enhanced_weathering.utils.pairing import (
    pair_locations,
    paired_column_names,
    require_complete_pairs,
)


def _make_samples(loc_ids: list[str], ca_values: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        "measurement_location_reference_id": loc_ids,
        "mass_fraction_ca": ca_values,
    })


def test_pair_locations_basic_inner_join() -> None:
    """Only locations present in both periods are kept."""
    baseline = _make_samples(["A", "B", "C"], [100.0, 200.0, 300.0])
    rp = _make_samples(["B", "C", "D"], [210.0, 310.0, 400.0])

    result = pair_locations(baseline, rp, value_columns=["mass_fraction_ca"])

    assert len(result.paired) == 2
    assert set(result.paired["measurement_location_reference_id"]) == {"B", "C"}
    assert result.n_baseline_only == 1
    assert result.n_reporting_period_only == 1


def test_pair_locations_all_matched() -> None:
    """When all locations match, no orphans are reported."""
    baseline = _make_samples(["A", "B"], [100.0, 200.0])
    rp = _make_samples(["A", "B"], [110.0, 220.0])

    result = pair_locations(baseline, rp, value_columns=["mass_fraction_ca"])

    assert len(result.paired) == 2
    assert result.n_baseline_only == 0
    assert result.n_reporting_period_only == 0


def test_pair_locations_averages_duplicates() -> None:
    """Multiple samples at the same location are averaged."""
    baseline = _make_samples(["A", "A"], [100.0, 200.0])
    rp = _make_samples(["A"], [300.0])

    result = pair_locations(baseline, rp, value_columns=["mass_fraction_ca"])

    assert len(result.paired) == 1
    assert result.paired["baseline_mass_fraction_ca"].iloc[0] == pytest.approx(150.0)
    assert result.paired["reporting_period_mass_fraction_ca"].iloc[0] == pytest.approx(300.0)


def test_pair_locations_no_overlap() -> None:
    """When no locations overlap, paired DataFrame is empty."""
    baseline = _make_samples(["A"], [100.0])
    rp = _make_samples(["B"], [200.0])

    result = pair_locations(baseline, rp, value_columns=["mass_fraction_ca"])

    assert len(result.paired) == 0
    assert result.n_baseline_only == 1
    assert result.n_reporting_period_only == 1


def test_pair_locations_multiple_columns() -> None:
    """Pairing works with multiple value columns simultaneously."""
    baseline = pd.DataFrame({
        "measurement_location_reference_id": ["A", "B"],
        "mass_fraction_ca": [100.0, 200.0],
        "mass_fraction_mg": [50.0, 75.0],
    })
    rp = pd.DataFrame({
        "measurement_location_reference_id": ["A", "B"],
        "mass_fraction_ca": [110.0, 220.0],
        "mass_fraction_mg": [55.0, 80.0],
    })

    result = pair_locations(
        baseline,
        rp,
        value_columns=["mass_fraction_ca", "mass_fraction_mg"],
    )

    assert len(result.paired) == 2
    assert "baseline_mass_fraction_ca" in result.paired.columns
    assert "reporting_period_mass_fraction_mg" in result.paired.columns


def test_require_complete_pairs_accepts_a_fully_measured_frame() -> None:
    paired = pair_locations(
        _make_samples(["a", "b"], [100.0, 110.0]),
        _make_samples(["a", "b"], [120.0, 130.0]),
        ["mass_fraction_ca"],
    ).paired

    require_complete_pairs(
        paired,
        ["baseline_mass_fraction_ca", "reporting_period_mass_fraction_ca"],
    )


def test_require_complete_pairs_raises_on_a_null_measurement() -> None:
    """A null survives the inner join, so it is a missing measurement, not an unmatched location."""
    paired = pair_locations(
        _make_samples(["a", "b"], [100.0, float("nan")]),
        _make_samples(["a", "b"], [120.0, 130.0]),
        ["mass_fraction_ca"],
    ).paired

    with pytest.raises(ValueError, match="null measurements"):
        require_complete_pairs(
            paired,
            ["baseline_mass_fraction_ca", "reporting_period_mass_fraction_ca"],
        )


def test_require_complete_pairs_counts_affected_locations_not_null_measurements() -> None:
    """One location null in two columns is one affected location, not two.

    Summing the per-column null counts would double-count it, overstating how much of the
    frame is affected in exactly the case the message exists to quantify.
    """
    baseline = pd.DataFrame({
        "measurement_location_reference_id": ["a", "b", "c"],
        "mass_fraction_ca": [float("nan"), 110.0, 120.0],
        "mass_fraction_mg": [float("nan"), 55.0, 60.0],
    })
    reporting_period = pd.DataFrame({
        "measurement_location_reference_id": ["a", "b", "c"],
        "mass_fraction_ca": [130.0, 140.0, 150.0],
        "mass_fraction_mg": [65.0, 70.0, 75.0],
    })
    paired = pair_locations(
        baseline,
        reporting_period,
        ["mass_fraction_ca", "mass_fraction_mg"],
    ).paired

    with pytest.raises(ValueError, match=r"1 of 3 paired location\(s\)"):
        require_complete_pairs(
            paired,
            paired_column_names(["mass_fraction_ca", "mass_fraction_mg"]),
        )


def test_require_complete_pairs_raises_on_a_missing_column() -> None:
    paired = pair_locations(
        _make_samples(["a"], [100.0]),
        _make_samples(["a"], [120.0]),
        ["mass_fraction_ca"],
    ).paired

    with pytest.raises(ValueError, match="not found in the paired data"):
        require_complete_pairs(paired, ["baseline_mass_fraction_mg"])
