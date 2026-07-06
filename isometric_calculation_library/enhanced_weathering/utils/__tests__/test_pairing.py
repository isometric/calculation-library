# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

import pandas as pd
import pytest

from isometric_calculation_library.enhanced_weathering.utils.pairing import (
    pair_locations,
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
