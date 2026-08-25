# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

import inspect
import warnings

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon

from isometric_calculation_library.enhanced_weathering.utils.spatial import (
    assign_and_split_by_plot_type,
    assign_area_type,
    calculate_area_hectares_by_plot_type,
    resolve_plot_columns,
)

_DEPLOYMENT_POLYGON = Polygon([(0, 0), (0.01, 0), (0.01, 0.01), (0, 0.01)])
_CONTROL_POLYGON = Polygon([(1, 1), (1.01, 1), (1.01, 1.01), (1, 1.01)])


def _site_plots() -> gpd.GeoDataFrame:
    """Plots as a site input supplies them: ``plot_type`` and ``geometry``."""
    return gpd.GeoDataFrame(
        {"plot_type": ["deployment", "control"]},
        geometry=[_DEPLOYMENT_POLYGON, _CONTROL_POLYGON],
        crs="EPSG:4326",
    )


def _csv_plots() -> gpd.GeoDataFrame:
    """The same plots under the CSV-era column names ``Type`` and ``Geometry``."""
    return gpd.GeoDataFrame(
        {
            "Type": ["deployment", "control"],
            "Geometry": [_DEPLOYMENT_POLYGON, _CONTROL_POLYGON],
        },
        geometry="Geometry",
        crs="EPSG:4326",
    )


def _samples() -> pd.DataFrame:
    """One sample inside each polygon, and one outside both."""
    return pd.DataFrame({
        "longitude": [0.005, 1.005, 5.0],
        "latitude": [0.005, 1.005, 5.0],
        "mass_fraction_ca": [100.0, 200.0, 300.0],
    })


def _current_line() -> int:
    """Line number of the call site, so a stacklevel assertion needs no hardcoded line."""
    frame = inspect.currentframe()
    if frame is None or frame.f_back is None:
        raise RuntimeError("No caller frame; this interpreter does not support introspection.")
    return frame.f_back.f_lineno


def test_resolve_plot_columns_prefers_site_field_names() -> None:
    """Site fields win when a frame carries both spellings, and warn about neither."""
    both = gpd.GeoDataFrame(
        {"plot_type": ["deployment"], "Type": ["control"]},
        geometry=[_DEPLOYMENT_POLYGON],
        crs="EPSG:4326",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert resolve_plot_columns(both) == ("plot_type", "geometry")


def test_resolve_plot_columns_warns_on_the_pre_site_names() -> None:
    """``Type``/``Geometry`` still resolve, but say they are going away."""
    with pytest.warns(DeprecationWarning, match="which is deprecated"):
        assert resolve_plot_columns(_csv_plots()) == ("Type", "Geometry")


def test_deprecation_is_reported_against_the_calling_line() -> None:
    """The warning names the caller's line, whether called directly or via a helper.

    Two different stack depths reach the same warning, so a single ``stacklevel`` cannot
    serve both; getting it wrong points a model author at library internals.
    """
    with warnings.catch_warnings(record=True) as direct:
        warnings.simplefilter("always")
        direct_call_line = _current_line() + 1
        resolve_plot_columns(_csv_plots())

    with warnings.catch_warnings(record=True) as via_helper:
        warnings.simplefilter("always")
        helper_call_line = _current_line() + 1
        calculate_area_hectares_by_plot_type(_csv_plots(), plot_types=("deployment",))

    assert direct[0].filename == __file__
    assert via_helper[0].filename == __file__
    assert direct[0].lineno == direct_call_line
    assert via_helper[0].lineno == helper_call_line


def test_resolve_plot_columns_raises_when_plot_type_is_absent() -> None:
    """A frame with geometry but no plot type names what it looked for."""
    plots = gpd.GeoDataFrame(geometry=[_DEPLOYMENT_POLYGON], crs="EPSG:4326")
    with pytest.raises(ValueError, match="no plot type column"):
        resolve_plot_columns(plots)


def test_assign_area_type_accepts_site_fields() -> None:
    """A site-shaped plots frame assigns the same types the CSV-shaped one does."""
    from_sites = assign_area_type(_samples(), _site_plots())
    with pytest.warns(DeprecationWarning, match="which is deprecated"):
        from_csv = assign_area_type(_samples(), _csv_plots())

    assert from_sites["plot_type"].tolist()[:2] == ["deployment", "control"]
    assert from_sites["plot_type"].isna().iloc[2]
    assert from_sites["plot_type"].tolist() == from_csv["plot_type"].tolist()


def test_assign_and_split_by_plot_type_accepts_site_fields() -> None:
    """Splitting works off site fields, and reports the sample outside every polygon."""
    result = assign_and_split_by_plot_type(_samples(), _samples(), _site_plots())

    assert sorted(result.splits.keys()) == ["control", "deployment"]
    assert result.n_baseline_unassigned == 1
    assert result.n_reporting_period_unassigned == 1
    deployment_baseline, _ = result.splits["deployment"]
    assert deployment_baseline["mass_fraction_ca"].tolist() == [100.0]


def test_calculate_area_hectares_by_plot_type_accepts_site_fields() -> None:
    """Areas are identical whichever column names the plots arrive under."""
    from_sites = calculate_area_hectares_by_plot_type(
        _site_plots(),
        plot_types=("deployment", "control"),
    )
    with pytest.warns(DeprecationWarning, match="which is deprecated"):
        from_csv = calculate_area_hectares_by_plot_type(
            _csv_plots(),
            plot_types=("deployment", "control"),
        )

    assert from_sites == from_csv
    assert from_sites["deployment"] > 0
