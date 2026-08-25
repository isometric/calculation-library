# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

import warnings
from collections.abc import Mapping, Sequence
from typing import Literal, NamedTuple

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

PlotType = Literal["control", "treatment", "deployment"]

PLOT_TYPE_COLUMN = "plot_type"
"""Column a plots frame carries its plot type under."""

GEOMETRY_COLUMN = "geometry"
"""Column a plots frame carries its polygons under."""

_DEPRECATED_PLOT_TYPE_COLUMN = "Type"
_DEPRECATED_GEOMETRY_COLUMN = "Geometry"


def resolve_plot_columns(
    plots: gpd.GeoDataFrame,
    *,
    _extra_stack_depth: int = 0,
) -> tuple[str, str]:
    """Find which columns a plots frame carries its plot type and geometry under.

    ``plot_type`` and ``geometry`` are the supported names: plot boundaries come from
    ``ENHANCED_WEATHERING_FIELD`` sites, which supply those.

    ``Type`` and ``Geometry`` are still read, but warn. They are the column names inside the
    GeoJSON files the pre-site models were given, so those models cannot be moved off them by
    editing code alone - the uploaded files have to be reissued. Until that happens, breaking
    them would make an already-issued quantification unreproducible. The fallback goes once no
    model depends on it.

    Args:
        plots: Plot polygons to inspect the columns of.
        _extra_stack_depth: Frames between this call and the code a deprecation should be
            reported against. Defaults to reporting against this function's own caller;
            the helpers in this module pass 1, being an intermediate frame themselves.
    """

    def resolve(preferred: str, deprecated: str, what: str) -> str:
        if preferred in plots.columns:
            return preferred
        if deprecated in plots.columns:
            warnings.warn(
                f"plots supplies its {what} as {deprecated!r}, which is deprecated. Site "
                f"inputs supply {preferred!r}, which is what this function expects; "
                f"{deprecated!r} is read for models still on pre-site GeoJSON inputs and "
                "will stop being accepted.",
                DeprecationWarning,
                # resolve -> resolve_plot_columns -> the caller being reported against.
                stacklevel=3 + _extra_stack_depth,
            )
            return deprecated
        raise ValueError(
            f"plots has no {what} column: expected {preferred!r}, found {sorted(plots.columns)}.",
        )

    return (
        resolve(PLOT_TYPE_COLUMN, _DEPRECATED_PLOT_TYPE_COLUMN, "plot type"),
        resolve(GEOMETRY_COLUMN, _DEPRECATED_GEOMETRY_COLUMN, "geometry"),
    )


class SplitByPlotTypeResult(NamedTuple):
    """Result of assigning samples to plot types and splitting by type."""

    splits: Mapping[PlotType, tuple[pd.DataFrame, pd.DataFrame]]
    """Per-plot-type ``(baseline, reporting_period)`` DataFrames."""

    n_baseline_unassigned: int
    """Number of baseline samples that fell outside all plot polygons."""

    n_reporting_period_unassigned: int
    """Number of reporting-period samples that fell outside all plot polygons."""


def assign_area_type(
    samples: pd.DataFrame,
    plots: gpd.GeoDataFrame,
    output_column: str = "plot_type",
) -> pd.DataFrame:
    """Assign area type to samples based on spatial location.

    Args:
        samples: DataFrame with latitude and longitude columns.
        plots: GeoDataFrame of plot polygons with ``plot_type`` and ``geometry`` columns.
            ``Type``/``Geometry`` are still read but deprecated, see
            ``resolve_plot_columns``.
        output_column: Name of the output column for the assigned type.
    """
    type_column, geometry_column = resolve_plot_columns(plots, _extra_stack_depth=1)
    samples_gdf = gpd.GeoDataFrame(
        samples,
        geometry=[
            Point(lon, lat)
            for lon, lat in zip(samples["longitude"], samples["latitude"], strict=True)
        ],
        crs="EPSG:4326",
    )

    # Renamed rather than joined on as-is: an input already using "plot_type" would collide
    # with the output column the join is about to write.
    plots_for_join = plots[[type_column, geometry_column]].rename(
        columns={type_column: "_plot_type", geometry_column: "_geometry"},
    )
    plots_for_join = gpd.GeoDataFrame(plots_for_join).set_geometry("_geometry")

    joined = gpd.sjoin(
        samples_gdf,
        plots_for_join,
        how="left",
        predicate="within",
    )

    # Drop duplicates from overlapping polygons, keeping first match
    joined = joined[~joined.index.duplicated(keep="first")]

    samples = samples.copy()
    samples[output_column] = joined["_plot_type"].str.lower().to_numpy()
    return samples


def assign_and_split_by_plot_type(
    baseline_samples: pd.DataFrame,
    reporting_period_samples: pd.DataFrame,
    plots: gpd.GeoDataFrame,
) -> SplitByPlotTypeResult:
    """Assign area types via spatial join and split into per-plot-type pairs.

    Samples that don't fall within any plot polygon are dropped.  The
    number of dropped samples per period is reported in the result so
    that callers can surface it in data reports.

    Args:
        baseline_samples: Baseline soil samples with ``latitude`` and
            ``longitude`` columns.
        reporting_period_samples: End-of-reporting-period soil samples with
            the same columns.
        plots: Plot geometries with ``plot_type`` and ``geometry`` columns.
            ``Type``/``Geometry`` are still read but deprecated, see
            ``resolve_plot_columns``.
    """
    # Resolved here rather than left to the two assign_area_type calls, so a deprecated
    # spelling is reported once, against this function's caller, instead of twice from a frame
    # deeper than any single stacklevel can point past.
    type_column, geometry_column = resolve_plot_columns(plots, _extra_stack_depth=1)
    plots = gpd.GeoDataFrame(
        plots.rename(
            columns={type_column: PLOT_TYPE_COLUMN, geometry_column: GEOMETRY_COLUMN},
        ),
    ).set_geometry(GEOMETRY_COLUMN)

    baseline_assigned = assign_area_type(baseline_samples, plots)
    reporting_period_assigned = assign_area_type(reporting_period_samples, plots)

    n_baseline_unassigned = int(baseline_assigned["plot_type"].isna().sum())
    n_reporting_period_unassigned = int(reporting_period_assigned["plot_type"].isna().sum())

    baseline_clean = baseline_assigned.dropna(subset=["plot_type"])
    reporting_period_clean = reporting_period_assigned.dropna(subset=["plot_type"])

    plot_types = set(baseline_clean["plot_type"].unique()) | set(
        reporting_period_clean["plot_type"].unique(),
    )

    splits = dict[PlotType, tuple[pd.DataFrame, pd.DataFrame]]()
    for plot_type in sorted(plot_types):
        splits[plot_type] = (
            baseline_clean[baseline_clean["plot_type"] == plot_type],
            reporting_period_clean[reporting_period_clean["plot_type"] == plot_type],
        )
    return SplitByPlotTypeResult(
        splits=splits,
        n_baseline_unassigned=n_baseline_unassigned,
        n_reporting_period_unassigned=n_reporting_period_unassigned,
    )


def calculate_area_hectares_by_plot_type(
    plots: gpd.GeoDataFrame,
    plot_types: Sequence[PlotType] = ("deployment", "treatment"),
) -> Mapping[PlotType, float]:
    """Calculate total area in hectares for each plot type.

    Projects geographic coordinates to UTM before computing area.

    Args:
        plots: GeoDataFrame of plot polygons with ``plot_type`` and ``geometry`` columns.
            ``Type``/``Geometry`` are still read but deprecated, see
            ``resolve_plot_columns``.
        plot_types: Plot types to compute areas for.
    """
    type_column, geometry_column = resolve_plot_columns(plots, _extra_stack_depth=1)
    plots_work = plots.copy()
    plots_work = plots_work.set_geometry(geometry_column)

    if plots_work.crs is not None and plots_work.crs.is_geographic:
        plots_projected = plots_work.to_crs(plots_work.estimate_utm_crs())
    else:
        plots_projected = plots_work

    type_values = plots_projected[type_column].str.lower()

    area_hectares = dict[PlotType, float]()
    for plot_type in plot_types:
        mask = type_values == plot_type
        if mask.any():
            area_m2 = plots_projected.loc[mask, geometry_column].area.sum()
            area_hectares[plot_type] = area_m2 / 10_000

    return area_hectares
