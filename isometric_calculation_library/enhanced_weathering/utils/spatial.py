# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

from collections.abc import Mapping, Sequence
from typing import Literal, NamedTuple

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

PlotType = Literal["control", "treatment", "deployment"]


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
        plots: GeoDataFrame with 'Type' and 'Geometry' columns.
        output_column: Name of the output column for the assigned type.
    """
    samples_gdf = gpd.GeoDataFrame(
        samples,
        geometry=[
            Point(lon, lat)
            for lon, lat in zip(samples["longitude"], samples["latitude"], strict=True)
        ],
        crs="EPSG:4326",
    )

    plots_for_join = plots[["Type", "Geometry"]].copy()
    plots_for_join = plots_for_join.set_geometry("Geometry")

    joined = gpd.sjoin(
        samples_gdf,
        plots_for_join,
        how="left",
        predicate="within",
    )

    # Drop duplicates from overlapping polygons, keeping first match
    joined = joined[~joined.index.duplicated(keep="first")]

    samples = samples.copy()
    samples[output_column] = joined["Type"].str.lower().to_numpy()
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
        plots: Plot geometries with ``Type`` and ``Geometry`` columns.
    """
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
        plots: GeoDataFrame with 'Type' and 'Geometry' columns.
        plot_types: Plot types to compute areas for.
    """
    plots_work = plots.copy()
    plots_work = plots_work.set_geometry("Geometry")

    if plots_work.crs is not None and plots_work.crs.is_geographic:
        plots_projected = plots_work.to_crs(plots_work.estimate_utm_crs())
    else:
        plots_projected = plots_work

    type_values = plots_projected["Type"].str.lower()

    area_hectares = dict[PlotType, float]()
    for plot_type in plot_types:
        mask = type_values == plot_type
        if mask.any():
            area_m2 = plots_projected.loc[mask, "Geometry"].area.sum()
            area_hectares[plot_type] = area_m2 / 10_000

    return area_hectares
