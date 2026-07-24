# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/
# See LICENSE file for additional permissions and contact details

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, NamedTuple

import numpy as np
from scipy import stats

from isometric_calculation_library.biosphere.reforestation_dynamic_baselining.spectral_matcher import (
    PixelMatchResult,
)
from isometric_calculation_library.biosphere.utils.raster import sample_raster_vectorized


class BenchmarkResult(NamedTuple):
    """Benchmark comparison result for a single project pixel."""

    project_plot_coordinates: tuple[float, float]
    difference: float
    project_value_1: float
    project_value_2: float
    control_plots_mean_difference: float
    control_plots_mean_value_1: float
    control_plots_mean_value_2: float


@dataclass(frozen=True)
class PairedTTestResult:
    """Result of a one-sided paired t-test."""

    t_statistic: float
    p_value: float
    mean_difference: float
    std_difference: float
    n_pairs: int
    significant_at_05: bool
    significant_at_01: bool


def perform_paired_ttest(
    project_differences: Sequence[float],
    control_differences: Sequence[float],
    alternative: Literal["greater", "less"] = "greater",
) -> PairedTTestResult:
    """
    Perform a one-sided paired t-test to compare project and control differences.

    Tests whether project differences are significantly different from control differences.

    Args:
        project_differences: Differences for project pixels.
        control_differences: Mean differences for matched control pixels.
        alternative: Direction of the test.
            - "greater": H1 is that project_differences > control_differences (project outperforms control)
            - "less": H1 is that project_differences < control_differences (control outperforms project)

    Returns:
        PairedTTestResult with test statistics and significance indicators.
    """
    project_array = np.array(project_differences)
    control_array = np.array(control_differences)

    # Remove pairs where either value is NaN
    valid_mask = ~(np.isnan(project_array) | np.isnan(control_array))
    project_valid = project_array[valid_mask]
    control_valid = control_array[valid_mask]

    n_pairs = len(project_valid)
    if n_pairs < 2:
        return PairedTTestResult(
            t_statistic=np.nan,
            p_value=np.nan,
            mean_difference=np.nan,
            std_difference=np.nan,
            n_pairs=n_pairs,
            significant_at_05=False,
            significant_at_01=False,
        )

    # Paired differences (project - control)
    paired_differences = project_valid - control_valid
    mean_difference = float(np.mean(paired_differences))
    std_difference = float(np.std(paired_differences, ddof=1))

    # Perform paired t-test
    t_statistic, p_value_two_sided = stats.ttest_rel(project_valid, control_valid)

    # Convert to one-sided p-value based on alternative hypothesis
    match alternative:
        case "greater":
            # H1: project > control (positive t-statistic supports this)
            p_value = p_value_two_sided / 2 if t_statistic > 0 else 1 - p_value_two_sided / 2
        case "less":
            # H1: project < control (negative t-statistic supports this)
            p_value = p_value_two_sided / 2 if t_statistic < 0 else 1 - p_value_two_sided / 2

    return PairedTTestResult(
        t_statistic=t_statistic.item(),
        p_value=p_value.item(),
        mean_difference=mean_difference,
        std_difference=std_difference,
        n_pairs=n_pairs,
        significant_at_05=(p_value < 0.05).item(),
        significant_at_01=(p_value < 0.01).item(),
    )


def _validate_wgs84_coordinate_ranges(latitudes: np.ndarray, longitudes: np.ndarray) -> None:
    if latitudes.size > 0 and (np.any(latitudes < -90) or np.any(latitudes > 90)):
        raise ValueError("Latitude values outside [-90, 90] found - expected WGS84 coordinates.")
    if longitudes.size > 0 and (np.any(longitudes < -180) or np.any(longitudes > 180)):
        raise ValueError("Longitude values outside [-180, 180] found - expected WGS84 coordinates.")


def calculate_intra_plot_difference(
    matched_data: Sequence[PixelMatchResult],
    raster_path_1: str,
    raster_path_2: str,
) -> list[BenchmarkResult]:
    """Calculate differences between two rasters at matched pixel locations.

    Assumes match coordinates (as produced by SpectralMatcher, whose target_crs is
    always WGS84) are (longitude, latitude) pairs in EPSG:4326.
    """
    if len(matched_data) == 0:
        return []

    # Extract all centroids at once
    project_centroids = list[tuple[float, float]]()
    control_centroids = list[tuple[float, float]]()
    control_counts = list[int]()

    for match in matched_data:
        longitude, latitude = (
            match.project_pixel_coordinates[0],
            match.project_pixel_coordinates[1],
        )
        project_centroids.append(
            (latitude, longitude),
        )  # store as (latitude, longitude) for sampling function

        # Control plot centroids
        donor_matches = match.matches
        control_counts.append(len(donor_matches))

        for donor_match in donor_matches:
            longitude, latitude = (
                donor_match.donor_pixel_coordinates[0],
                donor_match.donor_pixel_coordinates[1],
            )
            control_centroids.append((latitude, longitude))  # store as (latitude, longitude)

    # Convert to numpy arrays
    project_latitudes, project_longitudes = (
        zip(*project_centroids, strict=True) if project_centroids else ([], [])
    )
    control_latitudes, control_longitudes = (
        zip(*control_centroids, strict=True) if control_centroids else ([], [])
    )

    project_latitudes = np.array(project_latitudes)
    project_longitudes = np.array(project_longitudes)
    control_latitudes = np.array(control_latitudes)
    control_longitudes = np.array(control_longitudes)

    _validate_wgs84_coordinate_ranges(project_latitudes, project_longitudes)
    _validate_wgs84_coordinate_ranges(control_latitudes, control_longitudes)

    # Sample both rasters for all points at once
    project_values_1 = sample_raster_vectorized(
        raster_path_1,
        project_latitudes,
        project_longitudes,
    )
    project_values_2 = sample_raster_vectorized(
        raster_path_2,
        project_latitudes,
        project_longitudes,
    )
    project_differences = project_values_2 - project_values_1

    control_values_1 = sample_raster_vectorized(
        raster_path_1,
        control_latitudes,
        control_longitudes,
    )
    control_values_2 = sample_raster_vectorized(
        raster_path_2,
        control_latitudes,
        control_longitudes,
    )
    control_differences = control_values_2 - control_values_1

    # Split control arrays into groups matching each project pixel
    split_indices = np.cumsum(control_counts[:-1])
    control_difference_groups = np.split(control_differences, split_indices)
    control_values_1_groups = np.split(control_values_1, split_indices)
    control_values_2_groups = np.split(control_values_2, split_indices)

    results = list[BenchmarkResult]()
    for i, match in enumerate(matched_data):
        group_differences = control_difference_groups[i]
        group_values_1 = control_values_1_groups[i]
        group_values_2 = control_values_2_groups[i]

        results.append(
            BenchmarkResult(
                project_plot_coordinates=match.project_pixel_coordinates,
                difference=float(project_differences[i]),
                project_value_1=float(project_values_1[i]),
                project_value_2=float(project_values_2[i]),
                control_plots_mean_difference=float(np.nanmean(group_differences))
                if len(group_differences) > 0
                else float("nan"),
                control_plots_mean_value_1=float(np.nanmean(group_values_1))
                if len(group_values_1) > 0
                else float("nan"),
                control_plots_mean_value_2=float(np.nanmean(group_values_2))
                if len(group_values_2) > 0
                else float("nan"),
            ),
        )

    return results
