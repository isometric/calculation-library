# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

import math
from pathlib import Path

import numpy as np
import pytest
import rasterio
from numpy.testing import assert_allclose
from rasterio.transform import from_origin

from isometric_calculation_library.biosphere.reforestation_dynamic_baselining.performance_benchmark_calculator import (
    calculate_intra_plot_difference,
    perform_paired_ttest,
)
from isometric_calculation_library.biosphere.reforestation_dynamic_baselining.spectral_matcher import (
    DonorMatch,
    PixelMatchResult,
)

# --- perform_paired_ttest ---


def test_perform_paired_ttest_detects_significant_positive_difference() -> None:
    project_differences = [10.0] * 10
    control_differences = [1.0] * 10

    result = perform_paired_ttest(project_differences, control_differences, alternative="greater")

    assert result.mean_difference == pytest.approx(9.0)
    assert result.n_pairs == 10
    assert result.significant_at_05
    assert result.significant_at_01
    assert result.p_value < 0.01


def test_perform_paired_ttest_not_significant_when_no_real_difference() -> None:
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 1, 20)
    project_differences = list(noise)
    control_differences = list(noise + rng.normal(0, 0.01, 20))

    result = perform_paired_ttest(project_differences, control_differences, alternative="greater")

    assert not result.significant_at_05


def test_perform_paired_ttest_drops_nan_pairs() -> None:
    project_differences = [10.0, 10.0, float("nan"), 10.0, 10.0]
    control_differences = [1.0, 1.0, 1.0, float("nan"), 1.0]

    result = perform_paired_ttest(project_differences, control_differences)

    assert result.n_pairs == 3


def test_perform_paired_ttest_returns_nan_result_for_fewer_than_two_valid_pairs() -> None:
    result = perform_paired_ttest([10.0], [1.0])

    assert result.n_pairs == 1
    assert math.isnan(result.t_statistic)
    assert math.isnan(result.p_value)
    assert not result.significant_at_05
    assert not result.significant_at_01


def test_perform_paired_ttest_less_alternative_detects_negative_difference() -> None:
    project_differences = [1.0] * 10
    control_differences = [10.0] * 10

    result = perform_paired_ttest(project_differences, control_differences, alternative="less")

    assert result.significant_at_05


def _write_raster(path: Path, values: np.ndarray, nodata: float | None = None) -> str:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=values.shape[0],
        width=values.shape[1],
        count=1,
        dtype=np.float32,
        crs="EPSG:4326",
        transform=from_origin(0, 4, 1, 1),
        nodata=nodata,
    ) as destination:
        destination.write(values.astype(np.float32), 1)
    return str(path)


# --- calculate_intra_plot_difference ---


def _pixel_match(
    project_coordinates: tuple[float, float],
    donor_coordinates_list: list[tuple[float, float]],
) -> PixelMatchResult:
    donor_matches = [
        DonorMatch(
            distance=0.0,
            normalized_distance=0.0,
            donor_pixel_coordinates=donor_coordinates,
            donor_pixel_values={},
        )
        for donor_coordinates in donor_coordinates_list
    ]
    return PixelMatchResult(
        project_pixel_coordinates=project_coordinates,
        project_pixel_values={},
        matches=donor_matches,
    )


def test_calculate_intra_plot_difference_returns_empty_list_for_no_matches() -> None:
    assert calculate_intra_plot_difference([], "unused_1.tif", "unused_2.tif") == []


def test_calculate_intra_plot_difference_computes_expected_difference(tmp_path: Path) -> None:
    baseline_path = _write_raster(tmp_path / "baseline.tif", np.zeros((4, 4), dtype=np.float32))
    monitoring_values = np.arange(16, dtype=np.float32).reshape(4, 4) + 1
    monitoring_path = _write_raster(tmp_path / "monitoring.tif", monitoring_values)

    # Project pixel at row 0, col 0 (value 1.0); one donor match at row 3, col 3 (value 16.0).
    matched_data = [_pixel_match((0.5, 3.5), [(3.5, 0.5)])]

    results = calculate_intra_plot_difference(
        matched_data,
        str(baseline_path),
        str(monitoring_path),
    )

    assert len(results) == 1
    result = results[0]
    assert result.project_value_1 == pytest.approx(0.0)
    assert result.project_value_2 == pytest.approx(1.0)
    assert result.difference == pytest.approx(1.0)
    assert result.control_plots_mean_value_2 == pytest.approx(16.0)
    assert result.control_plots_mean_difference == pytest.approx(16.0)


def test_calculate_intra_plot_difference_averages_multiple_donor_matches(tmp_path: Path) -> None:
    baseline_path = _write_raster(tmp_path / "baseline.tif", np.zeros((4, 4), dtype=np.float32))
    monitoring_values = np.arange(16, dtype=np.float32).reshape(4, 4) + 1
    monitoring_path = _write_raster(tmp_path / "monitoring.tif", monitoring_values)

    # Donor matches at row 3, col 3 (value 16.0) and row 1, col 1 (value 6.0) -> mean 11.0.
    matched_data = [_pixel_match((0.5, 3.5), [(3.5, 0.5), (1.5, 2.5)])]

    results = calculate_intra_plot_difference(
        matched_data,
        str(baseline_path),
        str(monitoring_path),
    )

    assert_allclose(results[0].control_plots_mean_difference, 11.0)


def test_calculate_intra_plot_difference_rejects_out_of_range_coordinates(tmp_path: Path) -> None:
    """Coordinates outside plausible WGS84 ranges signal a caller passed projected coordinates."""
    baseline_path = _write_raster(tmp_path / "baseline.tif", np.zeros((4, 4), dtype=np.float32))
    monitoring_path = _write_raster(
        tmp_path / "monitoring.tif",
        np.zeros((4, 4), dtype=np.float32),
    )

    # Longitude 500 is not a valid WGS84 coordinate.
    matched_data = [_pixel_match((500.0, 3.5), [(3.5, 0.5)])]

    with pytest.raises(ValueError, match="WGS84"):
        calculate_intra_plot_difference(matched_data, str(baseline_path), str(monitoring_path))
