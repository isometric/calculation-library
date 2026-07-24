# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
import shapely
from rasterio.transform import from_origin
from sklearn.preprocessing import StandardScaler

from isometric_calculation_library.biosphere.reforestation_dynamic_baselining.spectral_matcher import (
    SpectralMatcher,
)

_PROJECTED_CRS = "EPSG:32633"


def _write_raster(
    path: Path,
    values: np.ndarray,
    transform: rasterio.Affine,
    crs: str = _PROJECTED_CRS,
) -> str:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=values.shape[0],
        width=values.shape[1],
        count=1,
        dtype=np.float32,
        crs=crs,
        transform=transform,
        nodata=np.nan,
    ) as destination:
        destination.write(values.astype(np.float32), 1)
    return str(path)


def _column_value_raster(tmp_path: Path, num_rows: int = 4, num_columns: int = 10) -> str:
    """A raster spanning x:[0, num_columns*10], y:[0, num_rows*10] at 10m resolution.

    Every pixel's value equals its column index, repeated identically across all rows.
    """
    values = np.tile(np.arange(num_columns, dtype=np.float32), (num_rows, 1))
    transform = from_origin(0, num_rows * 10, 10, 10)
    return _write_raster(tmp_path / "column_value.tif", values, transform)


def _box(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    crs: str = _PROJECTED_CRS,
) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(geometry=[shapely.box(xmin, ymin, xmax, ymax)], crs=crs)


# --- SpectralMatcher: CRS determination ---


def test_spectral_matcher_keeps_already_projected_crs_as_working_crs(tmp_path: Path) -> None:
    raster_path = _column_value_raster(tmp_path)
    donor_zone = _box(0, 0, 100, 40)
    project_boundary = _box(20, 0, 30, 10)

    matcher = SpectralMatcher(
        project_boundary=project_boundary,
        donor_zone=donor_zone,
        raster_paths=[raster_path],
        target_resolution=10.0,
    )

    assert matcher.working_crs == str(donor_zone.crs)
    assert matcher.target_crs == "EPSG:4326"


def test_spectral_matcher_derives_utm_working_crs_from_geographic_donor_zone(
    tmp_path: Path,
) -> None:
    """A geographic donor zone near 10°E, 50°N should resolve to UTM zone 32N (EPSG:32632)."""
    raster_path = _write_raster(
        tmp_path / "geographic.tif",
        np.ones((2, 2), dtype=np.float32),
        from_origin(9.999, 50.001, 0.001, 0.001),
        crs="EPSG:4326",
    )
    donor_zone = _box(9.999, 49.999, 10.001, 50.001, crs="EPSG:4326")
    project_boundary = _box(9.9995, 49.9995, 10.0, 50.0, crs="EPSG:4326")

    # target_resolution is applied in the working CRS's units (metres, once UTM is derived),
    # not the donor zone's original geographic units.
    matcher = SpectralMatcher(
        project_boundary=project_boundary,
        donor_zone=donor_zone,
        raster_paths=[raster_path],
        target_resolution=50.0,
    )

    assert matcher.working_crs == "EPSG:32632"


def test_spectral_matcher_clamps_utm_zone_at_antimeridian(tmp_path: Path) -> None:
    """A donor zone centered exactly on longitude 180 would derive zone 61 unclamped."""
    raster_path = _write_raster(
        tmp_path / "geographic.tif",
        np.ones((2, 2), dtype=np.float32),
        from_origin(179.999, 50.001, 0.001, 0.001),
        crs="EPSG:4326",
    )
    donor_zone = _box(179.999, 49.999, 180.001, 50.001, crs="EPSG:4326")
    project_boundary = _box(179.9995, 49.9995, 180.0, 50.0, crs="EPSG:4326")

    matcher = SpectralMatcher(
        project_boundary=project_boundary,
        donor_zone=donor_zone,
        raster_paths=[raster_path],
        target_resolution=50.0,
    )

    assert matcher.working_crs == "EPSG:32660"


def test_spectral_matcher_raises_when_donor_zone_has_no_crs(tmp_path: Path) -> None:
    raster_path = _column_value_raster(tmp_path)
    donor_zone = gpd.GeoDataFrame(geometry=[shapely.box(0, 0, 100, 40)])
    project_boundary = _box(20, 0, 30, 10)

    with pytest.raises(ValueError, match="CRS"):
        SpectralMatcher(
            project_boundary=project_boundary,
            donor_zone=donor_zone,
            raster_paths=[raster_path],
            target_resolution=10.0,
        )


# --- SpectralMatcher: grid and mask setup ---


def test_setup_master_grid_matches_donor_zone_bounds_at_target_resolution(tmp_path: Path) -> None:
    raster_path = _column_value_raster(tmp_path)
    donor_zone = _box(0, 0, 100, 40)
    project_boundary = _box(20, 0, 30, 10)

    matcher = SpectralMatcher(
        project_boundary=project_boundary,
        donor_zone=donor_zone,
        raster_paths=[raster_path],
        target_resolution=10.0,
    )

    assert matcher.master_width == 10
    assert matcher.master_height == 4


def test_setup_master_grid_rounds_bounds_outward_to_resolution_multiples(tmp_path: Path) -> None:
    raster_path = _column_value_raster(tmp_path)
    # Bounds span 95m (not a multiple of the 10m resolution) - should round up to 100m -> 10 cells.
    donor_zone = _box(0, 0, 95, 40)
    project_boundary = _box(20, 0, 30, 10)

    matcher = SpectralMatcher(
        project_boundary=project_boundary,
        donor_zone=donor_zone,
        raster_paths=[raster_path],
        target_resolution=10.0,
    )

    assert matcher.master_width == 10


def test_create_donor_zone_mask_excludes_project_boundary(tmp_path: Path) -> None:
    raster_path = _column_value_raster(tmp_path)
    donor_zone = _box(0, 0, 100, 40)
    # Bottom row (y:[0,10]), third column (x:[20,30]) - row index 3, column index 2 on the master grid.
    project_boundary = _box(20, 0, 30, 10)

    matcher = SpectralMatcher(
        project_boundary=project_boundary,
        donor_zone=donor_zone,
        raster_paths=[raster_path],
        target_resolution=10.0,
    )

    assert matcher.donor_zone_mask.shape == (4, 10)
    assert matcher.donor_zone_mask.sum() == 4 * 10 - 1
    assert not matcher.donor_zone_mask[3, 2]


# --- SpectralMatcher: prepare_for_sklearn scaler ordering contract ---


def test_prepare_for_sklearn_rejects_project_zone_before_donor_zone(tmp_path: Path) -> None:
    raster_path = _column_value_raster(tmp_path)
    donor_zone = _box(0, 0, 100, 40)
    project_boundary = _box(20, 0, 30, 10)

    matcher = SpectralMatcher(
        project_boundary=project_boundary,
        donor_zone=donor_zone,
        raster_paths=[raster_path],
        target_resolution=10.0,
    )
    raster_stack = np.ones((1, matcher.master_height, matcher.master_width), dtype=np.float32)

    with pytest.raises(ValueError, match="is_donor_zone=True"):
        matcher.prepare_for_sklearn(raster_stack, is_donor_zone=False)


def test_prepare_for_sklearn_allows_project_zone_first_with_prefit_scaler(tmp_path: Path) -> None:
    raster_path = _column_value_raster(tmp_path)
    donor_zone = _box(0, 0, 100, 40)
    project_boundary = _box(20, 0, 30, 10)
    prefit_scaler = StandardScaler().fit(np.array([[0.0], [1.0]]))

    matcher = SpectralMatcher(
        project_boundary=project_boundary,
        donor_zone=donor_zone,
        raster_paths=[raster_path],
        target_resolution=10.0,
        scaler=prefit_scaler,
    )
    raster_stack = np.ones((1, matcher.master_height, matcher.master_width), dtype=np.float32)

    # Should not raise: a pre-fitted scaler removes the ordering requirement.
    matcher.prepare_for_sklearn(raster_stack, is_donor_zone=False)


# --- SpectralMatcher: end-to-end dynamic matching ---


def test_run_dynamic_matching_finds_exact_match_for_repeated_column_value(tmp_path: Path) -> None:
    """Column 2's value (2.0) still exists in donor zone rows 0-2, so the excluded project
    pixel at row 3 should find a zero-distance nearest neighbor there.
    """
    raster_path = _column_value_raster(tmp_path)
    donor_zone = _box(0, 0, 100, 40)
    project_boundary = _box(20, 0, 30, 10)

    matcher = SpectralMatcher(
        project_boundary=project_boundary,
        donor_zone=donor_zone,
        raster_paths=[raster_path],
        target_resolution=10.0,
        n_neighbors=1,
    )
    result = matcher.run_dynamic_matching()

    assert len(result.match_details) == 1
    project_pixel = result.match_details[0]
    assert list(project_pixel.project_pixel_values.values()) == [2.0]
    assert len(project_pixel.matches) == 1
    assert project_pixel.matches[0].distance == pytest.approx(0.0)
    assert project_pixel.matches[0].donor_pixel_values == project_pixel.project_pixel_values


def test_run_dynamic_matching_respects_num_neighbors(tmp_path: Path) -> None:
    raster_path = _column_value_raster(tmp_path)
    donor_zone = _box(0, 0, 100, 40)
    project_boundary = _box(20, 0, 30, 10)

    matcher = SpectralMatcher(
        project_boundary=project_boundary,
        donor_zone=donor_zone,
        raster_paths=[raster_path],
        target_resolution=10.0,
        n_neighbors=3,
    )
    result = matcher.run_dynamic_matching()

    assert len(result.match_details[0].matches) == 3
    # Matches should be sorted by increasing distance (nearest neighbor first).
    normalized_distances = [match.normalized_distance for match in result.match_details[0].matches]
    assert normalized_distances == sorted(normalized_distances)


def test_run_dynamic_matching_excludes_nan_pixels_from_donor_zone(tmp_path: Path) -> None:
    """A NaN donor pixel sharing the project pixel's value should not be matched to."""
    num_rows, num_columns = 4, 10
    values = np.tile(np.arange(num_columns, dtype=np.float32), (num_rows, 1))
    values[0, 2] = np.nan  # Blank out one of the other column-2 (value=2) donor pixels.
    raster_path = _write_raster(
        tmp_path / "with_nan.tif",
        values,
        from_origin(0, num_rows * 10, 10, 10),
    )
    donor_zone = _box(0, 0, 100, 40)
    project_boundary = _box(20, 0, 30, 10)

    matcher = SpectralMatcher(
        project_boundary=project_boundary,
        donor_zone=donor_zone,
        raster_paths=[raster_path],
        target_resolution=10.0,
        n_neighbors=1,
    )
    result = matcher.run_dynamic_matching()

    # Rows 1-2 of column 2 (value=2.0) are still valid donor pixels, so the match is still exact.
    assert result.match_details[0].matches[0].distance == pytest.approx(0.0)
