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

from isometric_calculation_library.biosphere.utils import raster as raster_module
from isometric_calculation_library.biosphere.utils.raster import sample_raster_vectorized


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


def _grid_value_raster(tmp_path: Path) -> str:
    """A 4x4, 1-degree-per-pixel raster spanning lon:[0,4], lat:[0,4]. value[row, col] = row*4+col."""
    values = np.arange(16, dtype=np.float32).reshape(4, 4)
    return _write_raster(tmp_path / "grid_value.tif", values)


def test_sample_raster_vectorized_returns_values_at_pixel_centers(tmp_path: Path) -> None:
    raster_path = _grid_value_raster(tmp_path)

    # Pixel centers: row 0 -> latitude 3.5; row 3 -> latitude 0.5. col 0 -> longitude 0.5; col 3 -> longitude 3.5.
    values = sample_raster_vectorized(
        raster_path,
        latitudes=np.array([3.5, 0.5]),
        longitudes=np.array([0.5, 3.5]),
    )

    assert_allclose(values, [0.0, 15.0])


def test_sample_raster_vectorized_returns_nan_outside_raster_bounds(tmp_path: Path) -> None:
    raster_path = _grid_value_raster(tmp_path)

    values = sample_raster_vectorized(
        raster_path,
        latitudes=np.array([50.0]),
        longitudes=np.array([50.0]),
    )

    assert math.isnan(values[0])


def test_sample_raster_vectorized_returns_nan_at_nodata_pixel(tmp_path: Path) -> None:
    values = np.arange(16, dtype=np.float32).reshape(4, 4)
    values[0, 0] = -9999.0
    raster_path = _write_raster(tmp_path / "with_nodata.tif", values, nodata=-9999.0)

    sampled = sample_raster_vectorized(
        raster_path,
        latitudes=np.array([3.5]),
        longitudes=np.array([0.5]),
    )

    assert math.isnan(sampled[0])


# --- Large-raster (individual sampling) path, forced via a monkeypatched MAX_IN_MEMORY_PIXELS ---


def test_sample_raster_vectorized_large_raster_path_returns_nan_at_nodata_pixel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(raster_module, "MAX_IN_MEMORY_PIXELS", 0)
    values = np.arange(16, dtype=np.float32).reshape(4, 4)
    values[0, 0] = -9999.0
    raster_path = _write_raster(tmp_path / "with_nodata.tif", values, nodata=-9999.0)

    sampled = sample_raster_vectorized(
        raster_path,
        latitudes=np.array([3.5]),
        longitudes=np.array([0.5]),
    )

    assert math.isnan(sampled[0])


def test_sample_raster_vectorized_large_raster_path_returns_nan_outside_raster_bounds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test: rasterio.sample() returns a raw fill value (e.g. 0.0) for an
    out-of-bounds point when the raster has no nodata value, so bounds must be checked
    explicitly rather than inferred from a nodata match.
    """
    monkeypatch.setattr(raster_module, "MAX_IN_MEMORY_PIXELS", 0)
    raster_path = _grid_value_raster(tmp_path)

    sampled = sample_raster_vectorized(
        raster_path,
        latitudes=np.array([50.0]),
        longitudes=np.array([50.0]),
    )

    assert math.isnan(sampled[0])


def test_sample_raster_vectorized_large_raster_path_returns_value_at_valid_pixel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(raster_module, "MAX_IN_MEMORY_PIXELS", 0)
    raster_path = _grid_value_raster(tmp_path)

    sampled = sample_raster_vectorized(
        raster_path,
        latitudes=np.array([3.5]),
        longitudes=np.array([0.5]),
    )

    assert_allclose(sampled, [0.0])


def test_sample_raster_vectorized_preserves_input_shape(tmp_path: Path) -> None:
    raster_path = _grid_value_raster(tmp_path)

    values = sample_raster_vectorized(
        raster_path,
        latitudes=np.array([[3.5, 3.5], [0.5, 0.5]]),
        longitudes=np.array([[0.5, 1.5], [2.5, 3.5]]),
    )

    assert values.shape == (2, 2)
