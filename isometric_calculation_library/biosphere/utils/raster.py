# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

"""Raster sampling utilities."""

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import rowcol

MAX_IN_MEMORY_PIXELS = 10_000_000
"""Rasters at or above this pixel count are sampled point-by-point instead of read into memory."""


def sample_raster_vectorized(
    raster_path: str,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
) -> np.ndarray:
    """Sample a raster at WGS84 (latitude, longitude) coordinates.

    Coordinates outside the raster bounds, or landing on a nodata pixel, sample as NaN.
    """
    latitudes = np.asarray(latitudes)
    longitudes = np.asarray(longitudes)
    original_shape = latitudes.shape
    latitudes_flat = latitudes.flatten()
    longitudes_flat = longitudes.flatten()

    with rasterio.open(raster_path) as source_dataset:
        if source_dataset.crs is None or source_dataset.crs.to_epsg() != 4326:
            transformer = Transformer.from_crs("EPSG:4326", source_dataset.crs, always_xy=True)
            x_coordinates, y_coordinates = transformer.transform(longitudes_flat, latitudes_flat)
        else:
            x_coordinates, y_coordinates = longitudes_flat, latitudes_flat

        rows, columns = rowcol(source_dataset.transform, x_coordinates, y_coordinates)

        if source_dataset.width * source_dataset.height < MAX_IN_MEMORY_PIXELS:
            data = source_dataset.read(1)

            valid_mask = (
                (rows >= 0)
                & (rows < source_dataset.height)
                & (columns >= 0)
                & (columns < source_dataset.width)
            )

            values = np.full(len(rows), np.nan, dtype=np.float64)
            valid_indices = np.where(valid_mask)[0]

            if len(valid_indices) > 0:
                valid_rows = rows[valid_indices]
                valid_columns = columns[valid_indices]
                sampled = data[valid_rows, valid_columns]

                if source_dataset.nodata is not None:
                    sampled = np.where(sampled == source_dataset.nodata, np.nan, sampled)

                values[valid_indices] = sampled
        else:
            # For large rasters, sample individually instead of reading the whole raster into memory.
            valid_mask = (
                (rows >= 0)
                & (rows < source_dataset.height)
                & (columns >= 0)
                & (columns < source_dataset.width)
            )

            coordinates = list(zip(x_coordinates, y_coordinates, strict=True))
            sampled_iter = source_dataset.sample(coordinates)
            # Bounds-check explicitly rather than relying on rasterio's out-of-bounds fill value
            # matching source_dataset.nodata, since that fill value is returned regardless of
            # whether the raster even defines a nodata value.
            raw_values = np.array([value[0] for value in sampled_iter])
            values = np.where(valid_mask, raw_values, np.nan)

            if source_dataset.nodata is not None:
                values = np.where(values == source_dataset.nodata, np.nan, values)

    return values.reshape(original_shape)
