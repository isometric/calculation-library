# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import geopandas as gpd
import numpy as np
import pyproj
import rasterio
import rasterio.warp
from rasterio import mask, transform
from rasterio.enums import Resampling
from rasterio.features import rasterize
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from isometric_calculation_library.utils.types import Np1DArray, Np2DArray

if TYPE_CHECKING:
    from rasterio.transform import Affine

logger = logging.getLogger(__name__)


class DonorMatch(NamedTuple):
    """A single donor pixel match result."""

    distance: float
    normalized_distance: float
    donor_pixel_coordinates: tuple[float, float]
    donor_pixel_values: dict[str, float]


class PixelMatchResult(NamedTuple):
    """Match result for a single project pixel."""

    project_pixel_coordinates: tuple[float, float]
    project_pixel_values: dict[str, float]
    matches: Sequence[DonorMatch]


@dataclass(frozen=True)
class CrsDetermination:
    """Working and target coordinate reference systems for grid calculations."""

    working_crs: str
    """Projected CRS used internally so grid distances are measured in metres."""
    target_crs: str
    """CRS in which match coordinates are ultimately reported."""


@dataclass(frozen=True)
class MasterGrid:
    """Master grid parameters derived from the donor zone bounds."""

    transform: "Affine"
    width: int
    height: int


@dataclass(frozen=True)
class ProcessedRasterStacks:
    """Raster band stacks sampled onto the master grid for both zones."""

    large_grid_stack: np.ndarray
    """Shape (num_rasters, master_height, master_width), covering the donor zone."""
    smaller_zone_stack: np.ndarray
    """Shape (num_rasters, master_height, master_width), covering the project boundary."""
    raster_names: tuple[str, ...]


@dataclass(frozen=True)
class PreparedZoneData:
    """Flattened, normalized, valid-pixel data for one zone, ready for k-NN matching."""

    normalized_data: Np2DArray[np.floating]
    valid_indices: Np1DArray[np.integer]
    scaler: StandardScaler


@dataclass(frozen=True)
class NearestNeighborsMatch:
    """Nearest-neighbor distances and indices from a k-NN query."""

    distances: Np2DArray[np.floating]
    indices: Np2DArray[np.integer]


@dataclass(frozen=True)
class DynamicMatchingResult:
    """Full result of the dynamic matching workflow."""

    match_details: Sequence[PixelMatchResult]
    scaler: StandardScaler


class SpectralMatcher:
    """Matches project-boundary pixels to spectrally similar donor pixels in a surrounding zone.

    Rasters are sampled onto a common master grid covering the donor zone, then k-NN
    matching (on standardized band values) finds, for each project pixel, the nearest
    donor pixels outside the project boundary. Used to build a counterfactual baseline
    for reforestation dynamic baselining.
    """

    def __init__(
        self,
        project_boundary: gpd.GeoDataFrame,
        donor_zone: gpd.GeoDataFrame,
        raster_paths: Sequence[str],
        target_resolution: float = 30.0,
        scaler: StandardScaler | None = None,
        n_neighbors: int = 10,
    ) -> None:
        super().__init__()
        self.project_boundary = project_boundary
        self.donor_zone = donor_zone
        self.original_crs = self.donor_zone.crs
        if self.original_crs is None:
            raise ValueError("Donor zone GeoDataFrame must have a valid CRS.")
        project_boundary_reprojected = self.project_boundary.to_crs(self.original_crs)
        logger.info("Excluding project area from donor zone...")
        self.donor_zone = gpd.overlay(
            self.donor_zone,
            project_boundary_reprojected,
            how="difference",
        )

        self.raster_paths = raster_paths
        self.target_resolution = target_resolution
        self.scaler = scaler
        self.n_neighbors = n_neighbors
        # True once the scaler is fit and safe to `transform()` with: immediately if a
        # pre-fitted scaler was provided, otherwise only after prepare_for_sklearn is
        # called with is_donor_zone=True.
        self._scaler_ready = scaler is not None

        crs_determination = self._determine_working_crs()
        self.working_crs = crs_determination.working_crs
        self.target_crs = crs_determination.target_crs

        master_grid = self.setup_master_grid()
        self.master_transform = master_grid.transform
        self.master_width = master_grid.width
        self.master_height = master_grid.height

        self.donor_zone_mask = self.create_donor_zone_mask()

    def _determine_working_crs(self) -> CrsDetermination:
        """Determine appropriate working CRS for grid calculations."""
        target_crs = "EPSG:4326"
        # original_crs is guaranteed non-None by __init__ validation so we're ignoring pyright here
        if self.original_crs.is_geographic:  # pyright: ignore[reportOptionalMemberAccess]
            logger.info(f"Geographic CRS detected: {self.original_crs}")
            # Get center point of donor zone to determine UTM zone
            center_point = self.donor_zone.geometry.union_all().centroid
            center_longitude = center_point.x
            center_latitude = center_point.y

            # Calculate UTM zone using standard formula: zone = floor((longitude + 180) / 6) + 1
            # Clamped to 60 since longitude=180 (the antimeridian) otherwise yields an
            # out-of-range zone 61.
            utm_zone = min(int((center_longitude + 180) / 6) + 1, 60)
            hemisphere = "north" if center_latitude >= 0 else "south"

            # EPSG codes: 326xx for northern hemisphere, 327xx for southern
            working_crs = f"EPSG:{32600 + utm_zone if hemisphere == 'north' else 32700 + utm_zone}"
            logger.info(
                f"Using UTM zone {utm_zone}{hemisphere[0].upper()} ({working_crs}) for grid calculations",
            )

            return CrsDetermination(working_crs=working_crs, target_crs=target_crs)
        logger.info(f"Projected CRS detected: {self.original_crs}")
        return CrsDetermination(working_crs=str(self.original_crs), target_crs=target_crs)

    def setup_master_grid(self) -> MasterGrid:
        """Set up the master grid parameters based on donor zone bounds."""
        # Project to working CRS for grid calculations
        donor_zone_projected = self.donor_zone.to_crs(self.working_crs)

        # Unify all geometries to get the total bounds, robust for MultiPolygons
        unified_donor_zone = donor_zone_projected.union_all()
        min_x, min_y, max_x, max_y = unified_donor_zone.bounds

        # Align bounds to the grid
        min_x = np.floor(min_x / self.target_resolution) * self.target_resolution
        max_x = np.ceil(max_x / self.target_resolution) * self.target_resolution
        min_y = np.floor(min_y / self.target_resolution) * self.target_resolution
        max_y = np.ceil(max_y / self.target_resolution) * self.target_resolution

        width = int((max_x - min_x) / self.target_resolution)
        height = int((max_y - min_y) / self.target_resolution)
        master_transform = transform.from_origin(
            min_x,
            max_y,
            self.target_resolution,
            self.target_resolution,
        )

        logger.info(f"Master grid: {width}W x {height}H @ {self.target_resolution}m")
        return MasterGrid(transform=master_transform, width=width, height=height)

    def create_donor_zone_mask(self) -> np.ndarray:
        """Create a boolean mask for the donor zone on the master grid."""
        logger.info("Creating donor zone mask...")
        # Project to working CRS for rasterization
        donor_zone_projected = self.donor_zone.to_crs(self.working_crs)

        rasterized = rasterize(
            shapes=donor_zone_projected.geometry,
            out_shape=(self.master_height, self.master_width),
            transform=self.master_transform,
            fill=0,
            all_touched=True,
            dtype=np.uint8,
        )
        if rasterized is None:
            raise ValueError("Failed to rasterize donor zone.")
        return rasterized.astype(bool)

    def process_raster_to_grid(
        self,
        source_raster_path: str,
        clip_boundary: gpd.GeoDataFrame,
        resampling_method: Resampling = Resampling.bilinear,
    ) -> Np2DArray[np.floating]:
        """Clips and reprojects a source raster to match the master grid."""
        with rasterio.open(source_raster_path) as source_dataset:
            clip_geometries = clip_boundary.to_crs(source_dataset.crs).geometry.values

            clipped_array, clipped_transform = mask.mask(
                dataset=source_dataset,
                shapes=clip_geometries,
                crop=True,
                nodata=source_dataset.nodata,
                all_touched=True,
            )

            destination_array = np.full(
                (self.master_height, self.master_width),
                fill_value=np.nan,
                dtype=np.float32,
            )

            rasterio.warp.reproject(
                source=clipped_array,
                destination=destination_array,
                src_transform=clipped_transform,
                src_crs=source_dataset.crs,
                dst_transform=self.master_transform,
                dst_crs=self.working_crs,
                resampling=resampling_method,
                src_nodata=source_dataset.nodata,
                dst_nodata=np.nan,
            )

        return destination_array

    def process_all_rasters(self) -> ProcessedRasterStacks:
        """Process all rasters for both donor zone and project boundary."""
        logger.info("--- Processing rasters ---")

        large_grid_bands = list[Np2DArray[np.floating]]()
        smaller_zone_bands = list[Np2DArray[np.floating]]()
        raster_names = list[str]()

        for raster_path in self.raster_paths:
            logger.info(f"Processing {raster_path}...")
            raster_names.append(Path(raster_path).name)

            large_grid_bands.append(self.process_raster_to_grid(raster_path, self.donor_zone))
            smaller_zone_bands.append(
                self.process_raster_to_grid(raster_path, self.project_boundary),
            )

        large_grid_stack = np.stack(large_grid_bands)
        smaller_zone_stack = np.stack(smaller_zone_bands)

        logger.info(f"Large grid stack shape: {large_grid_stack.shape}")
        logger.info(f"Smaller zone stack shape: {smaller_zone_stack.shape}")

        return ProcessedRasterStacks(
            large_grid_stack=large_grid_stack,
            smaller_zone_stack=smaller_zone_stack,
            raster_names=tuple(raster_names),
        )

    def _analyze_nan_contributions(
        self,
        data_flat: np.ndarray,
        invalid_indices: np.ndarray,
        total_zone_pixels: int,
        zone_name: str,
    ) -> None:
        """Logs statistics about NaN values in invalid pixels."""
        invalid_data = data_flat[invalid_indices]
        is_nan_mask = np.isnan(invalid_data)
        num_bands = is_nan_mask.shape[1]

        logger.info(f"    Contribution to invalid pixels by band (as % of {zone_name} pixels):")
        nan_counts_per_band = is_nan_mask.sum(axis=0)
        for band_index, count in enumerate(nan_counts_per_band):
            if count > 0:
                percentage = (count / total_zone_pixels) * 100
                logger.info(
                    f"      - Band {band_index} ({Path(self.raster_paths[band_index]).name}): "
                    f"{percentage:.2f}%",
                )

        logger.info(f"    Exclusive NaN contributions by band (as % of {zone_name} pixels):")
        nans_per_pixel = is_nan_mask.sum(axis=1)
        single_nan_pixels = is_nan_mask[nans_per_pixel == 1]
        if single_nan_pixels.shape[0] > 0:
            exclusive_counts = single_nan_pixels.sum(axis=0)
            for band_index, count in enumerate(exclusive_counts):
                if count > 0:
                    percentage = (count / total_zone_pixels) * 100
                    logger.info(
                        f"      - Band {band_index} ({Path(self.raster_paths[band_index]).name}) "
                        f"is the sole NaN in {percentage:.2f}% of pixels.",
                    )
        else:
            logger.info("      - No pixels with a single NaN band found.")

        all_nan_pixels_count = np.sum(nans_per_pixel == num_bands)
        if all_nan_pixels_count > 0:
            percentage = (all_nan_pixels_count / total_zone_pixels) * 100
            logger.info(
                f"    Found {percentage:.2f}% of pixels are NaN across all bands.",
            )

    def prepare_for_sklearn(
        self,
        raster_stack: np.ndarray,
        is_donor_zone: bool = False,
    ) -> PreparedZoneData:
        """Flatten raster stack, normalize, and return valid indices and scaler.

        Must be called with is_donor_zone=True before any call with is_donor_zone=False
        (unless a pre-fitted scaler was passed to SpectralMatcher), since the first call
        fits self.scaler on donor zone data and the project zone is then normalized
        against that same fit.
        """
        data = np.moveaxis(raster_stack, 0, -1)
        data_flat = data.reshape(-1, data.shape[-1])
        valid_mask = ~np.isnan(data_flat).any(axis=1)

        if is_donor_zone:
            zone_name = "donor zone"
            donor_mask_flat = self.donor_zone_mask.flatten()
            zone_mask = donor_mask_flat
            total_zone_pixels = np.sum(donor_mask_flat)
            original_valid_pixel_count = np.sum(valid_mask)
            valid_mask &= donor_mask_flat
            new_valid_pixel_count = np.sum(valid_mask)
            logger.info(
                f"  Filtered large grid from {original_valid_pixel_count} to "
                f"{new_valid_pixel_count} pixels using donor zone mask.",
            )
        else:
            zone_name = "project zone"
            project_rasterized = rasterize(
                shapes=self.project_boundary.to_crs(self.working_crs).geometry,
                out_shape=(self.master_height, self.master_width),
                transform=self.master_transform,
                fill=0,
                all_touched=True,
                dtype=np.uint8,
            )
            if project_rasterized is None:
                raise ValueError("Failed to rasterize project boundary.")
            zone_mask = project_rasterized.astype(bool).flatten()
            total_zone_pixels = np.sum(zone_mask)
            valid_mask &= zone_mask

        valid_indices = np.where(valid_mask)[0]
        # Consider only invalid pixels that are within the zone of interest
        invalid_indices_in_zone = np.where(~valid_mask & zone_mask)[0]
        logger.info(
            f"  Found {len(invalid_indices_in_zone)} invalid pixels within the "
            f"{zone_name} ({total_zone_pixels} total pixels).",
        )

        if len(invalid_indices_in_zone) > 0 and total_zone_pixels > 0:
            self._analyze_nan_contributions(
                data_flat,
                invalid_indices_in_zone,
                total_zone_pixels,
                zone_name,
            )

        valid_data = data_flat[valid_indices]

        if is_donor_zone:
            if self.scaler is None:
                self.scaler = StandardScaler()
                normalized_data = self.scaler.fit_transform(valid_data)
            else:
                normalized_data = self.scaler.transform(valid_data)
            self._scaler_ready = True
        else:
            if not self._scaler_ready or self.scaler is None:
                raise ValueError(
                    "prepare_for_sklearn must be called with is_donor_zone=True before "
                    "is_donor_zone=False, so the scaler is fit on donor zone data "
                    "(unless a pre-fitted scaler was passed to SpectralMatcher).",
                )
            normalized_data = self.scaler.transform(valid_data)

        if not isinstance(normalized_data, np.ndarray):
            raise TypeError(f"Expected ndarray, got {type(normalized_data)}")

        return PreparedZoneData(
            normalized_data=normalized_data,
            valid_indices=valid_indices,
            scaler=self.scaler,
        )

    def find_nearest_neighbors(
        self,
        smaller_zone_data: Np2DArray[np.floating],
        large_grid_data: Np2DArray[np.floating],
    ) -> NearestNeighborsMatch:
        """Find nearest neighbors for smaller zone pixels in the larger zone."""
        nearest_neighbors_model = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm="auto")
        nearest_neighbors_model.fit(large_grid_data)
        logger.info("Model fitting complete")

        distances, indices = nearest_neighbors_model.kneighbors(smaller_zone_data)

        return NearestNeighborsMatch(distances=distances, indices=indices)

    def get_match_details(
        self,
        distances: Np2DArray[np.floating],
        indices: Np2DArray[np.integer],
        smaller_zone_valid_indices: Np1DArray[np.integer],
        large_grid_valid_indices: Np1DArray[np.integer],
        smaller_zone_stack: np.ndarray,
        large_grid_stack: np.ndarray,
        raster_names: Sequence[str],
    ) -> list[PixelMatchResult]:
        """Construct detailed information about each match."""
        logger.info("--- Constructing match details ---")
        matches = list[PixelMatchResult]()

        smaller_zone_flat = np.moveaxis(smaller_zone_stack, 0, -1).reshape(
            -1,
            smaller_zone_stack.shape[0],
        )
        large_grid_flat = np.moveaxis(large_grid_stack, 0, -1).reshape(
            -1,
            large_grid_stack.shape[0],
        )

        # Create transformer once if CRS conversion is needed
        transformer: pyproj.Transformer | None = None
        if self.working_crs != self.target_crs:
            transformer = pyproj.Transformer.from_crs(
                self.working_crs,
                self.target_crs,
                always_xy=True,
            )

        for i, smaller_zone_pixel_index in enumerate(smaller_zone_valid_indices):
            smaller_zone_row, smaller_zone_col = np.unravel_index(
                smaller_zone_pixel_index,
                (self.master_height, self.master_width),
            )
            smaller_zone_coordinates = self.master_transform * (
                smaller_zone_col + 0.5,
                smaller_zone_row + 0.5,
            )

            # Convert coordinates to target CRS if different from working CRS
            if transformer is not None:
                smaller_zone_coordinates = transformer.transform(*smaller_zone_coordinates)

            donor_matches_for_pixel = list[DonorMatch]()

            for neighbor_rank in range(distances.shape[1]):
                large_grid_pixel_flat_index = large_grid_valid_indices[indices[i, neighbor_rank]]
                large_grid_row, large_grid_col = np.unravel_index(
                    large_grid_pixel_flat_index,
                    (self.master_height, self.master_width),
                )
                donor_pixel_coordinates = self.master_transform * (
                    large_grid_col + 0.5,
                    large_grid_row + 0.5,
                )

                # Convert coordinates to target CRS if different from working CRS
                if transformer is not None:
                    donor_pixel_coordinates = transformer.transform(*donor_pixel_coordinates)

                # Calculate raw Euclidean distance
                raw_distance = np.linalg.norm(
                    smaller_zone_flat[smaller_zone_pixel_index]
                    - large_grid_flat[large_grid_pixel_flat_index],
                )

                donor_matches_for_pixel.append(
                    DonorMatch(
                        distance=float(raw_distance),
                        normalized_distance=float(distances[i, neighbor_rank]),
                        donor_pixel_coordinates=donor_pixel_coordinates,
                        donor_pixel_values=dict(
                            zip(
                                raster_names,
                                large_grid_flat[large_grid_pixel_flat_index],
                                strict=True,
                            ),
                        ),
                    ),
                )

            matches.append(
                PixelMatchResult(
                    project_pixel_coordinates=smaller_zone_coordinates,
                    project_pixel_values=dict(
                        zip(
                            raster_names,
                            smaller_zone_flat[smaller_zone_pixel_index],
                            strict=True,
                        ),
                    ),
                    matches=tuple(donor_matches_for_pixel),
                ),
            )
        return matches

    def run_dynamic_matching(self) -> DynamicMatchingResult:
        """Run the complete dynamic matching workflow."""
        logger.info("Starting dynamic matching workflow...")

        raster_stacks = self.process_all_rasters()

        # Prepare and normalize data for k-NN
        logger.info("--- Preparing data for ML ---")
        large_grid_prepared = self.prepare_for_sklearn(
            raster_stacks.large_grid_stack,
            is_donor_zone=True,
        )
        logger.info("--- Scaler statistics ---")
        logger.info(f"Means per band: {large_grid_prepared.scaler.mean_}")
        logger.info(f"Variances per band: {large_grid_prepared.scaler.var_}")

        smaller_zone_prepared = self.prepare_for_sklearn(raster_stacks.smaller_zone_stack)

        logger.info(
            f"Found {smaller_zone_prepared.normalized_data.shape[0]} valid pixels in smaller zone",
        )
        logger.info(
            f"Found {large_grid_prepared.normalized_data.shape[0]} valid pixels in large grid",
        )

        nearest_neighbors_match = self.find_nearest_neighbors(
            smaller_zone_prepared.normalized_data,
            large_grid_prepared.normalized_data,
        )

        match_details = self.get_match_details(
            nearest_neighbors_match.distances,
            nearest_neighbors_match.indices,
            smaller_zone_prepared.valid_indices,
            large_grid_prepared.valid_indices,
            raster_stacks.smaller_zone_stack,
            raster_stacks.large_grid_stack,
            raster_stacks.raster_names,
        )

        logger.info("--- Workflow complete ---")
        return DynamicMatchingResult(
            match_details=tuple(match_details),
            scaler=large_grid_prepared.scaler,
        )
