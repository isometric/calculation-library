# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

"""Monte Carlo perturbation of field plot geometry.

Plot georeferencing error is applied as two independent components:

* **GPS error** — offsets the plot centre rigidly, in metres.
* **Field measurement error** — deforms the plot's shape, as a fraction of the
  characteristic length measured in the field, since such errors compound with
  distance.

Deformation is applied to each vertex radius with angular autocorrelation, which
keeps polygons free of self-intersection and makes the realised distortion
insensitive to how densely the outline is vertex-sampled.
"""

from dataclasses import dataclass
from enum import StrEnum, auto

import numpy as np
import shapely

from isometric_calculation_library.utils.types import Np1DArray, Np2DArray

DEFAULT_ANGULAR_CORRELATION_SCALE = np.pi / 4
"""Angular correlation length (radians) of radial deformation.

Controls how smoothly deformation varies around the plot outline. Smaller
values approach independent per-vertex noise and give star-shaped polygons.
"""

_MIN_RADIUS_FRACTION = 0.05
"""Floor on a deformed vertex radius, as a fraction of the plot's mean radius.

Prevents a large negative draw from collapsing a vertex through the centre.
"""

_CHOLESKY_RIDGE = 1e-10
"""Relative ridge added to the correlation matrix diagonal before factorisation.

The squared-exponential kernel is numerically singular beyond ~32 vertices, where
plain Cholesky fails.
"""


class PlotShape(StrEnum):
    """Layout of a field plot, determining how its area relates to its measurement length."""

    SQUARE = auto()
    """Square plot; the measurement length is the side length."""

    CIRCLE = auto()
    """Circular plot; the measurement length is the radius."""


def plot_area_m2(shape: PlotShape, measurement_length_m: float) -> float:
    """Nominal plot area (m²) implied by a plot's shape and characteristic measurement length."""
    match shape:
        case PlotShape.SQUARE:
            return measurement_length_m**2
        case PlotShape.CIRCLE:
            return float(np.pi * measurement_length_m**2)


@dataclass(frozen=True)
class PlotDeformer:
    """Precomputed geometry for drawing deformed variants of one plot polygon.

    The polar decomposition and correlation factorisation are computed once and
    reused across simulations, rather than rebuilt per draw.

    Construct via :func:`build_plot_deformer`.
    """

    theta: Np1DArray[np.float64]
    """Angle of each vertex about the plot centre (radians)."""

    radius: Np1DArray[np.float64]
    """Distance of each vertex from the plot centre (m)."""

    centre: Np1DArray[np.float64]
    """Plot centre as ``(x, y)`` in the polygon's projected CRS."""

    deformation_factor: Np2DArray[np.float64] | None
    """Lower-triangular Cholesky factor of the radial deformation covariance.

    ``None`` when the deformation standard error is zero, i.e. GPS offset only.
    """

    gps_standard_error_m: float
    """Standard error of the plot centre position (m), applied per axis."""

    min_radius_m: float
    """Floor applied to deformed vertex radii."""

    def deform(self, rng: np.random.Generator) -> shapely.Polygon:
        """Draw one deformed, GPS-offset variant of the plot polygon."""
        radius = self.radius
        if self.deformation_factor is not None:
            deviation = self.deformation_factor @ rng.standard_normal(len(self.theta))
            radius = np.clip(radius + deviation, self.min_radius_m, None)

        # Offset a copy; `+=` would mutate the shared `centre` field.
        centre = (
            self.centre + rng.normal(0, self.gps_standard_error_m, size=2)
            if self.gps_standard_error_m > 0
            else self.centre
        )

        return shapely.Polygon(
            np.column_stack(
                [
                    centre[0] + radius * np.cos(self.theta),
                    centre[1] + radius * np.sin(self.theta),
                ],
            ),
        )


def build_plot_deformer(
    polygon: shapely.Polygon,
    *,
    measurement_length_m: float,
    deformation_fraction: float,
    gps_standard_error_m: float,
    angular_correlation_scale: float = DEFAULT_ANGULAR_CORRELATION_SCALE,
) -> PlotDeformer:
    """Prepare a reusable deformer for *polygon*.

    The polygon must be in a projected (metre) CRS. Only the exterior ring is used.

    Args:
        polygon: Plot outline in a projected CRS.
        measurement_length_m: Characteristic field measurement used to lay the
            plot out — side length for square plots, radius for circular ones.
        deformation_fraction: Field measurement error as a fraction of
            *measurement_length_m*.
        gps_standard_error_m: Per-axis standard error of the recorded plot centre.
        angular_correlation_scale: Angular correlation length (radians) of the
            radial deformation.
    """
    if polygon.area <= 0:
        raise ValueError(
            "Plot polygon must enclose a positive area to be deformed"
            f" (got area {polygon.area}); check the polygon is in a projected CRS",
        )

    coords = np.asarray(polygon.exterior.coords[:-1], dtype=np.float64)

    # The area centroid stands in for the surveyed centre monument the GPS reading
    # is taken at; the two coincide for regular plot layouts.
    centre = np.asarray(polygon.centroid.coords[0], dtype=np.float64)

    offsets = coords - centre
    theta = np.arctan2(offsets[:, 1], offsets[:, 0])
    radius = np.hypot(offsets[:, 0], offsets[:, 1])

    deformation_standard_error_m = measurement_length_m * deformation_fraction
    deformation_factor = (
        _radial_deformation_cholesky(
            theta,
            standard_error_m=deformation_standard_error_m,
            angular_correlation_scale=angular_correlation_scale,
        )
        if deformation_standard_error_m > 0
        else None
    )

    return PlotDeformer(
        theta=theta,
        radius=radius,
        centre=centre,
        deformation_factor=deformation_factor,
        gps_standard_error_m=gps_standard_error_m,
        min_radius_m=float(radius.mean()) * _MIN_RADIUS_FRACTION,
    )


def _radial_deformation_cholesky(
    theta: Np1DArray[np.float64],
    *,
    standard_error_m: float,
    angular_correlation_scale: float,
) -> Np2DArray[np.float64]:
    """Cholesky factor of the angularly-autocorrelated radial deformation covariance.

    Uses a squared-exponential kernel in chordal angular distance,
    ``2 sin²(Δθ/2)``, which wraps around the outline.
    """
    delta = theta[:, np.newaxis] - theta[np.newaxis, :]
    covariance = standard_error_m**2 * np.exp(
        -2 * np.sin(delta / 2) ** 2 / angular_correlation_scale**2,
    )
    ridge = np.eye(len(theta)) * standard_error_m**2 * _CHOLESKY_RIDGE
    return np.linalg.cholesky(covariance + ridge)
