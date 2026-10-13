# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

import numpy as np
import pytest
import shapely
from numpy.testing import assert_allclose

from isometric_calculation_library.biosphere.monte_carlo.plot_geometry import (
    PlotDeformer,
    PlotShape,
    build_plot_deformer,
    plot_area_m2,
)

# Representative georeferencing errors: 2.38 m GPS offset, 2.34% shape deformation.
_GPS_SE_M = 2.38
_DEFORMATION_FRACTION = 0.0234

_CIRCLE_RADIUS_M = float(np.sqrt(100 / np.pi))
"""Radius of a nominally 100 m² circular plot."""


def _square(side_m: float = 40.0) -> shapely.Polygon:
    return shapely.Polygon(
        [(0, 0), (side_m, 0), (side_m, side_m), (0, side_m), (0, 0)],
    )


def _circle(quad_segs: int = 4) -> shapely.Polygon:
    return shapely.Point(0, 0).buffer(_CIRCLE_RADIUS_M, quad_segs=quad_segs)


def _deformer(
    polygon: shapely.Polygon,
    measurement_length_m: float,
    **kwargs: float,
) -> PlotDeformer:
    return build_plot_deformer(
        polygon,
        measurement_length_m=measurement_length_m,
        deformation_fraction=kwargs.get("deformation_fraction", _DEFORMATION_FRACTION),
        gps_standard_error_m=kwargs.get("gps_standard_error_m", _GPS_SE_M),
    )


# --- plot_area_m2 ---


def test_plot_area_m2_square() -> None:
    assert plot_area_m2(PlotShape.SQUARE, 40.0) == pytest.approx(1600.0)


def test_plot_area_m2_circle() -> None:
    """A circular plot's area comes from pi*r^2, not r^2."""
    assert plot_area_m2(PlotShape.CIRCLE, _CIRCLE_RADIUS_M) == pytest.approx(100.0)


def test_plot_area_m2_circle_differs_from_square_by_pi() -> None:
    """A radius read as a side length gives an area 3.14x too small."""
    radius = 5.0
    assert plot_area_m2(PlotShape.CIRCLE, radius) / plot_area_m2(
        PlotShape.SQUARE,
        radius,
    ) == pytest.approx(np.pi)


# --- build_plot_deformer ---


def test_build_plot_deformer_rejects_zero_area_polygon() -> None:
    collapsed = shapely.Polygon([(0, 0), (1, 1), (2, 2), (0, 0)])
    with pytest.raises(ValueError, match="positive area"):
        _deformer(collapsed, 40.0)


def test_build_plot_deformer_recovers_polar_coordinates() -> None:
    deformer = _deformer(_circle(), _CIRCLE_RADIUS_M)
    assert_allclose(deformer.radius, _CIRCLE_RADIUS_M, rtol=1e-9)
    assert_allclose(deformer.centre, [0.0, 0.0], atol=1e-9)


def test_build_plot_deformer_no_factor_when_deformation_zero() -> None:
    deformer = _deformer(_square(), 40.0, deformation_fraction=0.0)
    assert deformer.deformation_factor is None


# --- PlotDeformer.deform ---


@pytest.mark.parametrize("quad_segs", [2, 4, 8, 16, 32])
def test_deform_always_produces_valid_circular_polygons(quad_segs: int) -> None:
    """Deformation yields valid polygons at any vertex density."""
    deformer = _deformer(_circle(quad_segs), _CIRCLE_RADIUS_M)
    rng = np.random.default_rng(0)
    assert all(deformer.deform(rng).is_valid for _ in range(500))


def test_deform_produces_valid_square_polygons() -> None:
    deformer = _deformer(_square(), 40.0)
    rng = np.random.default_rng(0)
    assert all(deformer.deform(rng).is_valid for _ in range(500))


def test_deform_preserves_vertex_count() -> None:
    circle = _circle()
    deformer = _deformer(circle, _CIRCLE_RADIUS_M)
    deformed = deformer.deform(np.random.default_rng(0))
    assert len(deformed.exterior.coords) == len(circle.exterior.coords)


def test_deform_is_deterministic_for_seeded_rng() -> None:
    deformer = _deformer(_square(), 40.0)
    first = deformer.deform(np.random.default_rng(7))
    second = deformer.deform(np.random.default_rng(7))
    assert first.equals_exact(second, tolerance=0)


def test_deform_area_is_unbiased() -> None:
    """Deformation leaves mean area unchanged."""
    circle = _circle()
    deformer = _deformer(circle, _CIRCLE_RADIUS_M)
    rng = np.random.default_rng(1)
    areas = np.array([deformer.deform(rng).area for _ in range(4000)])
    assert areas.mean() == pytest.approx(circle.area, rel=0.01)


@pytest.mark.parametrize("quad_segs", [4, 8, 16, 32])
def test_deform_area_spread_insensitive_to_vertex_density(quad_segs: int) -> None:
    """Area spread depends on the plot, not on how finely its outline is sampled."""
    deformer = _deformer(_circle(quad_segs), _CIRCLE_RADIUS_M)
    rng = np.random.default_rng(2)
    areas = np.array([deformer.deform(rng).area for _ in range(3000)])
    coefficient_of_variation = areas.std() / areas.mean()
    assert coefficient_of_variation == pytest.approx(0.028, abs=0.006)


def test_deform_centre_offset_matches_gps_standard_error() -> None:
    """The plot centre is offset by Normal(0, gps_standard_error_m) per axis."""
    square = _square()
    deformer = _deformer(square, 40.0, deformation_fraction=0.0)
    rng = np.random.default_rng(3)
    original = np.asarray(square.centroid.coords[0])
    offsets = np.array(
        [np.asarray(deformer.deform(rng).centroid.coords[0]) - original for _ in range(4000)],
    )
    assert_allclose(offsets.mean(axis=0), [0.0, 0.0], atol=0.15)
    assert_allclose(offsets.std(axis=0), [_GPS_SE_M, _GPS_SE_M], rtol=0.06)


def test_deform_scales_with_measurement_length() -> None:
    """Deformation is a fraction of the measurement length, so longer plots deform more."""
    rng = np.random.default_rng(4)
    small = _deformer(_square(40.0), 40.0, gps_standard_error_m=0.0)
    large = _deformer(_square(40.0), 80.0, gps_standard_error_m=0.0)
    small_spread = np.std([small.deform(rng).area for _ in range(2000)])
    large_spread = np.std([large.deform(rng).area for _ in range(2000)])
    assert large_spread > small_spread * 1.5


def test_deform_without_gps_error_barely_moves_centre() -> None:
    """With GPS error off, the centre only drifts as a side effect of deformation."""
    square = _square()
    deformer = _deformer(square, 40.0, gps_standard_error_m=0.0)
    rng = np.random.default_rng(5)
    offsets = np.array(
        [
            np.asarray(deformer.deform(rng).centroid.coords[0])
            - np.asarray(square.centroid.coords[0])
            for _ in range(2000)
        ],
    )
    assert_allclose(offsets.mean(axis=0), [0.0, 0.0], atol=0.1)
    assert np.all(offsets.std(axis=0) < _GPS_SE_M * 0.3)


def test_deform_with_no_error_returns_original_shape() -> None:
    square = _square()
    deformer = _deformer(square, 40.0, deformation_fraction=0.0, gps_standard_error_m=0.0)
    deformed = deformer.deform(np.random.default_rng(6))
    assert deformed.area == pytest.approx(square.area)
