# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

import numpy as np
import pytest
from more_itertools import one

from isometric_calculation_library.enhanced_weathering.utils.tracer import (
    compute_application_rate_from_tracer,
    compute_fraction_dissolved,
    compute_mass_ratio_from_immobile_tracer,
    compute_post_application_concentration,
)


def test_mass_ratio_no_change_gives_zero() -> None:
    """When tracer doesn't change between baseline and reporting period, mass ratio is zero."""
    result = compute_mass_ratio_from_immobile_tracer(
        feedstock_tracer_mg_kg=1000.0,
        soil_baseline_tracer_mg_kg=np.array([50.0, 50.0]),
        soil_end_of_reporting_period_tracer_mg_kg=np.array([50.0, 50.0]),
    )
    np.testing.assert_allclose(result, [0.0, 0.0])


def test_mass_ratio_positive_when_tracer_increases() -> None:
    """Tracer increase from 50 to 60 with feedstock at 1000: m = (60-50)/(1000-60)."""
    result = compute_mass_ratio_from_immobile_tracer(
        feedstock_tracer_mg_kg=1000.0,
        soil_baseline_tracer_mg_kg=np.array([50.0]),
        soil_end_of_reporting_period_tracer_mg_kg=np.array([60.0]),
    )
    expected = (60 - 50) / (1000 - 60)
    assert result == pytest.approx([expected])


def test_mass_ratio_zero_denominator_gives_nan() -> None:
    """Feedstock tracer equal to soil reporting-period tracer zeroes the denominator,
    so that replicate is NaN (not inf), while other replicates stay finite."""
    result = compute_mass_ratio_from_immobile_tracer(
        feedstock_tracer_mg_kg=60.0,
        soil_baseline_tracer_mg_kg=np.array([50.0, 50.0]),
        soil_end_of_reporting_period_tracer_mg_kg=np.array([60.0, 55.0]),
    )
    assert np.isnan(result[0])
    assert np.isfinite(result[1])


def test_post_application_concentration_mixing() -> None:
    """C_post = (C_bl + m * C_feed) / (1 + m)."""
    m = np.array([0.01])
    c_bl = np.array([100.0])
    c_feed = 5000.0
    result = compute_post_application_concentration(
        feedstock_soil_mass_ratio=m,
        soil_baseline_mg_kg=c_bl,
        feedstock_mg_kg=c_feed,
    )
    expected = (100.0 + 0.01 * 5000.0) / (1 + 0.01)
    assert result == pytest.approx([expected])


def test_post_application_no_feedstock_equals_baseline() -> None:
    """When mass ratio is zero, post-application equals baseline."""
    result = compute_post_application_concentration(
        feedstock_soil_mass_ratio=np.array([0.0]),
        soil_baseline_mg_kg=np.array([100.0]),
        feedstock_mg_kg=5000.0,
    )
    assert result == pytest.approx([100.0])


def test_fraction_dissolved_full_dissolution() -> None:
    """f_d = ((1+m)/m) * (C_post - C_rp) / C_feed. For f_d=1: C_rp = C_post - m*C_feed/(1+m)."""
    m = np.array([0.01])
    c_feed = 5000.0
    c_bl = np.array([100.0])
    c_post = (c_bl + m * c_feed) / (1 + m)
    # Solve for C_rp when f_d = 1: C_rp = C_post - m * C_feed / (1 + m)
    c_rp = c_post - m * c_feed / (1 + m)

    result = compute_fraction_dissolved(
        feedstock_soil_mass_ratio=m,
        post_application_concentration_mg_kg=c_post,
        soil_end_of_reporting_period_mg_kg=c_rp,
        feedstock_mg_kg=c_feed,
    )
    assert result == pytest.approx([1.0], rel=1e-10)


def test_fraction_dissolved_no_dissolution() -> None:
    """If no dissolution, C_rp = C_post and f_d = 0."""
    m = np.array([0.01])
    c_feed = 5000.0
    c_post = (np.array([100.0]) + m * c_feed) / (1 + m)
    c_rp = c_post  # no change

    result = compute_fraction_dissolved(
        feedstock_soil_mass_ratio=m,
        post_application_concentration_mg_kg=c_post,
        soil_end_of_reporting_period_mg_kg=c_rp,
        feedstock_mg_kg=c_feed,
    )
    assert result == pytest.approx([0.0])


def test_fraction_dissolved_control_correction_delta_reduces_dissolution() -> None:
    """A positive delta is background loss, so it must reduce the fraction dissolved."""
    m = np.array([0.01])
    c_feed = 5000.0
    c_post = (np.array([100.0]) + m * c_feed) / (1 + m)
    c_rp = c_post * 0.95
    delta = 5.0

    uncorrected = compute_fraction_dissolved(
        feedstock_soil_mass_ratio=m,
        post_application_concentration_mg_kg=c_post,
        soil_end_of_reporting_period_mg_kg=c_rp,
        feedstock_mg_kg=c_feed,
    )
    corrected = compute_fraction_dissolved(
        feedstock_soil_mass_ratio=m,
        post_application_concentration_mg_kg=c_post,
        soil_end_of_reporting_period_mg_kg=c_rp,
        feedstock_mg_kg=c_feed,
        control_correction_delta_mg_kg=delta,
    )

    assert one(corrected) < one(uncorrected)
    expected = ((1 + m) / m) * ((c_post - c_rp - delta) / c_feed)
    np.testing.assert_allclose(corrected, expected)


def test_fraction_dissolved_zero_delta_is_neutral() -> None:
    """The default delta of 0.0 leaves the uncorrected result unchanged."""
    m = np.array([0.01])
    c_feed = 5000.0
    c_post = (np.array([100.0]) + m * c_feed) / (1 + m)
    c_rp = c_post * 0.95

    np.testing.assert_allclose(
        compute_fraction_dissolved(
            feedstock_soil_mass_ratio=m,
            post_application_concentration_mg_kg=c_post,
            soil_end_of_reporting_period_mg_kg=c_rp,
            feedstock_mg_kg=c_feed,
        ),
        compute_fraction_dissolved(
            feedstock_soil_mass_ratio=m,
            post_application_concentration_mg_kg=c_post,
            soil_end_of_reporting_period_mg_kg=c_rp,
            feedstock_mg_kg=c_feed,
            control_correction_delta_mg_kg=0.0,
        ),
    )


def test_fraction_dissolved_ratio_scales_post_application() -> None:
    """The deprecated ratio form scales C_post, unchanged from its original behaviour."""
    m = np.array([0.01])
    c_feed = 5000.0
    c_post = (np.array([100.0]) + m * c_feed) / (1 + m)
    c_rp = np.array([120.0])
    cc = 0.9

    result = compute_fraction_dissolved(
        feedstock_soil_mass_ratio=m,
        post_application_concentration_mg_kg=c_post,
        soil_end_of_reporting_period_mg_kg=c_rp,
        feedstock_mg_kg=c_feed,
        control_correction_ratio=cc,
    )

    expected = ((1 + m) / m) * ((c_post * cc - c_rp) / c_feed)
    np.testing.assert_allclose(result, expected)


def test_fraction_dissolved_neutral_ratio_and_zero_delta_are_uncorrected() -> None:
    """Both corrections default to neutral, so omitting them leaves f_d uncorrected."""
    m = np.array([0.01])
    c_feed = 5000.0
    c_post = (np.array([100.0]) + m * c_feed) / (1 + m)
    c_rp = np.array([120.0])

    np.testing.assert_allclose(
        compute_fraction_dissolved(
            feedstock_soil_mass_ratio=m,
            post_application_concentration_mg_kg=c_post,
            soil_end_of_reporting_period_mg_kg=c_rp,
            feedstock_mg_kg=c_feed,
        ),
        ((1 + m) / m) * ((c_post - c_rp) / c_feed),
    )


def test_fraction_dissolved_inf_replaced_with_nan() -> None:
    """When mass ratio is zero, result should be NaN (not inf)."""
    result = compute_fraction_dissolved(
        feedstock_soil_mass_ratio=np.array([0.0]),
        post_application_concentration_mg_kg=np.array([100.0]),
        soil_end_of_reporting_period_mg_kg=np.array([90.0]),
        feedstock_mg_kg=5000.0,
    )
    assert np.isnan(result[0])


def test_application_rate_from_tracer() -> None:
    """app_rate = m * BD * depth * 100."""
    result = compute_application_rate_from_tracer(
        feedstock_soil_mass_ratio=np.array([0.005]),
        soil_bulk_density_kg_m3=np.array([1000.0]),
        depth_cm=30.0,
    )
    expected = 0.005 * 1000.0 * 30.0 * 100
    assert result == pytest.approx([expected])
