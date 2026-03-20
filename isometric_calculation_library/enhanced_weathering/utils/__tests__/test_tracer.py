# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

import numpy as np
import pytest
from more_itertools import one

from isometric_calculation_library.enhanced_weathering.utils.tracer import (
    compute_application_rate_from_tracer,
    compute_control_correction_ratio,
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


def test_control_correction_ratio() -> None:
    """cc = C_rp_ctrl / C_bl_ctrl."""
    result = compute_control_correction_ratio(
        control_baseline_mg_kg=np.array([100.0, 200.0]),
        control_end_of_reporting_period_mg_kg=np.array([90.0, 210.0]),
    )
    np.testing.assert_allclose(result, [0.9, 1.05])


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


def test_fraction_dissolved_with_control_correction() -> None:
    """Control correction scales the post-application concentration."""
    m = np.array([0.01])
    c_feed = 5000.0
    c_post = (np.array([100.0]) + m * c_feed) / (1 + m)
    c_rp = c_post * 0.95  # small decrease
    cc = 1.0  # no background change

    result_no_cc = compute_fraction_dissolved(
        feedstock_soil_mass_ratio=m,
        post_application_concentration_mg_kg=c_post,
        soil_end_of_reporting_period_mg_kg=c_rp,
        feedstock_mg_kg=c_feed,
        control_correction_ratio=cc,
    )
    # With cc > 1, it inflates post-app, so more dissolution is attributed
    cc_inflated = 1.05
    result_with_cc = compute_fraction_dissolved(
        feedstock_soil_mass_ratio=m,
        post_application_concentration_mg_kg=c_post,
        soil_end_of_reporting_period_mg_kg=c_rp,
        feedstock_mg_kg=c_feed,
        control_correction_ratio=cc_inflated,
    )
    assert one(result_with_cc) > one(result_no_cc) + 1e-6


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
