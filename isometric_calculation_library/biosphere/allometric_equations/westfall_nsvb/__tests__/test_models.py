# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

import numpy as np
import pytest
from numpy.testing import assert_allclose

from isometric_calculation_library.biosphere.allometric_equations.westfall_nsvb.coefficients import (
    get_nsvb_model,
)
from isometric_calculation_library.biosphere.allometric_equations.westfall_nsvb.models import (
    NsvbCoefficientLevel,
    NsvbComponentModel,
    NsvbModel,
    NsvbModelForm,
    is_softwood,
    segment_break_diameter_in,
)

CM_PER_INCH = 2.54
METRES_PER_FOOT = 0.3048
TONNES_PER_POUND = 0.45359237 / 1000

SHORTLEAF_PINE = 110
RED_MAPLE = 316
NOBLE_FIR = 22


A_1, B_1, C_1 = 0.186963564, 2.19009994, 0.725056549
"""Shortleaf pine wood and bark, ecodivision-pooled: Eq 1 (table S8a)."""

A_2, B_2, B1_2, C_2 = 0.177868614, 2.180014222, 1.976884888, 0.78319536
"""Slash pine wood and bark in natural stands: Eq 2 (table S8a)."""

A_3, A1_3, B_3, C_3, C1_3 = 0.358815475, 1.849604207, 0.149984321, 0.841295541, 0.332484731
"""Slash pine wood and bark in plantations: Eq 3 (table S8a)."""

A_4, B_4, B1_4, C_4 = 0.315730276, 1.853839844, -0.024745685, 0.740557379
"""Red maple wood and bark, ecodivision-pooled: Eq 4 (table S8a)."""

A_5, B_5, C_5 = 0.772534536, 2.184545296, 0.575832011
"""True fir and hemlock Jenkins group wood and bark: Eq 5 (table S8b)."""


def component(model: NsvbModelForm, **kwargs: float) -> NsvbComponentModel:
    defaults: dict[str, float] = {"a": 1.0, "b": 2.0, "c": 0.5, "sigma": 1.0}
    return NsvbComponentModel(
        model=model,
        coefficient_level=NsvbCoefficientLevel.SPECIES,
        **{**defaults, **kwargs},
    )


# --- NsvbComponentModel.compute_biomass_pounds, one test per published model form ---


def test_model_1_schumacher_hall() -> None:
    """Eq 1: biomass = a * D^b * H^c."""
    model = component(NsvbModelForm.SCHUMACHER_HALL, a=A_1, b=B_1, c=C_1)

    result = model.compute_biomass_pounds(np.array([12.0]), np.array([65.0]))

    assert_allclose(result, [A_1 * 12.0**B_1 * 65.0**C_1], rtol=1e-12)


def test_model_2_segmented_below_break() -> None:
    """Eq 2 below the segmentation point uses the b exponent, as in the paper (not nsvb.R's b1)."""
    model = component(NsvbModelForm.SEGMENTED, a=A_2, b=B_2, b1=B1_2, c=C_2, segment_break_in=9.0)

    result = model.compute_biomass_pounds(np.array([6.0]), np.array([40.0]))

    assert_allclose(result, [A_2 * 6.0**B_2 * 40.0**C_2], rtol=1e-12)


def test_model_2_segmented_above_break() -> None:
    """Eq 2 above the segmentation point: a * k^(b - b1) * D^b1 * H^c."""
    model = component(NsvbModelForm.SEGMENTED, a=A_2, b=B_2, b1=B1_2, c=C_2, segment_break_in=9.0)

    result = model.compute_biomass_pounds(np.array([20.0]), np.array([80.0]))

    assert_allclose(result, [A_2 * 9.0 ** (B_2 - B1_2) * 20.0**B1_2 * 80.0**C_2], rtol=1e-12)


def test_model_2_segmented_is_continuous_at_the_break() -> None:
    """The k^(b - b1) factor exists to make the two branches meet; check that it does."""
    model = component(NsvbModelForm.SEGMENTED, a=A_2, b=B_2, b1=B1_2, c=C_2, segment_break_in=9.0)

    result = model.compute_biomass_pounds(np.array([9.0 - 1e-9, 9.0]), np.array([60.0, 60.0]))

    assert_allclose(result[0], result[1], rtol=1e-8)


def test_model_3_continuously_variable() -> None:
    """Eq 3: the diameter exponent is a1 * (1 - exp(-b*D))^c1, rising towards a1."""
    model = component(
        NsvbModelForm.CONTINUOUSLY_VARIABLE,
        a=A_3,
        a1=A1_3,
        b=B_3,
        c=C_3,
        c1=C1_3,
    )

    result = model.compute_biomass_pounds(np.array([12.0]), np.array([65.0]))

    exponent = A1_3 * (1 - np.exp(-B_3 * 12.0)) ** C1_3
    assert_allclose(result, [A_3 * 12.0**exponent * 65.0**C_3], rtol=1e-12)


def test_model_3_diameter_exponent_approaches_a1() -> None:
    """The exponent asymptote is a1: a huge tree behaves like D^a1.

    This is what separates the paper's Eq 3 from the parenthesisation in the published
    nsvb.R, where the exponent would instead tend to a1^c1.
    """
    model = component(
        NsvbModelForm.CONTINUOUSLY_VARIABLE,
        a=A_3,
        a1=A1_3,
        b=B_3,
        c=C_3,
        c1=C1_3,
    )

    result = model.compute_biomass_pounds(np.array([500.0]), np.array([100.0]))

    assert_allclose(result, [A_3 * 500.0**A1_3 * 100.0**C_3], rtol=1e-3)


def test_model_4_modified_wiley() -> None:
    """Eq 4: biomass = a * D^b * H^c * exp(-b1 * D)."""
    model = component(NsvbModelForm.MODIFIED_WILEY, a=A_4, b=B_4, b1=B1_4, c=C_4)

    result = model.compute_biomass_pounds(np.array([12.0]), np.array([65.0]))

    assert_allclose(result, [A_4 * 12.0**B_4 * 65.0**C_4 * np.exp(-B1_4 * 12.0)], rtol=1e-12)


def test_model_5_scales_with_wood_specific_gravity() -> None:
    """Eq 5, the Jenkins-group fallback, multiplies through by WDSG."""
    model = component(
        NsvbModelForm.MODIFIED_SCHUMACHER_HALL,
        a=A_5,
        b=B_5,
        c=C_5,
        wood_specific_gravity=0.37,
    )

    result = model.compute_biomass_pounds(np.array([12.0]), np.array([65.0]))

    assert_allclose(result, [A_5 * 12.0**B_5 * 65.0**C_5 * 0.37], rtol=1e-12)


@pytest.mark.parametrize(
    ("model", "supplied", "missing"),
    [
        (NsvbModelForm.SEGMENTED, {}, "b1"),
        (NsvbModelForm.CONTINUOUSLY_VARIABLE, {"c1": C1_3}, "a1"),
        (NsvbModelForm.CONTINUOUSLY_VARIABLE, {"a1": A1_3}, "c1"),
        (NsvbModelForm.MODIFIED_WILEY, {}, "b1"),
        (NsvbModelForm.MODIFIED_SCHUMACHER_HALL, {}, "wood_specific_gravity"),
    ],
)
def test_model_form_missing_required_coefficient_raises(
    model: NsvbModelForm,
    supplied: dict[str, float],
    missing: str,
) -> None:
    """A form built without a coefficient it needs should name it rather than fail on None."""
    with pytest.raises(ValueError, match=f"requires {missing}"):
        component(model, **supplied).compute_biomass_pounds(np.array([12.0]), np.array([65.0]))


def test_model_3_is_zero_at_zero_diameter() -> None:
    """Eq 3's exponent vanishes with the diameter, so a * H^c is its limit; it must not leak."""
    model = component(
        NsvbModelForm.CONTINUOUSLY_VARIABLE,
        a=A_3,
        a1=A1_3,
        b=B_3,
        c=C_3,
        c1=C1_3,
    )

    result = model.compute_biomass_pounds(np.array([0.0, 12.0]), np.array([65.0, 65.0]))

    assert_allclose(result[0], 0.0)
    assert result[1] > 0.0


# --- unit conversion ---


def test_metric_matches_imperial() -> None:
    model = component(NsvbModelForm.SCHUMACHER_HALL, a=A_1, b=B_1, c=C_1)
    dbh_in, height_ft = np.array([6.0, 12.0, 30.0]), np.array([40.0, 65.0, 110.0])

    pounds = model.compute_biomass_pounds(dbh_in, height_ft)
    tonnes = model.compute_biomass_tonnes(dbh_in * CM_PER_INCH, height_ft * METRES_PER_FOOT)

    assert_allclose(tonnes, pounds * TONNES_PER_POUND, rtol=1e-12)


def test_biomass_increases_with_size() -> None:
    model = get_nsvb_model(SHORTLEAF_PINE)
    dbh = np.array([10.0, 20.0, 40.0])
    height = np.array([8.0, 15.0, 25.0])

    result = model.compute_agb_tonnes(dbh, height)

    assert result.shape == (3,)
    assert result[0] < result[1] < result[2]


# --- residual spread ---


def test_residual_sd_is_proportional_to_diameter() -> None:
    """NSVB fits with 1/D² weights, so Sigma is a residual SD per inch of diameter."""
    model = component(NsvbModelForm.SCHUMACHER_HALL, sigma=10.384548)

    result = model.residual_sd_tonnes(np.array([25.4, 50.8]))

    assert_allclose(
        result,
        [10.384548 * 10.0 * TONNES_PER_POUND, 10.384548 * 20.0 * TONNES_PER_POUND],
        rtol=1e-12,
    )


def test_agb_residual_sd_uses_the_summed_component_statistic() -> None:
    """AGB uncertainty comes from NSVB's own TT_DW_ADJ fit, not the two parts combined."""
    model = get_nsvb_model(SHORTLEAF_PINE)
    dbh = np.array([30.0])

    result = model.agb_residual_sd_tonnes(dbh)

    assert_allclose(result, model.agb_sigma * (30.0 / CM_PER_INCH) * TONNES_PER_POUND, rtol=1e-12)
    combined = np.hypot(
        model.wood_bark.residual_sd_tonnes(dbh),
        model.foliage.residual_sd_tonnes(dbh),
    )
    assert not np.isclose(result[0], combined[0])


def test_species_sigma_is_preferred_over_national() -> None:
    """Table S13 gives shortleaf pine its own Sigma; noble fir is unmodelled and falls to S12."""
    assert_allclose(get_nsvb_model(SHORTLEAF_PINE).agb_sigma, 9.20234009308803, rtol=1e-12)
    assert_allclose(get_nsvb_model(NOBLE_FIR).agb_sigma, 10.7568718776936, rtol=1e-12)


# --- NsvbModel.compute_agb_tonnes_with_error ---


def test_compute_agb_tonnes_with_error_shape_and_sign() -> None:
    model = get_nsvb_model(SHORTLEAF_PINE)

    result = model.compute_agb_tonnes_with_error(
        np.array([20.0, 30.0]),
        np.array([15.0, 22.0]),
        np.random.default_rng(42),
    )

    assert result.shape == (2,)
    assert np.all(result >= 0)


def test_compute_agb_tonnes_with_error_mean_matches_deterministic() -> None:
    """The error is additive and symmetric, so the mean should sit on the deterministic value."""
    model = get_nsvb_model(SHORTLEAF_PINE)
    dbh, height = np.full(50_000, 40.0), np.full(50_000, 25.0)

    result = model.compute_agb_tonnes_with_error(dbh, height, np.random.default_rng(42))

    deterministic = model.compute_agb_tonnes(np.array([40.0]), np.array([25.0]))[0]
    assert_allclose(result.mean(), deterministic, rtol=0.01)


def test_compute_agb_tonnes_with_error_is_clipped() -> None:
    """A small tree has a relative residual large enough for both clips to bind."""
    model = get_nsvb_model(SHORTLEAF_PINE)
    dbh, height = np.full(5_000, 5.0), np.full(5_000, 4.0)

    result = model.compute_agb_tonnes_with_error(dbh, height, np.random.default_rng(42))

    deterministic = model.compute_agb_tonnes(np.array([5.0]), np.array([4.0]))[0]
    assert result.min() >= 0
    assert result.max() <= NsvbModel.AGB_CLIP_FACTOR * deterministic * (1 + 1e-12)


def test_compute_agb_tonnes_with_error_deterministic_given_a_seed() -> None:
    model = get_nsvb_model(SHORTLEAF_PINE)
    dbh, height = np.array([30.0]), np.array([20.0])

    first = model.compute_agb_tonnes_with_error(dbh, height, np.random.default_rng(99))
    second = model.compute_agb_tonnes_with_error(dbh, height, np.random.default_rng(99))

    assert_allclose(first, second)


def test_compute_agb_tonnes_with_linearized_error_matches_drawn_error() -> None:
    model = get_nsvb_model(SHORTLEAF_PINE)
    dbh, height = np.array([30.0]), np.array([20.0])
    error = np.array([0.01])

    result = model.compute_agb_tonnes_with_linearized_error(
        dbh,
        height,
        allometric_error_tonnes=error,
    )

    assert_allclose(result, model.compute_agb_tonnes(dbh, height) + error, rtol=1e-12)


def test_compute_agb_tonnes_with_linearized_error_is_clipped() -> None:
    """The caller supplies the error here, so both clips have to hold for any value it passes."""
    model = get_nsvb_model(SHORTLEAF_PINE)
    dbh, height = np.array([30.0, 30.0]), np.array([20.0, 20.0])
    deterministic = model.compute_agb_tonnes(dbh, height)

    result = model.compute_agb_tonnes_with_linearized_error(
        dbh,
        height,
        allometric_error_tonnes=np.array([-10.0, 10.0]) * deterministic,
    )

    assert_allclose(result[0], 0.0)
    assert_allclose(result[1], NsvbModel.AGB_CLIP_FACTOR * deterministic[1], rtol=1e-12)


# --- segmentation point ---


def test_segment_break_splits_softwoods_from_hardwoods() -> None:
    assert is_softwood(SHORTLEAF_PINE)
    assert not is_softwood(RED_MAPLE)
    assert_allclose(segment_break_diameter_in(SHORTLEAF_PINE), 9.0)
    assert_allclose(segment_break_diameter_in(RED_MAPLE), 11.0)
    assert_allclose(get_nsvb_model(SHORTLEAF_PINE).wood_bark.segment_break_in, 9.0)
    assert_allclose(get_nsvb_model(RED_MAPLE).wood_bark.segment_break_in, 11.0)
