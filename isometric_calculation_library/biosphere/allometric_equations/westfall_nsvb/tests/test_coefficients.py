# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

import numpy as np
import pytest
from more_itertools import first
from numpy.testing import assert_allclose

from isometric_calculation_library.biosphere.allometric_equations.westfall_nsvb.coefficients import (
    create_nsvb_model_generator,
    get_nsvb_model,
    published_divisions,
)
from isometric_calculation_library.biosphere.allometric_equations.westfall_nsvb.models import (
    NsvbCoefficientLevel,
    NsvbModel,
    NsvbModelForm,
    StandOrigin,
)
from isometric_calculation_library.biosphere.allometric_equations.westfall_nsvb.species import (
    get_species,
)

SHORTLEAF_PINE = 110
SLASH_PINE = 111
DOUGLAS_FIR = 202
NOBLE_FIR = 22
UTAH_JUNIPER = 65
SPRUCE_PINE = 115

NATIONAL_SIGMA = {
    "wood_bark": 10.3845477850393,
    "foliage": 1.16555203273024,
    "agb": 10.7568718776936,
}
"""National Sigma per component (table S12), used where a species' own fit has none."""


# --- coefficient selection ---


def test_ecodivision_fit_is_used_when_published() -> None:
    model = get_nsvb_model(SHORTLEAF_PINE, division="230")

    assert model.wood_bark.coefficient_level is NsvbCoefficientLevel.SPECIES_ECODIVISION
    assert_allclose(model.wood_bark.a, 0.057622485, rtol=1e-12)


def test_unpublished_ecodivision_falls_back_to_the_pooled_fit() -> None:
    """ "If a species occurs in an ecodivision not explicitly listed, the entry having no
    ecodivision noted is used" (Westfall et al. 2024).
    """
    pooled = get_nsvb_model(SHORTLEAF_PINE)
    elsewhere = get_nsvb_model(SHORTLEAF_PINE, division="M999")

    assert elsewhere.wood_bark.coefficient_level is NsvbCoefficientLevel.SPECIES
    assert_allclose(elsewhere.wood_bark.a, pooled.wood_bark.a)


def test_species_without_a_fit_falls_back_to_its_jenkins_group() -> None:
    model = get_nsvb_model(NOBLE_FIR)

    assert model.wood_bark.coefficient_level is NsvbCoefficientLevel.JENKINS_GROUP
    assert model.wood_bark.model is NsvbModelForm.MODIFIED_SCHUMACHER_HALL
    assert_allclose(model.wood_bark.a, 0.772534536, rtol=1e-12)
    assert model.wood_bark.wood_specific_gravity == get_species(NOBLE_FIR).wood_specific_gravity

    # The two components fall back independently, so foliage has to be checked separately.
    assert model.foliage.coefficient_level is NsvbCoefficientLevel.JENKINS_GROUP
    assert model.foliage.model is NsvbModelForm.SCHUMACHER_HALL
    assert_allclose(model.foliage.a, 1.166751267, rtol=1e-12)


def test_woodland_species_has_no_above_ground_fit() -> None:
    with pytest.raises(KeyError, match="woodland group"):
        get_nsvb_model(UTAH_JUNIPER)


def test_components_are_looked_up_independently() -> None:
    """NSVB fits Douglas-fir wood and bark with Eq 1 but its foliage with Eq 2."""
    model = get_nsvb_model(DOUGLAS_FIR)

    assert model.wood_bark.model is NsvbModelForm.SCHUMACHER_HALL
    assert model.foliage.model is NsvbModelForm.SEGMENTED


def test_single_tree_species_falls_back_to_the_national_sigma() -> None:
    """NSVB publishes Sigma as infinite for species fitted on one felled tree.

    Spruce pine is one of those for all three components. Propagating the published value
    would make every downstream standard error infinite, so the national Sigma is used.
    """
    model = get_nsvb_model(SPRUCE_PINE)

    assert_allclose(model.wood_bark.sigma, NATIONAL_SIGMA["wood_bark"], rtol=1e-12)
    assert_allclose(model.foliage.sigma, NATIONAL_SIGMA["foliage"], rtol=1e-12)
    assert_allclose(model.agb_sigma, NATIONAL_SIGMA["agb"], rtol=1e-12)
    assert np.all(np.isfinite(model.agb_residual_sd_tonnes(np.array([30.0]))))


# --- stand origin ---


def test_stand_origin_defaults_to_planted() -> None:
    assert get_nsvb_model(SLASH_PINE) == get_nsvb_model(
        SLASH_PINE,
        stand_origin=StandOrigin.PLANTED,
    )


def test_stand_origin_selects_a_different_model_form() -> None:
    """Slash pine is segmented in natural stands and continuously variable in plantations."""
    natural = get_nsvb_model(SLASH_PINE, stand_origin=StandOrigin.NATURAL)
    planted = get_nsvb_model(SLASH_PINE, stand_origin=StandOrigin.PLANTED)

    assert natural.wood_bark.model is NsvbModelForm.SEGMENTED
    assert planted.wood_bark.model is NsvbModelForm.CONTINUOUSLY_VARIABLE


# --- published_divisions ---


def test_published_divisions_includes_the_pooled_fit_first() -> None:
    divisions = published_divisions(SHORTLEAF_PINE)

    assert first(divisions) is None
    assert set(divisions) == {None, "230", "M230"}


def test_published_divisions_unions_both_components() -> None:
    """Douglas-fir foliage is fitted for M210 but its wood and bark is not."""
    assert "M210" in published_divisions(DOUGLAS_FIR)


def test_published_divisions_for_a_species_without_a_fit() -> None:
    assert published_divisions(NOBLE_FIR) == [None]


def test_published_divisions_unknown_code_raises() -> None:
    """An unknown code must not pass for a pooled-only species, which also returns ``[None]``."""
    with pytest.raises(KeyError, match="SPCD 9999 not found"):
        published_divisions(9999)


def test_published_divisions_are_resolvable_for_the_stand_origin() -> None:
    """Slash pine's fits are keyed by stand origin, so the ecodivisions offered must be too."""
    for stand_origin in StandOrigin:
        divisions = published_divisions(SLASH_PINE, stand_origin=stand_origin)

        assert first(divisions) is None
        for division in divisions[1:]:
            model = get_nsvb_model(SLASH_PINE, division=division, stand_origin=stand_origin)
            assert NsvbCoefficientLevel.SPECIES_ECODIVISION in {
                model.wood_bark.coefficient_level,
                model.foliage.coefficient_level,
            }


# --- create_nsvb_model_generator ---


def test_generator_yields_published_coefficient_sets() -> None:
    gen = create_nsvb_model_generator(np.random.default_rng(42), SHORTLEAF_PINE)

    models = [gen() for _ in range(200)]

    assert all(isinstance(model, NsvbModel) for model in models)
    assert {model.division for model in models} == set(published_divisions(SHORTLEAF_PINE))


def test_generator_mean_is_close_to_the_pooled_fit() -> None:
    """Resampling ecodivisions perturbs AGB but should not shift it far off the pooled fit."""
    gen = create_nsvb_model_generator(np.random.default_rng(42), SHORTLEAF_PINE)
    dbh, height = np.array([30.0]), np.array([20.0])

    mean = np.mean([gen().compute_agb_tonnes(dbh, height)[0] for _ in range(1000)])

    assert_allclose(
        mean,
        get_nsvb_model(SHORTLEAF_PINE).compute_agb_tonnes(dbh, height)[0],
        rtol=0.05,
    )


def test_generator_is_deterministic_given_a_seed() -> None:
    first = create_nsvb_model_generator(np.random.default_rng(123), DOUGLAS_FIR)
    second = create_nsvb_model_generator(np.random.default_rng(123), DOUGLAS_FIR)

    assert [first().division for _ in range(20)] == [second().division for _ in range(20)]


def test_generator_differs_between_seeds() -> None:
    first = create_nsvb_model_generator(np.random.default_rng(1), DOUGLAS_FIR)
    second = create_nsvb_model_generator(np.random.default_rng(2), DOUGLAS_FIR)

    assert [first().division for _ in range(20)] != [second().division for _ in range(20)]


def test_generator_keeps_both_components_in_one_ecodivision() -> None:
    """One division is drawn per model, so each draw equals the model built at that division."""
    gen = create_nsvb_model_generator(np.random.default_rng(42), DOUGLAS_FIR)
    dbh, height = np.array([30.0]), np.array([20.0])

    for _ in range(50):
        model = gen()
        rebuilt = get_nsvb_model(DOUGLAS_FIR, division=model.division)
        assert_allclose(
            model.compute_wood_bark_tonnes(dbh, height),
            rebuilt.compute_wood_bark_tonnes(dbh, height),
        )
        assert_allclose(
            model.compute_foliage_tonnes(dbh, height),
            rebuilt.compute_foliage_tonnes(dbh, height),
        )


def test_generator_respects_stand_origin() -> None:
    gen = create_nsvb_model_generator(
        np.random.default_rng(42),
        SLASH_PINE,
        stand_origin=StandOrigin.PLANTED,
    )

    models = [gen() for _ in range(20)]

    assert all(model.stand_origin is StandOrigin.PLANTED for model in models)
    assert all(model.wood_bark.model is NsvbModelForm.CONTINUOUSLY_VARIABLE for model in models)


def test_generator_without_ecodivision_fits_is_constant() -> None:
    """Documented limitation: a species with only a pooled fit has nothing to resample."""
    gen = create_nsvb_model_generator(np.random.default_rng(42), NOBLE_FIR)
    dbh, height = np.array([30.0]), np.array([20.0])

    draws = [gen().compute_agb_tonnes(dbh, height)[0] for _ in range(20)]

    assert len(set(draws)) == 1
