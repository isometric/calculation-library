# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

import pytest
from numpy.testing import assert_allclose

from isometric_calculation_library.biosphere.allometric_equations.westfall_nsvb.species import (
    find_species_code,
    get_species,
)
from isometric_calculation_library.biosphere.allometric_equations.wood_density import (
    tree_type_to_species,
)

SHORTLEAF_PINE = 110
DOUGLAS_FIR = 202
FIR_SPP = 10
SWAMP_WHITE_OAK = 804


def test_get_species() -> None:
    species = get_species(SHORTLEAF_PINE)

    assert species.common_name == "shortleaf pine"
    assert species.genus == "Pinus"
    assert species.specific_epithet == "echinata"
    assert species.species == "Pinus echinata"
    assert species.jenkins_species_group_code == 4
    assert_allclose(species.wood_specific_gravity, 0.47)


def test_get_species_unknown_code_raises() -> None:
    with pytest.raises(KeyError, match="SPCD 9999 not found"):
        get_species(9999)


def test_find_species_code_is_case_insensitive() -> None:
    assert find_species_code("Pseudotsuga menziesii") == DOUGLAS_FIR
    assert find_species_code("pseudotsuga MENZIESII") == DOUGLAS_FIR


def test_find_species_code_resolves_a_bare_genus_to_the_genus_level_code() -> None:
    """FIA codes genera as well as species, spelling them ``"Abies spp."``."""
    assert find_species_code("Abies spp.") == FIR_SPP
    assert find_species_code("Abies") == FIR_SPP


def test_find_species_code_takes_a_converted_tree_type_qualifier() -> None:
    """The tree type an activity model carries reaches NSVB through the wood-density helper."""
    assert find_species_code(tree_type_to_species("taxonomic_rank:species:quercus_bicolor")) == (
        SWAMP_WHITE_OAK
    )
    assert find_species_code(tree_type_to_species("taxonomic_rank:genus:abies")) == FIR_SPP


def test_find_species_code_unknown_binomial_raises() -> None:
    with pytest.raises(KeyError, match="Eucalyptus regnans"):
        find_species_code("Eucalyptus regnans")


def test_find_species_code_ambiguous_binomial_raises_and_lists_the_codes() -> None:
    """FIA gives ten binomials more than one code, and they are not interchangeable.

    *Pinus elliottii* is both slash pine and Honduras pine, whose wood specific gravities
    differ by 26%, so resolving it to whichever row came first would be a silent error.
    """
    with pytest.raises(KeyError, match="ambiguous") as excinfo:
        find_species_code("Pinus elliottii")

    assert "111" in str(excinfo.value)
    assert "144" in str(excinfo.value)
