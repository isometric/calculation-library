# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

import pytest
from numpy.testing import assert_allclose

from ..wood_density import WoodDensityRecord, get_wood_density, list_species, tree_type_to_species


def test_get_wood_density_known_species() -> None:
    record = get_wood_density("Cecropia obtusa")
    assert isinstance(record, WoodDensityRecord)
    assert record.species == "Cecropia obtusa"
    assert_allclose(record.wood_density, 0.31, rtol=0.01)
    assert record.wood_density_sd > 0


def test_get_wood_density_unknown_species_raises() -> None:
    with pytest.raises(KeyError, match="not found"):
        get_wood_density("Nonexistent species")


def test_list_species_returns_sorted() -> None:
    species = list_species()
    assert len(species) > 0
    assert species == sorted(species)


def test_list_species_no_unidentified() -> None:
    species = list_species()
    assert "N.I." not in species


def test_all_records_have_positive_values() -> None:
    for sp in list_species():
        record = get_wood_density(sp)
        assert record.wood_density > 0, f"{sp} has non-positive wood density"
        assert record.wood_density_sd > 0, f"{sp} has non-positive SD"


def test_tree_type_to_species_with_species_prefix() -> None:
    assert tree_type_to_species("SPECIES_CECROPIA_OBTUSA") == "Cecropia obtusa"


def test_tree_type_to_species_with_genus_prefix() -> None:
    assert tree_type_to_species("GENUS_SALIX") == "Salix"


def test_tree_type_to_species_bare_name() -> None:
    assert tree_type_to_species("JUGLANS_NIGRA") == "Juglans nigra"


def test_tree_type_to_species_single_word() -> None:
    assert tree_type_to_species("SPECIES_CECROPIA") == "Cecropia"
