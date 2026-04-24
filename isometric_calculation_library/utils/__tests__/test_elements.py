# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/
from typing import get_args

import pytest
from molmass.elements import (  # noqa: TID251 - molmass is allowed as a test dependency, to verify production data
    ELEMENTS,
)

from ..elements import (
    ElementSymbol,
    atomic_number,
    atomic_weight,
    element_name,
    to_element_symbol,
)


def test_element_symbols_are_ordered_by_atomic_number() -> None:
    symbols = get_args(ElementSymbol.__value__)
    atomic_numbers = [atomic_number(s) for s in symbols]
    for i in range(len(atomic_numbers) - 1):
        assert atomic_numbers[i] < atomic_numbers[i + 1], (
            f"{symbols[i]} (Z={atomic_numbers[i]}) should come before "
            f"{symbols[i + 1]} (Z={atomic_numbers[i + 1]})"
        )


def test_all_molmass_elements_are_represented() -> None:
    defined = set(get_args(ElementSymbol.__value__))
    molmass_symbols = {e.symbol for e in ELEMENTS}
    assert defined == molmass_symbols


def test_to_element_symbol_accepts_all_valid_symbols() -> None:
    for symbol in get_args(ElementSymbol.__value__):
        assert to_element_symbol(symbol) == symbol


def test_to_element_symbol_rejects_invalid_symbol() -> None:
    with pytest.raises(ValueError, match="Unknown element symbol"):
        to_element_symbol("Xx")


def test_to_element_symbol_returns_none_for_invalid_symbol() -> None:
    assert to_element_symbol("Xx", raise_on_invalid=False) is None


def test_to_element_symbol_normalizes_case() -> None:
    assert to_element_symbol("fe", normalize=True) == "Fe"
    assert to_element_symbol("FE", normalize=True) == "Fe"


def test_to_element_symbol_rejects_unnormalized_case() -> None:
    with pytest.raises(ValueError, match="Unknown element symbol"):
        to_element_symbol("fe")


def test_atomic_number_matches_molmass() -> None:
    for symbol in get_args(ElementSymbol.__value__):
        ours = atomic_number(symbol)
        expected = ELEMENTS[symbol].number
        assert ours == expected, f"{symbol} - ours: {ours}, molmass: {expected}"


def test_atomic_weight_matches_molmass() -> None:
    for symbol in get_args(ElementSymbol.__value__):
        ours = atomic_weight(symbol)
        expected = ELEMENTS[symbol].mass
        assert ours == pytest.approx(expected, rel=1e-9), (
            f"{symbol} - ours: {ours}, molmass: {expected}"
        )


def test_element_name_matches_molmass() -> None:
    for symbol in get_args(ElementSymbol.__value__):
        ours = element_name(symbol)
        expected = ELEMENTS[symbol].name
        assert ours == expected, f"{symbol} - ours: {ours!r}, molmass: {expected!r}"
