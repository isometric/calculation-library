# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

"""Periodic table elements.

Uses proper chemical symbols as string literals, covering all elements
available in molmass.
"""

from functools import cache
from typing import Literal, NamedTuple, cast, get_args, overload

type ElementSymbol = Literal[
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
]
"""Chemical symbol for a periodic table element.

Matches the set of elements available in the molmass Python library.
"""


@cache
def _valid_element_symbols() -> frozenset[str]:
    return frozenset(get_args(ElementSymbol.__value__))


# If raise_on_invalid is explicitly set to True, we cannot return None
@overload
def to_element_symbol(
    symbol: str,
    *,
    normalize: bool = False,
    raise_on_invalid: Literal[True],
) -> ElementSymbol: ...


# If raise_on_invalid is explicitly set to False, we can return None
@overload
def to_element_symbol(
    symbol: str,
    *,
    normalize: bool = False,
    raise_on_invalid: Literal[False],
) -> ElementSymbol | None: ...


# If it's not set, the default is True as well
@overload
def to_element_symbol(
    symbol: str,
    *,
    normalize: bool = False,
) -> ElementSymbol: ...


def to_element_symbol(
    symbol: str,
    *,
    normalize: bool = False,
    raise_on_invalid: bool = True,
) -> ElementSymbol | None:
    """Validate and narrow a string to an Element literal.

    Args:
        symbol: The element symbol to validate.
        normalize: If True, capitalize the symbol before validation
            (e.g. "fe" -> "Fe").
        raise_on_invalid: If True, raise ValueError for unknown symbols.
            If False, return None instead.
    """
    if normalize:
        symbol = symbol.capitalize()

    if symbol not in _valid_element_symbols():
        if raise_on_invalid:
            raise ValueError(f"Unknown element symbol: {symbol!r}")
        return None
    # Safe: a test ensures the Element literal matches the dataset exactly.
    return cast("ElementSymbol", symbol)


class _ElementData(NamedTuple):
    number: int
    weight: float
    name: str


_elements_data: dict[ElementSymbol, _ElementData] = {
    "H": _ElementData(1, 1.007941, "Hydrogen"),
    "He": _ElementData(2, 4.002602, "Helium"),
    "Li": _ElementData(3, 6.94, "Lithium"),
    "Be": _ElementData(4, 9.0121831, "Beryllium"),
    "B": _ElementData(5, 10.811, "Boron"),
    "C": _ElementData(6, 12.01074, "Carbon"),
    "N": _ElementData(7, 14.006703, "Nitrogen"),
    "O": _ElementData(8, 15.999405, "Oxygen"),
    "F": _ElementData(9, 18.998403163, "Fluorine"),
    "Ne": _ElementData(10, 20.1797, "Neon"),
    "Na": _ElementData(11, 22.98976928, "Sodium"),
    "Mg": _ElementData(12, 24.3051, "Magnesium"),
    "Al": _ElementData(13, 26.9815385, "Aluminium"),
    "Si": _ElementData(14, 28.0855, "Silicon"),
    "P": _ElementData(15, 30.973761998, "Phosphorus"),
    "S": _ElementData(16, 32.0648, "Sulfur"),
    "Cl": _ElementData(17, 35.4529, "Chlorine"),
    "Ar": _ElementData(18, 39.948, "Argon"),
    "K": _ElementData(19, 39.0983, "Potassium"),
    "Ca": _ElementData(20, 40.078, "Calcium"),
    "Sc": _ElementData(21, 44.955908, "Scandium"),
    "Ti": _ElementData(22, 47.867, "Titanium"),
    "V": _ElementData(23, 50.9415, "Vanadium"),
    "Cr": _ElementData(24, 51.9961, "Chromium"),
    "Mn": _ElementData(25, 54.938044, "Manganese"),
    "Fe": _ElementData(26, 55.845, "Iron"),
    "Co": _ElementData(27, 58.933194, "Cobalt"),
    "Ni": _ElementData(28, 58.6934, "Nickel"),
    "Cu": _ElementData(29, 63.546, "Copper"),
    "Zn": _ElementData(30, 65.38, "Zinc"),
    "Ga": _ElementData(31, 69.723, "Gallium"),
    "Ge": _ElementData(32, 72.63, "Germanium"),
    "As": _ElementData(33, 74.921595, "Arsenic"),
    "Se": _ElementData(34, 78.971, "Selenium"),
    "Br": _ElementData(35, 79.9035, "Bromine"),
    "Kr": _ElementData(36, 83.798, "Krypton"),
    "Rb": _ElementData(37, 85.4678, "Rubidium"),
    "Sr": _ElementData(38, 87.62, "Strontium"),
    "Y": _ElementData(39, 88.90584, "Yttrium"),
    "Zr": _ElementData(40, 91.224, "Zirconium"),
    "Nb": _ElementData(41, 92.90637, "Niobium"),
    "Mo": _ElementData(42, 95.95, "Molybdenum"),
    "Tc": _ElementData(43, 97.9072, "Technetium"),
    "Ru": _ElementData(44, 101.07, "Ruthenium"),
    "Rh": _ElementData(45, 102.9055, "Rhodium"),
    "Pd": _ElementData(46, 106.42, "Palladium"),
    "Ag": _ElementData(47, 107.8682, "Silver"),
    "Cd": _ElementData(48, 112.414, "Cadmium"),
    "In": _ElementData(49, 114.818, "Indium"),
    "Sn": _ElementData(50, 118.71, "Tin"),
    "Sb": _ElementData(51, 121.76, "Antimony"),
    "Te": _ElementData(52, 127.6, "Tellurium"),
    "I": _ElementData(53, 126.90447, "Iodine"),
    "Xe": _ElementData(54, 131.293, "Xenon"),
    "Cs": _ElementData(55, 132.90545196, "Caesium"),
    "Ba": _ElementData(56, 137.327, "Barium"),
    "La": _ElementData(57, 138.90547, "Lanthanum"),
    "Ce": _ElementData(58, 140.116, "Cerium"),
    "Pr": _ElementData(59, 140.90766, "Praseodymium"),
    "Nd": _ElementData(60, 144.242, "Neodymium"),
    "Pm": _ElementData(61, 144.9128, "Promethium"),
    "Sm": _ElementData(62, 150.36, "Samarium"),
    "Eu": _ElementData(63, 151.964, "Europium"),
    "Gd": _ElementData(64, 157.25, "Gadolinium"),
    "Tb": _ElementData(65, 158.92535, "Terbium"),
    "Dy": _ElementData(66, 162.5, "Dysprosium"),
    "Ho": _ElementData(67, 164.93033, "Holmium"),
    "Er": _ElementData(68, 167.259, "Erbium"),
    "Tm": _ElementData(69, 168.93422, "Thulium"),
    "Yb": _ElementData(70, 173.054, "Ytterbium"),
    "Lu": _ElementData(71, 174.9668, "Lutetium"),
    "Hf": _ElementData(72, 178.49, "Hafnium"),
    "Ta": _ElementData(73, 180.94788, "Tantalum"),
    "W": _ElementData(74, 183.84, "Tungsten"),
    "Re": _ElementData(75, 186.207, "Rhenium"),
    "Os": _ElementData(76, 190.23, "Osmium"),
    "Ir": _ElementData(77, 192.217, "Iridium"),
    "Pt": _ElementData(78, 195.084, "Platinum"),
    "Au": _ElementData(79, 196.966569, "Gold"),
    "Hg": _ElementData(80, 200.592, "Mercury"),
    "Tl": _ElementData(81, 204.3834, "Thallium"),
    "Pb": _ElementData(82, 207.2, "Lead"),
    "Bi": _ElementData(83, 208.9804, "Bismuth"),
    "Po": _ElementData(84, 208.9824, "Polonium"),
    "At": _ElementData(85, 209.9871, "Astatine"),
    "Rn": _ElementData(86, 222.0176, "Radon"),
    "Fr": _ElementData(87, 223.0197, "Francium"),
    "Ra": _ElementData(88, 226.0254, "Radium"),
    "Ac": _ElementData(89, 227.0278, "Actinium"),
    "Th": _ElementData(90, 232.0377, "Thorium"),
    "Pa": _ElementData(91, 231.03588, "Protactinium"),
    "U": _ElementData(92, 238.02891, "Uranium"),
    "Np": _ElementData(93, 237.0482, "Neptunium"),
    "Pu": _ElementData(94, 244.0642, "Plutonium"),
    "Am": _ElementData(95, 243.0614, "Americium"),
    "Cm": _ElementData(96, 247.0704, "Curium"),
    "Bk": _ElementData(97, 247.0703, "Berkelium"),
    "Cf": _ElementData(98, 251.0796, "Californium"),
    "Es": _ElementData(99, 252.083, "Einsteinium"),
    "Fm": _ElementData(100, 257.0951, "Fermium"),
    "Md": _ElementData(101, 258.0984, "Mendelevium"),
    "No": _ElementData(102, 259.101, "Nobelium"),
    "Lr": _ElementData(103, 262.1096, "Lawrencium"),
    "Rf": _ElementData(104, 267.1218, "Rutherfordium"),
    "Db": _ElementData(105, 268.1257, "Dubnium"),
    "Sg": _ElementData(106, 271.1339, "Seaborgium"),
    "Bh": _ElementData(107, 272.1383, "Bohrium"),
    "Hs": _ElementData(108, 270.1343, "Hassium"),
    "Mt": _ElementData(109, 276.1516, "Meitnerium"),
}


def atomic_number(element: ElementSymbol) -> int:
    """Atomic number (number of protons) for the given element."""
    return _elements_data[element].number


def atomic_weight(element: ElementSymbol) -> float:
    """Atomic weight (numerically equal to the molar mass in g/mol).

    Sourced from the [molmass](https://pypi.org/project/molmass/) library, which computes these as a
    weighted average of isotopic masses by [natural abundance data from NIST](https://www.nist.gov/pml/atomic-weights-and-isotopic-compositions-relative-atomic-masses).
    This is [the exact dataset used](https://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl?ele=&all=all&ascii=ascii2&isotype=some).

    These values may differ slightly from [IUPAC standard atomic weights](https://www.ciaaw.org/atomic-weights.htm)
    for elements where IUPAC defines an interval rather than a single value (e.g. H, C, O, Mg), and
    are not the IUPAC abridged values.
    """
    return _elements_data[element].weight


def element_name(element: ElementSymbol) -> str:
    """Full English name, e.g. "Iron" for "Fe"."""
    return _elements_data[element].name
