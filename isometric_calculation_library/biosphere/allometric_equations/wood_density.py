# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

"""Species-level wood density lookup from the Global Wood Density Database.

Values are literature-derived mean wood densities (g/cm³) and their
standard deviations, used for Chave 2014 AGB estimation.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class WoodDensityRecord:
    """Wood density for a single species."""

    species: str
    wood_density: float
    """Mean wood density (g/cm³)."""
    wood_density_sd: float
    """Standard deviation of wood density (g/cm³)."""


_WOOD_DENSITY_PATH = Path(__file__).parent / "data" / "wood_density.csv"


def _load_wood_density_table() -> dict[str, WoodDensityRecord]:
    df = pd.read_csv(_WOOD_DENSITY_PATH)
    return {
        row["species"]: WoodDensityRecord(
            species=row["species"],
            wood_density=float(row["wood_density"]),
            wood_density_sd=float(row["wood_density_sd"]),
        )
        for _, row in df.iterrows()
    }


_cache: dict[str, dict[str, WoodDensityRecord]] = {}


def _get_table() -> dict[str, WoodDensityRecord]:
    if "table" not in _cache:
        _cache["table"] = _load_wood_density_table()
    return _cache["table"]


def get_wood_density(species: str) -> WoodDensityRecord:
    """Look up wood density for a species.

    Args:
        species: Binomial species name (e.g. "Cecropia obtusa").

    Raises:
        KeyError: If the species is not in the wood density table.
    """
    table = _get_table()
    if species not in table:
        raise KeyError(
            f"Species '{species}' not found in wood density table. "
            f"Available: {len(table)} species.",
        )
    return table[species]


def list_species() -> list[str]:
    """Return all species names in the wood density table."""
    return sorted(_get_table().keys())


def tree_type_to_species(tree_type: str) -> str:
    """Convert a tree type qualifier name to a binomial species name.

    Handles ``SPECIES_`` and ``GENUS_`` prefixed names as well as bare
    underscore-separated names.  The first word is capitalised; subsequent
    words are lowercased (standard binomial convention).

    Example::

        >>> tree_type_to_species("SPECIES_CECROPIA_OBTUSA")
        'Cecropia obtusa'
    """
    for prefix in ("SPECIES_", "GENUS_"):
        if tree_type.startswith(prefix):
            tree_type = tree_type[len(prefix) :]
            break

    return tree_type.replace("_", " ").capitalize()
