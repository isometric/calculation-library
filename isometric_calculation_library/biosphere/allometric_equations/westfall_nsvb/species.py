# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

"""FIA species reference lookup, which drives NSVB coefficient selection."""

from dataclasses import dataclass

from more_itertools import one

from isometric_calculation_library.biosphere.allometric_equations.westfall_nsvb.tables import (
    load_table,
)


@dataclass(frozen=True)
class NsvbSpecies:
    """Species reference entry (FIA ``REF_SPECIES``) for the columns NSVB lookup needs."""

    species_code: int
    """FIA species code (``SPCD``), which identifies the species in every NSVB table."""
    common_name: str
    genus: str
    specific_epithet: str
    """The epithet alone, as FIA stores it - ``"echinata"``, or ``"spp."`` for a genus-level code."""
    jenkins_species_group_code: int
    """Jenkins species group (``SPGRPCD``), the coarse grouping NSVB falls back to.

    Jenkins et al. 2003 sorted US trees into ten groups of broadly similar allometry, and NSVB
    fits one equation per group for species it has too few felled trees to fit individually.
    """
    wood_specific_gravity: float
    """Green-volume dry-weight wood specific gravity (``WOOD_SPGR_GREENVOL_DRYWT``)."""

    @property
    def species(self) -> str:
        """Binomial name, spelled as ``wood_density`` spells it, e.g. ``"Pinus echinata"``.

        FIA also codes genera, which come out as ``"Abies spp."`` - its own spelling for them.
        """
        return f"{self.genus} {self.specific_epithet}"


_cache = dict[int, NsvbSpecies]()


def _species_table() -> dict[int, NsvbSpecies]:
    if len(_cache) == 0:
        _cache.update(
            {
                int(row["spcd"]): NsvbSpecies(
                    species_code=int(row["spcd"]),
                    common_name=str(row["common_name"]),
                    genus=str(row["genus"]),
                    specific_epithet=str(row["species"]),
                    jenkins_species_group_code=int(row["jenkins_spgrpcd"]),
                    wood_specific_gravity=float(row["wdsg"]),
                )
                for row in load_table("species_reference").to_dict(orient="records")
            },
        )
    return _cache


def get_species(species_code: int) -> NsvbSpecies:
    """Look up an FIA species reference entry.

    Raises:
        KeyError: If the species code is not in the NSVB species reference.
    """
    table = _species_table()
    if species_code not in table:
        raise KeyError(
            f"SPCD {species_code} not found in the NSVB species reference ({len(table)} species).",
        )
    return table[species_code]


def find_species_code(species: str) -> int:
    """Find the FIA species code for a binomial name, e.g. ``"Pinus echinata"``.

    Takes the same spelling as ``wood_density.get_wood_density``, so a tree type carried by
    activity data reaches either lookup through ``wood_density.tree_type_to_species``. A bare
    genus resolves to FIA's genus-level code for it, since that is the only rank it can mean:
    ``find_species_code("Abies")`` and ``find_species_code("Abies spp.")`` both give SPCD 10.

    FIA codes a species per accepted name in most cases but not all: ten binomials in the
    reference carry two or three codes each, usually a species and a variety it once split
    from. Those are rejected rather than resolved arbitrarily, because the codes can differ
    materially - *Pinus elliottii* is both slash pine (111) and Honduras pine (144), whose
    wood specific gravities are 26% apart. Pass the code to :func:`get_species` directly to
    choose between them.

    Raises:
        KeyError: If no NSVB species matches the name, or if the name is one of the
            ambiguous binomials, in which case the message lists the codes to choose from.
    """
    wanted = species.strip()
    if " " not in wanted:
        wanted = f"{wanted} spp."
    matches = [
        entry
        for entry in _species_table().values()
        if entry.species.casefold() == wanted.casefold()
    ]
    if len(matches) == 0:
        raise KeyError(f"No NSVB species matches '{species}'.")
    if len(matches) > 1:
        candidates = ", ".join(f"{entry.species_code} ({entry.common_name})" for entry in matches)
        raise KeyError(
            f"'{species}' is ambiguous in the NSVB species reference, which gives it "
            f"{len(matches)} codes: {candidates}. Pass one to get_species instead.",
        )
    return one(matches).species_code
