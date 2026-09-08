# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

"""Coefficient selection: turning a species into a fitted above-ground NSVB model."""

from collections.abc import Callable, Sequence
from typing import NamedTuple

import numpy as np
import pandas as pd

from isometric_calculation_library.biosphere.allometric_equations.westfall_nsvb.models import (
    DEFAULT_STAND_ORIGIN,
    NsvbCoefficientLevel,
    NsvbComponentModel,
    NsvbModel,
    NsvbModelForm,
    StandOrigin,
    segment_break_diameter_in,
)
from isometric_calculation_library.biosphere.allometric_equations.westfall_nsvb.species import (
    NsvbSpecies,
    get_species,
)
from isometric_calculation_library.biosphere.allometric_equations.westfall_nsvb.tables import (
    load_table,
)

_WOODLAND_JENKINS_SPECIES_GROUP_CODE = 10
"""Jenkins group of woodland junipers and pinyons, which NSVB models outside this framework."""

_WOOD_BARK_COMPONENT = "TT_WDBK_DW_ADJ"
_FOLIAGE_COMPONENT = "FOL_DW"
_AGB_COMPONENT = "TT_DW_ADJ"


def _sigma(component: str, species_code: int) -> float:
    """Residual spread for a component, per species where NSVB reports it, else national.

    Species represented by a single felled tree have no residual degrees of freedom, so
    NSVB publishes an infinite Sigma for them. Those fall back to the national value too,
    which carries no information about the species but at least keeps the propagated
    uncertainty finite.

    Raises:
        KeyError: If the packaged fit statistics carry no national row for the component,
            which would leave the fallback with nothing to return.
    """
    stats = load_table("fit_statistics")
    stats = stats[stats["component"] == component]
    per_species = stats[stats["spcd"] == species_code]
    if not per_species.empty:
        value = float(per_species.iloc[0]["sigma"])
        if np.isfinite(value):
            return value
    national = stats[stats["spcd"].isna()]
    if national.empty:
        raise KeyError(f"No national NSVB Sigma for component {component} (table S12).")
    return float(national.iloc[0]["sigma"])


class _SelectedCoefficients(NamedTuple):
    """One row of a component coefficient table, and how specific that row is."""

    row: pd.Series
    level: NsvbCoefficientLevel


def _select_coefficients(
    table: str,
    species_code: int,
    division: str | None,
    stand_origin: StandOrigin,
) -> _SelectedCoefficients | None:
    """Apply NSVB's coefficient fallback within one component table.

    Returns None when the species has no usable row, which means the caller should fall
    back to the Jenkins species-group table.
    """
    rows = load_table(f"{table}_coefs_spcd")
    rows = rows[rows["spcd"] == species_code]
    if rows.empty:
        return None

    if rows["stdorgcd"].notna().any():
        rows = rows[rows["stdorgcd"] == int(stand_origin)]
        # Stand origin subdivides the species level rather than sitting beside it, so a
        # species fitted by origin but not for this one has no species-level fit to use
        # and belongs in the Jenkins group. The caller reads that off coefficient_level.
        if rows.empty:
            return None

    in_division = rows[rows["division"] == division] if division is not None else rows.iloc[:0]
    if not in_division.empty:
        return _SelectedCoefficients(
            in_division.iloc[0],
            NsvbCoefficientLevel.SPECIES_ECODIVISION,
        )

    # "If a species occurs in an ecodivision not explicitly listed, the entry having no
    # ecodivision noted is used" (Westfall et al. 2024, Results).
    pooled = rows[rows["division"].isna()]
    if pooled.empty:
        return None
    return _SelectedCoefficients(pooled.iloc[0], NsvbCoefficientLevel.SPECIES)


def _component_model(
    table: str,
    component: str,
    species: NsvbSpecies,
    division: str | None,
    stand_origin: StandOrigin,
) -> NsvbComponentModel:
    sigma = _sigma(component, species.species_code)
    selected = _select_coefficients(table, species.species_code, division, stand_origin)

    if selected is None:
        jenkins = load_table(f"{table}_coefs_jenkins")
        jenkins = jenkins[jenkins["jenkins_spgrpcd"] == species.jenkins_species_group_code]
        if jenkins.empty:
            raise KeyError(
                f"No NSVB {component} coefficients for SPCD {species.species_code}: it is absent "
                f"from the species table and its Jenkins group "
                f"{species.jenkins_species_group_code} is not modelled. Only the woodland group "
                f"({_WOODLAND_JENKINS_SPECIES_GROUP_CODE}) lacks a fit; woodland junipers and "
                f"pinyons are handled by separate NSVB equations not implemented here.",
            )
        row = jenkins.iloc[0]
        return NsvbComponentModel(
            model=NsvbModelForm(int(row["model"])),
            a=float(row["a"]),
            b=float(row["b"]),
            c=float(row["c"]),
            sigma=sigma,
            coefficient_level=NsvbCoefficientLevel.JENKINS_GROUP,
            wood_specific_gravity=species.wood_specific_gravity,
            segment_break_in=segment_break_diameter_in(species.species_code),
        )

    row, level = selected

    def optional(field: str) -> float | None:
        value = row[field]
        return None if pd.isna(value) else float(value)

    return NsvbComponentModel(
        model=NsvbModelForm(int(row["model"])),
        a=float(row["a"]),
        b=float(row["b"]),
        c=float(row["c"]),
        sigma=sigma,
        coefficient_level=level,
        a1=optional("a1"),
        b1=optional("b1"),
        c1=optional("c1"),
        wood_specific_gravity=species.wood_specific_gravity,
        segment_break_in=segment_break_diameter_in(species.species_code),
    )


def get_nsvb_model(
    species_code: int,
    *,
    division: str | None = None,
    stand_origin: StandOrigin = DEFAULT_STAND_ORIGIN,
) -> NsvbModel:
    """Build the above-ground NSVB model for a species.

    The two components are looked up independently, because NSVB models more species for
    foliage than for wood and bark - so one component can come from a species-specific
    fit while the other falls back to the Jenkins species group.

    Args:
        species_code: FIA species code.
        division: FIA ecodivision code, e.g. ``"230"`` or ``"M210"``. When the species has
            no fit for that ecodivision, the ecodivision-pooled fit is used.
        stand_origin: Used only by the two species fitted separately by stand origin,
            and defaults to ``DEFAULT_STAND_ORIGIN``.

    Raises:
        KeyError: If the species is not in the NSVB species reference, or has neither
            species-level nor Jenkins-group coefficients.
    """
    species = get_species(species_code)
    return NsvbModel(
        species_code=species_code,
        wood_bark=_component_model(
            "total_biomass",
            _WOOD_BARK_COMPONENT,
            species,
            division,
            stand_origin,
        ),
        foliage=_component_model("foliage", _FOLIAGE_COMPONENT, species, division, stand_origin),
        agb_sigma=_sigma(_AGB_COMPONENT, species_code),
        division=division,
        stand_origin=stand_origin,
    )


def published_divisions(
    species_code: int,
    *,
    stand_origin: StandOrigin = DEFAULT_STAND_ORIGIN,
) -> Sequence[str | None]:
    """Ecodivisions NSVB publishes above-ground coefficients for, for one species.

    ``None`` represents the ecodivision-pooled fit, which every modelled species has, and
    leads the result.

    Args:
        species_code: FIA species code.
        stand_origin: Restricts the result to the ecodivisions fitted for that origin, so
            that every ecodivision returned is one :func:`get_nsvb_model` can resolve.

    Raises:
        KeyError: If the species is not in the NSVB species reference. Without that check an
            unknown code would filter both tables to nothing and return the same ``[None]``
            as a genuinely pooled-only species.
    """
    get_species(species_code)
    divisions = set[str]()
    for table in ("total_biomass", "foliage"):
        rows = load_table(f"{table}_coefs_spcd")
        rows = rows[rows["spcd"] == species_code]
        if rows["stdorgcd"].notna().any():
            rows = rows[rows["stdorgcd"] == int(stand_origin)]
        divisions.update(rows["division"].dropna().astype(str))
    return [None, *sorted(divisions)]


def create_nsvb_model_generator(
    rng: np.random.Generator,
    species_code: int,
    *,
    stand_origin: StandOrigin = DEFAULT_STAND_ORIGIN,
) -> Callable[[], NsvbModel]:
    """Create a generator that yields NsvbModel instances with resampled coefficients.

    NSVB publishes no coefficient standard errors or covariance, so there is no bootstrap
    replicate table to draw from as there is for Chave 2014. Instead each call samples one
    of the alternative coefficient sets NSVB does publish for the species - its
    ecodivision-specific fits plus the ecodivision-pooled fit - drawn once per model so
    both components stay in the same ecodivision.

    This is a proxy, and a conservative one. Refitting the published Schumacher-Hall model
    on the open felled-tree data (Radtke et al. 2023) and bootstrapping it shows the spread
    across ecodivisions is about 2.5-3x the true parameter sampling spread, because it
    conflates genuine regional differences in allometry with sampling error. It also
    understates uncertainty for the many species with only an ecodivision-pooled fit, where
    this generator has nothing to resample and yields the same model every time.

    For a single tree, residual error dominates parameter error either way: nationally
    ``SD(PE%)`` is 21.5 percent for total AGB against a parameter coefficient of variation
    of 2-4 percent, so ``compute_agb_tonnes_with_error`` matters more than this generator.
    """
    divisions = published_divisions(species_code, stand_origin=stand_origin)

    def generator() -> NsvbModel:
        division = divisions[int(rng.integers(len(divisions)))]
        return get_nsvb_model(species_code, division=division, stand_origin=stand_origin)

    return generator
