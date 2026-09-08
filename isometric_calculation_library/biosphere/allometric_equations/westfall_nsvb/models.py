# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

"""The five NSVB model forms and the fitted models that carry them.

Two components are covered, each with its own coefficient table:

* total above-ground wood and bark, excluding foliage (``TT_WDBK_DW_ADJ``, table S8)
* foliage (``FOL_DW``, table S9)

Their sum is total above-ground biomass (``TT_DW_ADJ``), which NSVB reports fit statistics
for separately - so AGB uncertainty uses that component's own residual spread rather than
combining the two.

The published equations are in imperial units - diameter in inches, height in feet, dry
weight in pounds - and the coefficients are only valid there. The interface here is metric
and converts internally.
"""

from dataclasses import dataclass
from enum import IntEnum, StrEnum, auto
from typing import ClassVar

import numpy as np

from isometric_calculation_library.biosphere.utils.clipped_normal import clipped_normal
from isometric_calculation_library.utils.types import Np1DArray

_CM_PER_INCH = 2.54
_METRES_PER_FOOT = 0.3048
_TONNES_PER_POUND = 0.45359237 / 1000

_SOFTWOOD_SPECIES_CODE_MAX = 300
"""FIA species codes below this are softwoods; at or above it, hardwoods."""

_SOFTWOOD_SEGMENT_BREAK_IN = 9.0
_HARDWOOD_SEGMENT_BREAK_IN = 11.0
"""Segmentation point k of the segmented model (Westfall et al. 2024 Eq 2)."""


class NsvbModelForm(IntEnum):
    """The five functional forms NSVB fits, named and numbered as the paper names them.

    The value is the ``model`` column of the published coefficient tables, so a table row
    converts straight to a member and an unrecognised form fails as the table is read.
    """

    SCHUMACHER_HALL = 1
    SEGMENTED = 2
    CONTINUOUSLY_VARIABLE = 3
    MODIFIED_WILEY = 4
    MODIFIED_SCHUMACHER_HALL = 5


class NsvbCoefficientLevel(StrEnum):
    """How specific the coefficient set backing a component model is.

    NSVB falls back down this list: a species-and-ecodivision fit is preferred, then a
    fit pooled across ecodivisions, then the Jenkins species-group fit for species with
    too few sampled trees to model individually. The three levels are keyed in NSVB's
    tables by ``SPCD`` and ``DIVISION``, by ``SPCD`` alone, and by ``SPGRPCD``.
    """

    SPECIES_ECODIVISION = auto()
    SPECIES = auto()
    JENKINS_GROUP = auto()


class StandOrigin(IntEnum):
    """Whether a stand was planted by people or regenerated on its own.

    This is FIA's stand origin code (``STDORGCD``). Plantation trees are grown at a chosen
    spacing and often from selected stock, so they put on wood and foliage differently from
    trees that seeded themselves, which is why NSVB fits some species separately by origin.

    It is a property of the stand, not of the individual tree: every tree in a planted stand
    is ``PLANTED``, including any that seeded itself among them.

    Only slash pine (SPCD 111) and loblolly pine (SPCD 131) are fitted separately by
    stand origin; for every other species the coefficients are pooled over both.
    """

    NATURAL = 0
    """Naturally regenerated - the stand seeded or coppiced itself."""
    PLANTED = 1
    """Artificially regenerated - the stand was planted or direct-seeded."""


DEFAULT_STAND_ORIGIN = StandOrigin.PLANTED
"""Stand origin assumed when the caller does not give one.

Only slash and loblolly pine are fitted separately by origin, and the reforestation
inventories this serves are planted stands, so planted is assumed rather than demanded.
Pass ``StandOrigin.NATURAL`` explicitly for naturally regenerated stands.
"""


def is_softwood(species_code: int) -> bool:
    """Whether an FIA species code denotes a softwood, which several NSVB forms switch on."""
    return species_code < _SOFTWOOD_SPECIES_CODE_MAX


def segment_break_diameter_in(species_code: int) -> float:
    """Segmentation point k (inches) of the segmented model for a species."""
    return _SOFTWOOD_SEGMENT_BREAK_IN if is_softwood(species_code) else _HARDWOOD_SEGMENT_BREAK_IN


@dataclass(frozen=True)
class NsvbComponentModel:
    """One NSVB component equation, with the coefficients and residual spread it was fitted with.

    ``model`` selects the functional form (Westfall et al. 2024 Eqs 1-5); coefficients a
    form does not use are ``None``.
    """

    model: NsvbModelForm
    a: float
    b: float
    c: float
    sigma: float
    """Diameter-weighted residual standard deviation (table S12/S13 ``Sigma``).

    NSVB fits by weighted least squares with 1/D² weights, so residual variance is
    proportional to D² and ``Sigma`` is the residual SD per inch of diameter: a tree of
    diameter D inches has residual SD ``sigma * D`` pounds.
    """
    coefficient_level: NsvbCoefficientLevel
    a1: float | None = None
    b1: float | None = None
    c1: float | None = None
    wood_specific_gravity: float | None = None
    """Green-volume dry-weight wood specific gravity (``WDSG``), used only by model 5."""
    segment_break_in: float = _SOFTWOOD_SEGMENT_BREAK_IN
    """Segmentation point k, used only by model 2."""

    def compute_biomass_pounds(
        self,
        dbh_in: Np1DArray[np.floating],
        height_ft: Np1DArray[np.floating],
    ) -> Np1DArray[np.floating]:
        """Component dry weight in pounds, from diameter in inches and height in feet.

        This is NSVB's native form, useful for cross-checking against FIA outputs.
        """
        match self.model:
            case NsvbModelForm.SCHUMACHER_HALL:
                # Schumacher-Hall (Eq 1).
                return self.a * dbh_in**self.b * height_ft**self.c
            case NsvbModelForm.SEGMENTED:
                # Segmented (Eq 2): the diameter exponent switches from b to b1 at the
                # segmentation point, and the k^(b-b1) factor makes the branches meet there.
                if self.b1 is None:
                    raise ValueError(self._missing_coefficient_message("b1"))
                small = self.a * dbh_in**self.b * height_ft**self.c
                large = (
                    self.a
                    * self.segment_break_in ** (self.b - self.b1)
                    * dbh_in**self.b1
                    * height_ft**self.c
                )
                return np.where(dbh_in < self.segment_break_in, small, large)
            case NsvbModelForm.CONTINUOUSLY_VARIABLE:
                # Continuously variable (Eq 3): the diameter exponent rises with diameter
                # towards an asymptote of a1.
                if self.a1 is None:
                    raise ValueError(self._missing_coefficient_message("a1"))
                if self.c1 is None:
                    raise ValueError(self._missing_coefficient_message("c1"))
                exponent = self.a1 * (1 - np.exp(-self.b * dbh_in)) ** self.c1
                # The exponent vanishes with the diameter, so D^exponent tends to 1 rather
                # than 0 and this form alone has a positive limit, a * H^c, at D = 0. That
                # limit is real and not a floating-point artefact, but a tree of no
                # diameter has no biomass, so zero it as the other four forms do.
                return np.where(dbh_in > 0, self.a * dbh_in**exponent * height_ft**self.c, 0.0)
            case NsvbModelForm.MODIFIED_WILEY:
                # Modified Wiley (Eq 4).
                if self.b1 is None:
                    raise ValueError(self._missing_coefficient_message("b1"))
                return self.a * dbh_in**self.b * height_ft**self.c * np.exp(-self.b1 * dbh_in)
            case NsvbModelForm.MODIFIED_SCHUMACHER_HALL:
                # Modified Schumacher-Hall (Eq 5), the Jenkins species-group fallback.
                if self.wood_specific_gravity is None:
                    raise ValueError(self._missing_coefficient_message("wood_specific_gravity"))
                return self.a * dbh_in**self.b * height_ft**self.c * self.wood_specific_gravity

    def compute_biomass_tonnes(
        self,
        dbh_cm: Np1DArray[np.floating],
        height_m: Np1DArray[np.floating],
    ) -> Np1DArray[np.floating]:
        """Component dry weight in tonnes, from diameter in cm and height in m."""
        pounds = self.compute_biomass_pounds(dbh_cm / _CM_PER_INCH, height_m / _METRES_PER_FOOT)
        return pounds * _TONNES_PER_POUND

    def residual_sd_tonnes(self, dbh_cm: Np1DArray[np.floating]) -> Np1DArray[np.floating]:
        """Per-tree residual standard deviation in tonnes, which scales with diameter."""
        return self.sigma * (dbh_cm / _CM_PER_INCH) * _TONNES_PER_POUND

    def _missing_coefficient_message(self, field: str) -> str:
        return f"NSVB model form {self.model} requires {field}, which is not set."


@dataclass(frozen=True)
class NsvbModel:
    """Above-ground NSVB model for one species, ecodivision and stand origin.

    Build one with :func:`.coefficients.get_nsvb_model` rather than directly, so that the
    coefficient fallback and the fit-statistic lookup stay consistent.
    """

    AGB_CLIP_FACTOR: ClassVar[float] = 2.0
    """Upper clip on perturbed AGB, as a multiple of the deterministic value."""

    species_code: int
    wood_bark: NsvbComponentModel
    """Total above-ground wood and bark, excluding foliage (table S8)."""
    foliage: NsvbComponentModel
    """Foliage (table S9)."""
    agb_sigma: float
    """Diameter-weighted residual SD of total AGB including foliage (``TT_DW_ADJ``).

    Taken from NSVB's own fit statistics for the summed component rather than combined
    from the two parts, which would need an assumption about their correlation.
    """
    division: str | None = None
    """FIA ecodivision code (Cleland et al. 2007; Nowacki et al. 2003 for Alaska)."""
    stand_origin: StandOrigin = DEFAULT_STAND_ORIGIN
    """Stand origin the coefficients were selected for, which only slash and loblolly pine use."""

    def compute_wood_bark_tonnes(
        self,
        dbh_cm: Np1DArray[np.floating],
        height_m: Np1DArray[np.floating],
    ) -> Np1DArray[np.floating]:
        """Above-ground wood and bark dry weight in tonnes, excluding foliage."""
        return self.wood_bark.compute_biomass_tonnes(dbh_cm, height_m)

    def compute_foliage_tonnes(
        self,
        dbh_cm: Np1DArray[np.floating],
        height_m: Np1DArray[np.floating],
    ) -> Np1DArray[np.floating]:
        """Foliage dry weight in tonnes."""
        return self.foliage.compute_biomass_tonnes(dbh_cm, height_m)

    def compute_agb_tonnes(
        self,
        dbh_cm: Np1DArray[np.floating],
        height_m: Np1DArray[np.floating],
    ) -> Np1DArray[np.floating]:
        """Deterministic total above-ground biomass in tonnes, wood plus bark plus foliage."""
        return self.compute_wood_bark_tonnes(dbh_cm, height_m) + self.compute_foliage_tonnes(
            dbh_cm,
            height_m,
        )

    def agb_residual_sd_tonnes(self, dbh_cm: Np1DArray[np.floating]) -> Np1DArray[np.floating]:
        """Per-tree residual standard deviation of AGB in tonnes.

        NSVB's residual error is additive with a standard deviation proportional to
        diameter, so unlike Chave 2014 this needs no Monte Carlo linearisation - the
        published statistic is already an additive standard error.
        """
        return self.agb_sigma * (dbh_cm / _CM_PER_INCH) * _TONNES_PER_POUND

    def compute_agb_tonnes_with_error(
        self,
        dbh_cm: Np1DArray[np.floating],
        height_m: Np1DArray[np.floating],
        rng: np.random.Generator,
    ) -> Np1DArray[np.floating]:
        """AGB with additive heteroscedastic allometric error.

        Draws residuals with standard deviation ``agb_sigma * D``, matching the 1/D²
        weights NSVB was fitted with, and clips to ``[0, 2 * AGB]`` per tree. Small trees
        have the largest relative error, so the lower clip binds for them and the result
        is very slightly biased upwards there.
        """
        agb_tonnes = self.compute_agb_tonnes(dbh_cm, height_m)
        residual = clipped_normal(agb_tonnes.shape, rng) * self.agb_residual_sd_tonnes(dbh_cm)
        return (agb_tonnes + residual).clip(0, self.AGB_CLIP_FACTOR * agb_tonnes)

    def compute_agb_tonnes_with_linearized_error(
        self,
        dbh_cm: Np1DArray[np.floating],
        height_m: Np1DArray[np.floating],
        *,
        allometric_error_tonnes: Np1DArray[np.floating],
    ) -> Np1DArray[np.floating]:
        """AGB with pre-drawn additive allometric error per tree.

        Mirrors ``ChaveModel.compute_agb_tonnes_with_linearized_error`` so the two
        allometries are interchangeable in a Monte Carlo. The caller draws the error
        terms, e.g. ``clipped_normal(shape, rng) * model.agb_residual_sd_tonnes(dbh_cm)``.
        """
        agb_tonnes = self.compute_agb_tonnes(dbh_cm, height_m)
        return (agb_tonnes + allometric_error_tonnes).clip(0, self.AGB_CLIP_FACTOR * agb_tonnes)
