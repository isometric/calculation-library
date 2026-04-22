# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

"""DBH measurement error perturbation for Monte Carlo simulations."""

import numpy as np

from ..constants import DBH_ERROR_INTERCEPT, DBH_ERROR_SLOPE
from ..types import Np1DArray, Np2DArray
from .clipped_normal import clipped_normal

BLUNDER_PROBABILITY = 0.05
"""Probability of a gross measurement blunder on any single DBH reading."""

BLUNDER_SD_CM = 4.64
"""Standard deviation (cm) of gross DBH measurement blunders."""

DBH_CLIP_MIN_CM = 1.0
"""Minimum plausible DBH (cm)."""

DBH_CLIP_MAX_CM = 500.0
"""Maximum plausible DBH (cm)."""


def perturb_dbh(
    dbh_cm: Np1DArray[np.floating],
    num_sims: int,
    num_trees: int,
    rng: np.random.Generator,
) -> tuple[Np2DArray[np.floating], Np2DArray[np.floating], Np1DArray[np.floating]]:
    """Perturb DBH with measurement error and gross blunders.

    Returns (dbh_perturbed, dbh_perturbed_with_blunders, dbh_sd).
    """
    dbh_sd = DBH_ERROR_SLOPE * dbh_cm + DBH_ERROR_INTERCEPT
    dbh_err = clipped_normal((num_sims, num_trees), rng) * dbh_sd
    dbh_perturbed = (dbh_cm + dbh_err).clip(DBH_CLIP_MIN_CM, DBH_CLIP_MAX_CM)

    blunder_mask = rng.random((num_sims, num_trees)) < BLUNDER_PROBABILITY
    num_blunders = int(blunder_mask.sum())
    dbh_err[blunder_mask] = clipped_normal(num_blunders, rng) * BLUNDER_SD_CM
    dbh_with_blunders = (dbh_cm + dbh_err).clip(DBH_CLIP_MIN_CM, DBH_CLIP_MAX_CM)

    return dbh_perturbed, dbh_with_blunders, dbh_sd
