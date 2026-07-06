# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

"""Monte Carlo error propagation for tree inventory measurements.

Ported from the R BIOMASS package and Turmalina external notebooks. Propagates
uncertainty through DBH measurement error, height model error, wood density
species-level SD, and biomass-to-carbon ratio.

Allometric model error is handled by the allometric model classes themselves
(e.g. ``ChaveModel.compute_agb_tonnes_with_error``).
"""

import numpy as np

from isometric_calculation_library.biosphere.utils.clipped_normal import clipped_normal
from isometric_calculation_library.biosphere.utils.dbh import (
    DBH_CLIP_MAX_CM,
    DBH_CLIP_MIN_CM,
    perturb_dbh,
)
from isometric_calculation_library.biosphere.utils.height import (
    HEIGHT_CLIP_MAX_M,
    HEIGHT_CLIP_MIN_M,
    perturb_height,
)
from isometric_calculation_library.utils.types import Np1DArray, Np2DArray

DBH_HEIGHT_ERROR_CORRELATION = 0.74
"""Correlation between DBH and height measurement errors (Chave et al. 2004)."""

WOOD_DENSITY_MIN = 0.08
"""Minimum global wood density (g/cm3)."""

WOOD_DENSITY_MAX = 1.39
"""Maximum global wood density (g/cm3)."""

CARBON_RATIO_MEAN = 0.4713
"""Mean biomass-to-carbon ratio (Thomas and Martin 2012)."""

CARBON_RATIO_SD = 0.0206
"""SD of biomass-to-carbon ratio (Thomas and Martin 2012)."""

MONTE_CARLO_VARIANTS: dict[str, str] = {
    "height_cov": "height_m",
    "height_geometric": "height_m",
    "dbh_cov": "dbh_cm",
    "dbh_with_blunders": "dbh_cm",
}
"""Maps alternate Monte Carlo array names back to the base parameter names
expected by the biomass calculation."""


def _perturb_dbh_height_correlated(
    dbh_cm: Np1DArray[np.floating],
    dbh_sd: Np1DArray[np.floating],
    height_m: Np1DArray[np.floating],
    height_sd: Np1DArray[np.floating],
    num_sims: int,
    num_trees: int,
    rng: np.random.Generator,
) -> tuple[Np2DArray[np.floating], Np2DArray[np.floating]]:
    """Draw correlated DBH-height perturbations.

    Returns (dbh_correlated, height_correlated).
    """
    cov_matrix = [
        [1, DBH_HEIGHT_ERROR_CORRELATION],
        [DBH_HEIGHT_ERROR_CORRELATION, 1],
    ]
    # One vectorised draw of shape (num_sims, num_trees, 2); each [.., 0]/[.., 1]
    # pair is a correlated standard-normal sample shared by DBH and height.
    correlated = rng.multivariate_normal([0, 0], cov_matrix, size=(num_sims, num_trees))

    dbh_correlated = (dbh_cm + correlated[:, :, 0] * dbh_sd).clip(DBH_CLIP_MIN_CM, DBH_CLIP_MAX_CM)
    height_correlated = (height_m + correlated[:, :, 1] * height_sd).clip(
        HEIGHT_CLIP_MIN_M,
        HEIGHT_CLIP_MAX_M,
    )

    return dbh_correlated, height_correlated


def _perturb_wood_density(
    wood_density_g_cm3: Np1DArray[np.floating],
    wood_density_sd_g_cm3: Np1DArray[np.floating],
    num_sims: int,
    num_trees: int,
    rng: np.random.Generator,
) -> Np2DArray[np.floating]:
    """Perturb wood density using species-level SD, clipped to global range."""
    return (
        clipped_normal((num_sims, num_trees), rng) * wood_density_sd_g_cm3 + wood_density_g_cm3
    ).clip(
        WOOD_DENSITY_MIN,
        WOOD_DENSITY_MAX,
    )


def _sample_carbon_ratio(
    num_sims: int,
    num_trees: int,
    rng: np.random.Generator,
) -> Np2DArray[np.floating]:
    """Sample biomass-to-carbon ratio from Thomas and Martin 2012."""
    return clipped_normal((num_sims, num_trees), rng) * CARBON_RATIO_SD + CARBON_RATIO_MEAN


def inventory_monte_carlo(
    dbh_cm: Np1DArray[np.floating],
    wood_density_g_cm3: Np1DArray[np.floating],
    wood_density_sd_g_cm3: Np1DArray[np.floating],
    height_m: Np1DArray[np.floating],
    height_logsd: Np1DArray[np.floating],
    num_sims: int,
    rng: np.random.Generator,
) -> dict[str, Np2DArray[np.floating]]:
    """Monte Carlo error propagation for tree inventory measurements.

    Propagates measurement uncertainty only. Allometric model error is
    handled by the model classes (e.g. ``ChaveModel.compute_agb_tonnes_with_error``).

    Returns a dict of (num_sims, num_trees) arrays.
    """
    num_trees = len(dbh_cm)

    dbh_perturbed, dbh_with_blunders, dbh_sd = perturb_dbh(dbh_cm, num_sims, num_trees, rng)
    height_perturbed, height_geometric, height_sd = perturb_height(
        height_m,
        height_logsd,
        num_sims,
        num_trees,
        rng,
    )
    dbh_correlated, height_correlated = _perturb_dbh_height_correlated(
        dbh_cm,
        dbh_sd,
        height_m,
        height_sd,
        num_sims,
        num_trees,
        rng,
    )
    wood_density_perturbed = _perturb_wood_density(
        wood_density_g_cm3,
        wood_density_sd_g_cm3,
        num_sims,
        num_trees,
        rng,
    )
    carbon_ratio = _sample_carbon_ratio(num_sims, num_trees, rng)

    return {
        "dbh_cm": dbh_perturbed,
        "dbh_with_blunders": dbh_with_blunders,
        "wood_density": wood_density_perturbed,
        "height_m": height_perturbed,
        "height_geometric": height_geometric,
        "dbh_cov": dbh_correlated,
        "height_cov": height_correlated,
        "carbon_ratio": carbon_ratio,
    }
