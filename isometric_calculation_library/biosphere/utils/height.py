# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

"""Height measurement error perturbation for Monte Carlo simulations."""

import numpy as np

from ..types import Np1DArray, Np2DArray
from .clipped_normal import clipped_normal

HEIGHT_CLIP_MIN_M = 1.0
"""Minimum plausible height (m)."""

HEIGHT_CLIP_MAX_M = 100.0
"""Maximum plausible height (m)."""

HEIGHT_PROPORTIONAL_ERROR = 0.10
"""Proportional height measurement error (Chave et al. 2004)."""


def perturb_height(
    height_m: Np1DArray[np.floating],
    height_logsd: Np1DArray[np.floating],
    num_sims: int,
    num_trees: int,
    rng: np.random.Generator,
) -> tuple[Np2DArray[np.floating], Np2DArray[np.floating], Np1DArray[np.floating]]:
    """Perturb height with log-normal model error and proportional measurement error.

    Returns (height_perturbed, height_geometric, height_sd).
    """
    height_geometric = (
        np.exp(clipped_normal((num_sims, num_trees), rng) * height_logsd) * height_m
    ).clip(HEIGHT_CLIP_MIN_M, HEIGHT_CLIP_MAX_M)

    height_sd = height_m * HEIGHT_PROPORTIONAL_ERROR
    height_perturbed = (height_m + clipped_normal((num_sims, num_trees), rng) * height_sd).clip(
        HEIGHT_CLIP_MIN_M,
        HEIGHT_CLIP_MAX_M,
    )

    return height_perturbed, height_geometric, height_sd
