# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

"""Chave et al. 2014 allometric equation for above-ground biomass (AGB).

Implements the pantropical model from:
  Chave J, et al. (2014) "Improved allometric models to estimate the
  aboveground biomass of tropical trees." Global Change Biology 20(10):3177-90.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd

from isometric_calculation_library.biosphere.utils.clipped_normal import clipped_normal
from isometric_calculation_library.utils.types import Np1DArray


@dataclass(frozen=True)
class ChaveModel:
    """Chave et al. 2014 pantropical allometric model.

    Two parameterizations are supported, controlled by ``bootstrap_params``:

    Published (bootstrap_params=False, the default):
        AGB (kg) = gain * (wd * H * D²)^power
        where gain=0.0673, power=0.976

    Bootstrap (bootstrap_params=True):
        AGB (kg) = (gain * wd * H * D²)^power
        The R BIOMASS package fits ln(AGB) = power * (intercept + ln(wd * H * D²)),
        so gain = exp(intercept) goes inside the exponent.
    """

    ALLOMETRY_SIGMA: ClassVar[float] = 0.357
    """Residual standard error of the Chave 2014 model (Eq 4, sigma)."""

    gain: float = 0.0673
    power: float = 0.976
    bootstrap_params: bool = False
    """Whether gain/power come from the BIOMASS bootstrap table (gain-inside-exponent form)."""

    def compute_agb_tonnes(
        self,
        dbh_cm: Np1DArray[np.floating],
        height_m: Np1DArray[np.floating],
        wood_density_g_cm3: Np1DArray[np.floating],
    ) -> Np1DArray[np.floating]:
        """Deterministic AGB in tonnes.

        Computes AGB in kg via Chave 2014 Eq 4, then converts to tonnes.
        """
        compound = wood_density_g_cm3 * height_m * dbh_cm**2
        if self.bootstrap_params:
            agb_kg = (self.gain * compound) ** self.power
        else:
            agb_kg = self.gain * compound**self.power
        return agb_kg / 1000

    def compute_agb_tonnes_with_error(
        self,
        dbh_cm: Np1DArray[np.floating],
        height_m: Np1DArray[np.floating],
        wood_density_g_cm3: Np1DArray[np.floating],
        rng: np.random.Generator,
    ) -> Np1DArray[np.floating]:
        """AGB with log-normal multiplicative allometric error (Chave 2014, sigma=0.357)."""
        agb_tonnes = self.compute_agb_tonnes(dbh_cm, height_m, wood_density_g_cm3)
        err_factor = np.exp(clipped_normal(agb_tonnes.shape, rng) * self.ALLOMETRY_SIGMA)
        return (agb_tonnes * err_factor).clip(0)

    def compute_agb_tonnes_with_linearized_error(
        self,
        dbh_cm: Np1DArray[np.floating],
        height_m: Np1DArray[np.floating],
        wood_density_g_cm3: Np1DArray[np.floating],
        *,
        allometric_error_tonnes: Np1DArray[np.floating],
    ) -> Np1DArray[np.floating]:
        """AGB with pre-drawn additive allometric error per tree.

        The caller is responsible for drawing the error terms, e.g.
        ``clipped_normal(shape, rng) * linearize_allometric_se(...)``.
        Clips the result to ``[0, 2 * AGB]`` per tree, which is
        mean-preserving unlike the multiplicative log-normal approach.
        """
        agb_tonnes = self.compute_agb_tonnes(dbh_cm, height_m, wood_density_g_cm3)
        return (agb_tonnes + allometric_error_tonnes).clip(0, 2 * agb_tonnes)


CHAVE_DEFAULT = ChaveModel()
"""Default Chave 2014 model with published coefficients (gain=0.0673, power=0.976)."""

_LINEARIZE_NUM_SIMS = 1000
"""Number of MC draws used to linearize the log-normal allometric error."""


def linearize_allometric_se(
    model: ChaveModel,
    dbh_cm: Np1DArray[np.floating],
    height_m: Np1DArray[np.floating],
    wood_density_g_cm3: Np1DArray[np.floating],
    rng: np.random.Generator,
) -> float:
    """Convert the Chave 2014 log-normal allometric error into an additive SE.

    Runs a short MC with only the allometry error factor varied, computes
    the per-tree biomass deltas, clips to [5th, 95th] percentile, and returns
    the standard deviation. The result is a single scalar SE (in tonnes) that
    can be used with ``ChaveModel.compute_agb_tonnes_with_linearized_error``.

    This mirrors Mombak's ``linearize_allometry_errors`` approach: the additive
    formulation is mean-preserving, unlike the raw multiplicative log-normal.
    """
    baseline = model.compute_agb_tonnes(dbh_cm, height_m, wood_density_g_cm3)
    n_trees = len(dbh_cm)
    err_factors = np.exp(
        clipped_normal((_LINEARIZE_NUM_SIMS, n_trees), rng) * ChaveModel.ALLOMETRY_SIGMA,
    )
    perturbed = baseline * err_factors
    delta = perturbed - baseline
    delta = delta.clip(*np.percentile(delta, [5, 95]))
    return float(np.std(delta))


_CHAVE_BOOTSTRAP_PARAMS_PATH = Path(__file__).parent / "data" / "param_4.csv"


def create_chave_model_generator(
    rng: np.random.Generator,
) -> Callable[[], ChaveModel]:
    """Create a generator that yields ChaveModel instances with parameters sampled
    from the BIOMASS R package bootstrap replicates table (1001 replicates).

    The bootstrap table uses the gain-inside-exponent parameterization:
    ln(AGB) = power * (intercept + ln(wd * H * D²)), so the returned models
    have ``bootstrap_params=True``.
    """
    bootstrap_replicates = pd.read_csv(_CHAVE_BOOTSTRAP_PARAMS_PATH)

    def generator() -> ChaveModel:
        row = bootstrap_replicates.sample(1, replace=True, random_state=rng).iloc[0]
        return ChaveModel(
            gain=float(np.exp(row["intercept"])),
            power=float(row["logagbt"]),
            bootstrap_params=True,
        )

    return generator
