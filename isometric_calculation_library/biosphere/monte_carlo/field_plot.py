# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

"""Field plot abstraction for inventory-based biomass estimation."""

from dataclasses import dataclass

import numpy as np

from isometric_calculation_library.biosphere.allometric_equations.chave import ChaveModel
from isometric_calculation_library.biosphere.constants import CO2_TO_CARBON_RATIO, M2_PER_HECTARE
from isometric_calculation_library.utils.types import Np1DArray


@dataclass(frozen=True)
class TreeMeasurements:
    """Per-tree measurement arrays for a single field plot.

    All arrays are 1-D with length equal to the number of trees.
    """

    dbh_cm: Np1DArray[np.floating]
    height_m: Np1DArray[np.floating]
    wood_density_g_cm3: Np1DArray[np.floating]
    wood_density_sd_g_cm3: Np1DArray[np.floating]
    height_logsd: Np1DArray[np.floating]


@dataclass(frozen=True)
class FieldPlot:
    """A field plot with tree measurements and plot geometry."""

    plot_id: str
    trees: TreeMeasurements
    plot_size_m: float
    """Side length of the square plot in metres."""

    @property
    def num_trees(self) -> int:
        return len(self.trees.dbh_cm)

    @property
    def plot_area_ha(self) -> float:
        return (self.plot_size_m**2) / M2_PER_HECTARE

    def compute_tco2e_ha(
        self,
        *,
        model: ChaveModel,
        carbon_ratio: float | Np1DArray[np.floating],
    ) -> Np1DArray[np.floating]:
        """Compute tCO2e/ha for this plot using the given allometric model.

        Args:
            model: Allometric model to compute AGB.
            carbon_ratio: Biomass-to-carbon ratio. A scalar applies the same
                ratio to every tree; a 1-D array (one per tree) allows
                per-tree ratios.
        """
        agb_tonnes = model.compute_agb_tonnes(
            self.trees.dbh_cm,
            self.trees.height_m,
            self.trees.wood_density_g_cm3,
        )
        agb_per_ha = agb_tonnes / self.plot_area_ha
        return (agb_per_ha * carbon_ratio * CO2_TO_CARBON_RATIO).clip(0)

    def compute_tco2e_ha_with_error(
        self,
        *,
        model: ChaveModel,
        carbon_ratio: float | Np1DArray[np.floating],
        rng: np.random.Generator,
    ) -> Np1DArray[np.floating]:
        """Compute tCO2e/ha with allometric model error.

        Args:
            model: Allometric model to compute AGB.
            carbon_ratio: Biomass-to-carbon ratio. A scalar applies the same
                ratio to every tree; a 1-D array (one per tree) allows
                per-tree ratios.
            rng: Random number generator for allometric error sampling.
        """
        agb_tonnes = model.compute_agb_tonnes_with_error(
            self.trees.dbh_cm,
            self.trees.height_m,
            self.trees.wood_density_g_cm3,
            rng,
        )
        agb_per_ha = agb_tonnes / self.plot_area_ha
        return (agb_per_ha * carbon_ratio * CO2_TO_CARBON_RATIO).clip(0)
