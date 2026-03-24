# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

"""Application rate diagnostic check for enhanced weathering."""

import numpy as np
import pandas as pd

from ..spatial import PlotType
from ..types import Np1DArray


def build_application_rate_check(
    *,
    soil_based_application_rate_bootstrap_replicates_kg_ha: Np1DArray[np.floating],
    known_application_rate_kg_ha: float,
    plot_type: PlotType = "treatment",
) -> pd.DataFrame:
    """Build the application rate diagnostic DataFrame.

    Compares the tracer-derived bootstrapped application rate distribution
    against the actual known rate to flag large discrepancies.

    Args:
        soil_based_application_rate_bootstrap_replicates_kg_ha: Bootstrapped application rate
            samples in kg/ha derived from soil tracer mass balance.
        known_application_rate_kg_ha: Known feedstock application rate in kg/ha.
        plot_type: Label for the plot type.
    """
    soil_based_app_rate_t_ha = soil_based_application_rate_bootstrap_replicates_kg_ha / 1000
    known_app_rate_t_ha = known_application_rate_kg_ha / 1000
    soil_based_app_rate_mean = float(np.mean(soil_based_app_rate_t_ha))
    soil_based_app_rate_std = float(np.std(soil_based_app_rate_t_ha))
    within_3std = (
        bool(abs(known_app_rate_t_ha - soil_based_app_rate_mean) <= 3 * soil_based_app_rate_std)
        if soil_based_app_rate_std > 0
        else False
    )
    return pd.DataFrame([
        {
            "plot_type": plot_type,
            "known_app_rate_t_ha": known_app_rate_t_ha,
            "soil_based_app_rate_mean_t_ha": soil_based_app_rate_mean,
            "soil_based_app_rate_std_t_ha": soil_based_app_rate_std,
            "soil_based_app_rate_p5_t_ha": float(np.percentile(soil_based_app_rate_t_ha, 5)),
            "soil_based_app_rate_p16_t_ha": float(np.percentile(soil_based_app_rate_t_ha, 16)),
            "soil_based_app_rate_p84_t_ha": float(np.percentile(soil_based_app_rate_t_ha, 84)),
            "soil_based_app_rate_p95_t_ha": float(np.percentile(soil_based_app_rate_t_ha, 95)),
            "known_within_3std": within_3std,
            "deviation_in_std": float(
                abs(known_app_rate_t_ha - soil_based_app_rate_mean) / soil_based_app_rate_std,
            )
            if soil_based_app_rate_std > 0
            else float("inf"),
        },
    ])
