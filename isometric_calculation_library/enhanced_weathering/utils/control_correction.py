# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

"""Ratio-based control correction for enhanced weathering quantification."""

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from isometric_calculation_library.enhanced_weathering.utils.resampling import (
    compute_resampled_means_from_indices,
)
from isometric_calculation_library.enhanced_weathering.utils.types import (
    Np1DArray,
    Np2DArray,
    mass_fraction_column_name,
)


def compute_control_correction_ratio(
    *,
    control_baseline_mg_kg: Np1DArray[np.floating],
    control_end_of_reporting_period_mg_kg: Np1DArray[np.floating],
) -> Np1DArray[np.floating]:
    """Compute ratio-based control correction from control plot cation concentrations.

    cc = C_rp_ctrl / C_bl_ctrl

    No clamping is applied so that full uncertainty propagates through the bootstrap.
    """
    return control_end_of_reporting_period_mg_kg / control_baseline_mg_kg


def bootstrap_control_correction_ratios(
    *,
    ctrl_paired: pd.DataFrame,
    resampled_control_locations: Np2DArray[np.intp],
    elements: Sequence[str],
) -> Mapping[str, Np1DArray[np.floating]]:
    """Bootstrap control correction ratio distributions for each element.

    Bootstraps cation concentrations at control locations across both periods
    and returns the full ratio distribution for each element.  The caller
    decides how to summarise it (e.g. ``np.percentile(ratios["Ca"], 50)``).

    Args:
        ctrl_paired: Paired control DataFrame with ``bl_{col}`` and
            ``rp_{col}`` columns for each element (as produced by
            ``pair_locations``).
        resampled_control_locations: Bootstrap location indices of shape
            ``(n_runs, n_locations)`` from ``generate_bootstrap_location_indices``.
        elements: Element names (e.g. ``["Ca", "Mg"]``).
    """
    ratios = dict[str, Np1DArray[np.floating]]()
    for element in elements:
        col = mass_fraction_column_name(element)
        ctrl_bl_boot = compute_resampled_means_from_indices(
            ctrl_paired[f"bl_{col}"].to_numpy(),
            resampled_control_locations,
        )
        ctrl_rp_boot = compute_resampled_means_from_indices(
            ctrl_paired[f"rp_{col}"].to_numpy(),
            resampled_control_locations,
        )
        ratios[element] = compute_control_correction_ratio(
            control_baseline_mg_kg=ctrl_bl_boot,
            control_end_of_reporting_period_mg_kg=ctrl_rp_boot,
        )
    return ratios
