# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

"""Control correction for enhanced weathering quantification.

Supports both paired and unpaired designs. Both return an additive delta (mg/kg)
that is added to EOY soil concentrations before computing fraction dissolved.
"""

from collections.abc import Mapping, Sequence
from typing import NamedTuple

import numpy as np
import pandas as pd

from isometric_calculation_library.enhanced_weathering.utils.resampling import (
    compute_resampled_means_from_indices,
    generate_bootstrap_location_indices,
)
from isometric_calculation_library.enhanced_weathering.utils.statistical_checks.control_correction_significance import (
    ControlPlotAlkalinityChangeSignificanceTest,
    check_background_weathering_significance_paired,
    check_background_weathering_significance_unpaired,
)
from isometric_calculation_library.enhanced_weathering.utils.types import (
    Np1DArray,
    Np2DArray,
    mass_fraction_column_name,
)
from isometric_calculation_library.utils.elements import ElementSymbol

__all__ = [
    "ControlCorrectionDeltaResult",
    "ControlPlotAlkalinityChangeSignificanceTest",
    "apply_control_correction_delta_paired",
    "apply_control_correction_delta_unpaired",
    "bootstrap_control_correction_ratios",
    "check_background_weathering_significance_paired",
    "check_background_weathering_significance_unpaired",
    "compute_control_correction_ratio",
]


def compute_control_correction_ratio(
    *,
    control_baseline_mg_kg: Np1DArray[np.floating],
    control_end_of_reporting_period_mg_kg: Np1DArray[np.floating],
) -> Np1DArray[np.floating]:
    """Compute ratio-based control correction from control plot cation concentrations.

    cc = C_reporting_period_ctrl / C_baseline_ctrl

    No clamping is applied so that full uncertainty propagates through the bootstrap.
    """
    return control_end_of_reporting_period_mg_kg / control_baseline_mg_kg


def bootstrap_control_correction_ratios(
    *,
    ctrl_paired: pd.DataFrame,
    resampled_control_locations: Np2DArray[np.intp],
    elements: Sequence[ElementSymbol],
) -> Mapping[ElementSymbol, Np1DArray[np.floating]]:
    """Bootstrap control correction ratio distributions for each element.

    Bootstraps cation concentrations at control locations across both periods
    and returns the full ratio distribution for each element.  The caller
    decides how to summarise it (e.g. ``np.percentile(ratios["Ca"], 50)``).

    Args:
        ctrl_paired: Paired control DataFrame with ``baseline_{col}`` and
            ``reporting_period_{col}`` columns for each element (as produced by
            ``pair_locations``).
        resampled_control_locations: Bootstrap location indices of shape
            ``(n_runs, n_locations)`` from ``generate_bootstrap_location_indices``.
        elements: Element names (e.g. ``["Ca", "Mg"]``).
    """
    ratios = dict[ElementSymbol, Np1DArray[np.floating]]()
    for element in elements:
        col = mass_fraction_column_name(element)
        control_baseline_boot = compute_resampled_means_from_indices(
            ctrl_paired[f"baseline_{col}"].to_numpy(),
            resampled_control_locations,
        )
        control_reporting_period_boot = compute_resampled_means_from_indices(
            ctrl_paired[f"reporting_period_{col}"].to_numpy(),
            resampled_control_locations,
        )
        ratios[element] = compute_control_correction_ratio(
            control_baseline_mg_kg=control_baseline_boot,
            control_end_of_reporting_period_mg_kg=control_reporting_period_boot,
        )
    return ratios


class ControlCorrectionDeltaResult(NamedTuple):
    """Combined result of the control correction gate and bootstrap delta distribution."""

    element: ElementSymbol
    """Element (e.g. Ca, Mg)."""

    is_significant: bool
    """Whether the background weathering significance test passed."""

    cc_delta_point: float
    """Point estimate of additive cc delta (mg/kg). 0.0 if not significant."""

    cc_delta_distribution: Np1DArray[np.floating]
    """Full bootstrap distribution of additive cc deltas (mg/kg). All 0.0 if not significant.

    To apply: subtract this from the EOY-minus-RP difference before computing fraction dissolved.
    This shifts the CDR formula from f_d = ((1+m)/m)*(C_post - C_rp)/C_feed
    to f_d = ((1+m)/m)*(C_post - C_rp - delta)/C_feed.
    """

    test_statistic: float
    """Test statistic from the significance test."""

    p_value: float
    """P-value from the significance test."""

    n_control_baseline_samples: int
    """Number of control baseline samples used in the test."""

    n_control_reporting_period_samples: int
    """Number of control reporting period samples used in the test."""


def apply_control_correction_delta_paired(
    *,
    ctrl_paired: pd.DataFrame,
    elements: Sequence[ElementSymbol],
    rng: np.random.Generator,
    n_runs: int,
    alpha: float = 0.05,
    floor_at_zero: bool = True,
) -> list[ControlCorrectionDeltaResult]:
    """Gate on significance then compute the full bootstrap cc distribution (paired design).

    If the paired t-test is significant for an element, bootstraps the full
    additive cc delta distribution from the paired control data. Otherwise returns a
    constant distribution of 0.0 so that uncertainty propagation is uniform
    across all cases.

    Args:
        ctrl_paired: Paired control DataFrame with ``baseline_{col}`` and ``reporting_period_{col}``
            columns for each element.
        elements: Elements to process (e.g. ``["Ca", "Mg"]``).
        rng: Random number generator for bootstrap resampling.
        n_runs: Number of bootstrap iterations.
        alpha: Significance level.
        floor_at_zero: If True, floor each bootstrap delta at 0 so the correction
            only reduces CDR (no single run can inflate it).
    """
    significance_results = check_background_weathering_significance_paired(
        ctrl_paired=ctrl_paired,
        elements=elements,
        alpha=alpha,
    )

    ctrl_paired_finite = ctrl_paired.dropna()
    n_ctrl = len(ctrl_paired_finite)
    if n_ctrl == 0:
        raise ValueError(
            "ctrl_paired contains no rows with finite values for all elements. "
            "Cannot bootstrap control correction distribution.",
        )
    resampled_indices = generate_bootstrap_location_indices(rng, n_ctrl, n_runs)

    output = list[ControlCorrectionDeltaResult]()
    for result in significance_results:
        if result.is_significant:
            col = mass_fraction_column_name(result.element)
            baseline_boot = compute_resampled_means_from_indices(
                ctrl_paired_finite[f"baseline_{col}"].to_numpy(),
                resampled_indices,
            )
            reporting_period_boot = compute_resampled_means_from_indices(
                ctrl_paired_finite[f"reporting_period_{col}"].to_numpy(),
                resampled_indices,
            )
            # Additive delta: background change = reporting_period - baseline (may be positive or negative)
            cc_dist = reporting_period_boot - baseline_boot
            if floor_at_zero:
                cc_dist = np.maximum(cc_dist, 0.0)
        else:
            cc_dist = np.zeros(n_runs)

        output.append(
            ControlCorrectionDeltaResult(
                element=result.element,
                is_significant=result.is_significant,
                cc_delta_point=result.mean_reporting_period - result.mean_baseline
                if result.is_significant
                else 0.0,
                cc_delta_distribution=cc_dist,
                test_statistic=result.t_statistic,
                p_value=result.p_value,
                n_control_baseline_samples=result.n_baseline_samples,
                n_control_reporting_period_samples=result.n_reporting_period_samples,
            ),
        )

    return output


def apply_control_correction_delta_unpaired(
    *,
    control_reporting_period_samples: pd.DataFrame,
    control_baseline_samples: pd.DataFrame,
    elements: Sequence[ElementSymbol],
    rng: np.random.Generator,
    n_runs: int,
    alpha: float = 0.05,
    floor_at_zero: bool = True,
) -> list[ControlCorrectionDeltaResult]:
    """Gate on significance then bootstrap the additive control delta distribution (unpaired design).

    If the one-sided Welch's t-test is significant (control reporting period < control baseline),
    bootstraps the full delta distribution by independently resampling control reporting period and
    control baseline populations. delta = mean(baseline_resamp) - mean(reporting_period_resamp), optionally
    floored at 0.0 to prevent any single bootstrap run from inflating CDR.

    To apply the correction: add ``cc_delta_distribution`` to the reporting period soil concentration
    array before calling ``compute_fraction_dissolved``. This implements:
    ``f_d = ((1+m)/m) * (C_post - C_rp - delta) / C_feed``

    Args:
        control_reporting_period_samples: Control reporting period samples DataFrame.
        control_baseline_samples: Control baseline samples DataFrame.
        elements: Elements to process (e.g. ``["Ca", "Mg"]``).
        rng: Random number generator for bootstrap resampling.
        n_runs: Number of bootstrap iterations.
        alpha: Significance level.
        floor_at_zero: If True, floor each bootstrap delta at 0 so the correction
            only reduces CDR (no single run can inflate it).
    """
    significance_results = check_background_weathering_significance_unpaired(
        control_reporting_period_samples=control_reporting_period_samples,
        control_baseline_samples=control_baseline_samples,
        elements=elements,
        alpha=alpha,
    )

    output = list[ControlCorrectionDeltaResult]()
    for result in significance_results:
        if result.is_significant:
            col = mass_fraction_column_name(result.element)
            reporting_period_vals = (
                control_reporting_period_samples[col].dropna().to_numpy(dtype=float)
            )
            baseline_vals = control_baseline_samples[col].dropna().to_numpy(dtype=float)
            reporting_period_boot = rng.choice(
                reporting_period_vals,
                size=(n_runs, len(reporting_period_vals)),
                replace=True,
            ).mean(axis=1)
            baseline_boot = rng.choice(
                baseline_vals,
                size=(n_runs, len(baseline_vals)),
                replace=True,
            ).mean(axis=1)
            cc_dist = baseline_boot - reporting_period_boot
            if floor_at_zero:
                cc_dist = np.maximum(cc_dist, 0.0)
        else:
            cc_dist = np.zeros(n_runs)

        output.append(
            ControlCorrectionDeltaResult(
                element=result.element,
                is_significant=result.is_significant,
                cc_delta_point=result.mean_baseline - result.mean_reporting_period
                if result.is_significant
                else 0.0,
                cc_delta_distribution=cc_dist,
                test_statistic=result.t_statistic,
                p_value=result.p_value,
                n_control_baseline_samples=result.n_baseline_samples,
                n_control_reporting_period_samples=result.n_reporting_period_samples,
            ),
        )

    return output
