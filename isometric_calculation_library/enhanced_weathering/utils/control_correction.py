# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

"""Control correction for enhanced weathering quantification.

Prefer the additive delta (``apply_control_correction_delta_*``), which returns a signed
mg/kg offset — positive means the control lost cations — subtracted when computing fraction
dissolved.

``compute_control_correction_ratio`` and ``bootstrap_control_correction_ratios`` are the
older multiplicative form and are deprecated.
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
    ControlPlotChangeSignificanceTest,
    UnpairedControlPlotChangeSignificanceTest,
    check_background_weathering_significance_paired,
    check_background_weathering_significance_unpaired,
)
from isometric_calculation_library.enhanced_weathering.utils.types import (
    mass_fraction_column_name,
)
from isometric_calculation_library.utils.elements import ElementSymbol
from isometric_calculation_library.utils.types import Np1DArray, Np2DArray

__all__ = [
    "ControlCorrectionDeltaResult",
    "ControlPlotChangeSignificanceTest",
    "UnpairedControlPlotChangeSignificanceTest",
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

    Prefer ``apply_control_correction_delta_paired`` for new quantification.
    """
    return control_end_of_reporting_period_mg_kg / control_baseline_mg_kg


def bootstrap_control_correction_ratios(
    *,
    ctrl_paired: pd.DataFrame,
    resampled_control_locations: Np2DArray[np.intp],
    elements: Sequence[ElementSymbol],
) -> Mapping[ElementSymbol, Np1DArray[np.floating]]:
    """Bootstrap control correction ratio distributions for each element.

    Bootstraps cation concentrations at control locations across both periods and returns
    the full ratio distribution for each element. The caller decides how to summarise it
    (e.g. ``np.percentile(ratios["Ca"], 50)``).

    Prefer ``apply_control_correction_delta_paired`` for new quantification.

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
    """Point estimate of additive cc delta (mg/kg). 0.0 if not significant.

    Signed so that a positive delta means the control *lost* cations
    (delta = mean(C_baseline_ctrl) - mean(C_reporting_period_ctrl)).
    """

    cc_delta_distribution: Np1DArray[np.floating]
    """Full bootstrap distribution of additive cc deltas (mg/kg). All 0.0 if not significant.

    Pass this to ``compute_fraction_dissolved`` as ``control_correction_delta_mg_kg``:
    ``f_d = ((1+m)/m) * (C_post - C_rp - delta) / C_feed``. A positive delta therefore
    reduces the fraction dissolved, which is the intended effect when the control shows
    background cation loss.
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

    If the paired significance test passes for an element, bootstraps the full
    additive cc delta distribution from the paired control data. Otherwise returns a
    constant distribution of 0.0 so that uncertainty propagation is uniform
    across all cases.

    delta = mean(baseline_resamp) - mean(reporting_period_resamp), so a positive delta
    means the control lost cations and the correction reduces CDR.

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
            # Positive delta means the control lost cations, which reduces CDR.
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
                test_statistic=result.statistic,
                p_value=result.p_value,
                n_control_baseline_samples=result.n_pairs,
                n_control_reporting_period_samples=result.n_pairs,
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

    If the two-sided significance test passes, bootstraps the full delta distribution by
    independently resampling the control reporting-period and control baseline populations.
    delta = mean(baseline_resamp) - mean(reporting_period_resamp), so a positive delta means
    the control lost cations.

    Pass ``cc_delta_distribution`` to ``compute_fraction_dissolved`` as
    ``control_correction_delta_mg_kg``, which implements:
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
                test_statistic=result.statistic,
                p_value=result.p_value,
                n_control_baseline_samples=result.n_baseline_samples,
                n_control_reporting_period_samples=result.n_reporting_period_samples,
            ),
        )

    return output
