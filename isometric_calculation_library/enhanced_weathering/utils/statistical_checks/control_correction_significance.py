# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

"""Significance tests for alkalinity change in control plots."""

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, ttest_rel

from isometric_calculation_library.enhanced_weathering.utils.types import mass_fraction_column_name
from isometric_calculation_library.utils.elements import ElementSymbol


class ControlPlotAlkalinityChangeSignificanceTest(NamedTuple):
    """Result of a significance test for alkalinity change in control plots."""

    element: ElementSymbol
    """Element tested (e.g. Ca, Mg)."""

    t_statistic: float
    """Test statistic."""

    p_value: float
    """Two-sided p-value for paired designs; one-sided (H1: depletion) for unpaired."""

    is_significant: bool
    """Whether the alkalinity change is statistically significant at the given alpha."""

    n_baseline_samples: int
    """Number of baseline samples used (equals n_reporting_period_samples for paired design)."""

    n_reporting_period_samples: int
    """Number of reporting period samples used (equals n_baseline_samples for paired design)."""

    paired: bool
    """Whether the test used a paired design."""

    mean_baseline: float
    """Mean baseline concentration (mg/kg)."""

    mean_reporting_period: float
    """Mean reporting period concentration (mg/kg)."""


def check_background_weathering_significance_paired(
    *,
    ctrl_paired: pd.DataFrame,
    elements: Sequence[ElementSymbol],
    alpha: float = 0.05,
) -> list[ControlPlotAlkalinityChangeSignificanceTest]:
    """Test whether background weathering in control plots is statistically significant.

    Runs a paired t-test per element on control plot cation concentrations across
    baseline and reporting period.

    Args:
        ctrl_paired: Paired control DataFrame with ``bl_{col}`` and ``rp_{col}``
            columns (as produced by ``pair_locations``).
        elements: Element names to test (e.g. ``["Ca", "Mg"]``).
        alpha: Significance level for the two-sided test.
    """
    results = list[ControlPlotAlkalinityChangeSignificanceTest]()
    for element in elements:
        col = mass_fraction_column_name(element)
        baseline_col = f"bl_{col}"
        reporting_period_col = f"rp_{col}"

        missing = [c for c in (baseline_col, reporting_period_col) if c not in ctrl_paired.columns]
        if missing:
            raise ValueError(
                f"Expected columns {missing!r} not found in ctrl_paired "
                f"(available: {list(ctrl_paired.columns)!r}). "
                f"Ensure ctrl_paired was produced by pair_locations with the correct value_columns.",
            )

        baseline_values = ctrl_paired[baseline_col].to_numpy(dtype=float)
        reporting_period_values = ctrl_paired[reporting_period_col].to_numpy(dtype=float)

        valid = np.isfinite(baseline_values) & np.isfinite(reporting_period_values)
        baseline_valid = baseline_values[valid]
        reporting_period_valid = reporting_period_values[valid]
        n_pairs = len(baseline_valid)

        if n_pairs < 3:
            raise ValueError(
                f"Only {n_pairs} valid paired location(s) found for element {element!r} — "
                "at least 3 are required to run the background weathering significance test.",
            )

        mean_baseline = float(np.mean(baseline_valid))
        mean_reporting_period = float(np.mean(reporting_period_valid))

        t_stat, p_val = ttest_rel(baseline_valid, reporting_period_valid)
        is_significant = float(p_val) < alpha

        results.append(
            ControlPlotAlkalinityChangeSignificanceTest(
                element=element,
                t_statistic=float(t_stat),
                p_value=float(p_val),
                is_significant=is_significant,
                n_baseline_samples=n_pairs,
                n_reporting_period_samples=n_pairs,
                paired=True,
                mean_baseline=mean_baseline,
                mean_reporting_period=mean_reporting_period,
            ),
        )

    return results


def check_background_weathering_significance_unpaired(
    *,
    control_reporting_period_samples: pd.DataFrame,
    control_baseline_samples: pd.DataFrame,
    elements: Sequence[ElementSymbol],
    alpha: float = 0.05,
) -> list[ControlPlotAlkalinityChangeSignificanceTest]:
    """Test whether background weathering is significant using unpaired control samples.

    Applies a one-sided Welch's two-sample t-test (unequal variances) per element,
    testing whether control reporting period concentrations are *lower* than the control
    baseline (``alternative='less'``). Only depletion is actionable: if controls lose
    cations over time, the CDR estimate must be adjusted downward. Enrichment is not
    credited because the unpaired design cannot separate temporal enrichment from spatial
    heterogeneity between control and treatment areas.

    This is the appropriate test when control plots have only a single sampling timepoint
    (reporting period only) and no paired pre-deployment baseline is available. The control
    baseline is typically drawn from treatment plot baseline samples, which represent the
    landscape-wide soil chemistry before any intervention.

    Args:
        control_reporting_period_samples: DataFrame of control reporting period samples
            with a ``mass_fraction_{element}`` column for each element.
        control_baseline_samples: DataFrame of control baseline samples with the same
            columns (e.g. treatment plot baseline samples).
        elements: Element names to test (e.g. ``["Ca", "Mg"]``).
        alpha: Significance level for the one-sided test (protocol requires 0.05).
    """
    results = list[ControlPlotAlkalinityChangeSignificanceTest]()
    for element in elements:
        col = mass_fraction_column_name(element)

        for df, label in (
            (control_reporting_period_samples, "control_reporting_period"),
            (control_baseline_samples, "control_baseline"),
        ):
            if col not in df.columns:
                raise ValueError(
                    f"Column {col!r} not found in {label} samples "
                    f"(available: {list(df.columns)!r}).",
                )

        control_values = control_reporting_period_samples[col].to_numpy(dtype=float)
        baseline_values = control_baseline_samples[col].to_numpy(dtype=float)

        control_valid = control_values[np.isfinite(control_values)]
        baseline_valid = baseline_values[np.isfinite(baseline_values)]

        if len(control_valid) < 3:
            raise ValueError(
                f"Only {len(control_valid)} valid control reporting period sample(s) for element {element!r} — "
                "at least 3 are required.",
            )
        if len(baseline_valid) < 3:
            raise ValueError(
                f"Only {len(baseline_valid)} valid control baseline sample(s) for element {element!r} — "
                "at least 3 are required.",
            )

        mean_reporting_period = float(np.mean(control_valid))
        mean_baseline = float(np.mean(baseline_valid))

        # One-sided: alternative='less' tests H1: mean(control_rp) < mean(control_baseline)
        t_stat, p_val = ttest_ind(
            control_valid,
            baseline_valid,
            equal_var=False,
            alternative="less",
        )
        is_significant = float(p_val) < alpha

        results.append(
            ControlPlotAlkalinityChangeSignificanceTest(
                element=element,
                t_statistic=float(t_stat),
                p_value=float(p_val),
                is_significant=is_significant,
                n_baseline_samples=len(baseline_valid),
                n_reporting_period_samples=len(control_valid),
                paired=False,
                mean_baseline=mean_baseline,
                mean_reporting_period=mean_reporting_period,
            ),
        )

    return results
