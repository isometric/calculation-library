# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

"""Significance tests for cation change in control plots."""

from collections.abc import Sequence
from typing import Literal, NamedTuple

import numpy as np
import pandas as pd

from isometric_calculation_library.enhanced_weathering.utils.statistical_checks._significance import (
    run_paired_significance_test,
    run_unpaired_significance_test,
)
from isometric_calculation_library.enhanced_weathering.utils.types import mass_fraction_column_name
from isometric_calculation_library.utils.elements import ElementSymbol


class ControlPlotChangeSignificanceTest(NamedTuple):
    """Result of a paired significance test for control-plot cation change.

    Chooses a paired t-test when the paired differences pass a Shapiro-Wilk
    normality check, otherwise a Wilcoxon signed-rank test.
    """

    element: ElementSymbol
    """Element tested (e.g. Ca, Mg)."""

    test_name: Literal["paired_t_test", "wilcoxon_signed_rank"]
    """Which test was run, decided by the normality check on the paired differences."""

    differences_are_normal: bool
    """Whether the paired differences passed the Shapiro-Wilk normality check."""

    statistic: float
    """Test statistic."""

    p_value: float
    """Two-sided p-value: control change in either direction can be significant."""

    is_significant: bool
    """Whether the control-plot change is statistically significant at the given alpha."""

    n_pairs: int
    """Number of valid paired locations used in the test."""

    mean_baseline: float
    """Mean baseline concentration (mg/kg)."""

    mean_reporting_period: float
    """Mean reporting period concentration (mg/kg)."""


def check_background_weathering_significance_paired(
    *,
    ctrl_paired: pd.DataFrame,
    elements: Sequence[ElementSymbol],
    alpha: float = 0.05,
) -> list[ControlPlotChangeSignificanceTest]:
    """Test whether paired control-plot cation change is significant.

    For each element, tests whether the paired baseline-vs-reporting-period change
    at control locations is significant. A Shapiro-Wilk check on the paired
    differences selects the test: a paired t-test when the differences are normal,
    otherwise a Wilcoxon signed-rank test. The test is two-sided because a control
    correction ratio can legitimately move in either direction.

    Args:
        ctrl_paired: Paired control DataFrame with ``baseline_{col}`` and
            ``reporting_period_{col}`` columns (as produced by ``pair_locations``).
        elements: Element names to test (e.g. ``["Ca", "Mg"]``).
        alpha: Significance level for the two-sided test.
    """
    results = list[ControlPlotChangeSignificanceTest]()
    for element in elements:
        col = mass_fraction_column_name(element)
        baseline_col = f"baseline_{col}"
        reporting_period_col = f"reporting_period_{col}"

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
                "at least 3 are required to run the control-correction significance test.",
            )

        # Two-sided: a control-correction ratio can legitimately move in either direction.
        test = run_paired_significance_test(
            reporting_period_valid,
            baseline_valid,
            alternative="two-sided",
            significance_level=alpha,
        )
        results.append(
            ControlPlotChangeSignificanceTest(
                element=element,
                test_name=test.test_name,
                differences_are_normal=test.differences_are_normal,
                statistic=test.statistic,
                p_value=test.p_value,
                is_significant=test.p_value < alpha,
                n_pairs=n_pairs,
                mean_baseline=float(np.mean(baseline_valid)),
                mean_reporting_period=float(np.mean(reporting_period_valid)),
            ),
        )

    return results


class UnpairedControlPlotChangeSignificanceTest(NamedTuple):
    """Result of an unpaired significance test for control-plot cation change.

    Chooses Welch's t-test when both sample distributions pass a Shapiro-Wilk normality
    check, otherwise a Mann-Whitney U test. Two-sided: the control can move either way.
    """

    element: ElementSymbol
    """Element tested (e.g. Ca, Mg)."""

    test_name: Literal["welch_t_test", "mann_whitney_u"]
    """Which test was run, decided by the normality check on the two samples."""

    both_distributions_normal: bool
    """Whether both samples passed the Shapiro-Wilk normality check."""

    statistic: float
    """Test statistic."""

    p_value: float
    """Two-sided p-value: control change in either direction can be significant."""

    is_significant: bool
    """Whether the control-plot change is statistically significant at the given alpha."""

    n_baseline_samples: int
    """Number of valid control baseline samples used."""

    n_reporting_period_samples: int
    """Number of valid control reporting-period samples used."""

    mean_baseline: float
    """Mean baseline concentration (mg/kg)."""

    mean_reporting_period: float
    """Mean reporting period concentration (mg/kg)."""


def check_background_weathering_significance_unpaired(
    *,
    control_reporting_period_samples: pd.DataFrame,
    control_baseline_samples: pd.DataFrame,
    elements: Sequence[ElementSymbol],
    alpha: float = 0.05,
) -> list[UnpairedControlPlotChangeSignificanceTest]:
    """Test whether unpaired control-plot cation change is significant.

    For each element, tests whether the control reporting-period concentrations differ
    from the control baseline. A Shapiro-Wilk check on each sample selects the test:
    Welch's two-sample t-test when both are normal, otherwise Mann-Whitney U. The test
    is two-sided because background cation change is real in either direction.

    Whether an enrichment should then be allowed to increase CDR is a separate decision,
    made by the ``floor_at_zero`` argument on the ``apply_control_correction_delta_*``
    functions.

    Args:
        control_reporting_period_samples: Control reporting-period samples with a
            ``mass_fraction_{element}`` column for each element.
        control_baseline_samples: Control baseline samples with the same columns.
        elements: Element names to test (e.g. ``["Ca", "Mg"]``).
        alpha: Significance level for the two-sided test.
    """
    results = list[UnpairedControlPlotChangeSignificanceTest]()
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

        reporting_period_valid = control_reporting_period_samples[col].to_numpy(dtype=float)
        reporting_period_valid = reporting_period_valid[np.isfinite(reporting_period_valid)]
        baseline_valid = control_baseline_samples[col].to_numpy(dtype=float)
        baseline_valid = baseline_valid[np.isfinite(baseline_valid)]

        if len(reporting_period_valid) < 3 or len(baseline_valid) < 3:
            raise ValueError(
                f"Element {element!r} needs at least 3 valid control reporting-period and "
                f"baseline samples (got {len(reporting_period_valid)} and "
                f"{len(baseline_valid)}) to run the control-correction significance test.",
            )

        # Two-sided: background change is real whichever way the control moved.
        test = run_unpaired_significance_test(
            reporting_period_valid,
            baseline_valid,
            alternative="two-sided",
            significance_level=alpha,
        )
        results.append(
            UnpairedControlPlotChangeSignificanceTest(
                element=element,
                test_name=test.test_name,
                both_distributions_normal=test.both_distributions_normal,
                statistic=test.statistic,
                p_value=test.p_value,
                is_significant=test.p_value < alpha,
                n_baseline_samples=len(baseline_valid),
                n_reporting_period_samples=len(reporting_period_valid),
                mean_baseline=float(np.mean(baseline_valid)),
                mean_reporting_period=float(np.mean(reporting_period_valid)),
            ),
        )

    return results
