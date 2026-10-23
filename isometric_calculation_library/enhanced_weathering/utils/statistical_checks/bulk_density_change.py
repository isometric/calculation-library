# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

"""Bulk density increase check between sampling events, for total cation accounting (TCA).

A TCA design measures three sampling events: baseline (t₀), post-application (t₁) and the end
of the reporting period (t₂). If bulk density increases significantly from one event to the
next, one density no longer describes both, and the stocks behind the weathered fraction need
separate densities.

``check_bulk_density_increase`` runs the one-tailed test for one pair of events.
``resolve_bulk_density_groups`` takes the t₀ → t₁ and t₁ → t₂ tests and decides which events
share a density:

| t₀ → t₁         | t₁ → t₂         | Densities for t₀ / t₁ / t₂ |
|-----------------|-----------------|----------------------------|
| No increase     | No increase     | Pool all three             |
| Significant     | No increase     | ρ₀ / ρ₁₂ / ρ₁₂             |
| No increase     | Significant     | ρ₀₁ / ρ₀₁ / ρ₂             |
| Significant     | Significant     | ρ₀ / ρ₁ / ρ₂               |

"Significant" means a significant increase. "No increase" covers no significant increase and
not testable.

``pool_bulk_density_by_group`` then gathers each group's measurements, ready to bootstrap.
"""

from collections.abc import Mapping
from typing import Literal, NamedTuple

import numpy as np

from isometric_calculation_library.enhanced_weathering.utils.statistical_checks._significance import (
    run_unpaired_significance_test,
)
from isometric_calculation_library.utils.types import Np1DArray

type BulkDensityChangeStatus = Literal[
    "significant_increase",
    "no_significant_increase",
    "not_testable",
]

type TcaSamplingEvent = Literal["baseline", "post_application", "reporting_period"]

TCA_SAMPLING_EVENTS: tuple[TcaSamplingEvent, ...] = (
    "baseline",
    "post_application",
    "reporting_period",
)
"""The three TCA sampling events, in time order."""

_MIN_SAMPLES_FOR_NORMALITY_CHECK = 3


class BulkDensityIncreaseTest(NamedTuple):
    """Outcome of the one-tailed test for a bulk density increase between two events."""

    status: BulkDensityChangeStatus
    """``not_testable`` when either event has fewer than ``min_samples`` measurements."""

    test_name: Literal["welch_t_test", "mann_whitney_u"] | None
    """``None`` when the test was not run."""

    statistic: float | None
    p_value: float | None
    n_earlier: int
    n_later: int
    mean_earlier_kg_m3: float | None
    mean_later_kg_m3: float | None
    significance_level: float

    def report_row(self, *, earlier_event: str, later_event: str) -> dict[str, object]:
        """One row of intermediate output describing this test."""
        return {
            "earlier_event": earlier_event,
            "later_event": later_event,
            "status": self.status,
            "test_name": self.test_name,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "significance_level": self.significance_level,
            "n_earlier": self.n_earlier,
            "n_later": self.n_later,
            "mean_earlier_kg_m3": self.mean_earlier_kg_m3,
            "mean_later_kg_m3": self.mean_later_kg_m3,
        }


def check_bulk_density_increase(
    earlier_kg_m3: Np1DArray[np.floating],
    later_kg_m3: Np1DArray[np.floating],
    *,
    significance_level: float = 0.05,
    min_samples: int = _MIN_SAMPLES_FOR_NORMALITY_CHECK,
) -> BulkDensityIncreaseTest:
    """Test whether bulk density increased significantly from one sampling event to the next.

    One-tailed (H1: later > earlier), so a decrease never counts as significant. Unpaired,
    because density cores are separate measurements, not re-visits of the same location. A
    Shapiro-Wilk check on each event selects Welch's t-test or a Mann-Whitney U test.

    If either event has fewer than ``min_samples`` measurements, returns ``not_testable``
    rather than raising, so the caller can treat the events as one density and report why.

    Args:
        earlier_kg_m3: Bulk density measurements at the earlier event (kg/m³).
        later_kg_m3: Bulk density measurements at the later event (kg/m³).
        significance_level: Significance level for the test.
        min_samples: Fewest measurements per event to run the test. At least 3, which the
            Shapiro-Wilk normality check needs.
    """
    if min_samples < _MIN_SAMPLES_FOR_NORMALITY_CHECK:
        raise ValueError(
            f"min_samples must be at least {_MIN_SAMPLES_FOR_NORMALITY_CHECK}, which the "
            f"Shapiro-Wilk normality check needs, got {min_samples}.",
        )
    earlier = np.asarray(earlier_kg_m3, dtype=float)
    later = np.asarray(later_kg_m3, dtype=float)
    if not (np.all(np.isfinite(earlier)) and np.all(np.isfinite(later))):
        raise ValueError(
            "Bulk density measurements must be finite. Drop missing values before testing, "
            "so the counts reported with the test are the measurements it used.",
        )

    mean_earlier = float(earlier.mean()) if len(earlier) > 0 else None
    mean_later = float(later.mean()) if len(later) > 0 else None
    if len(earlier) < min_samples or len(later) < min_samples:
        return BulkDensityIncreaseTest(
            status="not_testable",
            test_name=None,
            statistic=None,
            p_value=None,
            n_earlier=len(earlier),
            n_later=len(later),
            mean_earlier_kg_m3=mean_earlier,
            mean_later_kg_m3=mean_later,
            significance_level=significance_level,
        )

    test = run_unpaired_significance_test(
        later,
        earlier,
        alternative="greater",
        significance_level=significance_level,
    )
    return BulkDensityIncreaseTest(
        status=(
            "significant_increase"
            if test.p_value < significance_level
            else "no_significant_increase"
        ),
        test_name=test.test_name,
        statistic=test.statistic,
        p_value=test.p_value,
        n_earlier=len(earlier),
        n_later=len(later),
        mean_earlier_kg_m3=mean_earlier,
        mean_later_kg_m3=mean_later,
        significance_level=significance_level,
    )


def resolve_bulk_density_groups(
    baseline_to_post_application: BulkDensityIncreaseTest,
    post_application_to_reporting_period: BulkDensityIncreaseTest,
) -> Mapping[TcaSamplingEvent, str]:
    """Decide which sampling events share a bulk density, from the two increase tests.

    A significant increase separates the two events it compares. Every other pair of
    neighbouring events shares a density, and ``not_testable`` counts as no significant
    increase. Returns a group label per event, made by joining the events in the group with
    ``+``, for example ``post_application+reporting_period``.
    """
    separated_after = [
        baseline_to_post_application.status == "significant_increase",
        post_application_to_reporting_period.status == "significant_increase",
    ]
    groups = list[list[TcaSamplingEvent]]()
    current = list[TcaSamplingEvent]()
    for position, event in enumerate(TCA_SAMPLING_EVENTS):
        current.append(event)
        if position == len(separated_after) or separated_after[position]:
            groups.append(current)
            current = list[TcaSamplingEvent]()

    group_by_event = dict[TcaSamplingEvent, str]()
    for group in groups:
        label = "+".join(group)
        for event in group:
            group_by_event[event] = label
    return group_by_event


def pool_bulk_density_by_group(
    values_by_event: Mapping[TcaSamplingEvent, Np1DArray[np.floating]],
    group_by_event: Mapping[TcaSamplingEvent, str],
) -> Mapping[str, Np1DArray[np.floating]]:
    """Gather each density group's measurements from the events it contains.

    Each group's measurements are sorted, so a bootstrap of them doesn't depend on the order
    the events' measurements arrived in. Raises if a group has no measurements at all, because
    its density would be undefined.
    """
    missing_events = [event for event in TCA_SAMPLING_EVENTS if event not in values_by_event]
    if len(missing_events) > 0:
        raise ValueError(
            f"values_by_event has no entry for {missing_events!r}. Pass an empty array for an "
            "event without bulk density measurements.",
        )

    pooled = dict[str, Np1DArray[np.floating]]()
    for event in TCA_SAMPLING_EVENTS:
        label = group_by_event[event]
        values = np.asarray(values_by_event[event], dtype=float)
        pooled[label] = np.concatenate([pooled[label], values]) if label in pooled else values

    pooled = {label: np.sort(values) for label, values in pooled.items()}
    empty_groups = [label for label, values in pooled.items() if len(values) == 0]
    if len(empty_groups) > 0:
        raise ValueError(
            f"Bulk density group(s) {empty_groups!r} have no measurements, so their density "
            "is undefined.",
        )
    return pooled
