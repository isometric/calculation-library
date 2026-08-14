# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

"""Application rate diagnostic check for enhanced weathering.

``build_application_rate_check`` reports a comparison. ``resolve_application_rate`` decides
which rate to quantify with, conservatively preferring the lower of the operational and
soil-derived figures when the two disagree.

Both are agnostic about how the soil-derived rate was obtained. Callers derive it with
whichever method their design supports — see
``application_rate_derivation.derive_application_rate_from_post_application_samples`` and
``derive_application_rate_from_tracer`` — and pass the result in.
"""

from collections.abc import Mapping
from typing import NamedTuple

import numpy as np
import pandas as pd

from isometric_calculation_library.enhanced_weathering.utils.application_rate_derivation import (
    DerivedApplicationRate,
)
from isometric_calculation_library.enhanced_weathering.utils.spatial import PlotType
from isometric_calculation_library.utils.types import Np1DArray


def build_application_rate_check(
    *,
    soil_based_application_rate_bootstrap_replicates_kg_ha: Np1DArray[np.floating],
    known_application_rate_kg_ha: float,
    plot_type: PlotType = "treatment",
    n_std: float = 2.0,
) -> pd.DataFrame:
    """Build the application rate diagnostic DataFrame.

    Compares the tracer-derived bootstrapped application rate distribution
    against the actual known rate to flag large discrepancies.

    Args:
        soil_based_application_rate_bootstrap_replicates_kg_ha: Bootstrapped application rate
            samples in kg/ha derived from soil tracer mass balance.
        known_application_rate_kg_ha: Known feedstock application rate in kg/ha.
        plot_type: Label for the plot type.
        n_std: Number of standard deviations within which the known rate must fall.
            Default is 2; may be relaxed to 3 where scientifically justified.
    """
    soil_based_app_rate_t_ha = soil_based_application_rate_bootstrap_replicates_kg_ha / 1000
    known_app_rate_t_ha = known_application_rate_kg_ha / 1000
    soil_based_app_rate_mean = float(np.mean(soil_based_app_rate_t_ha))
    soil_based_app_rate_std = float(np.std(soil_based_app_rate_t_ha))
    within_n_std = (
        bool(abs(known_app_rate_t_ha - soil_based_app_rate_mean) <= n_std * soil_based_app_rate_std)
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
            f"known_within_{int(n_std) if n_std == int(n_std) else n_std}std": within_n_std,
            "deviation_in_std": float(
                abs(known_app_rate_t_ha - soil_based_app_rate_mean) / soil_based_app_rate_std,
            )
            if soil_based_app_rate_std > 0
            else float("inf"),
        },
    ])


class ApplicationRateDecision(NamedTuple):
    """Which application rate to quantify with, and the evidence for choosing it."""

    rate_kg_ha: float
    """The rate to use downstream: the operational rate, or the soil-derived median.

    Always a scalar. The soil-derived rate is itself a bootstrap distribution, but only its
    median is used when it wins — propagating its full spread would carry the inversion's own
    sampling noise into every rate-dependent step downstream, on top of the noise those steps
    already model independently.
    """

    rate_is_operational: bool
    """True when the operational rate is used, False when the soil-derived one is."""

    passes: bool
    """Whether the operational rate agreed with the soil-derived one."""

    report_row: Mapping[str, object]
    """One row of intermediate output describing the comparison and its outcome."""


def resolve_application_rate(
    *,
    derived_rate: DerivedApplicationRate,
    known_application_rate_kg_ha: float,
    n_std: float = 2.0,
) -> ApplicationRateDecision:
    """Compare an operational rate against a soil-derived one and pick which to quantify with.

    Agnostic about the derivation: pass the result of
    ``derive_application_rate_from_post_application_samples`` or
    ``derive_application_rate_from_tracer``. This function owns the summarising, the
    comparison, the choice and the report row, so the derivations carry no reporting logic.

    The acceptance window is ``n_std`` standard deviations of the soil-derived distribution,
    as the protocol specifies. Note that this is a moment-based window over a distribution
    that is skew-prone — these inversions are ratios whose denominator can approach zero — so
    tail replicates inflate the standard deviation and widen the window. A percentile
    interval would be scale-free, but the protocol text defines the check in standard
    deviations and the implementation follows it; ``soil_based_app_rate_p16_t_ha`` /
    ``_p84_t_ha`` are reported so a reviewer can see the distribution's actual shape and spot
    a case where the two would disagree.

    The rate is summarised by its **median** rather than its mean, since the mean of a
    skewed ratio distribution is unstable. If the operational rate falls within the window it
    is accepted unchanged. Otherwise the lower of the two is used, so an overstated
    operational rate cannot inflate a mixing ratio and hence CDR — as a scalar median, not
    the full distribution, so the inversion's own sampling noise doesn't propagate into every
    rate-dependent step downstream on top of what those steps already model.

    The check always runs: a soil-derived rate with no finite replicate raises rather than
    letting the operational rate stand unchecked, since an unverifiable rate that quietly
    passes through is indistinguishable in the outputs from one that was verified.

    Args:
        derived_rate: The soil-derived rate to check against.
        known_application_rate_kg_ha: The operational rate to check. Must be positive.
        n_std: Standard deviations within which the operational rate is accepted. Two per
            the protocol default; may be relaxed to three where scientifically justified.
    """
    if known_application_rate_kg_ha <= 0:
        raise ValueError(
            f"known_application_rate_kg_ha must be positive, got "
            f"{known_application_rate_kg_ha}. A non-positive operational rate cannot be "
            "compared against a soil-derived rate, and would give a zero or inverted "
            "mixing ratio downstream.",
        )

    if n_std <= 0:
        raise ValueError(
            f"n_std must be positive, got {n_std}; a non-positive window would reject "
            "every operational rate regardless of the measurement.",
        )

    known_rate_t_ha = known_application_rate_kg_ha / 1000
    within_column = f"known_within_{int(n_std) if n_std == int(n_std) else n_std}std"

    replicates = np.asarray(derived_rate.rate_distribution_kg_ha, dtype=float)
    finite = replicates[np.isfinite(replicates)]
    if len(finite) == 0:
        raise ValueError(
            f"The {derived_rate.derivation_method!r} rate distribution has no finite "
            f"replicate across its {len(replicates)} replicates, so the operational rate of "
            f"{known_application_rate_kg_ha} kg/ha cannot be checked against it.",
        )

    soil_based_t_ha = finite / 1000
    soil_based_p50_t_ha = float(np.percentile(soil_based_t_ha, 50))
    soil_based_std_t_ha = float(np.std(soil_based_t_ha))
    # A degenerate distribution carries no evidence that a differing operational rate is
    # consistent with it, so it fails rather than trivially passing on a zero-width window.
    passes = (
        bool(abs(known_rate_t_ha - soil_based_p50_t_ha) <= n_std * soil_based_std_t_ha)
        if soil_based_std_t_ha > 0
        else False
    )
    deviation = (
        float(abs(known_rate_t_ha - soil_based_p50_t_ha) / soil_based_std_t_ha)
        if soil_based_std_t_ha > 0
        else float("inf")
    )

    soil_based_p50_kg_ha = soil_based_p50_t_ha * 1000
    use_soil_based = not passes and soil_based_p50_kg_ha < known_application_rate_kg_ha
    effective_rate_kg_ha = soil_based_p50_kg_ha if use_soil_based else known_application_rate_kg_ha

    return ApplicationRateDecision(
        rate_kg_ha=effective_rate_kg_ha,
        rate_is_operational=not use_soil_based,
        passes=passes,
        report_row={
            "derivation_method": derived_rate.derivation_method,
            "n_baseline": derived_rate.n_baseline,
            "n_post_application": derived_rate.n_post_application,
            "paired": derived_rate.paired,
            "known_app_rate_t_ha": known_rate_t_ha,
            "soil_based_app_rate_p50_t_ha": soil_based_p50_t_ha,
            "soil_based_app_rate_std_t_ha": soil_based_std_t_ha,
            "soil_based_app_rate_p16_t_ha": float(np.percentile(soil_based_t_ha, 16)),
            "soil_based_app_rate_p84_t_ha": float(np.percentile(soil_based_t_ha, 84)),
            "deviation_in_std": deviation,
            within_column: passes,
            "rate_is_operational": not use_soil_based,
            "effective_app_rate_t_ha": effective_rate_kg_ha / 1000,
        },
    )
