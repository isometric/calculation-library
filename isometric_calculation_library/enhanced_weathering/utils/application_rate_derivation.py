# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

"""Soil-derived application rate, from post-application baselines or from an immobile tracer.

Both routes answer the same question — how much feedstock did this soil actually receive? —
by inverting the same mixing formula, and both return a ``DerivedApplicationRate`` so callers
can treat them interchangeably. What differs is the measurement that pins the mixing ratio:

- **Post-application baseline (BLP).** Sampled after spreading but before appreciable
  weathering, so the BL-to-BLP cation enrichment is due to the feedstock alone::

      m = (eq_blp - eq_bl) / (eq_feed - eq_blp)

  Every creditable cation the feedstock carries out of Ca, Mg, Na and K is pooled onto a
  charge-equivalent basis and inverted once, rather than inverted one at a time, since they
  dissolve congruently from the same rock and charge equivalents are the basis CDR is
  proportional to.

- **Immobile tracer.** A tracer that does not weather (Zr, Ti) traces the feedstock mass
  directly, so no pre-weathering sample is needed and the reporting-period soil can be used::

      m = (T_rp - T_bl) / (T_feed - T_rp)

Either way ``rate = m * bulk_density * depth_m * 10_000``, and either way the two sampling
events can be matched by location or resampled independently, per the ``paired`` argument.
Pass the result to
``statistical_checks.application_rate.resolve_application_rate``, which owns the comparison
against the operational rate, the choice between them, and the reporting.
"""

from collections.abc import Mapping, Sequence
from typing import NamedTuple

import numpy as np
import pandas as pd

from isometric_calculation_library.enhanced_weathering.utils.conversions import (
    Cation,
    compute_soil_mass_kg_ha,
    convert_cation_kg_to_charge_equivalents,
)
from isometric_calculation_library.enhanced_weathering.utils.data_cleaning import (
    check_measured_values,
)
from isometric_calculation_library.enhanced_weathering.utils.pairing import (
    pair_locations,
    paired_column_names,
    require_complete_pairs,
)
from isometric_calculation_library.enhanced_weathering.utils.resampling import (
    resample_columns_together,
    resample_events_together,
    resample_mean,
)
from isometric_calculation_library.enhanced_weathering.utils.tracer import (
    ImmobileTracer,
    compute_application_rate_from_tracer,
    compute_mass_ratio_from_immobile_tracer,
)
from isometric_calculation_library.enhanced_weathering.utils.types import (
    mass_fraction_column_name,
)
from isometric_calculation_library.utils.types import Np1DArray


class DerivedApplicationRate(NamedTuple):
    """A soil-derived application rate and enough provenance to report it.

    There is no "unavailable" state: a derivation that cannot invert the mass balance raises.
    Deciding whether a unit lacking the required samples should be skipped is the calling
    model's judgement to make explicitly.
    """

    rate_distribution_kg_ha: Np1DArray[np.floating]
    """Bootstrap distribution of the implied rate (kg/ha), with at least one finite replicate.

    Replicates whose mass balance is non-physical are NaN rather than dropped, so the array
    stays aligned with the caller's other bootstrap arrays.
    """

    derivation_method: str
    """How the rate was measured, for the outputs (e.g. ``blp_enrichment``, ``ti_tracer``)."""

    n_baseline: int
    """Baseline samples the inversion used."""

    n_post_application: int
    """Samples from the post-application event the inversion used.

    Which event that is depends on the method: the pre-weathering (BLP) samples for an
    enrichment inversion, the end-of-reporting-period samples for an immobile tracer.
    """

    paired: bool
    """Whether the two sampling events were matched by location before inverting.

    A paired design resamples both events through the same bootstrap indices, so shared
    spatial variance cancels and ``n_baseline`` equals ``n_post_application``. An unpaired
    design resamples each event independently and the two counts can differ.
    """


def pool_cations_as_charge_equivalents(
    concentrations_mg_kg: Mapping[Cation, Np1DArray[np.floating]],
) -> Np1DArray[np.floating]:
    """Sum per-cation concentrations (mg/kg) onto a charge-equivalent basis (eq/kg).

    Expresses several cations as one alkalinity quantity, so a mass balance can be
    inverted once rather than per cation.

    Args:
        concentrations_mg_kg: Concentration array per cation. All arrays must be the same
            length and aligned (element i of each refers to the same sample or replicate).
    """
    if len(concentrations_mg_kg) == 0:
        raise ValueError(
            "concentrations_mg_kg is empty; at least one cation is required to pool "
            "charge equivalents.",
        )
    lengths = {len(values) for values in concentrations_mg_kg.values()}
    if len(lengths) > 1:
        raise ValueError(
            f"Concentration arrays have differing lengths {sorted(lengths)!r}; they must be "
            "aligned so that element i of each array refers to the same sample.",
        )

    total_eq_per_kg = np.zeros(next(iter(lengths)))
    for cation, values in concentrations_mg_kg.items():
        # mg/kg -> kg of cation per kg of soil, then to equivalents of charge.
        cation_kg_per_kg = np.asarray(values, dtype=float) / 1e6
        total_eq_per_kg += convert_cation_kg_to_charge_equivalents(
            cation_kg=cation_kg_per_kg,
            cation=cation,
        )
    return total_eq_per_kg


class _SamplingEvents(NamedTuple):
    """Bootstrapped mean of each sampling event, ready to feed a mass balance."""

    baseline: Mapping[str, Np1DArray[np.floating]]
    """Baseline replicate means, keyed by mass fraction column."""

    post_application: Mapping[str, Np1DArray[np.floating]]
    """Post-application replicate means, keyed by the same columns."""

    n_baseline: int
    n_post_application: int


def _resample_events_together(
    *,
    baseline_samples: pd.DataFrame,
    post_application_samples: pd.DataFrame,
    value_columns: Sequence[str],
    paired: bool,
    min_samples: int,
    n_runs: int,
    rng: np.random.Generator,
) -> _SamplingEvents:
    """Validate the two sampling events and hand them to ``resample_events_together``.

    Owns only what the derivation is entitled to decide: whether to inner-join by location,
    that every value used is measured, and the minimum sample count below which an inversion
    is refused. The resampling itself belongs to ``resampling``.
    """
    if paired:
        pairs = pair_locations(baseline_samples, post_application_samples, value_columns).paired
        require_complete_pairs(pairs, paired_column_names(value_columns))
        if len(pairs) < min_samples:
            raise ValueError(
                f"Only {len(pairs)} baseline/post-application pair(s) found, but at least "
                f"{min_samples} are required to invert the mass balance to an application "
                "rate.",
            )
        # pair_locations names its second argument's columns reporting_period_*, which here
        # holds the post-application samples.
        means = resample_events_together(
            rng,
            baseline_values={col: pairs[f"baseline_{col}"].to_numpy() for col in value_columns},
            reporting_period_values={
                col: pairs[f"reporting_period_{col}"].to_numpy() for col in value_columns
            },
            n_runs=n_runs,
            paired=True,
        )
        return _SamplingEvents(
            baseline=means.baseline,
            post_application=means.reporting_period,
            n_baseline=len(pairs),
            n_post_application=len(pairs),
        )

    def event_values(samples: pd.DataFrame, event: str) -> dict[str, Np1DArray[np.floating]]:
        if len(samples) < min_samples:
            raise ValueError(
                f"Only {len(samples)} {event} sample(s) found, but at least {min_samples} "
                "are required to invert the mass balance to an application rate.",
            )
        return {
            col: check_measured_values(samples, col, source=f"{event} sample")
            for col in value_columns
        }

    means = resample_events_together(
        rng,
        baseline_values=event_values(baseline_samples, "baseline"),
        reporting_period_values=event_values(post_application_samples, "post-application"),
        n_runs=n_runs,
        paired=False,
    )
    return _SamplingEvents(
        baseline=means.baseline,
        post_application=means.reporting_period,
        n_baseline=len(baseline_samples),
        n_post_application=len(post_application_samples),
    )


def derive_application_rate_from_post_application_samples(
    *,
    baseline_samples: pd.DataFrame,
    post_application_samples: pd.DataFrame,
    feedstock_samples: pd.DataFrame,
    bulk_density_values_kg_m3: Np1DArray[np.floating],
    depth_cm: float,
    cations: Sequence[Cation],
    n_runs: int,
    rng: np.random.Generator,
    paired: bool = True,
    min_samples: int = 3,
) -> DerivedApplicationRate:
    """Invert BL-to-BLP cation enrichment to a bootstrapped application rate.

    Args:
        baseline_samples: Pre-application soil samples, with a
            ``measurement_location_reference_id`` column and a mass fraction column per cation.
        post_application_samples: Post-application, pre-weathering (BLP) samples, same columns.
        feedstock_samples: Feedstock composition samples, same cation columns.
        bulk_density_values_kg_m3: Measured bulk densities to bootstrap over (kg/m3).
        depth_cm: Depth of the sampled soil layer, in centimetres.
        cations: Cations to pool (e.g. ``["Ca", "Mg", "Na", "K"]``).
        n_runs: Bootstrap replicates.
        rng: Random number generator.
        paired: Whether the sampling plan revisits the same locations in both events. When
            True the two events are inner-joined by location and resampled through shared
            bootstrap indices, so the spatial variance they share cancels; when False each
            event is resampled independently and the two may have different sample counts.
        min_samples: Minimum paired locations, or samples per event when unpaired, before an
            inversion is attempted.
    """
    method = "blp_enrichment"
    if len(cations) == 0:
        raise ValueError("cations is empty; at least one cation is required.")
    if len(bulk_density_values_kg_m3) == 0:
        raise ValueError(
            "bulk_density_values_kg_m3 is empty; bulk density sets the soil mass the "
            "feedstock mixes into, so the rate inversion cannot proceed without it.",
        )

    events = _resample_events_together(
        baseline_samples=baseline_samples,
        post_application_samples=post_application_samples,
        value_columns=[mass_fraction_column_name(cation) for cation in cations],
        paired=paired,
        min_samples=min_samples,
        n_runs=n_runs,
        rng=rng,
    )
    # The feedstock is one sampling event too: its cations are measured on the same physical
    # rock samples, so they share a draw for the same reason the soil events do.
    feedstock_means = resample_columns_together(
        rng,
        values={
            mass_fraction_column_name(cation): check_measured_values(
                feedstock_samples,
                mass_fraction_column_name(cation),
                source="feedstock sample",
            )
            for cation in cations
        },
        n_runs=n_runs,
        event="feedstock samples",
    )
    baseline_by_cation = dict[Cation, Np1DArray[np.floating]]()
    post_by_cation = dict[Cation, Np1DArray[np.floating]]()
    feedstock_by_cation = dict[Cation, Np1DArray[np.floating]]()
    for cation in cations:
        col = mass_fraction_column_name(cation)
        baseline_by_cation[cation] = events.baseline[col]
        post_by_cation[cation] = events.post_application[col]
        feedstock_by_cation[cation] = feedstock_means[col]

    eq_baseline = pool_cations_as_charge_equivalents(baseline_by_cation)
    eq_post = pool_cations_as_charge_equivalents(post_by_cation)
    eq_feedstock = pool_cations_as_charge_equivalents(feedstock_by_cation)

    bulk_density = resample_mean(rng, bulk_density_values_kg_m3, n_runs)

    # A non-positive denominator means the feedstock is no richer than the post-application
    # soil; no enrichment means no application. Neither yields a recoverable rate.
    denominator = eq_feedstock - eq_post
    with np.errstate(divide="ignore", invalid="ignore"):
        mass_ratio = np.where(
            (denominator > 0) & (eq_post > eq_baseline),
            (eq_post - eq_baseline) / denominator,
            np.nan,
        )
        rate_kg_ha = mass_ratio * compute_soil_mass_kg_ha(
            soil_bulk_density_kg_m3=bulk_density,
            depth_cm=depth_cm,
        )

    if not np.any(np.isfinite(rate_kg_ha)):
        raise ValueError(
            f"No physically recoverable rate from BL-BLP enrichment across {n_runs} bootstrap "
            f"replicates of {events.n_baseline} baseline and {events.n_post_application} "
            "post-application sample(s): every replicate had either no cation enrichment from "
            "baseline to post-application, or a feedstock no richer than the post-application "
            "soil. The samples do not evidence an application.",
        )
    return DerivedApplicationRate(
        rate_distribution_kg_ha=rate_kg_ha,
        derivation_method=method,
        n_baseline=events.n_baseline,
        n_post_application=events.n_post_application,
        paired=paired,
    )


def derive_application_rate_from_tracer(
    *,
    baseline_samples: pd.DataFrame,
    reporting_period_samples: pd.DataFrame,
    feedstock_samples: pd.DataFrame,
    bulk_density_values_kg_m3: Np1DArray[np.floating],
    depth_cm: float,
    tracer: ImmobileTracer,
    n_runs: int,
    rng: np.random.Generator,
    paired: bool = True,
    min_samples: int = 3,
) -> DerivedApplicationRate:
    """Invert an immobile tracer mass balance to a bootstrapped application rate.

    The tracer does not weather, so its enrichment traces feedstock mass directly and no
    pre-weathering sample is needed — the reporting-period soil can be used.

    Args:
        baseline_samples: Pre-application soil samples, with a
            ``measurement_location_reference_id`` column and the tracer's mass fraction column.
        reporting_period_samples: End-of-reporting-period samples, same columns.
        feedstock_samples: Feedstock composition samples, same tracer column.
        bulk_density_values_kg_m3: Measured bulk densities to bootstrap over (kg/m3).
        depth_cm: Depth of the sampled soil layer, in centimetres.
        tracer: Immobile tracer element (``"Zr"`` or ``"Ti"``).
        n_runs: Bootstrap replicates.
        rng: Random number generator.
        paired: Whether the sampling plan revisits the same locations in both events. When
            True the two events are inner-joined by location and resampled through shared
            bootstrap indices, so the spatial variance they share cancels; when False each
            event is resampled independently and the two may have different sample counts.
        min_samples: Minimum paired locations, or samples per event when unpaired, before an
            inversion is attempted.
    """
    method = f"{tracer.lower()}_tracer"
    if len(bulk_density_values_kg_m3) == 0:
        raise ValueError(
            "bulk_density_values_kg_m3 is empty; bulk density sets the soil mass the "
            "feedstock mixes into, so the rate inversion cannot proceed without it.",
        )

    col = mass_fraction_column_name(tracer)
    events = _resample_events_together(
        baseline_samples=baseline_samples,
        post_application_samples=reporting_period_samples,
        value_columns=[col],
        paired=paired,
        min_samples=min_samples,
        n_runs=n_runs,
        rng=rng,
    )
    mass_ratio = compute_mass_ratio_from_immobile_tracer(
        feedstock_tracer_mg_kg=resample_mean(
            rng,
            check_measured_values(feedstock_samples, col, source="feedstock sample"),
            n_runs,
        ),
        soil_baseline_tracer_mg_kg=events.baseline[col],
        soil_end_of_reporting_period_tracer_mg_kg=events.post_application[col],
    )
    # A negative ratio means the soil lost tracer, which an immobile tracer cannot do; it is
    # noise rather than a measurement of application.
    mass_ratio = np.where(mass_ratio > 0, mass_ratio, np.nan)
    rate_kg_ha = compute_application_rate_from_tracer(
        feedstock_soil_mass_ratio=mass_ratio,
        soil_bulk_density_kg_m3=resample_mean(rng, bulk_density_values_kg_m3, n_runs),
        depth_cm=depth_cm,
    )

    if not np.any(np.isfinite(rate_kg_ha)):
        raise ValueError(
            f"No physically recoverable rate from the {tracer} mass balance across {n_runs} "
            f"bootstrap replicates of {events.n_baseline} baseline and "
            f"{events.n_post_application} reporting-period sample(s): every replicate implied "
            f"the soil lost {tracer}, which an immobile tracer cannot do. The samples do not "
            "evidence an application.",
        )
    return DerivedApplicationRate(
        rate_distribution_kg_ha=rate_kg_ha,
        derivation_method=method,
        n_baseline=events.n_baseline,
        n_post_application=events.n_post_application,
        paired=paired,
    )
