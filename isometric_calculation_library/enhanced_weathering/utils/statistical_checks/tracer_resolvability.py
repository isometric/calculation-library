# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

"""Tracer resolvability check for enhanced weathering."""

import numpy as np
import pandas as pd

from isometric_calculation_library.enhanced_weathering.utils.spatial import PlotType
from isometric_calculation_library.enhanced_weathering.utils.tracer import ImmobileTracer
from isometric_calculation_library.enhanced_weathering.utils.types import (
    mass_fraction_column_name,
)
from isometric_calculation_library.utils.types import Np1DArray


def calculate_tracer_resolvability(
    *,
    soil_mass_kg: float,
    feedstock_mass_kg: float,
    feedstock_tracer_mg_kg: Np1DArray[np.floating],
    baseline_treatment_tracer_mg_kg: Np1DArray[np.floating],
) -> float:
    """Calculate resolvability index for an immobile tracer element.

    Measures how distinguishable the tracer signal is from background noise.
    Higher values indicate better resolvability.

    Uses standard errors (std / sqrt(n)) for both feedstock and soil
    uncertainties.

    Args:
        soil_mass_kg: Mass of soil in the sampling volume.
        feedstock_mass_kg: Mass of feedstock applied.
        feedstock_tracer_mg_kg: Feedstock tracer concentrations
            (one value per sample).
        baseline_treatment_tracer_mg_kg: Baseline treatment soil tracer
            concentrations (one value per sample).
    """
    total_mass_kg = feedstock_mass_kg + soil_mass_kg
    if not total_mass_kg > 0:
        raise ValueError(
            "Tracer resolvability requires a positive total mass (feedstock + soil), "
            f"got feedstock_mass_kg={feedstock_mass_kg} and soil_mass_kg={soil_mass_kg}. "
            "This typically means a plot type with no assigned area or samples (e.g. "
            "an absent plot type defaulting its area to zero); the feedstock mixing "
            "fraction is otherwise undefined and would silently yield a NaN index.",
        )
    feedstock_mass_fraction = feedstock_mass_kg / total_mass_kg
    mean_feedstock_tracer = float(np.mean(feedstock_tracer_mg_kg))
    mean_soil_tracer = float(np.mean(baseline_treatment_tracer_mg_kg))
    feedstock_tracer_standard_error = float(
        np.std(feedstock_tracer_mg_kg) / np.sqrt(len(feedstock_tracer_mg_kg)),
    )
    soil_tracer_standard_error = float(
        np.std(baseline_treatment_tracer_mg_kg) / np.sqrt(len(baseline_treatment_tracer_mg_kg)),
    )

    signal = abs(feedstock_mass_fraction * (mean_feedstock_tracer - mean_soil_tracer))
    noise = (
        2 * soil_tracer_standard_error
        + (feedstock_tracer_standard_error - soil_tracer_standard_error) * feedstock_mass_fraction
    )
    if not noise > 0:
        raise ValueError(
            f"Tracer resolvability noise term must be positive, got {noise}. The noise "
            "term is soil_se*(2 - feedstock_mass_fraction) + feedstock_se*feedstock_mass_fraction "
            "with feedstock_mass_fraction in [0, 1], so it can only reach zero when both "
            "standard errors are zero: single-sample or identical-value inputs giving a "
            "zero standard error. The tracer signal cannot be resolved from measurement noise.",
        )
    return signal / noise


def build_tracer_resolvability_df(
    *,
    baseline_samples: pd.DataFrame,
    feedstock_tracer_values: Np1DArray[np.floating],
    bulk_density_values: Np1DArray[np.floating],
    area_ha: float,
    application_rate_kg_ha: float,
    tracer: ImmobileTracer,
    sampling_depth_cm: float,
    plot_type: PlotType = "treatment",
) -> pd.DataFrame:
    """Compute tracer resolvability index and return as a DataFrame.

    Args:
        baseline_samples: Baseline soil samples for the given plot type.
        feedstock_tracer_values: Feedstock tracer concentrations in mg/kg.
        bulk_density_values: Bulk density measurements in kg/m3.
        area_ha: Area of the plot in hectares.
        application_rate_kg_ha: Known feedstock application rate in kg/ha.
        tracer: Immobile tracer element used.
        sampling_depth_cm: Sampling depth in cm.
        plot_type: Plot type label (e.g. treatment or deployment).
    """
    tracer_col = mass_fraction_column_name(tracer)
    mean_bd_kg_m3 = float(np.mean(bulk_density_values))
    soil_mass_per_ha_kg = mean_bd_kg_m3 * sampling_depth_cm * 100
    soil_mass_kg = soil_mass_per_ha_kg * area_ha
    feedstock_mass_kg = application_rate_kg_ha * area_ha
    baseline_tracer_values = baseline_samples[tracer_col].dropna().to_numpy()

    resolvability = calculate_tracer_resolvability(
        soil_mass_kg=soil_mass_kg,
        feedstock_mass_kg=feedstock_mass_kg,
        feedstock_tracer_mg_kg=feedstock_tracer_values,
        baseline_treatment_tracer_mg_kg=baseline_tracer_values,
    )
    return pd.DataFrame([
        {
            "plot_type": plot_type,
            "resolvability_index": resolvability,
            "soil_mass_kg": soil_mass_kg,
            "feedstock_mass_kg": feedstock_mass_kg,
            "n_baseline_samples": len(baseline_tracer_values),
        },
    ])
