# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

"""Tracer-corrected TCA CDR quantification — Ti tracer, treatment-only.

Uses per-location immobile Ti tracer mixing ratio to estimate feedstock
contribution. CDR is computed as fraction_dissolved x known application rate
for each cation (Ca, Mg), with ratio-based control correction (p50).

Total CDR is scaled by treatment area only (no deployment plots).
No outlier handling is applied. Reported at p16.
"""

from typing import Literal, TypedDict

from isometric_calculation_library.dependencies import geopandas as gpd
from isometric_calculation_library.dependencies import numpy as np
from isometric_calculation_library.dependencies import pandas as pd
from isometric_calculation_library.enhanced_weathering.utils.control_correction import (
    bootstrap_control_correction_ratios,
)
from isometric_calculation_library.enhanced_weathering.utils.conversions import (
    convert_cation_kg_to_co2_kg,
)
from isometric_calculation_library.enhanced_weathering.utils.pairing import pair_locations
from isometric_calculation_library.enhanced_weathering.utils.resampling import (
    bootstrap_bulk_density_paired,
    compute_resampled_means_from_indices,
    generate_bootstrap_location_indices,
    resample_mean,
    summarize_distributions,
)
from isometric_calculation_library.enhanced_weathering.utils.spatial import (
    assign_and_split_by_plot_type,
    calculate_area_hectares_by_plot_type,
)
from isometric_calculation_library.enhanced_weathering.utils.statistical_checks.application_rate import (
    build_application_rate_check,
)
from isometric_calculation_library.enhanced_weathering.utils.statistical_checks.tracer_resolvability import (
    build_tracer_resolvability_df,
)
from isometric_calculation_library.enhanced_weathering.utils.statistical_checks.weathering_signal import (
    run_significance_tests,
)
from isometric_calculation_library.enhanced_weathering.utils.tracer import (
    compute_application_rate_from_tracer,
    compute_fraction_dissolved,
    compute_mass_ratio_from_immobile_tracer,
    compute_post_application_concentration,
)
from isometric_calculation_library.enhanced_weathering.utils.types import (
    Np1DArray,
    mass_fraction_column_name,
)

type _Cation = Literal["Ca", "Mg"]
_CATIONS: list[_Cation] = ["Ca", "Mg"]
_N_BOOTSTRAP_RUNS = 200_000
_SEED = 42
_SAMPLING_DEPTH_CM = 30.0
_TRACER: Literal["Ti"] = "Ti"


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class IntermediateOutputs(TypedDict):
    """Intermediate outputs from CDR quantification."""

    data_cleaning_report: pd.DataFrame
    """Samples dropped during spatial assignment (not inside any plot polygon)."""

    distributions_summary: pd.DataFrame
    """Summary statistics for each bootstrap distribution."""

    area_hectares: pd.DataFrame
    """Area in hectares by plot type."""

    pairing_report: pd.DataFrame
    """Pairing statistics per cation."""

    application_rate_check: pd.DataFrame
    """Comparison of tracer-derived vs actual feedstock application rate."""

    tracer_resolvability: pd.DataFrame
    """Tracer resolvability index for treatment plots."""

    significance_test_results: pd.DataFrame
    """Statistical significance test results per cation."""


class ModelResult(TypedDict):
    """Result from CDR quantification model."""

    result: float
    """CDR at 16th percentile in tonnes CO2."""

    intermediate_outputs: IntermediateOutputs


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_application_rate_diagnostic(
    feedstock_soil_mass_ratio: Np1DArray[np.floating],
    bulk_density_means: Np1DArray[np.floating],
    sampling_depth_cm: float,
    known_application_rate_kg_ha: float,
) -> tuple[Np1DArray[np.floating], pd.DataFrame]:
    """Compute tracer-derived application rate and build diagnostic check."""
    app_rate_boot = compute_application_rate_from_tracer(
        feedstock_soil_mass_ratio=feedstock_soil_mass_ratio,
        soil_bulk_density_kg_m3=bulk_density_means,
        depth_cm=sampling_depth_cm,
    )
    application_rate_check = build_application_rate_check(
        soil_based_application_rate_bootstrap_replicates_kg_ha=app_rate_boot,
        known_application_rate_kg_ha=known_application_rate_kg_ha,
    )
    return app_rate_boot, application_rate_check


def _compute_control_correction_ratios(
    control_baseline: pd.DataFrame,
    control_reporting_period: pd.DataFrame,
    value_columns: list[str],
    cations: list[_Cation],
    rng: np.random.Generator,
    n_bootstrap_runs: int,
) -> tuple[dict[_Cation, float], int, int, int]:
    """Compute ratio-based control correction (p50) for each cation.

    Returns:
        Tuple of (control_correction_ratio_p50, n_control_paired,
        n_control_baseline_only, n_control_reporting_period_only).
    """
    control_pairing = pair_locations(control_baseline, control_reporting_period, value_columns)
    control_paired = control_pairing.paired
    n_control = len(control_paired)
    resampled_control_locations = generate_bootstrap_location_indices(
        rng,
        n_control,
        n_bootstrap_runs,
    )
    control_correction_distributions = bootstrap_control_correction_ratios(
        ctrl_paired=control_paired,
        resampled_control_locations=resampled_control_locations,
        elements=cations,
    )
    # Elements passed in are _Cation values, so the result keys are _Cation.
    control_correction_ratio_p50 = {
        element: float(np.percentile(dist, 50))
        for element, dist in control_correction_distributions.items()
    }
    return (
        control_correction_ratio_p50,  # pyright: ignore[reportReturnType]
        n_control,
        control_pairing.n_baseline_only,
        control_pairing.n_reporting_period_only,
    )


# ---------------------------------------------------------------------------
# Model entry point
# ---------------------------------------------------------------------------


def main(
    *,
    baseline_samples: pd.DataFrame,
    reporting_period_samples: pd.DataFrame,
    feedstock_samples: pd.DataFrame,
    bulk_density_samples_kg_m3: pd.DataFrame,
    plots: gpd.GeoDataFrame,
    known_application_rate_kg_ha: float,
) -> ModelResult:
    """Calculate CDR using Ti tracer-corrected TCA with treatment plots only.

    Uses immobile Ti tracer mass balance to estimate the feedstock-to-soil
    mixing ratio at each paired location, then computes the fraction of
    feedstock cations (Ca, Mg) that dissolved. CDR is derived from the
    dissolved fraction and a known application rate, with ratio-based
    control correction (p50). No outlier handling is applied.

    Args:
        baseline_samples: Baseline (pre-application) soil samples. DataFrame
            with columns ``mass_fraction_ca``, ``mass_fraction_mg``,
            ``mass_fraction_ti`` (all in mg/kg), ``latitude``, ``longitude``,
            and ``measurement_location_reference_id``.
        reporting_period_samples: End-of-reporting-period soil samples.
            Same columns as ``baseline_samples``.
        feedstock_samples: Feedstock geochemistry samples. DataFrame with
            columns ``mass_fraction_ca``, ``mass_fraction_mg``, and
            ``mass_fraction_ti`` (all in mg/kg).
        bulk_density_samples_kg_m3: Field bulk density measurements. DataFrame
            with a ``bulk_density`` column in kg/m3.
        plots: Plot geometries and types. GeoDataFrame with a ``Type``
            column (values: ``"control"``, ``"treatment"``) and a
            ``Geometry`` column containing polygon geometries in EPSG:4326.
            Samples are assigned to plots via spatial join.
        known_application_rate_kg_ha: Known feedstock application rate in kg/ha.
            Used to convert dissolved fraction to absolute CDR.
    """
    rng = np.random.default_rng(_SEED)

    # Bulk density bootstrap (paired: single draw shared across periods)
    bulk_density_values = bulk_density_samples_kg_m3["bulk_density"].dropna().to_numpy()
    bulk_density_means = bootstrap_bulk_density_paired(rng, bulk_density_values, _N_BOOTSTRAP_RUNS)

    # Spatial assignment and splitting (no outlier handling)
    split_result = assign_and_split_by_plot_type(baseline_samples, reporting_period_samples, plots)
    for required_plot_type in ("treatment", "control"):
        if required_plot_type not in split_result.splits:
            raise ValueError(
                f"No samples were assigned to any '{required_plot_type}' plot. "
                f"Assigned plot types: {sorted(split_result.splits.keys())!r}. "
                "Check that plots contain polygons with Type='treatment' and "
                "Type='control', and that sample coordinates fall within them.",
            )
    treatment_baseline, treatment_reporting_period = split_result.splits["treatment"]
    control_baseline, control_reporting_period = split_result.splits["control"]

    # Split-type keys come from a union across periods, so a plot type present in
    # only one period yields a (non-empty, empty) split that passes the key check
    # above but pairs to zero locations downstream and silently collapses the
    # bootstrap to NaN. Require both periods to be populated.
    for plot_type_name, plot_baseline, plot_reporting_period in (
        ("treatment", treatment_baseline, treatment_reporting_period),
        ("control", control_baseline, control_reporting_period),
    ):
        if plot_baseline.empty or plot_reporting_period.empty:
            raise ValueError(
                f"Plot type {plot_type_name!r} must have both baseline and "
                f"reporting-period samples, got {len(plot_baseline)} baseline and "
                f"{len(plot_reporting_period)} reporting-period samples. A plot type "
                "present in only one period pairs to zero locations and would "
                "otherwise yield a silent NaN result.",
            )

    data_cleaning_report = pd.DataFrame([
        {
            "step": "spatial_assignment",
            "period": "baseline",
            "n_input": len(baseline_samples),
            "n_dropped": split_result.n_baseline_unassigned,
            "n_retained": len(baseline_samples) - split_result.n_baseline_unassigned,
        },
        {
            "step": "spatial_assignment",
            "period": "reporting_period",
            "n_input": len(reporting_period_samples),
            "n_dropped": split_result.n_reporting_period_unassigned,
            "n_retained": len(reporting_period_samples)
            - split_result.n_reporting_period_unassigned,
        },
    ])

    area_hectares = calculate_area_hectares_by_plot_type(
        plots,
        plot_types=("control", "treatment"),
    )
    # The split guard above already proved treatment polygons exist, and
    # calculate_area_hectares_by_plot_type uses the same plots + case-folding,
    # so area_hectares["treatment"] is always populated here.
    treatment_area = area_hectares["treatment"]
    if treatment_area <= 0:
        raise ValueError(
            f"treatment_area must be positive, got {treatment_area}. Check the plot geometries.",
        )

    tracer_col = mass_fraction_column_name(_TRACER)
    ca_col = mass_fraction_column_name("Ca")
    mg_col = mass_fraction_column_name("Mg")
    value_columns = [tracer_col, ca_col, mg_col]

    feedstock_tracer_values = feedstock_samples[tracer_col].dropna().to_numpy()

    # Tracer resolvability
    tracer_resolvability_df = build_tracer_resolvability_df(
        baseline_samples=treatment_baseline,
        feedstock_tracer_values=feedstock_tracer_values,
        bulk_density_values=bulk_density_values,
        area_ha=treatment_area,
        application_rate_kg_ha=known_application_rate_kg_ha,
        tracer=_TRACER,
        sampling_depth_cm=_SAMPLING_DEPTH_CM,
    )

    # Feedstock bootstrap
    feedstock_tracer_boot = resample_mean(
        rng,
        feedstock_samples[tracer_col].dropna().to_numpy(),
        _N_BOOTSTRAP_RUNS,
    )
    feedstock_boot_by_cation: dict[_Cation, Np1DArray[np.floating]] = {
        "Ca": resample_mean(rng, feedstock_samples[ca_col].dropna().to_numpy(), _N_BOOTSTRAP_RUNS),
        "Mg": resample_mean(rng, feedstock_samples[mg_col].dropna().to_numpy(), _N_BOOTSTRAP_RUNS),
    }

    # Control correction (ratio p50)
    (
        control_correction_ratio_p50,
        n_control,
        n_control_baseline_only,
        n_control_reporting_period_only,
    ) = _compute_control_correction_ratios(
        control_baseline=control_baseline,
        control_reporting_period=control_reporting_period,
        value_columns=value_columns,
        cations=_CATIONS,
        rng=rng,
        n_bootstrap_runs=_N_BOOTSTRAP_RUNS,
    )

    # Treatment pairing and report
    pairing_rows = list[dict[str, str | int]]()
    distributions = dict[str, Np1DArray[np.floating]]()
    distributions["bulk_density_kg_m3"] = bulk_density_means

    treatment_pairing = pair_locations(
        treatment_baseline,
        treatment_reporting_period,
        value_columns,
    )
    treatment_paired = treatment_pairing.paired
    n_treatment = len(treatment_paired)

    for cation in _CATIONS:
        pairing_rows.append({
            "area_type": "treatment",
            "cation": cation,
            "n_paired": n_treatment,
            "n_baseline_only": treatment_pairing.n_baseline_only,
            "n_reporting_period_only": treatment_pairing.n_reporting_period_only,
        })

    for cation in _CATIONS:
        pairing_rows.append({
            "area_type": "control",
            "cation": cation,
            "n_paired": n_control,
            "n_baseline_only": n_control_baseline_only,
            "n_reporting_period_only": n_control_reporting_period_only,
        })

    # Shared bootstrap indices for treatment
    resampled_treatment_locations = generate_bootstrap_location_indices(
        rng,
        n_treatment,
        _N_BOOTSTRAP_RUNS,
    )

    # Tracer bootstrap
    tracer_baseline_boot = compute_resampled_means_from_indices(
        treatment_paired[f"bl_{tracer_col}"].to_numpy(),
        resampled_treatment_locations,
    )
    tracer_reporting_period_boot = compute_resampled_means_from_indices(
        treatment_paired[f"rp_{tracer_col}"].to_numpy(),
        resampled_treatment_locations,
    )

    mass_ratio = compute_mass_ratio_from_immobile_tracer(
        feedstock_tracer_mg_kg=feedstock_tracer_boot,
        soil_baseline_tracer_mg_kg=tracer_baseline_boot,
        soil_end_of_reporting_period_tracer_mg_kg=tracer_reporting_period_boot,
    )
    distributions["mass_ratio_treatment"] = mass_ratio

    # Application rate diagnostic
    app_rate_boot, application_rate_check = _compute_application_rate_diagnostic(
        mass_ratio,
        bulk_density_means,
        _SAMPLING_DEPTH_CM,
        known_application_rate_kg_ha,
    )
    distributions["app_rate_t_ha_treatment"] = app_rate_boot / 1000

    # CDR per cation
    co2_combined_kg_ha: Np1DArray[np.floating] = np.zeros(_N_BOOTSTRAP_RUNS)

    for cation in _CATIONS:
        cation_col = mass_fraction_column_name(cation)
        cation_baseline_boot = compute_resampled_means_from_indices(
            treatment_paired[f"bl_{cation_col}"].to_numpy(),
            resampled_treatment_locations,
        )
        cation_reporting_period_boot = compute_resampled_means_from_indices(
            treatment_paired[f"rp_{cation_col}"].to_numpy(),
            resampled_treatment_locations,
        )
        feedstock_boot = feedstock_boot_by_cation[cation]

        cation_post_application_concentration = compute_post_application_concentration(
            feedstock_soil_mass_ratio=mass_ratio,
            soil_baseline_mg_kg=cation_baseline_boot,
            feedstock_mg_kg=feedstock_boot,
        )

        fraction_dissolved = compute_fraction_dissolved(
            feedstock_soil_mass_ratio=mass_ratio,
            post_application_concentration_mg_kg=cation_post_application_concentration,
            soil_end_of_reporting_period_mg_kg=cation_reporting_period_boot,
            feedstock_mg_kg=feedstock_boot,
            control_correction_ratio=control_correction_ratio_p50[cation],
        )

        cdr_cation_kg_ha = fraction_dissolved * known_application_rate_kg_ha * feedstock_boot / 1e6

        co2_kg_ha = convert_cation_kg_to_co2_kg(
            cation_kg=cdr_cation_kg_ha,
            cation=cation,
        )
        co2_combined_kg_ha += co2_kg_ha

        distributions[f"fraction_dissolved_{cation}_treatment"] = fraction_dissolved
        distributions[f"cdr_{cation}_treatment_kg_ha"] = cdr_cation_kg_ha
        distributions[f"co2_{cation}_treatment_kg_ha"] = co2_kg_ha

    distributions["co2_combined_treatment_kg_ha"] = co2_combined_kg_ha

    # Total CDR: scale by treatment area
    total_co2_tonnes = co2_combined_kg_ha * treatment_area / 1000
    distributions["total_co2_tonnes"] = total_co2_tonnes

    final_p16 = float(np.nanpercentile(total_co2_tonnes, 16))

    # Statistical significance check
    significance_df = run_significance_tests(
        treatment_baseline=treatment_baseline,
        treatment_reporting_period=treatment_reporting_period,
        feedstock_samples=feedstock_samples,
        bulk_density_kg_m3=float(np.mean(bulk_density_values)),
        application_rate_kg_ha=known_application_rate_kg_ha,
        elements=_CATIONS,
        sampling_depth_cm=_SAMPLING_DEPTH_CM,
    )

    # Build summary
    distributions_summary = summarize_distributions(distributions)

    area_hectares_df = pd.DataFrame([
        {"plot_type": pt, "area_hectares": area} for pt, area in area_hectares.items()
    ])

    pairing_report = pd.DataFrame(pairing_rows)

    return {
        "result": final_p16,
        "intermediate_outputs": {
            "data_cleaning_report": data_cleaning_report,
            "distributions_summary": distributions_summary,
            "area_hectares": area_hectares_df,
            "pairing_report": pairing_report,
            "application_rate_check": application_rate_check,
            "tracer_resolvability": tracer_resolvability_df,
            "significance_test_results": significance_df,
        },
    }
