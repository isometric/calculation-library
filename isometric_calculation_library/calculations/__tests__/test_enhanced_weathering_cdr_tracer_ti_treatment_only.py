# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon

from isometric_calculation_library.calculations.enhanced_weathering_cdr_tracer_ti_treatment_only import (
    main,
)

# ---------------------------------------------------------------------------
# Synthetic test data
# ---------------------------------------------------------------------------

# Two simple rectangular plots: treatment and control, side by side.
_TREATMENT_POLY = Polygon([(0, 0), (0, 0.01), (0.01, 0.01), (0.01, 0)])
_CONTROL_POLY = Polygon([(0.02, 0), (0.02, 0.01), (0.03, 0.01), (0.03, 0)])


def _make_plots() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"Type": ["Treatment", "Control"], "Geometry": [_TREATMENT_POLY, _CONTROL_POLY]},
        geometry="Geometry",
        crs="EPSG:4326",
    )


def _make_soil_samples(
    *,
    n_locations: int,
    ca_mean: float,
    mg_mean: float,
    ti_mean: float,
    lat_base: float,
    lon_base: float,
    prefix: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate synthetic soil samples inside a plot region."""
    rows = list[dict[str, object]]()
    for i in range(n_locations):
        rows.append({
            "reference_id": f"{prefix}_sample_{i}",
            "measurement_location_reference_id": f"{prefix}_loc_{i}",
            "latitude": lat_base + rng.uniform(0, 0.005),
            "longitude": lon_base + rng.uniform(0, 0.005),
            "mass_fraction_ca": ca_mean + rng.normal(0, ca_mean * 0.05),
            "mass_fraction_mg": mg_mean + rng.normal(0, mg_mean * 0.05),
            "mass_fraction_ti": ti_mean + rng.normal(0, ti_mean * 0.05),
        })
    return pd.DataFrame(rows)


def _make_feedstock_samples(rng: np.random.Generator) -> pd.DataFrame:
    """Generate synthetic feedstock samples with high Ti and cation content."""
    n = 10
    return pd.DataFrame({
        "mass_fraction_ca": rng.normal(50_000, 2000, size=n),
        "mass_fraction_mg": rng.normal(30_000, 1500, size=n),
        "mass_fraction_ti": rng.normal(5000, 200, size=n),
    })


def _make_bulk_density_samples(rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame({
        "bulk_density": rng.normal(1200, 50, size=15),
    })


def _build_inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    gpd.GeoDataFrame,
]:
    """Build a complete set of synthetic inputs for the model.

    Simulates weathering: reporting period treatment has lower Ca/Mg (cations
    dissolved) and higher Ti (tracer enriched) compared to baseline.
    Control plots have stable concentrations across periods.
    """
    rng = np.random.default_rng(123)

    # Treatment: inside _TREATMENT_POLY (lon 0-0.01, lat 0-0.01)
    treat_bl = _make_soil_samples(
        n_locations=20,
        ca_mean=200,
        mg_mean=150,
        ti_mean=100,
        lat_base=0.001,
        lon_base=0.001,
        prefix="treat_bl",
        rng=rng,
    )
    treat_rp = _make_soil_samples(
        n_locations=20,
        ca_mean=180,
        mg_mean=135,
        ti_mean=110,
        lat_base=0.001,
        lon_base=0.001,
        prefix="treat_bl",  # same location IDs for pairing
        rng=rng,
    )

    # Control: inside _CONTROL_POLY (lon 0.02-0.03, lat 0-0.01)
    ctrl_bl = _make_soil_samples(
        n_locations=15,
        ca_mean=200,
        mg_mean=150,
        ti_mean=100,
        lat_base=0.001,
        lon_base=0.021,
        prefix="ctrl_bl",
        rng=rng,
    )
    ctrl_rp = _make_soil_samples(
        n_locations=15,
        ca_mean=200,
        mg_mean=150,
        ti_mean=100,
        lat_base=0.001,
        lon_base=0.021,
        prefix="ctrl_bl",  # same location IDs for pairing
        rng=rng,
    )

    baseline = pd.concat([treat_bl, ctrl_bl], ignore_index=True)
    reporting_period = pd.concat([treat_rp, ctrl_rp], ignore_index=True)

    feedstock = _make_feedstock_samples(rng)
    bulk_density = _make_bulk_density_samples(rng)
    plots = _make_plots()

    return baseline, reporting_period, feedstock, bulk_density, plots


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

APPLICATION_RATE_KG_HA = 15_000.0


def test_main_returns_expected_structure() -> None:
    """Model returns a result float and all expected intermediate outputs."""
    baseline, rp, feedstock, bd, plots = _build_inputs()

    result = main(
        baseline,
        rp,
        feedstock,
        bd,
        plots,
        known_application_rate_kg_ha=APPLICATION_RATE_KG_HA,
    )

    assert "result" in result
    assert isinstance(result["result"], float)

    intermediates = result["intermediate_outputs"]
    assert set(intermediates.keys()) == {
        "data_cleaning_report",
        "distributions_summary",
        "area_hectares",
        "pairing_report",
        "application_rate_check",
        "tracer_resolvability",
        "significance_test_results",
    }

    for value in intermediates.values():
        assert isinstance(value, pd.DataFrame)
        assert len(value) > 0


def test_main_result_is_finite() -> None:
    """Model result should be a finite number (not NaN or inf)."""
    baseline, rp, feedstock, bd, plots = _build_inputs()

    result = main(
        baseline,
        rp,
        feedstock,
        bd,
        plots,
        known_application_rate_kg_ha=APPLICATION_RATE_KG_HA,
    )

    assert np.isfinite(result["result"])


def test_main_positive_cdr_with_weathering_signal() -> None:
    """When cation concentrations decrease in treatment, CDR should be positive."""
    baseline, rp, feedstock, bd, plots = _build_inputs()

    result = main(
        baseline,
        rp,
        feedstock,
        bd,
        plots,
        known_application_rate_kg_ha=APPLICATION_RATE_KG_HA,
    )

    # p16 should still be positive given the strong weathering signal
    assert result["result"] > 0


def test_main_distributions_summary_contains_total_co2() -> None:
    """Distributions summary should include the total CO2 distribution."""
    baseline, rp, feedstock, bd, plots = _build_inputs()

    result = main(
        baseline,
        rp,
        feedstock,
        bd,
        plots,
        known_application_rate_kg_ha=APPLICATION_RATE_KG_HA,
    )

    summary = result["intermediate_outputs"]["distributions_summary"]
    distribution_names = set(summary["distribution_name"])

    assert "total_co2_tonnes" in distribution_names
    assert "mass_ratio_treatment" in distribution_names
    assert "bulk_density_kg_m3" in distribution_names
    assert "fraction_dissolved_Ca_treatment" in distribution_names
    assert "fraction_dissolved_Mg_treatment" in distribution_names


def test_main_area_hectares_has_treatment_and_control() -> None:
    """Area output should contain treatment and control plot types."""
    baseline, rp, feedstock, bd, plots = _build_inputs()

    result = main(
        baseline,
        rp,
        feedstock,
        bd,
        plots,
        known_application_rate_kg_ha=APPLICATION_RATE_KG_HA,
    )

    area_df = result["intermediate_outputs"]["area_hectares"]
    plot_types = set(area_df["plot_type"])

    assert "treatment" in plot_types
    assert "control" in plot_types


def test_main_pairing_report_has_treatment_and_control() -> None:
    """Pairing report should cover both treatment and control for both cations."""
    baseline, rp, feedstock, bd, plots = _build_inputs()

    result = main(
        baseline,
        rp,
        feedstock,
        bd,
        plots,
        known_application_rate_kg_ha=APPLICATION_RATE_KG_HA,
    )

    pairing = result["intermediate_outputs"]["pairing_report"]
    area_types = set(pairing["area_type"])

    assert area_types == {"treatment", "control"}
    assert set(pairing["cation"]) == {"Ca", "Mg"}


def test_main_tracer_resolvability_is_treatment_only() -> None:
    """Tracer resolvability should only be computed for treatment."""
    baseline, rp, feedstock, bd, plots = _build_inputs()

    result = main(
        baseline,
        rp,
        feedstock,
        bd,
        plots,
        known_application_rate_kg_ha=APPLICATION_RATE_KG_HA,
    )

    resolvability = result["intermediate_outputs"]["tracer_resolvability"]
    assert set(resolvability["plot_type"]) == {"treatment"}
    assert resolvability["resolvability_index"].iloc[0] > 0


def test_main_significance_tests_per_cation() -> None:
    """Significance test results should have one row per cation."""
    baseline, rp, feedstock, bd, plots = _build_inputs()

    result = main(
        baseline,
        rp,
        feedstock,
        bd,
        plots,
        known_application_rate_kg_ha=APPLICATION_RATE_KG_HA,
    )

    sig = result["intermediate_outputs"]["significance_test_results"]
    assert set(sig["cation"]) == {"Ca", "Mg"}
    assert all(sig["p_value"].between(0, 1))


def test_main_is_deterministic() -> None:
    """Running the model twice with the same inputs gives the same result."""
    baseline, rp, feedstock, bd, plots = _build_inputs()
    kwargs = {
        "known_application_rate_kg_ha": APPLICATION_RATE_KG_HA,
    }

    result_1 = main(baseline, rp, feedstock, bd, plots, **kwargs)
    result_2 = main(baseline, rp, feedstock, bd, plots, **kwargs)

    assert result_1["result"] == pytest.approx(result_2["result"])


def test_main_data_cleaning_report_tracks_dropped_samples() -> None:
    """Data cleaning report should show sample counts for spatial assignment."""
    baseline, rp, feedstock, bd, plots = _build_inputs()

    result = main(
        baseline,
        rp,
        feedstock,
        bd,
        plots,
        known_application_rate_kg_ha=APPLICATION_RATE_KG_HA,
    )

    report = result["intermediate_outputs"]["data_cleaning_report"]
    assert set(report["step"]) == {"spatial_assignment"}
    assert set(report["period"]) == {"baseline", "reporting_period"}

    for _, row in report.iterrows():
        assert row["n_input"] > 0
        assert row["n_dropped"] >= 0
        assert row["n_retained"] == row["n_input"] - row["n_dropped"]


def test_main_no_deployment_in_outputs() -> None:
    """No intermediate output should reference deployment plots."""
    baseline, rp, feedstock, bd, plots = _build_inputs()

    result = main(
        baseline,
        rp,
        feedstock,
        bd,
        plots,
        known_application_rate_kg_ha=APPLICATION_RATE_KG_HA,
    )

    summary = result["intermediate_outputs"]["distributions_summary"]
    distribution_names = list(summary["distribution_name"])
    assert not any("deployment" in name for name in distribution_names)

    pairing = result["intermediate_outputs"]["pairing_report"]
    assert "deployment" not in set(pairing["area_type"])

    resolvability = result["intermediate_outputs"]["tracer_resolvability"]
    assert "deployment" not in set(resolvability["plot_type"])
