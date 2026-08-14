# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

import numpy as np
import pandas as pd
import pytest

from isometric_calculation_library.enhanced_weathering.utils.application_rate_derivation import (
    derive_application_rate_from_post_application_samples,
    derive_application_rate_from_tracer,
    pool_cations_as_charge_equivalents,
)
from isometric_calculation_library.enhanced_weathering.utils.conversions import Cation

_DEPTH_CM = 10.0
_BULK_DENSITY = 1300.0
_SOIL_MASS_KG_HA = _BULK_DENSITY * (_DEPTH_CM / 100) * 1e4
_N_RUNS = 2_000

_FEEDSTOCK: dict[Cation, float] = {"Ca": 80_000.0, "Mg": 40_000.0}
_BASELINE: dict[Cation, float] = {"Ca": 6_000.0, "Mg": 3_000.0}


def _mix(baseline: float, feedstock: float, mass_ratio: float) -> float:
    """Forward mixing formula: the concentration a known rate would produce."""
    return (baseline + mass_ratio * feedstock) / (1 + mass_ratio)


def _samples(values: dict[str, float], n: int = 30, prefix: str = "loc") -> pd.DataFrame:
    """Noise-free samples, one row per location, so an inversion is exactly invertible."""
    return pd.DataFrame({
        "measurement_location_reference_id": [f"{prefix}_{i}" for i in range(n)],
        **{col: np.full(n, value) for col, value in values.items()},
    })


def _cation_samples(
    concentrations: dict[Cation, float],
    n: int = 30,
    prefix: str = "loc",
) -> pd.DataFrame:
    return _samples(
        {f"mass_fraction_{c.lower()}": v for c, v in concentrations.items()},
        n,
        prefix,
    )


def _bulk_density(n: int = 20, std: float = 0.0) -> np.ndarray:
    if std == 0:
        return np.full(n, _BULK_DENSITY)
    return np.random.default_rng(3).normal(_BULK_DENSITY, std, n)


def test_pool_charge_equivalents_weights_by_valence_and_molar_mass() -> None:
    """Equal mg/kg of Ca and Mg give different charge; Mg is lighter so contributes more."""
    pooled = pool_cations_as_charge_equivalents({
        "Ca": np.array([1000.0]),
        "Mg": np.array([1000.0]),
    })
    ca_only = pool_cations_as_charge_equivalents({"Ca": np.array([1000.0])})
    mg_only = pool_cations_as_charge_equivalents({"Mg": np.array([1000.0])})

    assert pooled[0] == pytest.approx(ca_only[0] + mg_only[0])
    assert mg_only[0] > ca_only[0]


def test_pool_charge_equivalents_rejects_misaligned_arrays() -> None:
    """Differing lengths would silently broadcast, so they must raise."""
    with pytest.raises(ValueError, match="differing lengths"):
        pool_cations_as_charge_equivalents({
            "Ca": np.array([1000.0, 1100.0]),
            "Mg": np.array([500.0]),
        })


def test_pool_charge_equivalents_rejects_empty() -> None:
    with pytest.raises(ValueError, match="at least one cation"):
        pool_cations_as_charge_equivalents({})


def test_post_application_recovers_the_rate_that_generated_the_enrichment() -> None:
    """BLP built forward from a known rate inverts back to that rate."""
    known_rate_kg_ha = 20_000.0
    mass_ratio = known_rate_kg_ha / _SOIL_MASS_KG_HA

    result = derive_application_rate_from_post_application_samples(
        baseline_samples=_cation_samples(_BASELINE),
        post_application_samples=_cation_samples({
            c: _mix(_BASELINE[c], _FEEDSTOCK[c], mass_ratio) for c in _FEEDSTOCK
        }),
        feedstock_samples=_cation_samples(_FEEDSTOCK, n=5),
        bulk_density_values_kg_m3=_bulk_density(),
        depth_cm=_DEPTH_CM,
        cations=["Ca", "Mg"],
        n_runs=_N_RUNS,
        rng=np.random.default_rng(42),
    )

    assert result.derivation_method == "blp_enrichment"
    assert result.n_baseline == 30
    assert result.n_post_application == 30
    assert result.paired
    assert float(np.nanpercentile(result.rate_distribution_kg_ha, 50)) == pytest.approx(
        known_rate_kg_ha,
        rel=1e-6,
    )


def test_post_application_unpaired_recovers_the_rate_from_unmatched_locations() -> None:
    """An unpaired design has no location correspondence and need not have equal counts."""
    known_rate_kg_ha = 20_000.0
    mass_ratio = known_rate_kg_ha / _SOIL_MASS_KG_HA

    result = derive_application_rate_from_post_application_samples(
        baseline_samples=_cation_samples(_BASELINE, n=30, prefix="bl"),
        post_application_samples=_cation_samples(
            {c: _mix(_BASELINE[c], _FEEDSTOCK[c], mass_ratio) for c in _FEEDSTOCK},
            n=18,
            prefix="blp",
        ),
        feedstock_samples=_cation_samples(_FEEDSTOCK, n=5),
        bulk_density_values_kg_m3=_bulk_density(),
        depth_cm=_DEPTH_CM,
        cations=["Ca", "Mg"],
        n_runs=_N_RUNS,
        rng=np.random.default_rng(42),
        paired=False,
    )

    assert result.n_baseline == 30
    assert result.n_post_application == 18
    assert not result.paired
    assert float(np.nanpercentile(result.rate_distribution_kg_ha, 50)) == pytest.approx(
        known_rate_kg_ha,
        rel=1e-6,
    )


def test_post_application_paired_rejects_the_locations_an_unpaired_design_accepts() -> None:
    """The same disjoint-location inputs pair to nothing, so the paired route must not run."""
    with pytest.raises(ValueError, match="at least 3 are required"):
        derive_application_rate_from_post_application_samples(
            baseline_samples=_cation_samples(_BASELINE, prefix="bl"),
            post_application_samples=_cation_samples(
                {"Ca": 10_000.0, "Mg": 5_000.0},
                prefix="blp",
            ),
            feedstock_samples=_cation_samples(_FEEDSTOCK, n=5),
            bulk_density_values_kg_m3=_bulk_density(),
            depth_cm=_DEPTH_CM,
            cations=["Ca", "Mg"],
            n_runs=100,
            rng=np.random.default_rng(42),
        )


def test_post_application_unpaired_rejects_null_measurements() -> None:
    """The no-nulls convention holds on the unpaired route too, where no join filters them."""
    post_application = _cation_samples({"Ca": 10_000.0, "Mg": 5_000.0}, prefix="blp")
    post_application.loc[0, "mass_fraction_ca"] = np.nan

    with pytest.raises(ValueError, match="null 'mass_fraction_ca'"):
        derive_application_rate_from_post_application_samples(
            baseline_samples=_cation_samples(_BASELINE, prefix="bl"),
            post_application_samples=post_application,
            feedstock_samples=_cation_samples(_FEEDSTOCK, n=5),
            bulk_density_values_kg_m3=_bulk_density(),
            depth_cm=_DEPTH_CM,
            cations=["Ca", "Mg"],
            n_runs=100,
            rng=np.random.default_rng(42),
            paired=False,
        )


def test_post_application_unpaired_rejects_too_few_samples_in_either_event() -> None:
    """A short event is named, so the caller knows which one is under-sampled."""
    with pytest.raises(ValueError, match="Only 2 post-application sample"):
        derive_application_rate_from_post_application_samples(
            baseline_samples=_cation_samples(_BASELINE, prefix="bl"),
            post_application_samples=_cation_samples(
                {"Ca": 10_000.0, "Mg": 5_000.0},
                n=2,
                prefix="blp",
            ),
            feedstock_samples=_cation_samples(_FEEDSTOCK, n=5),
            bulk_density_values_kg_m3=_bulk_density(),
            depth_cm=_DEPTH_CM,
            cations=["Ca", "Mg"],
            n_runs=100,
            rng=np.random.default_rng(42),
            paired=False,
        )


def test_post_application_pooling_is_charge_weighted_not_an_average() -> None:
    """Cations implying different rates resolve by charge, not by arithmetic mean."""
    result = derive_application_rate_from_post_application_samples(
        baseline_samples=_cation_samples(_BASELINE),
        post_application_samples=_cation_samples({
            "Ca": _mix(_BASELINE["Ca"], _FEEDSTOCK["Ca"], 10_000.0 / _SOIL_MASS_KG_HA),
            "Mg": _mix(_BASELINE["Mg"], _FEEDSTOCK["Mg"], 30_000.0 / _SOIL_MASS_KG_HA),
        }),
        feedstock_samples=_cation_samples(_FEEDSTOCK, n=5),
        bulk_density_values_kg_m3=_bulk_density(),
        depth_cm=_DEPTH_CM,
        cations=["Ca", "Mg"],
        n_runs=_N_RUNS,
        rng=np.random.default_rng(42),
    )

    p50 = float(np.nanpercentile(result.rate_distribution_kg_ha, 50))
    assert 10_000.0 < p50 < 30_000.0
    assert p50 != pytest.approx(20_000.0, rel=1e-3)


def test_post_application_rejects_too_few_pairs() -> None:
    """Below the pair threshold there is no rate, and no rate must not pass silently."""
    with pytest.raises(ValueError, match="at least 3 are required"):
        derive_application_rate_from_post_application_samples(
            baseline_samples=_cation_samples(_BASELINE, n=2),
            post_application_samples=_cation_samples({"Ca": 10_000.0, "Mg": 5_000.0}, n=2),
            feedstock_samples=_cation_samples(_FEEDSTOCK, n=5),
            bulk_density_values_kg_m3=_bulk_density(),
            depth_cm=_DEPTH_CM,
            cations=["Ca", "Mg"],
            n_runs=100,
            rng=np.random.default_rng(42),
            min_samples=3,
        )


def test_post_application_rejects_null_paired_measurements() -> None:
    """A null at a paired location must raise, not quietly shrink the location set."""
    baseline = _cation_samples(_BASELINE)
    post_application = _cation_samples({"Ca": 10_000.0, "Mg": 5_000.0})
    post_application.loc[0, "mass_fraction_ca"] = np.nan

    with pytest.raises(ValueError, match="null measurement"):
        derive_application_rate_from_post_application_samples(
            baseline_samples=baseline,
            post_application_samples=post_application,
            feedstock_samples=_cation_samples(_FEEDSTOCK, n=5),
            bulk_density_values_kg_m3=_bulk_density(),
            depth_cm=_DEPTH_CM,
            cations=["Ca", "Mg"],
            n_runs=100,
            rng=np.random.default_rng(42),
        )


def test_post_application_rejects_null_feedstock_measurements() -> None:
    """A null feedstock measurement must raise rather than be dropped from the mean."""
    feedstock = _cation_samples(_FEEDSTOCK, n=5)
    feedstock.loc[0, "mass_fraction_mg"] = np.nan

    with pytest.raises(ValueError, match="null 'mass_fraction_mg'"):
        derive_application_rate_from_post_application_samples(
            baseline_samples=_cation_samples(_BASELINE),
            post_application_samples=_cation_samples({"Ca": 10_000.0, "Mg": 5_000.0}),
            feedstock_samples=feedstock,
            bulk_density_values_kg_m3=_bulk_density(),
            depth_cm=_DEPTH_CM,
            cations=["Ca", "Mg"],
            n_runs=100,
            rng=np.random.default_rng(42),
        )


def test_post_application_rejects_no_enrichment_as_unrecoverable() -> None:
    """BLP equal to BL is not a physical application."""
    with pytest.raises(ValueError, match="No physically recoverable rate"):
        derive_application_rate_from_post_application_samples(
            baseline_samples=_cation_samples(_BASELINE),
            post_application_samples=_cation_samples(_BASELINE),
            feedstock_samples=_cation_samples(_FEEDSTOCK, n=5),
            bulk_density_values_kg_m3=_bulk_density(),
            depth_cm=_DEPTH_CM,
            cations=["Ca", "Mg"],
            n_runs=_N_RUNS,
            rng=np.random.default_rng(42),
        )


def test_post_application_rejects_feedstock_poorer_than_blp_as_unrecoverable() -> None:
    """A non-positive denominator has no solution."""
    with pytest.raises(ValueError, match="No physically recoverable rate"):
        derive_application_rate_from_post_application_samples(
            baseline_samples=_cation_samples(_BASELINE),
            post_application_samples=_cation_samples({"Ca": 90_000.0, "Mg": 50_000.0}),
            feedstock_samples=_cation_samples(_FEEDSTOCK, n=5),
            bulk_density_values_kg_m3=_bulk_density(),
            depth_cm=_DEPTH_CM,
            cations=["Ca", "Mg"],
            n_runs=_N_RUNS,
            rng=np.random.default_rng(42),
        )


def test_post_application_propagates_bulk_density_uncertainty() -> None:
    """The rate scales with soil mass, so bulk-density spread must widen it."""
    mass_ratio = 20_000.0 / _SOIL_MASS_KG_HA
    baseline = _cation_samples(_BASELINE)
    post_application = _cation_samples({
        c: _mix(_BASELINE[c], _FEEDSTOCK[c], mass_ratio) for c in _FEEDSTOCK
    })
    feedstock = _cation_samples(_FEEDSTOCK, n=5)

    fixed = derive_application_rate_from_post_application_samples(
        baseline_samples=baseline,
        post_application_samples=post_application,
        feedstock_samples=feedstock,
        bulk_density_values_kg_m3=_bulk_density(),
        depth_cm=_DEPTH_CM,
        cations=["Ca", "Mg"],
        n_runs=_N_RUNS,
        rng=np.random.default_rng(42),
    )
    varying = derive_application_rate_from_post_application_samples(
        baseline_samples=baseline,
        post_application_samples=post_application,
        feedstock_samples=feedstock,
        bulk_density_values_kg_m3=_bulk_density(std=100.0),
        depth_cm=_DEPTH_CM,
        cations=["Ca", "Mg"],
        n_runs=_N_RUNS,
        rng=np.random.default_rng(42),
    )

    assert np.nanstd(varying.rate_distribution_kg_ha) > np.nanstd(fixed.rate_distribution_kg_ha)


def test_post_application_requires_bulk_density() -> None:
    with pytest.raises(ValueError, match="bulk_density_values_kg_m3 is empty"):
        derive_application_rate_from_post_application_samples(
            baseline_samples=_cation_samples(_BASELINE),
            post_application_samples=_cation_samples({"Ca": 8_000.0, "Mg": 4_000.0}),
            feedstock_samples=_cation_samples(_FEEDSTOCK, n=5),
            bulk_density_values_kg_m3=np.empty(0),
            depth_cm=_DEPTH_CM,
            cations=["Ca", "Mg"],
            n_runs=100,
            rng=np.random.default_rng(42),
        )


def test_post_application_requires_at_least_one_cation() -> None:
    with pytest.raises(ValueError, match="cations is empty"):
        derive_application_rate_from_post_application_samples(
            baseline_samples=_cation_samples(_BASELINE),
            post_application_samples=_cation_samples({"Ca": 8_000.0, "Mg": 4_000.0}),
            feedstock_samples=_cation_samples(_FEEDSTOCK, n=5),
            bulk_density_values_kg_m3=_bulk_density(),
            depth_cm=_DEPTH_CM,
            cations=[],
            n_runs=100,
            rng=np.random.default_rng(42),
        )


def test_tracer_recovers_the_rate_that_generated_the_enrichment() -> None:
    """A reporting-period tracer built forward from a known rate inverts back to it."""
    known_rate_kg_ha = 20_000.0
    mass_ratio = known_rate_kg_ha / _SOIL_MASS_KG_HA
    baseline_ti, feedstock_ti = 4_000.0, 17_000.0

    result = derive_application_rate_from_tracer(
        baseline_samples=_samples({"mass_fraction_ti": baseline_ti}),
        reporting_period_samples=_samples({
            "mass_fraction_ti": _mix(baseline_ti, feedstock_ti, mass_ratio),
        }),
        feedstock_samples=_samples({"mass_fraction_ti": feedstock_ti}, n=5),
        bulk_density_values_kg_m3=_bulk_density(),
        depth_cm=_DEPTH_CM,
        tracer="Ti",
        n_runs=_N_RUNS,
        rng=np.random.default_rng(42),
    )

    assert result.derivation_method == "ti_tracer"
    assert float(np.nanpercentile(result.rate_distribution_kg_ha, 50)) == pytest.approx(
        known_rate_kg_ha,
        rel=1e-6,
    )


def test_tracer_labels_the_method_from_the_tracer_element() -> None:
    """The label distinguishes Zr from Ti in the outputs."""
    result = derive_application_rate_from_tracer(
        baseline_samples=_samples({"mass_fraction_zr": 200.0}),
        reporting_period_samples=_samples({"mass_fraction_zr": 300.0}),
        feedstock_samples=_samples({"mass_fraction_zr": 900.0}, n=5),
        bulk_density_values_kg_m3=_bulk_density(),
        depth_cm=_DEPTH_CM,
        tracer="Zr",
        n_runs=100,
        rng=np.random.default_rng(42),
    )

    assert result.derivation_method == "zr_tracer"


def test_tracer_unpaired_recovers_the_rate_from_unmatched_locations() -> None:
    """The tracer route takes the same unpaired designs the enrichment route does."""
    known_rate_kg_ha = 20_000.0
    mass_ratio = known_rate_kg_ha / _SOIL_MASS_KG_HA
    baseline_ti, feedstock_ti = 4_000.0, 17_000.0

    result = derive_application_rate_from_tracer(
        baseline_samples=_samples({"mass_fraction_ti": baseline_ti}, prefix="bl"),
        reporting_period_samples=_samples(
            {"mass_fraction_ti": _mix(baseline_ti, feedstock_ti, mass_ratio)},
            n=18,
            prefix="rp",
        ),
        feedstock_samples=_samples({"mass_fraction_ti": feedstock_ti}, n=5),
        bulk_density_values_kg_m3=_bulk_density(),
        depth_cm=_DEPTH_CM,
        tracer="Ti",
        n_runs=_N_RUNS,
        rng=np.random.default_rng(42),
        paired=False,
    )

    assert result.n_baseline == 30
    assert result.n_post_application == 18
    assert not result.paired
    assert float(np.nanpercentile(result.rate_distribution_kg_ha, 50)) == pytest.approx(
        known_rate_kg_ha,
        rel=1e-6,
    )


def test_tracer_rejects_tracer_loss_as_unrecoverable() -> None:
    """An immobile tracer cannot decrease, so a drop is noise rather than a measurement."""
    with pytest.raises(ValueError, match="Ti mass balance"):
        derive_application_rate_from_tracer(
            baseline_samples=_samples({"mass_fraction_ti": 5_000.0}),
            reporting_period_samples=_samples({"mass_fraction_ti": 4_000.0}),
            feedstock_samples=_samples({"mass_fraction_ti": 17_000.0}, n=5),
            bulk_density_values_kg_m3=_bulk_density(),
            depth_cm=_DEPTH_CM,
            tracer="Ti",
            n_runs=_N_RUNS,
            rng=np.random.default_rng(42),
        )


def test_both_derivations_agree_on_the_same_deployment() -> None:
    """Given consistent inputs, BLP and tracer routes recover the same rate.

    The two measurements are independent, so agreement here is a real cross-check that
    neither inversion carries a scale error.
    """
    known_rate_kg_ha = 20_000.0
    mass_ratio = known_rate_kg_ha / _SOIL_MASS_KG_HA
    baseline_ti, feedstock_ti = 4_000.0, 17_000.0

    from_blp = derive_application_rate_from_post_application_samples(
        baseline_samples=_cation_samples(_BASELINE),
        post_application_samples=_cation_samples({
            c: _mix(_BASELINE[c], _FEEDSTOCK[c], mass_ratio) for c in _FEEDSTOCK
        }),
        feedstock_samples=_cation_samples(_FEEDSTOCK, n=5),
        bulk_density_values_kg_m3=_bulk_density(),
        depth_cm=_DEPTH_CM,
        cations=["Ca", "Mg"],
        n_runs=_N_RUNS,
        rng=np.random.default_rng(42),
    )
    from_tracer = derive_application_rate_from_tracer(
        baseline_samples=_samples({"mass_fraction_ti": baseline_ti}),
        reporting_period_samples=_samples({
            "mass_fraction_ti": _mix(baseline_ti, feedstock_ti, mass_ratio),
        }),
        feedstock_samples=_samples({"mass_fraction_ti": feedstock_ti}, n=5),
        bulk_density_values_kg_m3=_bulk_density(),
        depth_cm=_DEPTH_CM,
        tracer="Ti",
        n_runs=_N_RUNS,
        rng=np.random.default_rng(42),
    )

    assert float(np.nanpercentile(from_blp.rate_distribution_kg_ha, 50)) == pytest.approx(
        float(np.nanpercentile(from_tracer.rate_distribution_kg_ha, 50)),
        rel=1e-6,
    )
