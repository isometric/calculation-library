# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

import numpy as np
import pytest

from isometric_calculation_library.enhanced_weathering.utils.application_rate_derivation import (
    DerivedApplicationRate,
)
from isometric_calculation_library.enhanced_weathering.utils.statistical_checks.application_rate import (
    build_application_rate_check,
    resolve_application_rate,
)


def _derived(
    replicates: np.ndarray,
    *,
    method: str = "blp_enrichment",
    n_pairs: int = 30,
) -> DerivedApplicationRate:
    return DerivedApplicationRate(
        rate_distribution_kg_ha=replicates,
        derivation_method=method,
        n_baseline=n_pairs,
        n_post_application=n_pairs,
        paired=True,
    )


def test_build_application_rate_check_columns() -> None:
    """DataFrame has all expected columns."""
    boot = np.random.default_rng(42).normal(15_000, 500, size=10_000)
    result = build_application_rate_check(
        soil_based_application_rate_bootstrap_replicates_kg_ha=boot,
        known_application_rate_kg_ha=15_000.0,
    )

    expected_cols = {
        "plot_type",
        "known_app_rate_t_ha",
        "soil_based_app_rate_mean_t_ha",
        "soil_based_app_rate_std_t_ha",
        "soil_based_app_rate_p5_t_ha",
        "soil_based_app_rate_p16_t_ha",
        "soil_based_app_rate_p84_t_ha",
        "soil_based_app_rate_p95_t_ha",
        "known_within_2std",
        "deviation_in_std",
    }
    assert set(result.columns) == expected_cols
    assert len(result) == 1


def test_build_application_rate_check_within_2std() -> None:
    """When actual rate matches bootstrap mean, it should be within 2 std."""
    boot = np.random.default_rng(42).normal(15_000, 500, size=10_000)
    result = build_application_rate_check(
        soil_based_application_rate_bootstrap_replicates_kg_ha=boot,
        known_application_rate_kg_ha=15_000.0,
    )

    assert result["known_within_2std"].iloc[0]


def test_build_application_rate_check_outside_2std() -> None:
    """When actual rate is far from bootstrap mean, it should be outside 2 std."""
    boot = np.full(10_000, 15_000.0) + np.random.default_rng(42).normal(0, 100, size=10_000)
    result = build_application_rate_check(
        soil_based_application_rate_bootstrap_replicates_kg_ha=boot,
        known_application_rate_kg_ha=50_000.0,
    )

    assert not result["known_within_2std"].iloc[0]


def test_build_application_rate_check_zero_std() -> None:
    """When bootstrap has zero variance, deviation_in_std should be inf."""
    boot = np.full(100, 15_000.0)
    result = build_application_rate_check(
        soil_based_application_rate_bootstrap_replicates_kg_ha=boot,
        known_application_rate_kg_ha=50_000.0,
    )

    assert result["deviation_in_std"].iloc[0] == float("inf")
    assert not result["known_within_2std"].iloc[0]


def test_resolve_keeps_the_operational_rate_when_the_check_passes() -> None:
    """Agreement means the operational scalar is used, carrying no rate uncertainty."""
    boot = np.random.default_rng(42).normal(15_000, 500, size=10_000)
    decision = resolve_application_rate(
        derived_rate=_derived(boot, method="blp_enrichment"),
        known_application_rate_kg_ha=15_000.0,
    )

    assert decision.passes
    assert decision.rate_is_operational
    assert decision.rate_kg_ha == pytest.approx(15_000.0)
    assert decision.report_row["derivation_method"] == "blp_enrichment"


def test_resolve_switches_to_the_soil_based_median_when_it_is_lower() -> None:
    """A disagreement where the soil-derived median is lower switches to that median.

    The switch is to the median as a scalar, not to the full soil-derived distribution.
    """
    boot = np.random.default_rng(42).normal(10_000, 200, size=10_000)
    decision = resolve_application_rate(
        derived_rate=_derived(boot, method="blp_enrichment"),
        known_application_rate_kg_ha=30_000.0,
    )

    assert not decision.passes
    assert not decision.rate_is_operational
    # A scalar median, not the distribution: its own sampling noise should not propagate.
    assert isinstance(decision.rate_kg_ha, float)
    assert decision.rate_kg_ha == pytest.approx(float(np.percentile(boot, 50)))


def test_resolve_keeps_the_operational_rate_when_soil_based_is_higher() -> None:
    """Disagreeing upward is not a reason to credit more, so the operational rate stands."""
    boot = np.random.default_rng(42).normal(50_000, 200, size=10_000)
    decision = resolve_application_rate(
        derived_rate=_derived(boot, method="ti_tracer"),
        known_application_rate_kg_ha=15_000.0,
    )

    assert not decision.passes
    assert decision.rate_is_operational
    assert decision.rate_kg_ha == pytest.approx(15_000.0)


def test_resolve_summarises_a_skewed_distribution_by_its_median() -> None:
    """The reported rate is the median, so a long right tail does not inflate it.

    Uses a skewed distribution whose mean sits far above its median; the median is what
    gets reported and what the lower-of-the-two decision is made on.
    """
    rng = np.random.default_rng(42)
    boot = np.concatenate([rng.normal(10_000, 100, size=9_900), np.full(100, 5_000_000.0)])
    decision = resolve_application_rate(
        derived_rate=_derived(boot, method="blp_enrichment"),
        known_application_rate_kg_ha=4_000_000.0,
    )

    median_kg_ha = float(np.percentile(boot, 50))
    assert decision.rate_kg_ha == pytest.approx(median_kg_ha)
    # Had the mean been used, it would sit ~5x higher than the median.
    assert median_kg_ha < float(np.mean(boot))
    # The operational rate is far above even the inflated std, so the soil-derived rate wins.
    assert not decision.passes
    assert not decision.rate_is_operational


def test_resolve_raises_when_no_replicate_is_physical() -> None:
    """An unverifiable rate raises rather than letting the operational rate stand unchecked."""
    with pytest.raises(ValueError, match="no finite replicate"):
        resolve_application_rate(
            derived_rate=_derived(np.full(1_000, np.nan), method="blp_enrichment"),
            known_application_rate_kg_ha=15_000.0,
        )


def test_resolve_ignores_nan_replicates_when_summarising() -> None:
    """NaN replicates must not poison the percentiles used for the decision."""
    clean = np.random.default_rng(42).normal(10_000, 200, size=5_000)
    padded = np.concatenate([clean, np.full(5_000, np.nan)])

    from_clean = resolve_application_rate(
        derived_rate=_derived(clean, method="blp_enrichment"),
        known_application_rate_kg_ha=30_000.0,
    )
    from_padded = resolve_application_rate(
        derived_rate=_derived(padded, method="blp_enrichment"),
        known_application_rate_kg_ha=30_000.0,
    )

    assert from_padded.rate_kg_ha == pytest.approx(from_clean.rate_kg_ha)


def test_resolve_rejects_a_non_positive_operational_rate() -> None:
    """A zero or negative operational rate is a data error, not something to quantify with."""
    boot = np.random.default_rng(42).normal(15_000, 500, size=1_000)
    for bad_rate in (0.0, -5_000.0):
        with pytest.raises(ValueError, match="must be positive"):
            resolve_application_rate(
                derived_rate=_derived(boot, method="blp_enrichment"),
                known_application_rate_kg_ha=bad_rate,
            )


def test_resolve_zero_variance_fails_a_differing_rate() -> None:
    """A degenerate distribution carries no evidence that a differing rate is consistent."""
    decision = resolve_application_rate(
        derived_rate=_derived(np.full(1_000, 10_000.0), method="blp_enrichment"),
        known_application_rate_kg_ha=30_000.0,
    )

    assert not decision.passes
    assert not decision.rate_is_operational


def test_resolve_window_widens_with_a_tail_inflated_spread() -> None:
    """Documents a known property of the protocol's std-based window.

    The window is n_std empirical standard deviations, per the protocol text. Because these
    inversions are skewed, tail replicates inflate that std and widen the window, so a rate
    far from the median can still pass. The p16/p84 columns are reported precisely so a
    reviewer can see the distribution's real shape in such a case.
    """
    rng = np.random.default_rng(1)
    boot = np.concatenate([rng.normal(10_000, 200, 9_900), rng.uniform(1e6, 5e6, 100)])
    decision = resolve_application_rate(
        derived_rate=_derived(boot, method="blp_enrichment"),
        known_application_rate_kg_ha=100_000.0,
    )

    # 10x the soil-derived median, yet inside 2 inflated standard deviations, so the
    # operational rate is accepted and is what gets quantified with.
    assert decision.passes
    assert decision.rate_is_operational
    assert decision.rate_kg_ha == pytest.approx(100_000.0)
    row = decision.report_row
    soil_p50 = row["soil_based_app_rate_p50_t_ha"]
    assert isinstance(soil_p50, float)
    assert soil_p50 == pytest.approx(float(np.percentile(boot, 50)) / 1000)
    p16, p84 = row["soil_based_app_rate_p16_t_ha"], row["soil_based_app_rate_p84_t_ha"]
    assert isinstance(p16, float)
    assert isinstance(p84, float)
    # The reported percentiles make the tight core visible despite the wide window.
    assert p84 - p16 < 1.0


def test_resolve_n_std_names_its_column_and_widens_acceptance() -> None:
    """Relaxing to 3 std both renames the column and can flip a borderline case."""
    boot = np.random.default_rng(42).normal(10_000, 1_000, size=10_000)
    strict = resolve_application_rate(
        derived_rate=_derived(boot, method="blp_enrichment"),
        known_application_rate_kg_ha=12_500.0,
        n_std=2.0,
    )
    relaxed = resolve_application_rate(
        derived_rate=_derived(boot, method="blp_enrichment"),
        known_application_rate_kg_ha=12_500.0,
        n_std=3.0,
    )

    assert "known_within_2std" in strict.report_row
    assert "known_within_3std" in relaxed.report_row
    assert not strict.passes
    assert relaxed.passes


def test_resolve_rejects_a_non_positive_window() -> None:
    boot = np.random.default_rng(42).normal(15_000, 500, size=1_000)
    with pytest.raises(ValueError, match="n_std must be positive"):
        resolve_application_rate(
            derived_rate=_derived(boot, method="blp_enrichment"),
            known_application_rate_kg_ha=15_000.0,
            n_std=0.0,
        )
