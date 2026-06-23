# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

import numpy as np
import pandas as pd
import pytest

from isometric_calculation_library.enhanced_weathering.utils.feedstock_weighting import (
    BootstrapWeightedFeedstockResult,
    bootstrap_weighted_feedstock,
    compute_plot_coverage_weights,
    compute_weighted_feedstock_composition,
)


def _make_feedstock() -> pd.DataFrame:
    """Three batches, two samples each, with known composition values."""
    return pd.DataFrame([
        {"feedstock_batch_id": "batch_a", "ca": 100.0, "mg": 50.0},
        {"feedstock_batch_id": "batch_a", "ca": 120.0, "mg": 60.0},
        {"feedstock_batch_id": "batch_b", "ca": 200.0, "mg": 80.0},
        {"feedstock_batch_id": "batch_b", "ca": 220.0, "mg": 90.0},
        {"feedstock_batch_id": "batch_c", "ca": 300.0, "mg": 40.0},
        {"feedstock_batch_id": "batch_c", "ca": 320.0, "mg": 50.0},
    ])


# -- compute_weighted_feedstock_composition ------------------------------------


def test_weighted_feedstock_returns_single_row() -> None:
    result = compute_weighted_feedstock_composition(
        _make_feedstock(),
        {"batch_a": 1.0, "batch_b": 1.0, "batch_c": 1.0},
        ["ca", "mg"],
    )
    assert len(result) == 1
    assert set(result.columns) == {"ca", "mg"}


def test_weighted_feedstock_equal_weights_is_mean_of_batch_means() -> None:
    """With equal weights, result equals mean of per-batch means."""
    result = compute_weighted_feedstock_composition(
        _make_feedstock(),
        {"batch_a": 1.0, "batch_b": 1.0, "batch_c": 1.0},
        ["ca", "mg"],
    )
    # Batch means: a=(110, 55), b=(210, 85), c=(310, 45)
    assert result["ca"].iloc[0] == pytest.approx((110.0 + 210.0 + 310.0) / 3, rel=1e-6)
    assert result["mg"].iloc[0] == pytest.approx((55.0 + 85.0 + 45.0) / 3, rel=1e-6)


def test_weighted_feedstock_weights_are_normalised() -> None:
    """Passing weights [2, 2, 2] gives same result as [1, 1, 1]."""
    df = _make_feedstock()
    r1 = compute_weighted_feedstock_composition(
        df,
        {"batch_a": 1.0, "batch_b": 1.0, "batch_c": 1.0},
        ["ca"],
    )
    r2 = compute_weighted_feedstock_composition(
        df,
        {"batch_a": 2.0, "batch_b": 2.0, "batch_c": 2.0},
        ["ca"],
    )
    assert r1["ca"].iloc[0] == pytest.approx(r2["ca"].iloc[0])


def test_weighted_feedstock_unequal_weights_shift_composition() -> None:
    """Higher weight on batch_c (high Ca) increases the weighted Ca mean."""
    df = _make_feedstock()
    equal = compute_weighted_feedstock_composition(
        df,
        {"batch_a": 1.0, "batch_b": 1.0, "batch_c": 1.0},
        ["ca"],
    )
    skewed = compute_weighted_feedstock_composition(
        df,
        {"batch_a": 1.0, "batch_b": 1.0, "batch_c": 10.0},
        ["ca"],
    )
    assert skewed["ca"].iloc[0] > equal["ca"].iloc[0]


def test_weighted_feedstock_custom_batch_column() -> None:
    df = pd.DataFrame([
        {"crusher_id": "A", "ca": 100.0},
        {"crusher_id": "B", "ca": 200.0},
    ])
    result = compute_weighted_feedstock_composition(
        df,
        {"A": 1.0, "B": 1.0},
        ["ca"],
        batch_column="crusher_id",
    )
    assert result["ca"].iloc[0] == pytest.approx(150.0)


# -- compute_plot_coverage_weights ---------------------------------------------


def _make_sample_locations() -> pd.DataFrame:
    return pd.DataFrame([
        {"feedstock_batch_id": "batch_a", "location_id": "loc_1"},
        {"feedstock_batch_id": "batch_a", "location_id": "loc_2"},
        {"feedstock_batch_id": "batch_a", "location_id": "loc_3"},
        {"feedstock_batch_id": "batch_a", "location_id": "loc_4"},
        {"feedstock_batch_id": "batch_a", "location_id": "loc_4"},  # duplicate → 4 unique
        {"feedstock_batch_id": "batch_b", "location_id": "loc_5"},
        {"feedstock_batch_id": "batch_b", "location_id": "loc_6"},
        {"feedstock_batch_id": "batch_c", "location_id": "loc_7"},
    ])


def test_coverage_weights_returns_dict_str_float() -> None:
    weights = compute_plot_coverage_weights(
        _make_sample_locations(),
        "feedstock_batch_id",
        "location_id",
    )
    for k, v in weights.items():
        assert isinstance(k, str)
        assert isinstance(v, float)


def test_coverage_weights_counts_unique_locations() -> None:
    """Weight is unique-location count, not sample count."""
    weights = compute_plot_coverage_weights(
        _make_sample_locations(),
        "feedstock_batch_id",
        "location_id",
    )
    assert weights["batch_a"] == pytest.approx(4.0)
    assert weights["batch_b"] == pytest.approx(2.0)
    assert weights["batch_c"] == pytest.approx(1.0)


def test_coverage_weights_all_batches_covered() -> None:
    weights = compute_plot_coverage_weights(
        _make_sample_locations(),
        "feedstock_batch_id",
        "location_id",
    )
    assert set(weights.keys()) == {"batch_a", "batch_b", "batch_c"}


# -- bootstrap_weighted_feedstock ----------------------------------------------


def test_bootstrap_distributions_shape() -> None:
    """bootstrap_distributions has shape (n_runs,) per column."""
    rng = np.random.default_rng(42)
    result = bootstrap_weighted_feedstock(
        _make_feedstock(),
        {"batch_a": 2.0, "batch_b": 1.0, "batch_c": 1.0},
        ["ca", "mg"],
        batch_column="feedstock_batch_id",
        rng=rng,
        n_runs=100,
    )
    assert isinstance(result, BootstrapWeightedFeedstockResult)
    assert result.bootstrap_distributions["ca"].shape == (100,)
    assert result.bootstrap_distributions["mg"].shape == (100,)


def test_bootstrap_point_estimate_matches_compute_weighted() -> None:
    """Point estimate should match compute_weighted_feedstock_composition for same inputs."""
    batch_weights = {"batch_a": 2.0, "batch_b": 1.0, "batch_c": 1.0}
    feedstock = _make_feedstock()

    point_only = compute_weighted_feedstock_composition(feedstock, batch_weights, ["ca", "mg"])
    bootstrap_result = bootstrap_weighted_feedstock(
        feedstock,
        batch_weights,
        ["ca", "mg"],
        batch_column="feedstock_batch_id",
        rng=np.random.default_rng(0),
        n_runs=10,
    )
    pd.testing.assert_frame_equal(
        bootstrap_result.point_estimate.reset_index(drop=True),
        point_only.reset_index(drop=True),
        check_exact=False,
        rtol=1e-10,
    )


def test_bootstrap_noise_is_reproducible() -> None:
    """Same rng and noise_rng seeds produce identical bootstrap distributions."""
    feedstock = _make_feedstock()
    result_a = bootstrap_weighted_feedstock(
        feedstock_samples=feedstock,
        batch_weights={"batch_a": 1.0, "batch_b": 1.0, "batch_c": 1.0},
        value_columns=["ca"],
        batch_column="feedstock_batch_id",
        rng=np.random.default_rng(0),
        n_runs=500,
        noise_rng=np.random.default_rng(99),
        noise_fractions={"ca": 0.1},
    )
    result_b = bootstrap_weighted_feedstock(
        feedstock_samples=feedstock,
        batch_weights={"batch_a": 1.0, "batch_b": 1.0, "batch_c": 1.0},
        value_columns=["ca"],
        batch_column="feedstock_batch_id",
        rng=np.random.default_rng(0),
        n_runs=500,
        noise_rng=np.random.default_rng(99),
        noise_fractions={"ca": 0.1},
    )
    np.testing.assert_array_equal(
        result_a.bootstrap_distributions["ca"],
        result_b.bootstrap_distributions["ca"],
    )


def test_bootstrap_noise_differs_from_no_noise() -> None:
    """With noise, bootstrap distributions differ from the noiseless case."""
    feedstock = _make_feedstock()
    no_noise = bootstrap_weighted_feedstock(
        feedstock_samples=feedstock,
        batch_weights={"batch_a": 1.0, "batch_b": 1.0, "batch_c": 1.0},
        value_columns=["ca"],
        batch_column="feedstock_batch_id",
        rng=np.random.default_rng(0),
        n_runs=500,
    )
    with_noise = bootstrap_weighted_feedstock(
        feedstock_samples=feedstock,
        batch_weights={"batch_a": 1.0, "batch_b": 1.0, "batch_c": 1.0},
        value_columns=["ca"],
        batch_column="feedstock_batch_id",
        rng=np.random.default_rng(0),
        n_runs=500,
        noise_rng=np.random.default_rng(99),
        noise_fractions={"ca": 0.1},
    )
    assert not np.allclose(
        no_noise.bootstrap_distributions["ca"],
        with_noise.bootstrap_distributions["ca"],
    )


def test_bootstrap_noise_widens_distribution() -> None:
    """Noise increases the std of the bootstrap distribution."""
    feedstock = _make_feedstock()
    no_noise = bootstrap_weighted_feedstock(
        feedstock_samples=feedstock,
        batch_weights={"batch_a": 1.0, "batch_b": 1.0, "batch_c": 1.0},
        value_columns=["ca"],
        batch_column="feedstock_batch_id",
        rng=np.random.default_rng(0),
        n_runs=10_000,
    )
    with_noise = bootstrap_weighted_feedstock(
        feedstock_samples=feedstock,
        batch_weights={"batch_a": 1.0, "batch_b": 1.0, "batch_c": 1.0},
        value_columns=["ca"],
        batch_column="feedstock_batch_id",
        rng=np.random.default_rng(0),
        n_runs=10_000,
        noise_rng=np.random.default_rng(1),
        noise_fractions={"ca": 0.1},
    )
    assert float(np.std(with_noise.bootstrap_distributions["ca"])) > float(
        np.std(no_noise.bootstrap_distributions["ca"]),
    )


def test_bootstrap_noise_rng_none_ignores_noise_fraction() -> None:
    """noise_fractions is ignored when noise_rng is None."""
    feedstock = _make_feedstock()
    base = bootstrap_weighted_feedstock(
        feedstock_samples=feedstock,
        batch_weights={"batch_a": 1.0, "batch_b": 1.0, "batch_c": 1.0},
        value_columns=["ca"],
        batch_column="feedstock_batch_id",
        rng=np.random.default_rng(0),
        n_runs=500,
    )
    with_fraction_no_rng = bootstrap_weighted_feedstock(
        feedstock_samples=feedstock,
        batch_weights={"batch_a": 1.0, "batch_b": 1.0, "batch_c": 1.0},
        value_columns=["ca"],
        batch_column="feedstock_batch_id",
        rng=np.random.default_rng(0),
        n_runs=500,
        noise_fractions={"ca": 0.1},
    )
    np.testing.assert_array_equal(
        base.bootstrap_distributions["ca"],
        with_fraction_no_rng.bootstrap_distributions["ca"],
    )


def test_bootstrap_partial_batch_renormalises() -> None:
    """A batch in weights with no samples is silently skipped; remaining weights sum to 1."""
    # batch_MISSING has no samples — its weight should be excluded and the rest renormalised.
    # With equal weights 1:1, dropping one leaves batch_a with full weight → result = batch_a mean.
    feedstock = _make_feedstock()
    batch_a_ca_mean = float(feedstock[feedstock["feedstock_batch_id"] == "batch_a"]["ca"].mean())

    result = bootstrap_weighted_feedstock(
        feedstock,
        {"batch_a": 1.0, "batch_MISSING": 1.0},
        ["ca"],
        batch_column="feedstock_batch_id",
        rng=np.random.default_rng(0),
        n_runs=10,
    )
    assert result.point_estimate["ca"].iloc[0] == pytest.approx(batch_a_ca_mean)


def test_weighted_composition_raises_on_zero_total_weight() -> None:
    with pytest.raises(ValueError, match="positive value"):
        compute_weighted_feedstock_composition(
            _make_feedstock(),
            batch_weights={},
            value_columns=["ca", "mg"],
        )


def test_weighted_composition_raises_when_no_weighted_batch_present() -> None:
    with pytest.raises(ValueError, match="present in feedstock_samples"):
        compute_weighted_feedstock_composition(
            _make_feedstock(),
            batch_weights={"batch_z": 1.0},
            value_columns=["ca", "mg"],
        )


def test_bootstrap_raises_on_zero_total_weight() -> None:
    with pytest.raises(ValueError, match="positive value"):
        bootstrap_weighted_feedstock(
            _make_feedstock(),
            batch_weights={},
            value_columns=["ca", "mg"],
            batch_column="feedstock_batch_id",
            rng=np.random.default_rng(0),
            n_runs=10,
        )


def test_bootstrap_raises_when_no_batch_has_complete_samples() -> None:
    with pytest.raises(ValueError, match="complete"):
        bootstrap_weighted_feedstock(
            _make_feedstock(),
            batch_weights={"batch_z": 1.0},
            value_columns=["ca", "mg"],
            batch_column="feedstock_batch_id",
            rng=np.random.default_rng(0),
            n_runs=10,
        )
