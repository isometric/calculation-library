# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

import numpy as np
from numpy.testing import assert_allclose

from ...constants import DBH_ERROR_INTERCEPT, DBH_ERROR_SLOPE
from ...utils.dbh import DBH_CLIP_MAX_CM, DBH_CLIP_MIN_CM
from ...utils.height import HEIGHT_CLIP_MAX_M, HEIGHT_CLIP_MIN_M
from ..inventory import (
    CARBON_RATIO_MEAN,
    CARBON_RATIO_SD,
    MONTE_CARLO_VARIANTS,
    WOOD_DENSITY_MAX,
    WOOD_DENSITY_MIN,
    inventory_monte_carlo,
)

_TREE_DATA_KEYS = {
    "dbh_cm",
    "dbh_with_big",
    "wood_density",
    "height_m",
    "height_geometric",
    "dbh_cov",
    "height_cov",
    "carbon_ratio",
}


def _make_tree_data(n_trees: int = 5) -> dict[str, np.ndarray]:
    return {
        "dbh_cm": np.full(n_trees, 20.0),
        "wood_density_g_cm3": np.full(n_trees, 0.5),
        "wood_density_sd_g_cm3": np.full(n_trees, 0.07),
        "height_m": np.full(n_trees, 15.0),
        "height_logsd": np.full(n_trees, 0.1),
    }


# --- inventory_monte_carlo ---


def test_inventory_monte_carlo_output_keys() -> None:
    mc = inventory_monte_carlo(**_make_tree_data(), num_sims=10, rng=np.random.default_rng(42))
    assert set(mc.keys()) == _TREE_DATA_KEYS


def test_inventory_monte_carlo_output_shapes() -> None:
    n_trees = 5
    num_sims = 50
    mc = inventory_monte_carlo(
        **_make_tree_data(n_trees),
        num_sims=num_sims,
        rng=np.random.default_rng(42),
    )

    for key, arr in mc.items():
        assert arr.shape == (num_sims, n_trees), f"{key} has wrong shape {arr.shape}"


def test_inventory_monte_carlo_dbh_within_bounds() -> None:
    mc = inventory_monte_carlo(**_make_tree_data(), num_sims=100, rng=np.random.default_rng(42))
    assert np.all(mc["dbh_cm"] >= DBH_CLIP_MIN_CM)
    assert np.all(mc["dbh_cm"] <= DBH_CLIP_MAX_CM)
    assert np.all(mc["dbh_with_big"] >= DBH_CLIP_MIN_CM)
    assert np.all(mc["dbh_with_big"] <= DBH_CLIP_MAX_CM)


def test_inventory_monte_carlo_height_within_bounds() -> None:
    mc = inventory_monte_carlo(**_make_tree_data(), num_sims=100, rng=np.random.default_rng(42))
    assert np.all(mc["height_m"] >= HEIGHT_CLIP_MIN_M)
    assert np.all(mc["height_m"] <= HEIGHT_CLIP_MAX_M)
    assert np.all(mc["height_geometric"] >= HEIGHT_CLIP_MIN_M)
    assert np.all(mc["height_geometric"] <= HEIGHT_CLIP_MAX_M)


def test_inventory_monte_carlo_wood_density_within_bounds() -> None:
    mc = inventory_monte_carlo(**_make_tree_data(), num_sims=100, rng=np.random.default_rng(42))
    assert np.all(mc["wood_density"] >= WOOD_DENSITY_MIN)
    assert np.all(mc["wood_density"] <= WOOD_DENSITY_MAX)


def test_inventory_monte_carlo_dbh_error_uses_shared_constants() -> None:
    """The DBH SD formula should use the shared Chave 2004 constants."""
    expected_sd = DBH_ERROR_SLOPE * 20.0 + DBH_ERROR_INTERCEPT
    mc = inventory_monte_carlo(
        dbh_cm=np.array([20.0]),
        wood_density_g_cm3=np.array([0.5]),
        wood_density_sd_g_cm3=np.array([0.07]),
        height_m=np.array([15.0]),
        height_logsd=np.array([0.1]),
        num_sims=10_000,
        rng=np.random.default_rng(42),
    )
    observed_sd = np.std(mc["dbh_cm"][:, 0] - 20.0)
    assert_allclose(observed_sd, expected_sd, rtol=0.1)


def test_inventory_monte_carlo_carbon_ratio_distribution() -> None:
    mc = inventory_monte_carlo(**_make_tree_data(), num_sims=10_000, rng=np.random.default_rng(42))
    assert_allclose(np.mean(mc["carbon_ratio"]), CARBON_RATIO_MEAN, atol=0.005)
    assert_allclose(np.std(mc["carbon_ratio"]), CARBON_RATIO_SD, atol=0.005)


def test_inventory_monte_carlo_deterministic() -> None:
    data = _make_tree_data()
    mc1 = inventory_monte_carlo(**data, num_sims=10, rng=np.random.default_rng(7))
    mc2 = inventory_monte_carlo(**data, num_sims=10, rng=np.random.default_rng(7))
    for key in mc1:
        assert_allclose(mc1[key], mc2[key])


# --- MONTE_CARLO_VARIANTS ---


def test_monte_carlo_variants_maps_alternate_names_to_base() -> None:
    assert MONTE_CARLO_VARIANTS["height_cov"] == "height_m"
    assert MONTE_CARLO_VARIANTS["height_geometric"] == "height_m"
    assert MONTE_CARLO_VARIANTS["dbh_cov"] == "dbh_cm"
    assert MONTE_CARLO_VARIANTS["dbh_with_big"] == "dbh_cm"
