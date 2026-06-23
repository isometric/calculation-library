# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

import numpy as np
from numpy.testing import assert_allclose

from isometric_calculation_library.biosphere.allometric_equations.chave import (
    CHAVE_DEFAULT,
    ChaveModel,
)
from isometric_calculation_library.biosphere.constants import (
    CARBON_FRACTION,
    CO2_TO_CARBON_RATIO,
    M2_PER_HECTARE,
)
from isometric_calculation_library.biosphere.monte_carlo.field_plot import (
    FieldPlot,
    TreeMeasurements,
)


def _make_trees(n: int = 3) -> TreeMeasurements:
    return TreeMeasurements(
        dbh_cm=np.array([10.0, 20.0, 30.0][:n]),
        height_m=np.array([8.0, 15.0, 22.0][:n]),
        wood_density_g_cm3=np.array([0.4, 0.5, 0.6][:n]),
        wood_density_sd_g_cm3=np.array([0.05, 0.07, 0.08][:n]),
        height_logsd=np.array([0.1, 0.1, 0.12][:n]),
    )


def _make_plot(n: int = 3, plot_size: float = 40.0) -> FieldPlot:
    return FieldPlot(plot_id="P1", trees=_make_trees(n), plot_size_m=plot_size)


# --- TreeMeasurements ---


def test_tree_measurements_is_frozen() -> None:
    trees = _make_trees()
    try:
        trees.dbh_cm = np.array([1.0])  # pyright: ignore[reportAttributeAccessIssue]
        raise AssertionError("Should be frozen")
    except AttributeError:
        pass


# --- FieldPlot ---


def test_field_plot_num_trees() -> None:
    plot = _make_plot(n=3)
    assert plot.num_trees == 3


def test_field_plot_compute_tco2e_ha_returns_per_tree() -> None:
    plot = _make_plot(n=3)
    result = plot.compute_tco2e_ha(model=CHAVE_DEFAULT, carbon_ratio=CARBON_FRACTION)
    assert result.shape == (3,)
    assert np.all(result > 0)


def test_field_plot_compute_tco2e_ha_matches_manual_calculation() -> None:
    """FieldPlot.compute_tco2e_ha should match manual AGB -> tCO2e/ha conversion."""
    plot = _make_plot()
    plot_result = plot.compute_tco2e_ha(model=CHAVE_DEFAULT, carbon_ratio=CARBON_FRACTION)

    agb_tonnes = CHAVE_DEFAULT.compute_agb_tonnes(
        plot.trees.dbh_cm,
        plot.trees.height_m,
        plot.trees.wood_density_g_cm3,
    )
    plot_area_ha = (plot.plot_size_m**2) / M2_PER_HECTARE
    expected = agb_tonnes / plot_area_ha * CARBON_FRACTION * CO2_TO_CARBON_RATIO
    assert_allclose(plot_result, expected)


def test_field_plot_plot_size_affects_result() -> None:
    plot_40 = _make_plot(plot_size=40.0)
    plot_100 = _make_plot(plot_size=100.0)

    result_40 = plot_40.compute_tco2e_ha(model=CHAVE_DEFAULT, carbon_ratio=CARBON_FRACTION)
    result_100 = plot_100.compute_tco2e_ha(model=CHAVE_DEFAULT, carbon_ratio=CARBON_FRACTION)

    assert_allclose(result_40 / result_100, [(100 / 40) ** 2] * 3, rtol=1e-10)


def test_field_plot_custom_model() -> None:
    plot = _make_plot()
    custom = ChaveModel(gain=0.065, power=0.975)
    result = plot.compute_tco2e_ha(model=custom, carbon_ratio=CARBON_FRACTION)
    assert result.shape == (3,)
    assert np.all(result > 0)


def test_field_plot_compute_tco2e_ha_with_error() -> None:
    plot = _make_plot()
    result = plot.compute_tco2e_ha_with_error(
        model=CHAVE_DEFAULT,
        carbon_ratio=CARBON_FRACTION,
        rng=np.random.default_rng(42),
    )
    assert result.shape == (3,)
    assert np.all(result >= 0)


def test_field_plot_compute_tco2e_ha_with_error_deterministic() -> None:
    plot = _make_plot()
    r1 = plot.compute_tco2e_ha_with_error(
        model=CHAVE_DEFAULT,
        carbon_ratio=CARBON_FRACTION,
        rng=np.random.default_rng(99),
    )
    r2 = plot.compute_tco2e_ha_with_error(
        model=CHAVE_DEFAULT,
        carbon_ratio=CARBON_FRACTION,
        rng=np.random.default_rng(99),
    )
    assert_allclose(r1, r2)
