# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

import numpy as np
from numpy.testing import assert_allclose

from ..chave import CHAVE_DEFAULT, ChaveModel, create_chave_model_generator

# --- ChaveModel.compute_agb_tonnes ---


def test_compute_agb_tonnes_single_tree() -> None:
    """Chave 2014 default: AGB = 0.0673 * (wd * H * D^2)^0.976 / 1000."""
    dbh = np.array([20.0])
    height = np.array([15.0])
    wd = np.array([0.5])

    result = CHAVE_DEFAULT.compute_agb_tonnes(dbh, height, wd)

    expected = 0.0673 * (0.5 * 15 * 400) ** 0.976 / 1000
    assert_allclose(result, [expected], rtol=1e-10)


def test_compute_agb_tonnes_vectorized() -> None:
    dbh = np.array([10.0, 20.0, 30.0])
    height = np.array([8.0, 15.0, 22.0])
    wd = np.array([0.4, 0.5, 0.6])

    result = CHAVE_DEFAULT.compute_agb_tonnes(dbh, height, wd)

    assert result.shape == (3,)
    assert result[0] < result[1] < result[2]


def test_compute_agb_tonnes_custom_params() -> None:
    model = ChaveModel(gain=0.065, power=0.975)
    dbh = np.array([20.0])
    height = np.array([15.0])
    wd = np.array([0.5])

    result = model.compute_agb_tonnes(dbh, height, wd)

    expected = 0.065 * (0.5 * 15 * 400) ** 0.975 / 1000
    assert_allclose(result, [expected], rtol=1e-10)


def test_compute_agb_tonnes_bootstrap_params() -> None:
    """Bootstrap parameterization: AGB = (gain * wd * H * D^2)^power / 1000."""
    model = ChaveModel(gain=0.065, power=0.975, bootstrap_params=True)
    dbh = np.array([20.0])
    height = np.array([15.0])
    wd = np.array([0.5])

    result = model.compute_agb_tonnes(dbh, height, wd)

    expected = (0.065 * 0.5 * 15 * 400) ** 0.975 / 1000
    assert_allclose(result, [expected], rtol=1e-10)


def test_bootstrap_params_mean_matches_published() -> None:
    """Bootstrap models should produce unbiased AGB relative to the published equation."""
    rng = np.random.default_rng(42)
    gen = create_chave_model_generator(rng)
    models = [gen() for _ in range(1001)]

    dbh = np.array([20.0])
    height = np.array([15.0])
    wd = np.array([0.5])

    published = CHAVE_DEFAULT.compute_agb_tonnes(dbh, height, wd)[0]
    bootstrap_mean = np.mean([m.compute_agb_tonnes(dbh, height, wd)[0] for m in models])

    assert_allclose(bootstrap_mean, published, rtol=0.02)


def test_compute_agb_tonnes_positive_for_positive_inputs() -> None:
    rng = np.random.default_rng(42)
    dbh = rng.uniform(5, 50, 100)
    height = rng.uniform(3, 30, 100)
    wd = rng.uniform(0.2, 1.0, 100)

    result = CHAVE_DEFAULT.compute_agb_tonnes(dbh, height, wd)
    assert np.all(result > 0)


# --- ChaveModel.compute_agb_tonnes_with_error ---


def test_compute_agb_tonnes_with_error_shape() -> None:
    dbh = np.array([20.0, 30.0])
    height = np.array([15.0, 22.0])
    wd = np.array([0.5, 0.6])

    result = CHAVE_DEFAULT.compute_agb_tonnes_with_error(dbh, height, wd, np.random.default_rng(42))
    assert result.shape == (2,)
    assert np.all(result >= 0)


def test_compute_agb_tonnes_with_error_log_normal_median() -> None:
    """Log-normal error should have median close to the deterministic value."""
    dbh = np.full(10_000, 20.0)
    height = np.full(10_000, 15.0)
    wd = np.full(10_000, 0.5)

    result = CHAVE_DEFAULT.compute_agb_tonnes_with_error(dbh, height, wd, np.random.default_rng(42))
    deterministic = CHAVE_DEFAULT.compute_agb_tonnes(
        np.array([20.0]),
        np.array([15.0]),
        np.array([0.5]),
    )[0]

    assert_allclose(np.median(result), deterministic, rtol=0.05)


def test_compute_agb_tonnes_with_error_deterministic() -> None:
    dbh = np.array([20.0])
    height = np.array([15.0])
    wd = np.array([0.5])

    r1 = CHAVE_DEFAULT.compute_agb_tonnes_with_error(dbh, height, wd, np.random.default_rng(99))
    r2 = CHAVE_DEFAULT.compute_agb_tonnes_with_error(dbh, height, wd, np.random.default_rng(99))
    assert_allclose(r1, r2)


# --- create_chave_model_generator ---


def test_create_chave_model_generator_returns_bootstrap_model() -> None:
    gen = create_chave_model_generator(np.random.default_rng(42))
    model = gen()
    assert isinstance(model, ChaveModel)
    assert model.bootstrap_params is True


def test_create_chave_model_generator_gain_is_positive() -> None:
    gen = create_chave_model_generator(np.random.default_rng(42))
    models = [gen() for _ in range(100)]
    assert all(m.gain > 0 for m in models)


def test_create_chave_model_generator_power_range() -> None:
    gen = create_chave_model_generator(np.random.default_rng(42))
    models = [gen() for _ in range(1000)]
    mean_power = np.mean([m.power for m in models])
    assert 0.95 < mean_power < 1.0


def test_create_chave_model_generator_deterministic() -> None:
    gen1 = create_chave_model_generator(np.random.default_rng(123))
    gen2 = create_chave_model_generator(np.random.default_rng(123))

    m1 = gen1()
    m2 = gen2()
    assert_allclose(m1.gain, m2.gain)
    assert_allclose(m1.power, m2.power)


def test_create_chave_model_generator_different_seeds() -> None:
    gen1 = create_chave_model_generator(np.random.default_rng(1))
    gen2 = create_chave_model_generator(np.random.default_rng(2))

    m1 = gen1()
    m2 = gen2()
    assert m1.gain != m2.gain or m1.power != m2.power


# --- CHAVE_DEFAULT ---


def test_chave_default_has_published_coefficients() -> None:
    assert_allclose(CHAVE_DEFAULT.gain, 0.0673)
    assert_allclose(CHAVE_DEFAULT.power, 0.976)
    assert CHAVE_DEFAULT.bootstrap_params is False
