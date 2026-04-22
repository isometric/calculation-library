# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

import numpy as np
from numpy.testing import assert_allclose

from ..clipped_normal import clipped_normal


def test_clipped_normal_output_shape_tuple() -> None:
    result = clipped_normal((100, 50), np.random.default_rng(42))
    assert result.shape == (100, 50)


def test_clipped_normal_output_shape_scalar() -> None:
    result = clipped_normal(1000, np.random.default_rng(42))
    assert result.shape == (1000,)


def test_clipped_normal_clipping_bounds() -> None:
    result = clipped_normal(100_000, np.random.default_rng(42), sigma=3)
    assert np.all(result >= -3)
    assert np.all(result <= 3)


def test_clipped_normal_default_sigma_6() -> None:
    result = clipped_normal(100_000, np.random.default_rng(42))
    assert np.all(result >= -6)
    assert np.all(result <= 6)


def test_clipped_normal_approximately_standard_normal() -> None:
    result = clipped_normal(100_000, np.random.default_rng(42))
    assert abs(np.mean(result)) < 0.02
    assert abs(np.std(result) - 1.0) < 0.02


def test_clipped_normal_deterministic() -> None:
    r1 = clipped_normal(100, np.random.default_rng(99))
    r2 = clipped_normal(100, np.random.default_rng(99))
    assert_allclose(r1, r2)
