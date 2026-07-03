# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

"""Clipped normal distribution utility for Monte Carlo simulations."""

import numpy as np


def clipped_normal(
    size: int | tuple[int, ...],
    rng: np.random.Generator,
    *,
    clip_sigmas: float = 6,
) -> np.ndarray:
    """Standard normal draws clipped to +/- clip_sigmas standard deviations (default 6)."""
    return rng.standard_normal(size).clip(-clip_sigmas, clip_sigmas)
