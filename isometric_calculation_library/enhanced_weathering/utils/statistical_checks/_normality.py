# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

import numpy as np
from scipy import stats

from isometric_calculation_library.utils.types import Np1DArray


def check_normality(
    samples: Np1DArray[np.floating],
    significance_level: float = 0.05,
) -> bool:
    """Check normality using Shapiro-Wilk test.

    Returns True if the null hypothesis of normality is not rejected.
    Requires at least 3 samples; returns False otherwise.
    """
    if len(samples) < 3:
        return False
    _, p_value = stats.shapiro(samples)
    return float(p_value) >= significance_level
