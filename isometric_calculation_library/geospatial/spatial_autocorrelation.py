# Copyright (c) 2026 Isometric HQ Ltd
# Licensed under PolyForm Noncommercial 1.0.0
# https://polyformproject.org/licenses/noncommercial/1.0.0/

"""Spatial autocorrelation testing for bootstrap validity.

Implements Global Moran's I with a permutation test per protocol Eq. 15,
Benjamini-Hochberg correction for multiple testing, and effective sample
size (n_eff) derivation for bootstrap resampling adjustment.

When spatial autocorrelation is detected in paired differences,
the bootstrap should resample n_eff < n locations per replicate to account for
reduced independent information content.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypedDict

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

from isometric_calculation_library.enhanced_weathering.utils.types import Np1DArray

_METERS_PER_DEGREE = 111_000
"""Approximate metres per degree of latitude (and longitude at equator)."""


@dataclass(frozen=True)
class MoransIResult:
    """Result from Moran's I permutation test for one variable."""

    variable: str
    """Column name of the variable tested (e.g. a tracer delta column)."""
    observed_i: float
    """Observed Moran's I statistic: 0 = no autocorrelation, 1 = perfect positive, -1 = perfect negative."""
    expected_i: float
    """Mean Moran's I across permutations (approximates the null expectation)."""
    std_i: float
    """Standard deviation of the permutation distribution."""
    z_score: float
    """Standardised effect size: (observed_i - expected_i) / std_i."""
    p_value: float
    """Benjamini-Hochberg adjusted p-value."""
    significant: bool
    """True if Benjamini-Hochberg adjusted p < 0.05 AND |z_I| >= 2."""


class PerVariableDetails(TypedDict):
    """Per-variable Moran's I details stored in NeffResult."""

    I_obs: float
    """Observed Moran's I statistic."""
    z_score: float
    """Standardised effect size."""
    p_adj: float
    """Benjamini-Hochberg adjusted p-value."""
    significant: bool
    """True if BH-adjusted p < 0.05 AND |z_I| >= 2."""
    d_eff: float
    """Design effect: n / n_eff."""
    n_eff: float
    """Effective sample size for this variable."""


@dataclass(frozen=True)
class NeffResult:
    """Effective sample size derived from spatial autocorrelation analysis."""

    n: int
    """Original number of locations."""

    n_eff: float
    """Effective sample size (minimum across variables). Floored at 1.0."""

    n_eff_int: int
    """Integer n_eff for use in bootstrap (rounded, minimum 2).

    Note: floored at 2 rather than 1 because resampling fewer than 2 observations
    is degenerate. This means n_eff_int may exceed n_eff when n_eff is in (1.0, 1.5).
    """

    per_variable: dict[str, PerVariableDetails]
    """Per-variable details: I_obs, z_score, p_adj, significant, d_eff, n_eff."""

    morans_results: list[MoransIResult]
    """Full Moran's I test results for each variable."""


def _build_inverse_distance_weights(coords: np.ndarray) -> np.ndarray:
    """Build row-normalised inverse-distance weight matrix.

    Converts lat/lon to approximate metres before computing distances:
      lat_m = lat * _METERS_PER_DEGREE
      lon_m = lon * _METERS_PER_DEGREE * cos(mean_lat)

    Weight w_ij = 1/d_ij, w_ii = 0, row-normalised so sum_j w_ij = 1.
    """
    lat_rad = np.radians(coords[:, 0].mean())
    coords_m = np.empty_like(coords)
    coords_m[:, 0] = coords[:, 0] * _METERS_PER_DEGREE
    coords_m[:, 1] = coords[:, 1] * _METERS_PER_DEGREE * np.cos(lat_rad)

    dist_matrix = squareform(pdist(coords_m, metric="euclidean"))

    with np.errstate(divide="ignore"):
        w = np.where(dist_matrix > 0, 1.0 / dist_matrix, 0.0)
    np.fill_diagonal(w, 0.0)

    row_sums = w.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    w /= row_sums

    return w


def _compute_morans_i(values: np.ndarray, weights: np.ndarray) -> float:
    """Compute Global Moran's I (Protocol Eq. 15).

    For row-normalised weights where W = n, the formula simplifies to:
      I = sum_i sum_j w_ij (y_i - ybar)(y_j - ybar) / sum_i (y_i - ybar)^2
    """
    y_dev = values - values.mean()
    denominator = float(np.dot(y_dev, y_dev))
    if denominator == 0:
        return 0.0
    numerator = float(np.dot(y_dev, weights @ y_dev))
    return numerator / denominator


def compute_morans_i_permutation_test(
    paired: pd.DataFrame,
    variables: Sequence[str],
    n_permutations: int = 1_999,
    rng: np.random.Generator | None = None,
) -> list[MoransIResult]:
    """Moran's I permutation test for spatial autocorrelation (Protocol Eq. 15).

    Tests values at measurement locations using inverse-distance weights.
    Significance requires BOTH:
      - Benjamini-Hochberg adjusted p-value < 0.05
      - |z_I| >= 2 (standardised effect size)

    Args:
        paired: DataFrame with ``latitude``, ``longitude``, and variable columns.
        variables: Column names to test (typically EoRP - baseline differences).
        n_permutations: Number of random permutations (protocol requires >= 1000).
        rng: Random number generator. Pass an explicit seeded generator for
            reproducibility; omitting this will produce non-deterministic results.
    """
    if rng is None:
        raise ValueError(
            "rng is required — pass an explicit np.random.Generator for reproducibility. "
            "Example: rng=np.random.default_rng(42)",
        )
    coords = paired[["latitude", "longitude"]].to_numpy()
    weights = _build_inverse_distance_weights(coords)

    raw_results = list[tuple[str, float, float, float, float, float]]()

    for var in variables:
        values = paired[var].to_numpy().astype(np.float64)
        observed_i = _compute_morans_i(values, weights)

        # Permutation distribution: randomise values across fixed locations
        perm_i = np.empty(n_permutations)
        for p in range(n_permutations):
            perm_values = rng.permutation(values)
            perm_i[p] = _compute_morans_i(perm_values, weights)

        expected_i = float(perm_i.mean())
        std_i = float(perm_i.std())
        if std_i == 0:
            raise ValueError(
                f"Permutation distribution std is zero for variable {var!r} — "
                "all permutations produced identical Moran's I values. "
                "Check that the input variable is not constant.",
            )
        z_score = (observed_i - expected_i) / std_i

        # Two-sided p-value: fraction of permutations whose deviation from the null mean
        # is at least as large as the observed deviation (Protocol Eq. 16). Centring on
        # expected_i (rather than 0) is consistent with the z-score definition and avoids
        # underestimating p when the permutation distribution is asymmetric.
        # +1 in numerator and denominator gives a conservative, unbiased estimate.
        n_extreme = int(np.sum(np.abs(perm_i - expected_i) >= abs(observed_i - expected_i)))
        p_value = (n_extreme + 1) / (n_permutations + 1)

        raw_results.append((var, observed_i, expected_i, std_i, z_score, p_value))

    # Benjamini-Hochberg correction
    p_values = np.array([r[5] for r in raw_results])
    bh_adjusted = _benjamini_hochberg(p_values)

    results = list[MoransIResult]()
    for i, (var, obs_i, exp_i, std_i, z, _raw_p) in enumerate(raw_results):
        adj_p = float(bh_adjusted[i])
        significant = (adj_p < 0.05) and (abs(z) >= 2.0)
        results.append(
            MoransIResult(
                variable=var,
                observed_i=obs_i,
                expected_i=exp_i,
                std_i=std_i,
                z_score=z,
                p_value=adj_p,
                significant=significant,
            ),
        )

    return results


def _benjamini_hochberg(p_values: Np1DArray[np.floating]) -> Np1DArray[np.floating]:
    """Apply Benjamini-Hochberg FDR correction to p-values."""
    n_tests = len(p_values)
    if n_tests == 0:
        return p_values.copy()

    sorted_indices = np.argsort(p_values)
    bh_adjusted = np.empty(n_tests)

    for rank_idx, orig_idx in enumerate(sorted_indices):
        rank = rank_idx + 1
        bh_adjusted[orig_idx] = p_values[orig_idx] * n_tests / rank

    # Enforce monotonicity (step down from largest rank)
    for i in range(n_tests - 2, -1, -1):
        idx = sorted_indices[i]
        idx_next = sorted_indices[i + 1]
        bh_adjusted[idx] = min(bh_adjusted[idx], bh_adjusted[idx_next])

    return np.clip(bh_adjusted, 0.0, 1.0)


def compute_neff_from_morans_i(
    paired: pd.DataFrame,
    variables: Sequence[str],
    n_permutations: int = 1_999,
    rng: np.random.Generator | None = None,
) -> NeffResult:
    """Compute effective sample size from Moran's I spatial autocorrelation.

    Procedure:
      1. Compute Moran's I for each variable (EoRP - baseline differences).
      2. Run permutation test to assess significance (Benjamini-Hochberg corrected).
      3. For significant variables with I > 0: n_eff = n * (1 - I) / (1 + I).
      4. Take minimum n_eff across variables (most conservative).

    Args:
        paired: DataFrame with ``latitude``, ``longitude``, and variable columns.
        variables: Column names to test.
        n_permutations: Number of permutations (>= 1000 per protocol).
        rng: Random number generator. Pass an explicit seeded generator for
            reproducibility; omitting this will produce non-deterministic results.
    """
    if rng is None:
        raise ValueError(
            "rng is required — pass an explicit np.random.Generator for reproducibility. "
            "Example: rng=np.random.default_rng(42)",
        )
    morans_results = compute_morans_i_permutation_test(
        paired,
        variables,
        n_permutations=n_permutations,
        rng=rng,
    )

    n = len(paired)
    per_variable = dict[str, PerVariableDetails]()
    n_effs = list[float]()

    for mr in morans_results:
        if mr.significant and mr.observed_i > 0:
            n_eff = max(n * (1 - mr.observed_i) / (1 + mr.observed_i), 1.0)
        else:
            n_eff = float(n)

        deff = n / n_eff

        per_variable[mr.variable] = PerVariableDetails(
            I_obs=mr.observed_i,
            z_score=mr.z_score,
            p_adj=mr.p_value,
            significant=mr.significant,
            d_eff=deff,
            n_eff=n_eff,
        )
        n_effs.append(n_eff)

    min_n_eff = min(n_effs) if n_effs else float(n)

    return NeffResult(
        n=n,
        n_eff=min_n_eff,
        n_eff_int=max(2, int(np.round(min_n_eff))),
        per_variable=per_variable,
        morans_results=morans_results,
    )
