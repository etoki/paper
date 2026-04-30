"""Bootstrap confidence intervals with Section 5.1 / M4 fallback chain.

CI selection priority (per Methods Clarifications Log Section 3.4 = M4):
    1. Degenerate cells (X_c ∈ {0, N_c}): exact Clopper-Pearson interval.
    2. BCa (primary, when |a| ≤ 10 and computation succeeds).
    3. BC (bias-corrected) bootstrap fallback.
    4. Percentile bootstrap (final fallback).

BCa acceleration parameter (m4): jackknife per Efron (1987) Eq. 6.6.

Cell-stratified resampling (M3 + v2.0 Section 5.4 step 1): individuals
are resampled with replacement WITHIN each cell to preserve cell
membership marginals; centroids and population weights are FIXED.

Bootstrap iteration counts (m3):
    - Per-cell CI default: B = 2,000
    - Headline national MAPE CI: B = 10,000
    - Counterfactual ΔP_x CI: B = 2,000

Random seed: 20260429 (per v2.0 Section 2.4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Sequence

import numpy as np
from numpy.random import Generator
from scipy import stats

from .utils_io import BOOTSTRAP_HEADLINE_MAPE, BOOTSTRAP_PER_CELL, make_rng

# ====================================================================
# Constants
# ====================================================================

ACCELERATION_LIMIT = 10.0
"""Per M4 + v2.0 Section 5.1: |a| > 10 triggers BC fallback."""

CI_LEVEL = 0.95
"""Default CI coverage."""

CIMethod = Literal["clopper_pearson", "bca", "bc", "percentile", "eb_posterior"]
"""Tag identifying which CI method was actually used for a cell."""


@dataclass(frozen=True)
class CellCI:
    """Confidence interval for a single cell's binary-outcome propensity."""

    point: float
    """Point estimate p̂_c = X_c / N_c."""

    lower: float
    """Lower bound of the 95% CI."""

    upper: float
    """Upper bound of the 95% CI."""

    method: CIMethod
    """Which CI method was actually used (per M4 priority chain)."""

    n: int
    """Cell sample size."""

    successes: int
    """X_c, the count of binary outcome = 1."""


# ====================================================================
# 1) Clopper-Pearson exact binomial interval (M4 degenerate-cell fallback)
# ====================================================================


def clopper_pearson_interval(
    successes: int, n: int, level: float = CI_LEVEL
) -> tuple[float, float]:
    """Exact two-sided Clopper-Pearson interval for binomial proportion.

    Used when X_c ∈ {0, N_c} (degenerate cell) per clarifications log
    Section 3.4 (M4). Built on scipy ``binomtest`` for numerical
    correctness at boundary cases.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if successes < 0 or successes > n:
        raise ValueError("successes must satisfy 0 ≤ X ≤ n")
    test = stats.binomtest(successes, n)
    ci = test.proportion_ci(confidence_level=level, method="exact")
    return float(ci.low), float(ci.high)


# ====================================================================
# 2) Jackknife acceleration parameter (m4: Efron 1987 Eq. 6.6)
# ====================================================================


def jackknife_acceleration(
    statistic_fn: Callable[[np.ndarray], float],
    data: np.ndarray,
) -> float:
    """Compute BCa acceleration parameter `a` via jackknife.

    Per m4 (clarifications log Section 4.4):

        a = Σ (θ_·− θ_(i))^3 / [6 (Σ (θ_· − θ_(i))^2)^(3/2)]

    where θ_(i) is the leave-one-out estimate and θ_· is the jackknife
    mean. Returns 0.0 if the denominator is zero / near-zero (degenerate).
    """
    arr = np.asarray(data)
    n = len(arr)
    if n < 2:
        return 0.0
    jackknife_estimates = np.empty(n, dtype=float)
    idx_full = np.arange(n)
    for i in range(n):
        mask = idx_full != i
        jackknife_estimates[i] = float(statistic_fn(arr[mask]))
    theta_dot = float(np.mean(jackknife_estimates))
    diffs = theta_dot - jackknife_estimates
    numerator = float(np.sum(diffs**3))
    denominator = 6.0 * float(np.sum(diffs**2)) ** 1.5
    if denominator <= 0 or not np.isfinite(denominator):
        return 0.0
    return numerator / denominator


# ====================================================================
# 3) BCa CI computation (primary method)
# ====================================================================


def bca_interval(
    bootstrap_estimates: np.ndarray,
    point_estimate: float,
    statistic_fn: Callable[[np.ndarray], float],
    data: np.ndarray,
    level: float = CI_LEVEL,
) -> tuple[float, float, float]:
    """Compute BCa CI given a bootstrap distribution and original data.

    Returns ``(lower, upper, a)``. The acceleration ``a`` is returned
    separately so callers can decide whether to fall back to BC when
    ``|a| > ACCELERATION_LIMIT``.

    Per m4: jackknife per Efron 1987 Eq. 6.6.
    """
    boot = np.asarray(bootstrap_estimates, dtype=float)
    if len(boot) < 2:
        raise ValueError("bootstrap distribution must have ≥ 2 samples")
    alpha = (1.0 - level) / 2.0

    # Bias correction z₀ = Φ^{-1}(p) where p = (#{boot < theta_hat}) / B
    p_below = float(np.mean(boot < point_estimate))
    p_below = np.clip(p_below, 1.0 / (2 * len(boot)), 1.0 - 1.0 / (2 * len(boot)))
    z0 = float(stats.norm.ppf(p_below))

    # Acceleration via jackknife on original data
    a = jackknife_acceleration(statistic_fn, data)

    # Adjusted quantile probabilities
    z_alpha_lo = stats.norm.ppf(alpha)
    z_alpha_hi = stats.norm.ppf(1 - alpha)
    p_lo = float(stats.norm.cdf(z0 + (z0 + z_alpha_lo) / (1 - a * (z0 + z_alpha_lo))))
    p_hi = float(stats.norm.cdf(z0 + (z0 + z_alpha_hi) / (1 - a * (z0 + z_alpha_hi))))
    p_lo = float(np.clip(p_lo, 0.0, 1.0))
    p_hi = float(np.clip(p_hi, 0.0, 1.0))

    lower = float(np.quantile(boot, p_lo))
    upper = float(np.quantile(boot, p_hi))
    return lower, upper, a


# ====================================================================
# 4) BC (bias-corrected only) fallback
# ====================================================================


def bc_interval(
    bootstrap_estimates: np.ndarray,
    point_estimate: float,
    level: float = CI_LEVEL,
) -> tuple[float, float]:
    """Bias-corrected (no acceleration) bootstrap CI."""
    boot = np.asarray(bootstrap_estimates, dtype=float)
    alpha = (1.0 - level) / 2.0
    p_below = float(np.mean(boot < point_estimate))
    p_below = np.clip(p_below, 1.0 / (2 * len(boot)), 1.0 - 1.0 / (2 * len(boot)))
    z0 = float(stats.norm.ppf(p_below))
    z_alpha_lo = stats.norm.ppf(alpha)
    z_alpha_hi = stats.norm.ppf(1 - alpha)
    p_lo = float(stats.norm.cdf(2 * z0 + z_alpha_lo))
    p_hi = float(stats.norm.cdf(2 * z0 + z_alpha_hi))
    return float(np.quantile(boot, p_lo)), float(np.quantile(boot, p_hi))


# ====================================================================
# 5) Percentile (final fallback)
# ====================================================================


def percentile_interval(
    bootstrap_estimates: np.ndarray,
    level: float = CI_LEVEL,
) -> tuple[float, float]:
    """Naive percentile bootstrap CI."""
    boot = np.asarray(bootstrap_estimates, dtype=float)
    alpha = (1.0 - level) / 2.0
    return (
        float(np.quantile(boot, alpha)),
        float(np.quantile(boot, 1 - alpha)),
    )


# ====================================================================
# 6) M4 priority chain: full CI selection logic for binary cell data
# ====================================================================


def cell_proportion_ci(
    successes: int,
    n: int,
    rng: Generator | None = None,
    n_bootstrap: int = BOOTSTRAP_PER_CELL,
    level: float = CI_LEVEL,
) -> CellCI:
    """Compute a 95% CI for a binary-outcome cell proportion p̂ = X / n.

    Implements the M4 priority chain:
        1. If X ∈ {0, n} → exact Clopper-Pearson.
        2. Else compute BCa; if |a| > 10 or BCa fails numerically, use BC.
        3. If BC also fails (rare), use percentile.

    The actually-used method is recorded in ``CellCI.method`` so that
    the supplementary table (per M4 last paragraph) can document which
    fallback was triggered for each cell.
    """
    if rng is None:
        rng = make_rng(extra_offset=hash(("cell_proportion_ci", successes, n)) % 1_000_000)
    if n <= 0:
        raise ValueError("n must be positive")
    if successes < 0 or successes > n:
        raise ValueError("successes must satisfy 0 ≤ X ≤ n")

    p_hat = successes / n

    # 1) Degenerate-cell fallback
    if successes == 0 or successes == n:
        lo, hi = clopper_pearson_interval(successes, n, level=level)
        return CellCI(
            point=p_hat,
            lower=lo,
            upper=hi,
            method="clopper_pearson",
            n=n,
            successes=successes,
        )

    # Bootstrap distribution of the proportion (cell-stratified
    # resampling at the individual level within this single cell).
    indicators = np.zeros(n, dtype=int)
    indicators[:successes] = 1  # arbitrary order; resample makes order irrelevant
    boot = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        sample = rng.choice(indicators, size=n, replace=True)
        boot[b] = float(np.mean(sample))

    # 2) BCa primary
    try:
        lo, hi, a = bca_interval(
            bootstrap_estimates=boot,
            point_estimate=p_hat,
            statistic_fn=lambda x: float(np.mean(x)),
            data=indicators,
            level=level,
        )
        if not (np.isnan(lo) or np.isnan(hi)) and abs(a) <= ACCELERATION_LIMIT:
            return CellCI(
                point=p_hat,
                lower=lo,
                upper=hi,
                method="bca",
                n=n,
                successes=successes,
            )
    except (ValueError, FloatingPointError):
        pass  # fall through

    # 3) BC fallback
    try:
        lo, hi = bc_interval(boot, p_hat, level=level)
        if not (np.isnan(lo) or np.isnan(hi)):
            return CellCI(
                point=p_hat,
                lower=lo,
                upper=hi,
                method="bc",
                n=n,
                successes=successes,
            )
    except (ValueError, FloatingPointError):
        pass

    # 4) Percentile final fallback
    lo, hi = percentile_interval(boot, level=level)
    return CellCI(
        point=p_hat,
        lower=lo,
        upper=hi,
        method="percentile",
        n=n,
        successes=successes,
    )


# ====================================================================
# 7) Cell-stratified bootstrap for aggregate statistics (M3, m3)
# ====================================================================


def cell_stratified_bootstrap(
    cell_data: Sequence[np.ndarray],
    statistic_fn: Callable[[Sequence[np.ndarray]], float],
    n_bootstrap: int = BOOTSTRAP_HEADLINE_MAPE,
    rng: Generator | None = None,
) -> np.ndarray:
    """Cell-stratified resampling for aggregate (e.g., MAPE) statistics.

    Per v2.0 Section 5.4 + clarifications log M3:
    - Resampling preserves cell membership marginals.
    - Centroids are NOT bootstrapped (fixed parameters per M3).
    - Population weights are NOT bootstrapped (fixed inputs to MAPE).
    - The supplied ``statistic_fn`` consumes resampled cell data and
      returns the aggregate statistic for that bootstrap iteration.

    Parameters
    ----------
    cell_data : sequence of ndarrays
        One array per cell, each containing the within-cell observations
        (e.g., binary outcomes for the harassment propensity). Length is
        14 (main) or 28 (sensitivity).
    statistic_fn : callable
        Maps a sequence of resampled cell arrays to a scalar statistic.
    n_bootstrap : int
        Iteration count (per m3: 10,000 for headline MAPE; 2,000 default).
    rng : Generator, optional

    Returns
    -------
    ndarray of length ``n_bootstrap`` containing bootstrap statistics.
    """
    if rng is None:
        rng = make_rng(extra_offset=300)
    boot_stats = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        resampled = [
            rng.choice(cell, size=len(cell), replace=True) for cell in cell_data
        ]
        boot_stats[b] = float(statistic_fn(resampled))
    return boot_stats


# ====================================================================
# 8) Tier classification helper (v2.0 Section 5.4 4-tier hierarchy)
# ====================================================================


@dataclass(frozen=True)
class TierClassification:
    """4-tier H1 classification result (v2.0 Section 5.4)."""

    tier: Literal["Strict SUCCESS", "Standard SUCCESS", "PARTIAL SUCCESS", "FAILURE"]
    point_mape: float
    ci_lower: float
    ci_upper: float
    explanation: str


def classify_h1_tier(
    point_mape: float,
    ci_lower: float,
    ci_upper: float,
) -> TierClassification:
    """Apply the v2.0 4-tier hierarchy to a (point, CI) pair.

    Tiers:
        Strict SUCCESS = point ≤ 30 AND CI upper ≤ 30
        Standard SUCCESS = point ≤ 30 (CI upper > 30 permitted)
        PARTIAL SUCCESS = 30 < point ≤ 60
        FAILURE = point > 60
    """
    if point_mape <= 30.0 and ci_upper <= 30.0:
        tier = "Strict SUCCESS"
        explanation = (
            f"point MAPE = {point_mape:.2f}%; 95% CI = [{ci_lower:.2f}%, "
            f"{ci_upper:.2f}%]. CI upper bound ≤ 30%; result rules out "
            "PARTIAL/FAILURE regions. Strict confirmatory."
        )
    elif point_mape <= 30.0:
        tier = "Standard SUCCESS"
        explanation = (
            f"point MAPE = {point_mape:.2f}%; 95% CI = [{ci_lower:.2f}%, "
            f"{ci_upper:.2f}%]. CI overlapped the PARTIAL region. "
            "Pre-registered ambiguity Tier; weak confirmatory."
        )
    elif point_mape <= 60.0:
        tier = "PARTIAL SUCCESS"
        explanation = (
            f"point MAPE = {point_mape:.2f}%; 95% CI = [{ci_lower:.2f}%, "
            f"{ci_upper:.2f}%]. Mixed evidence."
        )
    else:
        tier = "FAILURE"
        explanation = (
            f"point MAPE = {point_mape:.2f}%; 95% CI = [{ci_lower:.2f}%, "
            f"{ci_upper:.2f}%]. Point estimate exceeded the 60% failure "
            "threshold; published as failure-mode discovery (Section 7.3)."
        )
    return TierClassification(
        tier=tier,
        point_mape=point_mape,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        explanation=explanation,
    )
