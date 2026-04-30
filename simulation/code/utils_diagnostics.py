"""Diagnostic utilities for MoM, positivity, and assignment checks.

Source specification:
- v2.0 master Section 5.2 (Method-of-moments hyperprior + Stan triangulation)
- Clarifications Log Section 4.1 (m1 — bootstrap-stabilized MoM threshold)
- Clarifications Log Section 4.2 (m2 — 3rd MoM trigger: α̂+β̂ > 5 × median N_k)
- Clarifications Log Section 4.5 (m5 — positivity quantitative criterion)
- Clarifications Log Section 3.2 (M2 — soft-assignment sensitivity)

Random seed: 20260429 (per v2.0 Section 2.4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.random import Generator

from .utils_io import make_rng

# ====================================================================
# Constants per clarifications log
# ====================================================================

MOM_VARIANCE_RATIO_THRESHOLD = 0.05
"""m1: σ̂² / [μ̂(1−μ̂)] threshold below which MoM is rejected."""

MOM_PSEUDOCOUNT_MAX = 100.0
"""v2.0 m1 (existing): max(α̂, β̂) > 100 → reject MoM as overshrunk."""

MOM_PSEUDOCOUNT_MEDIAN_N_RATIO = 5.0
"""m2: α̂+β̂ > 5 × median(N_k) → reject MoM as symmetric overshrink."""

MOM_BOOTSTRAP_ITERATIONS = 200
"""m1: cell-bootstrap iterations for MoM threshold stabilization."""

MOM_BOOTSTRAP_UPPER_QUANTILE = 0.975
"""m1: 97.5th percentile of bootstrapped variance ratio is the rule's input."""

POSITIVITY_RATIO_THRESHOLD = 0.10
"""m5: ρ_{c,x} < 0.10 → cell flagged as positivity-violated for counterfactual x."""

POSITIVITY_FLAGGED_WEIGHT_MAX = 0.20
"""m5: flagged_weight ≥ 20% → counterfactual ΔP_x downgraded confirmatory → exploratory."""

# ====================================================================
# Method-of-moments diagnostics (m1, m2, plus existing v2.0 trigger)
# ====================================================================


@dataclass(frozen=True)
class MoMDiagnostic:
    """Result of MoM hyperprior diagnostic (clarifications log Sections 4.1, 4.2)."""

    mu_hat: float
    """Mean of 14-cell observed propensities."""

    sigma2_hat: float
    """Variance of 14-cell observed propensities."""

    alpha_hat: float
    """MoM Beta hyperparameter α̂."""

    beta_hat: float
    """MoM Beta hyperparameter β̂."""

    variance_ratio: float
    """σ̂² / [μ̂(1−μ̂)] (Trigger 1 input)."""

    variance_ratio_upper95: float
    """Upper 95% bootstrap bound of variance ratio (m1)."""

    pseudocount_total: float
    """α̂ + β̂."""

    pseudocount_median_n_ratio: float
    """(α̂ + β̂) / median(N_k)."""

    median_cell_n: float
    """Median cell N over 28-cell partition."""

    trigger1_fired: bool
    """m1: variance_ratio_upper95 < threshold (variance collapse)."""

    trigger2_fired: bool
    """v2.0 existing: max(α̂, β̂) > 100 (asymmetric overshrink)."""

    trigger3_fired: bool
    """m2: pseudocount_median_n_ratio > 5 (symmetric overshrink)."""

    @property
    def reject_mom(self) -> bool:
        """True if any trigger fires → switch to Stan hierarchical Bayes."""
        return self.trigger1_fired or self.trigger2_fired or self.trigger3_fired

    @property
    def reasons(self) -> tuple[str, ...]:
        """Human-readable list of why MoM was rejected (empty if accepted)."""
        out = []
        if self.trigger1_fired:
            out.append(
                f"m1 variance ratio upper95 = {self.variance_ratio_upper95:.4f} "
                f"< {MOM_VARIANCE_RATIO_THRESHOLD}"
            )
        if self.trigger2_fired:
            out.append(
                f"v2.0 max(α̂, β̂) = {max(self.alpha_hat, self.beta_hat):.1f} "
                f"> {MOM_PSEUDOCOUNT_MAX}"
            )
        if self.trigger3_fired:
            out.append(
                f"m2 (α̂+β̂)/median(N_k) = {self.pseudocount_median_n_ratio:.2f} "
                f"> {MOM_PSEUDOCOUNT_MEDIAN_N_RATIO}"
            )
        return tuple(out)


def compute_mom_hyperprior(propensities: np.ndarray) -> tuple[float, float, float, float]:
    """Compute Beta-Binomial hyperprior (α̂, β̂) from cell propensities.

    Per v2.0 Section 5.2 method-of-moments formula.

    Parameters
    ----------
    propensities : ndarray
        Cell-level observed propensities {p̂_j} for j = 1..K (typically K=14).

    Returns
    -------
    mu_hat, sigma2_hat, alpha_hat, beta_hat : 4-tuple of floats
        Sample moments and the implied Beta hyperparameters.
    """
    p = np.asarray(propensities, dtype=float)
    if p.ndim != 1 or len(p) < 2:
        raise ValueError("propensities must be a 1-D array with length ≥ 2.")
    mu = float(np.mean(p))
    sigma2 = float(np.var(p, ddof=1))
    if sigma2 <= 0 or mu <= 0 or mu >= 1:
        # Degenerate case; downstream MoM diagnostic will flag this.
        return mu, sigma2, float("nan"), float("nan")
    common_factor = mu * (1.0 - mu) / sigma2 - 1.0
    if common_factor <= 0:
        return mu, sigma2, float("nan"), float("nan")
    alpha = mu * common_factor
    beta = (1.0 - mu) * common_factor
    return mu, sigma2, alpha, beta


def diagnose_mom(
    propensities: np.ndarray,
    cell_sizes_28: np.ndarray,
    rng: Generator | None = None,
) -> MoMDiagnostic:
    """Run all three MoM rejection triggers (m1, v2.0 existing, m2).

    Parameters
    ----------
    propensities : ndarray
        14-cell observed propensities.
    cell_sizes_28 : ndarray
        N_k for each of the 28 cells (used in m2 trigger via median).
    rng : Generator, optional
        Defaults to seeded ``make_rng()`` for reproducibility.

    Returns
    -------
    MoMDiagnostic
    """
    if rng is None:
        rng = make_rng(extra_offset=200)  # offset isolates this stream

    mu, sigma2, alpha, beta = compute_mom_hyperprior(propensities)
    var_ratio = sigma2 / (mu * (1.0 - mu)) if 0 < mu < 1 else float("nan")

    # m1: bootstrap stabilization of variance ratio
    p_arr = np.asarray(propensities, dtype=float)
    var_ratios_boot = np.empty(MOM_BOOTSTRAP_ITERATIONS, dtype=float)
    for b in range(MOM_BOOTSTRAP_ITERATIONS):
        sample = rng.choice(p_arr, size=len(p_arr), replace=True)
        m_b = float(np.mean(sample))
        v_b = float(np.var(sample, ddof=1))
        if 0 < m_b < 1 and v_b > 0:
            var_ratios_boot[b] = v_b / (m_b * (1.0 - m_b))
        else:
            var_ratios_boot[b] = np.nan
    var_ratio_upper = float(
        np.nanquantile(var_ratios_boot, MOM_BOOTSTRAP_UPPER_QUANTILE)
    )

    cell_n_28 = np.asarray(cell_sizes_28, dtype=float)
    median_n_28 = float(np.median(cell_n_28))
    pseudocount_total = (
        (alpha + beta) if not (np.isnan(alpha) or np.isnan(beta)) else float("nan")
    )
    pseudo_ratio = (
        pseudocount_total / median_n_28 if median_n_28 > 0 else float("nan")
    )

    trigger1 = (
        not np.isnan(var_ratio_upper)
        and var_ratio_upper < MOM_VARIANCE_RATIO_THRESHOLD
    )
    trigger2 = (
        not (np.isnan(alpha) or np.isnan(beta))
        and max(alpha, beta) > MOM_PSEUDOCOUNT_MAX
    )
    trigger3 = (
        not np.isnan(pseudo_ratio)
        and pseudo_ratio > MOM_PSEUDOCOUNT_MEDIAN_N_RATIO
    )

    return MoMDiagnostic(
        mu_hat=mu,
        sigma2_hat=sigma2,
        alpha_hat=alpha,
        beta_hat=beta,
        variance_ratio=var_ratio,
        variance_ratio_upper95=var_ratio_upper,
        pseudocount_total=pseudocount_total,
        pseudocount_median_n_ratio=pseudo_ratio,
        median_cell_n=median_n_28,
        trigger1_fired=trigger1,
        trigger2_fired=trigger2,
        trigger3_fired=trigger3,
    )


# ====================================================================
# Soft-assignment diagnostic (M2)
# ====================================================================


def soft_assign_weights(
    distances: np.ndarray,
    tau: float,
) -> np.ndarray:
    """Convert a cluster-distance matrix to softmax soft-assignment weights.

    Per Methods Clarifications Log Section 3.2 (M2):
        w_ij(τ) = exp(-d_ij / τ) / Σ_k exp(-d_ik / τ)

    Parameters
    ----------
    distances : ndarray, shape (N, K)
        Distances from each individual to each centroid.
    tau : float
        Temperature parameter (τ = c × median nearest-neighbor distance,
        for c ∈ {0.5, 1.0, 2.0} per the M2 sensitivity sweep).

    Returns
    -------
    weights : ndarray, shape (N, K)
        Each row sums to 1.0; w_ij is the soft-assignment weight of
        individual i to cluster j.
    """
    if tau <= 0:
        raise ValueError("tau must be strictly positive.")
    d = np.asarray(distances, dtype=float)
    # Numerically stable softmax with temperature
    scaled = -d / tau
    scaled -= scaled.max(axis=1, keepdims=True)  # subtract row max for stability
    exp_d = np.exp(scaled)
    return exp_d / exp_d.sum(axis=1, keepdims=True)


def median_nn_distance(distances: np.ndarray) -> float:
    """Median nearest-neighbor distance across individuals (M2 calibration anchor)."""
    nn = np.min(np.asarray(distances, dtype=float), axis=1)
    return float(np.median(nn))


# ====================================================================
# Positivity diagnostic (m5)
# ====================================================================


@dataclass
class PositivityDiagnostic:
    """Per-counterfactual positivity report (clarifications log Section 4.5)."""

    counterfactual: Literal["A", "B", "C"]
    cell_ratios: np.ndarray = field(repr=False)
    """ρ_{c,x} for each cell c (length-14)."""

    cell_weights: np.ndarray = field(repr=False)
    """Population weights W_c for each cell."""

    flagged_mask: np.ndarray = field(repr=False)
    """Boolean array: True where ρ_{c,x} < POSITIVITY_RATIO_THRESHOLD."""

    flagged_weight_share: float
    """Σ W_c (flagged) / Σ W_c (total)."""

    @property
    def downgrade_to_exploratory(self) -> bool:
        """m5 downgrade rule: True if flagged_weight_share ≥ 20%."""
        return self.flagged_weight_share >= POSITIVITY_FLAGGED_WEIGHT_MAX


def compute_positivity_ratio_b(
    observed_in_target: np.ndarray,
    expected_post_intervention: np.ndarray,
) -> np.ndarray:
    """ρ_{c,B} = observed-target / expected-post-intervention (m5).

    Counterfactual A and C are trivially positive (ρ ≡ 1) per clarifications
    log Section 4.5; only Counterfactual B requires this diagnostic in
    practice.
    """
    obs = np.asarray(observed_in_target, dtype=float)
    exp = np.asarray(expected_post_intervention, dtype=float)
    out = np.zeros_like(obs, dtype=float)
    nonzero = exp > 0
    out[nonzero] = obs[nonzero] / exp[nonzero]
    out[~nonzero] = 0.0  # zero expected count → ρ = 0 → flagged
    return out


def evaluate_positivity(
    counterfactual: Literal["A", "B", "C"],
    cell_ratios: np.ndarray,
    cell_weights: np.ndarray,
) -> PositivityDiagnostic:
    """Apply the m5 quantitative positivity criterion + downgrade rule."""
    rho = np.asarray(cell_ratios, dtype=float)
    w = np.asarray(cell_weights, dtype=float)
    flagged = rho < POSITIVITY_RATIO_THRESHOLD
    total_w = w.sum()
    flagged_share = float(w[flagged].sum() / total_w) if total_w > 0 else 0.0
    return PositivityDiagnostic(
        counterfactual=counterfactual,
        cell_ratios=rho,
        cell_weights=w,
        flagged_mask=flagged,
        flagged_weight_share=flagged_share,
    )


# ====================================================================
# Sanity-check helpers (used across stages)
# ====================================================================


def cell_size_summary(cell_sizes: np.ndarray) -> dict[str, float]:
    """Summarize cell sizes (used in Stage 0 diagnostics + reporting)."""
    n = np.asarray(cell_sizes, dtype=float)
    return {
        "min": float(n.min()),
        "max": float(n.max()),
        "median": float(np.median(n)),
        "n_cells_lt_10": int((n < 10).sum()),
        "n_cells_lt_20": int((n < 20).sum()),
        "n_cells_eq_0": int((n == 0).sum()),
    }
