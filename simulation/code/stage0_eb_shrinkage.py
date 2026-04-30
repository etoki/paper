"""Stage 0 step 3: 28-cell Empirical Bayes shrinkage (sensitivity tier).

Specification:
- v2.0 master Section 5.2 (Beta-Binomial conjugate, MoM hyperprior).
- Methods Clarifications Log Section 3.1 (M1): bootstrap re-estimate
  (α̂, β̂) PER ITERATION; naive plug-in EB with bootstrap propagation
  (Carlin & Louis 2000; Efron 2014).
- Methods Clarifications Log Section 4.1 (m1): MoM threshold
  bootstrap-stabilized (200 cell-bootstrap iters; reject if upper 95%
  bound of variance ratio < 0.05).
- Methods Clarifications Log Section 4.2 (m2): 3rd MoM trigger:
  α̂ + β̂ > 5 × median(N_k for 28 cells).
- Methods Clarifications Log Section 3.4 (M4): for 28-cell tier, the
  EB posterior credible interval is the PRIMARY CI (the prior pulls
  the posterior off the {0, 1} boundary even when X_k = 0 or N_k);
  bootstrap-BCa is the AUXILIARY triangulation.

Inputs:
- output/supplementary/stage0_cell_propensity.h5 (14-cell estimates +
  raw counts)
- harassment data + type assignment (for 28-cell partition by role)

Output:
- output/supplementary/stage0_eb_shrinkage.h5 with arrays:
    - cell_ids_28 (28,) int
    - cell_n_28 (28,) int
    - cell_x_power_28 (28,) int
    - posterior_mean_power (28,) float: E[p_k | X_k, N_k] from MoM/Stan
    - posterior_lo_power (28,) float, posterior_hi_power (28,) float
    - mom_diagnostic_per_iter: (n_bootstrap, 4) float (μ̂, σ̂², α̂, β̂)
    - mom_rejected (bool)
    - mom_reasons (str)
  Metadata:
    - n_iterations, mom_rejected_per_iter_count, etc.

Random seed: 20260429.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import beta as beta_dist

from .utils_diagnostics import (
    compute_mom_hyperprior,
    diagnose_mom,
)
from .utils_io import (
    BOOTSTRAP_PER_CELL,
    N_CELLS_MAIN,
    N_CELLS_SENSITIVITY,
    load_artifacts,
    load_harassment,
    make_rng,
    save_artifacts,
    standard_metadata,
)
from .stage0_cell_propensity import (
    GENDER_COL,
    POWER_HARASSMENT_COL,
    binarize_outcome,
)

# ====================================================================
# Constants per v2.0 Section 5.2 + clarifications log
# ====================================================================

ROLE_COMPOSITE_TOP_PCT = 0.15
"""v2.0 Section 5.2 + research plan Part 11.2: top 15% of (C + 0.5*X)
composite assigned to manager (role=1). Matches MHLW Labor Force ~12-15%
manager rate."""


# ====================================================================
# Role estimation (literature-based main; D1 sensitivity in Stage 3)
# ====================================================================


def estimate_role_literature_rule(c_scores: np.ndarray, x_scores: np.ndarray) -> np.ndarray:
    """Top 15% of (C + 0.5*X) composite → manager (role=1).

    Per v2.0 Section 5.2 main role-estimation rule. Returns binary array
    of shape (N,).
    """
    composite = np.asarray(c_scores, dtype=float) + 0.5 * np.asarray(x_scores, dtype=float)
    cutoff = float(np.quantile(composite, 1.0 - ROLE_COMPOSITE_TOP_PCT))
    return (composite >= cutoff).astype(np.int8)


def cell_id_28(type_idx: int, gender_idx: int, role_idx: int) -> int:
    """Encode 28-cell index ∈ {0..27}: cell = (type * 2 + gender) * 2 + role."""
    if not (0 <= type_idx < 7):
        raise ValueError(f"type_idx must be in [0,7); got {type_idx}.")
    if gender_idx not in (0, 1):
        raise ValueError(f"gender_idx must be 0 or 1; got {gender_idx}.")
    if role_idx not in (0, 1):
        raise ValueError(f"role_idx must be 0 or 1; got {role_idx}.")
    return (type_idx * 2 + gender_idx) * 2 + role_idx


# ====================================================================
# Empirical Bayes posterior with bootstrap re-estimation (M1)
# ====================================================================


def eb_posterior_credible_interval(
    alpha: float, beta: float, x: int, n: int, level: float = 0.95
) -> tuple[float, float, float]:
    """Return (mean, lo, hi) for Beta(alpha+x, beta+n-x).

    Per M4 + clarifications log Section 3.4: for the 28-cell tier this
    posterior credible interval is the PRIMARY CI because the prior
    pulls the posterior off {0, 1} boundary.
    """
    if np.isnan(alpha) or np.isnan(beta):
        return float("nan"), float("nan"), float("nan")
    a_post = alpha + x
    b_post = beta + n - x
    mean = a_post / (a_post + b_post)
    lower_q = (1.0 - level) / 2.0
    upper_q = 1.0 - lower_q
    lo = float(beta_dist.ppf(lower_q, a_post, b_post))
    hi = float(beta_dist.ppf(upper_q, a_post, b_post))
    return float(mean), lo, hi


# ====================================================================
# Pipeline
# ====================================================================


def run(
    type_assignment_path: str | Path,
    cell_propensity_path: str | Path,
    output_path: str | Path,
    n_bootstrap: int = BOOTSTRAP_PER_CELL,
    sd_offset: float = 0.5,
) -> None:
    """Compute 28-cell EB shrinkage with M1 bootstrap (α̂, β̂) propagation."""
    # Load inputs
    type_arrays, _ = load_artifacts(type_assignment_path)
    hard_assignment = type_arrays["hard_assignment"]

    cell_arrays, _ = load_artifacts(cell_propensity_path)
    point_power_14 = cell_arrays["point_power"]  # 14-cell observed propensities

    # Reload raw harassment data to compute role + 28-cell counts
    harassment = load_harassment()
    df = harassment.df

    # Validate columns
    for col in [GENDER_COL, POWER_HARASSMENT_COL, "C", "X"]:
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' in harassment data.")

    gender = df[GENDER_COL].to_numpy().astype(np.int8)
    role = estimate_role_literature_rule(
        df["C"].to_numpy(), df["X"].to_numpy()
    )
    bin_power = binarize_outcome(df[POWER_HARASSMENT_COL].to_numpy(), sd_offset)

    # Build 28-cell counts
    cell_ids_28 = np.zeros(N_CELLS_SENSITIVITY, dtype=np.int8)
    cell_n_28 = np.zeros(N_CELLS_SENSITIVITY, dtype=np.int32)
    cell_x_28 = np.zeros(N_CELLS_SENSITIVITY, dtype=np.int32)

    for c in range(N_CELLS_SENSITIVITY):
        cell_ids_28[c] = c
        # Decode: cell = ((type * 2) + gender) * 2 + role
        type_idx = c // 4
        gender_idx = (c // 2) % 2
        role_idx = c % 2
        mask = (
            (hard_assignment == type_idx)
            & (gender == gender_idx)
            & (role == role_idx)
        )
        n_c = int(mask.sum())
        cell_n_28[c] = n_c
        cell_x_28[c] = int(bin_power[mask].sum()) if n_c > 0 else 0

    # MoM diagnostic on 14-cell data with 28-cell N for m2 trigger
    diag = diagnose_mom(propensities=point_power_14, cell_sizes_28=cell_n_28)

    # Per-iteration (α̂, β̂) bootstrap (M1): re-estimate hyperprior on each
    # bootstrap iteration and propagate through to the 28-cell posterior.
    rng = make_rng(extra_offset=20_000)  # isolate Stage 0 Step 3 stream
    posterior_means_boot = np.zeros((n_bootstrap, N_CELLS_SENSITIVITY), dtype=np.float32)
    mom_param_log = np.zeros((n_bootstrap, 4), dtype=np.float32)  # (μ̂, σ̂², α̂, β̂) per iter

    for b in range(n_bootstrap):
        # Resample 14-cell propensities with replacement (mirrors the
        # M3 cell-level bootstrap principle at the hyperprior scale).
        boot_props = rng.choice(point_power_14, size=N_CELLS_MAIN, replace=True)
        mu_b, sigma2_b, alpha_b, beta_b = compute_mom_hyperprior(boot_props)
        mom_param_log[b] = [mu_b, sigma2_b, alpha_b, beta_b]
        if np.isnan(alpha_b) or np.isnan(beta_b):
            # Degenerate hyperprior in this bootstrap iteration; fall
            # back to point-estimate (α̂, β̂) from full data.
            alpha_b, beta_b = diag.alpha_hat, diag.beta_hat
        if np.isnan(alpha_b) or np.isnan(beta_b):
            # Even the global MoM is degenerate → use uniform prior
            alpha_b, beta_b = 1.0, 1.0
        for c in range(N_CELLS_SENSITIVITY):
            n_c = int(cell_n_28[c])
            x_c = int(cell_x_28[c])
            if n_c == 0:
                # Empty cell; posterior = prior mean
                posterior_means_boot[b, c] = alpha_b / (alpha_b + beta_b)
            else:
                posterior_means_boot[b, c] = (alpha_b + x_c) / (alpha_b + beta_b + n_c)

    # Aggregate: point estimate from full-data hyperprior, CI from
    # the bootstrap distribution of posterior means.
    post_mean = np.zeros(N_CELLS_SENSITIVITY, dtype=np.float32)
    post_lo = np.zeros(N_CELLS_SENSITIVITY, dtype=np.float32)
    post_hi = np.zeros(N_CELLS_SENSITIVITY, dtype=np.float32)

    full_alpha = diag.alpha_hat if not np.isnan(diag.alpha_hat) else 1.0
    full_beta = diag.beta_hat if not np.isnan(diag.beta_hat) else 1.0

    for c in range(N_CELLS_SENSITIVITY):
        x_c = int(cell_x_28[c])
        n_c = int(cell_n_28[c])
        if n_c == 0:
            post_mean[c] = full_alpha / (full_alpha + full_beta)
            # Use full-data prior credible interval as CI for empty cells
            mean_, lo_, hi_ = eb_posterior_credible_interval(full_alpha, full_beta, 0, 0)
            post_lo[c] = lo_
            post_hi[c] = hi_
        else:
            mean_, lo_, hi_ = eb_posterior_credible_interval(full_alpha, full_beta, x_c, n_c)
            post_mean[c] = mean_
            # Bootstrap CI is auxiliary; we report posterior credible
            # interval as primary per M4 (28-cell tier).
            post_lo[c] = lo_
            post_hi[c] = hi_

    arrays = {
        "cell_ids_28": cell_ids_28,
        "cell_n_28": cell_n_28,
        "cell_x_power_28": cell_x_28,
        "posterior_mean_power": post_mean,
        "posterior_lo_power": post_lo,
        "posterior_hi_power": post_hi,
        "mom_param_log": mom_param_log,
        "posterior_means_boot": posterior_means_boot.mean(axis=0),  # bootstrap mean per cell
    }

    metadata = standard_metadata(
        stage="stage0_eb_shrinkage",
        extra={
            "n_cells_sensitivity": int(N_CELLS_SENSITIVITY),
            "bootstrap_iterations": int(n_bootstrap),
            "mom_rejected": bool(diag.reject_mom),
            "mom_reasons": "; ".join(diag.reasons) if diag.reasons else "(MoM accepted)",
            "mom_alpha_hat_full": float(diag.alpha_hat),
            "mom_beta_hat_full": float(diag.beta_hat),
            "mom_variance_ratio": float(diag.variance_ratio),
            "mom_variance_ratio_upper95": float(diag.variance_ratio_upper95),
            "mom_pseudocount_total": float(diag.pseudocount_total),
            "mom_pseudocount_median_n_ratio": float(diag.pseudocount_median_n_ratio),
            "primary_ci_method": "EB posterior credible interval (M4: 28-cell tier)",
            "auxiliary_ci_method": "bootstrap re-estimate (alpha,beta) (M1)",
            "stan_triangulation": "TODO: cmdstanpy hierarchical Bayes (auxiliary, optional)",
        },
    )

    save_artifacts(output_path, arrays=arrays, metadata=metadata)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--type-assignment",
        type=Path,
        default=Path("output/supplementary/stage0_type_assignment.h5"),
    )
    parser.add_argument(
        "--cell-propensity",
        type=Path,
        default=Path("output/supplementary/stage0_cell_propensity.h5"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/supplementary/stage0_eb_shrinkage.h5"),
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=BOOTSTRAP_PER_CELL,
        help="Bootstrap iterations for (α̂, β̂) re-estimation (M1)",
    )
    parser.add_argument(
        "--sd-offset",
        type=float,
        default=0.5,
        help="Binarization threshold offset (must match Stage 0 step 2)",
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    if args.seed is not None:
        import warnings

        warnings.warn(
            "Seed override detected; v2.0 fixes seed=20260429.", stacklevel=2
        )
    run(
        type_assignment_path=args.type_assignment,
        cell_propensity_path=args.cell_propensity,
        output_path=args.output,
        n_bootstrap=args.n_bootstrap,
        sd_offset=args.sd_offset,
    )


if __name__ == "__main__":
    main()
