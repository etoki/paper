"""Stage 3: Sensitivity sweeps over V, f1, f2, EB scale, etc.

Specification:
- v2.0 master Section 6.4 (sensitivity master table).
- Methods Clarifications Log Section 3.2 (M2): soft-assignment τ sweep.
- Methods Clarifications Log Section 4.1, 4.2 (m1, m2): MoM rule
  diagnostics already covered upstream in Stage 0 step 3.

Strategy: one-at-a-time (OAT) sweep around the locked main configuration.
Per v2.0 Section 6.4 the full Cartesian product of pre-registered ranges
yields ≈ 25,920 combinations; OAT-from-baseline is the canonical reduction
in registered reports (Saltelli 2008, Iooss & Lemaître 2015) and is
explicitly permitted by Section 6.4 as the "headline sensitivity table".
The full Cartesian sweep is triggered by --full-cartesian for archival
runs; it is recorded as a TODO with computational-cost rationale below.

Parameters swept in-process (each varied while others held at main):
- Binarization threshold ∈ {mean+0.25 SD, mean+0.5 SD (main), mean+1.0 SD}
- Soft-assignment τ factor ∈ {0.5, 1.0, 2.0} × median NN distance (M2)
- EB shrinkage scale ∈ {0.5×, 1.0×, 2.0×} prior pseudocount
- Cluster-proportion perturbation ∈ {-0.10, 0.0 (main), +0.10} mass shift
  from low-prevalence to high-prevalence clusters (proxy for V/f1/f2
  uncertainty in MHLW reweighting; full V/f1/f2 sweep in Phase 1 actual
  impl)

Parameters deferred (TODO with rationale):
- V (victim multiplier) ∈ {2, 3, 4, 5}: requires MHLW Labor Force Survey
  2022 raw incident counts (Phase 1 gating).
- Cluster K ∈ {4, 5, 6, 7, 8}: requires re-running k-means on N=13,668
  HEXACO data; centroids are pre-registered fixed parameters per M3.
- Role-estimation models ∈ {linear, tree-based, literature}: applied at
  Stage 0 step 3 (28-cell EB) only; main pipeline aggregates raw 14-cell
  propensities, so a role-model OAT sweep is recorded but emits 0 effect
  on headline MAPE in current pipeline.

Output:
- output/supplementary/stage3_sensitivity.h5 with per-row arrays:
    - sweep_parameter (R,) bytes: parameter family label
    - sweep_value (R,) bytes: configured value label
    - point_mape_FY2016 (R,) float
    - ci_lo_FY2016 (R,), ci_hi_FY2016 (R,)
    - tier (R,) bytes
    - n_used (R,) int

Random seed: 20260429.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .stage0_cell_propensity import (
    BINARIZATION_THRESHOLDS,
    GENDER_COL,
    GENDER_HARASSMENT_COL,
    POWER_HARASSMENT_COL,
    binarize_outcome,
)
from .stage0_type_assignment import (
    SOFT_TEMPERATURE_FACTORS,
    euclidean_distances_to_centroids,
    hard_assign,
)
from .stage1_population_aggregation import aggregate_national_prevalence
from .stage2_validation import absolute_percentage_error
from .utils_bootstrap import (
    cell_stratified_bootstrap,
    classify_h1_tier,
    percentile_interval,
)
from .utils_diagnostics import (
    compute_mom_hyperprior,
    median_nn_distance,
    soft_assign_weights,
)
from .utils_io import (
    HEXACO_DOMAINS,
    MHLW_VALIDATION_TARGETS,
    N_CELLS_MAIN,
    N_CLUSTERS,
    load_centroids,
    load_harassment,
    make_rng,
    save_artifacts,
    standard_metadata,
)

# ====================================================================
# Constants per v2.0 Section 6.4
# ====================================================================

EB_SCALE_FACTORS = (0.5, 1.0, 2.0)
"""EB shrinkage prior pseudocount multiplier (v2.0 Section 6.4)."""

WEIGHT_PERTURBATIONS = (-0.10, 0.0, 0.10)
"""Cluster proportion mass-shift perturbation: high-prev cluster gets +δ,
low-prev cluster gets -δ. Proxy for V/f1/f2 uncertainty (Phase 1 gating)."""

DEFAULT_CLUSTER_PROPORTIONS = np.array(
    [0.117, 0.158, 0.168, 0.129, 0.154, 0.136, 0.139]
)
"""Placeholder cluster proportions matching Stage 1 (m8 limitation: taken
as-observed from N=13,668 IEEE-published clustering until MHLW Phase 1
reweight is implemented)."""

DEFAULT_GENDER_PROPORTIONS = np.array([0.5, 0.5])
"""Placeholder gender proportions matching Stage 1."""

SWEEP_BOOTSTRAP_ITERATIONS = 2_000
"""Per-row bootstrap budget for OAT sweep. Lower than B=10,000 headline so
the full sweep completes in ~2 minutes; recorded in metadata for audit."""


# ====================================================================
# Helpers: rebuild cell propensities under override
# ====================================================================


def _build_cell_data(
    bin_outcome: np.ndarray,
    type_assignment: np.ndarray,
    gender: np.ndarray,
) -> list[np.ndarray]:
    """Group binary outcomes by 14 cells (type * 2 + gender)."""
    cell_data: list[np.ndarray] = []
    for c in range(N_CELLS_MAIN):
        type_idx = c // 2
        gender_idx = c % 2
        mask = (type_assignment == type_idx) & (gender == gender_idx)
        cell_data.append(bin_outcome[mask].astype(int))
    return cell_data


def _construct_cell_weights(
    cluster_proportions: np.ndarray, gender_proportions: np.ndarray
) -> np.ndarray:
    """Replicate Stage 1's W_c = cluster × gender Cartesian product."""
    weights = np.zeros(N_CELLS_MAIN, dtype=float)
    for type_idx in range(N_CLUSTERS):
        for gender_idx in range(2):
            cell = type_idx * 2 + gender_idx
            weights[cell] = (
                cluster_proportions[type_idx] * gender_proportions[gender_idx]
            )
    return weights


def _evaluate_combination(
    cell_data: list[np.ndarray],
    cell_weights: np.ndarray,
    rng,
    n_bootstrap: int = SWEEP_BOOTSTRAP_ITERATIONS,
    eb_scale: float = 1.0,
) -> tuple[float, float, float, str, int]:
    """Compute (point_mape_FY2016, ci_lo, ci_hi, tier, n_used) for one config.

    EB scale (when ≠ 1.0) applies a Beta-Binomial conjugate shrinkage to each
    cell's bootstrap proportion using the method-of-moments hyperprior with
    pseudocount × eb_scale. EB scale = 1.0 reproduces the unshrunk Stage
    1+2 pipeline exactly.
    """
    cell_p_obs = np.array(
        [float(np.mean(arr)) if len(arr) > 0 else 0.0 for arr in cell_data]
    )
    cell_n_obs = np.array([len(arr) for arr in cell_data], dtype=int)

    if abs(eb_scale - 1.0) > 1e-9:
        cell_x_obs = np.array(
            [int(arr.sum()) for arr in cell_data], dtype=int
        )
        nonempty = cell_n_obs > 0
        if nonempty.sum() >= 2:
            _mu, _s2, alpha_hat, beta_hat = compute_mom_hyperprior(
                cell_p_obs[nonempty]
            )
            if np.isfinite(alpha_hat) and np.isfinite(beta_hat):
                alpha_eb = alpha_hat * eb_scale
                beta_eb = beta_hat * eb_scale
                cell_p_eb = np.zeros(N_CELLS_MAIN, dtype=float)
                for c in range(N_CELLS_MAIN):
                    if cell_n_obs[c] > 0:
                        cell_p_eb[c] = (cell_x_obs[c] + alpha_eb) / (
                            cell_n_obs[c] + alpha_eb + beta_eb
                        )
                cell_p_obs = cell_p_eb

    point_pred = aggregate_national_prevalence(cell_p_obs, cell_weights)
    observed = MHLW_VALIDATION_TARGETS["FY2016"]["value"]
    point_ape = absolute_percentage_error(point_pred, observed)

    def aggregate_statistic(resampled):
        cell_p_boot = np.array(
            [
                float(np.mean(arr)) if len(arr) > 0 else 0.0
                for arr in resampled
            ]
        )
        if abs(eb_scale - 1.0) > 1e-9:
            cell_x_boot = np.array(
                [int(arr.sum()) for arr in resampled], dtype=int
            )
            cell_n_boot = np.array(
                [len(arr) for arr in resampled], dtype=int
            )
            nonempty = cell_n_boot > 0
            if nonempty.sum() >= 2:
                _m, _v, a_b, b_b = compute_mom_hyperprior(cell_p_boot[nonempty])
                if np.isfinite(a_b) and np.isfinite(b_b):
                    a_e = a_b * eb_scale
                    b_e = b_b * eb_scale
                    shrunk = np.zeros(N_CELLS_MAIN, dtype=float)
                    for c in range(N_CELLS_MAIN):
                        if cell_n_boot[c] > 0:
                            shrunk[c] = (cell_x_boot[c] + a_e) / (
                                cell_n_boot[c] + a_e + b_e
                            )
                    cell_p_boot = shrunk
        return aggregate_national_prevalence(cell_p_boot, cell_weights)

    boot_prevalence = cell_stratified_bootstrap(
        cell_data=cell_data,
        statistic_fn=aggregate_statistic,
        n_bootstrap=n_bootstrap,
        rng=rng,
    )
    boot_ape = np.array(
        [absolute_percentage_error(p, observed) for p in boot_prevalence]
    )
    ci_lo, ci_hi = percentile_interval(boot_ape)

    tier = classify_h1_tier(
        point_mape=float(point_ape),
        ci_lower=float(ci_lo),
        ci_upper=float(ci_hi),
    )
    n_used = int(sum(cell_n_obs))
    return float(point_ape), float(ci_lo), float(ci_hi), tier.tier, n_used


def _soft_cell_data(
    bin_outcome: np.ndarray,
    soft_weights: np.ndarray,
    gender: np.ndarray,
) -> list[np.ndarray]:
    """Build per-cell binary arrays under soft assignment by replicating
    each individual according to their integer-rounded soft weight per
    cluster. This preserves a Bernoulli-like outcome at the cell level
    while reflecting the M2 sensitivity to assignment fuzziness.

    For an individual with soft weight w_k in cluster k, contribute
    round(w_k × N_replicates) copies of (cluster=k, outcome) where
    N_replicates = 100 (sufficient resolution for τ ∈ [0.5, 2.0] ×
    median NN distance).
    """
    n_replicates = 100
    cell_data: list[list[int]] = [[] for _ in range(N_CELLS_MAIN)]
    for i in range(len(bin_outcome)):
        for k in range(N_CLUSTERS):
            count = int(round(soft_weights[i, k] * n_replicates))
            if count == 0:
                continue
            cell = k * 2 + int(gender[i])
            cell_data[cell].extend([int(bin_outcome[i])] * count)
    return [np.array(cell, dtype=int) for cell in cell_data]


# ====================================================================
# Pipeline
# ====================================================================


def run(output_path: str | Path, n_bootstrap: int = SWEEP_BOOTSTRAP_ITERATIONS) -> None:
    """Run OAT sensitivity sweep and persist results."""
    rng = make_rng(extra_offset=50_000)

    # Reusable inputs
    harassment = load_harassment()
    centroid_data = load_centroids()
    df = harassment.df
    hexaco = harassment.hexaco_matrix
    centroids = centroid_data.matrix
    gender = df[GENDER_COL].to_numpy().astype(np.int8)

    distances = euclidean_distances_to_centroids(hexaco, centroids)
    hard = hard_assign(distances)
    median_nn = median_nn_distance(distances)

    # Main configuration baseline (eb_scale=1.0, sd_offset=0.5, hard, no
    # weight perturbation) — included for reference comparison
    bin_main = binarize_outcome(df[POWER_HARASSMENT_COL].to_numpy(), 0.5)
    cell_data_main = _build_cell_data(bin_main, hard, gender)
    weights_main = _construct_cell_weights(
        DEFAULT_CLUSTER_PROPORTIONS, DEFAULT_GENDER_PROPORTIONS
    )

    rows: list[tuple[str, str, float, float, float, str, int]] = []

    # ----- Baseline (main locked configuration) -----
    p, lo, hi, tier, n_used = _evaluate_combination(
        cell_data_main, weights_main, rng, n_bootstrap=n_bootstrap
    )
    rows.append(("baseline", "main_v2.0", p, lo, hi, tier, n_used))

    # ----- Sweep 1: binarization threshold -----
    for label, sd_offset in BINARIZATION_THRESHOLDS.items():
        bin_swp = binarize_outcome(df[POWER_HARASSMENT_COL].to_numpy(), sd_offset)
        cell_data_swp = _build_cell_data(bin_swp, hard, gender)
        p, lo, hi, tier, n_used = _evaluate_combination(
            cell_data_swp, weights_main, rng, n_bootstrap=n_bootstrap
        )
        rows.append(("binarization_threshold", label, p, lo, hi, tier, n_used))

    # ----- Sweep 2: soft-assignment τ factor (M2) -----
    for factor in SOFT_TEMPERATURE_FACTORS:
        tau = factor * median_nn
        if tau <= 0:
            continue
        soft_w = soft_assign_weights(distances, tau)
        cell_data_soft = _soft_cell_data(bin_main, soft_w, gender)
        p, lo, hi, tier, n_used = _evaluate_combination(
            cell_data_soft, weights_main, rng, n_bootstrap=n_bootstrap
        )
        rows.append(
            ("soft_tau_factor", f"{factor:.1f}x_median_NN", p, lo, hi, tier, n_used)
        )

    # ----- Sweep 3: EB shrinkage scale -----
    for scale in EB_SCALE_FACTORS:
        p, lo, hi, tier, n_used = _evaluate_combination(
            cell_data_main,
            weights_main,
            rng,
            n_bootstrap=n_bootstrap,
            eb_scale=scale,
        )
        rows.append(("eb_scale", f"{scale:.1f}x_pseudocount", p, lo, hi, tier, n_used))

    # ----- Sweep 4: cluster-proportion mass shift (V/f1/f2 proxy) -----
    # Locate highest-prevalence and lowest-prevalence cluster from baseline
    # cluster-level marginal (averaged over both genders) to determine
    # direction of the perturbation.
    cell_p_main = np.array(
        [
            float(np.mean(arr)) if len(arr) > 0 else 0.0
            for arr in cell_data_main
        ]
    )
    cluster_p = np.array(
        [(cell_p_main[2 * k] + cell_p_main[2 * k + 1]) / 2.0 for k in range(N_CLUSTERS)]
    )
    high_idx = int(np.argmax(cluster_p))
    low_idx = int(np.argmin(cluster_p))

    for delta in WEIGHT_PERTURBATIONS:
        cluster_props_swp = DEFAULT_CLUSTER_PROPORTIONS.copy()
        # Bound delta so neither cluster becomes negative
        max_delta = min(
            cluster_props_swp[low_idx] - 1e-3,
            DEFAULT_CLUSTER_PROPORTIONS[high_idx] - 1e-3,
        )
        d = float(np.clip(delta, -max_delta, max_delta))
        cluster_props_swp[high_idx] += d
        cluster_props_swp[low_idx] -= d
        weights_swp = _construct_cell_weights(
            cluster_props_swp, DEFAULT_GENDER_PROPORTIONS
        )
        p, lo, hi, tier, n_used = _evaluate_combination(
            cell_data_main, weights_swp, rng, n_bootstrap=n_bootstrap
        )
        rows.append(
            (
                "cluster_weight_perturbation",
                f"delta_{d:+.2f}_high_minus_low",
                p,
                lo,
                hi,
                tier,
                n_used,
            )
        )

    # Pack rows into HDF5-compatible arrays
    n_rows = len(rows)
    sweep_parameter = np.array([r[0].encode("ascii") for r in rows], dtype="S40")
    sweep_value = np.array([r[1].encode("ascii") for r in rows], dtype="S40")
    point_mape = np.array([r[2] for r in rows], dtype=np.float64)
    ci_lo = np.array([r[3] for r in rows], dtype=np.float64)
    ci_hi = np.array([r[4] for r in rows], dtype=np.float64)
    tier_arr = np.array([r[5].encode("ascii") for r in rows], dtype="S20")
    n_used_arr = np.array([r[6] for r in rows], dtype=np.int32)

    save_artifacts(
        output_path,
        arrays={
            "sweep_parameter": sweep_parameter,
            "sweep_value": sweep_value,
            "point_mape_FY2016": point_mape,
            "ci_lo_FY2016": ci_lo,
            "ci_hi_FY2016": ci_hi,
            "tier": tier_arr,
            "n_used": n_used_arr,
        },
        metadata=standard_metadata(
            stage="stage3_sensitivity",
            extra={
                "sweep_strategy": "OAT_one_at_a_time_around_main",
                "n_rows": int(n_rows),
                "n_bootstrap_per_row": int(n_bootstrap),
                "binarization_range": ",".join(BINARIZATION_THRESHOLDS.keys()),
                "soft_tau_factors": ",".join(
                    f"{f:.1f}" for f in SOFT_TEMPERATURE_FACTORS
                ),
                "eb_scale_factors": ",".join(f"{s:.1f}" for s in EB_SCALE_FACTORS),
                "weight_perturbation_deltas": ",".join(
                    f"{d:+.2f}" for d in WEIGHT_PERTURBATIONS
                ),
                "deferred_v_range": "2,3,4,5 (gated on MHLW Phase 1 raw incident counts)",
                "deferred_k_range": "4,5,6,7,8 (centroids are M3-fixed; re-clustering N=13,668 out of scope)",
                "deferred_role_models": "linear,tree-based,literature (only used at Stage 0 step 3 EB; main pipeline aggregates raw 14-cell propensities)",
                "primary_target": "MHLW H28 FY2016 (32.5%)",
                "ci_method": "percentile",
                "median_nn_distance": float(median_nn),
                "high_prev_cluster_idx": int(high_idx),
                "low_prev_cluster_idx": int(low_idx),
            },
        ),
    )

    # Console summary
    print("[Stage 3] OAT sensitivity sweep")
    print(f"  Median NN distance = {median_nn:.4f}")
    print(
        f"  {'parameter':>30s}  {'value':>30s}  {'MAPE':>8s}  {'CI':>22s}  {'tier':>17s}"
    )
    for param, value, p, lo, hi, tier, _n in rows:
        print(
            f"  {param:>30s}  {value:>30s}  {p:>7.2f}%  [{lo:>7.2f}, {hi:>7.2f}]  {tier:>17s}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/supplementary/stage3_sensitivity.h5"),
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=SWEEP_BOOTSTRAP_ITERATIONS,
        help="Per-row bootstrap iterations (default 2,000)",
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    if args.seed is not None and args.seed != 20260429:
        import warnings
        warnings.warn(
            "Seed override; v2.0 fixes seed=20260429.", stacklevel=2
        )
    run(args.output, n_bootstrap=args.n_bootstrap)


if __name__ == "__main__":
    main()
