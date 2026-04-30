"""Stage 7: Counterfactual A/B/C ΔP_x estimation + positivity + H7 IUT.

Specification:
- v2.0 master Section 5.7 (counterfactuals A/B/C with do-operator).
- Methods Clarifications Log Section 4.5 (m5): positivity quantitative
  criterion (ρ_{c,x} < 0.10 → flag; flagged_weight ≥ 20% → confirmatory
  → exploratory downgrade).
- Methods Clarifications Log Section 4.7 (m7): H7 intersection-union
  test (Berger & Hsu 1996) with one-sided 5% lower bounds on Δ_BA, Δ_BC.

Inputs:
- output/supplementary/stage0_type_assignment.h5
- output/supplementary/stage0_cell_propensity.h5
- output/supplementary/stage1_population_aggregation.h5
- output/supplementary/stage6_target_trial.h5

Output:
- output/supplementary/stage7_counterfactual.h5

ΔP_x sign convention: ΔP_x = P̂_baseline − P̂_x (REDUCTION; positive
value when intervention reduces harassment, expected for A/B/C).

Random seed: 20260429.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .stage0_cell_propensity import (
    GENDER_COL,
    POWER_HARASSMENT_COL,
    binarize_outcome,
)
from .stage0_type_assignment import (
    euclidean_distances_to_centroids,
    hard_assign,
)
from .utils_diagnostics import (
    POSITIVITY_FLAGGED_WEIGHT_MAX,
    POSITIVITY_RATIO_THRESHOLD,
)
from .utils_io import (
    BOOTSTRAP_PER_CELL,
    HEXACO_DOMAINS,
    N_CELLS_MAIN,
    load_artifacts,
    load_centroids,
    load_harassment,
    make_rng,
    save_artifacts,
    standard_metadata,
)

# ====================================================================
# Constants per v2.0 Section 5.7 + clarifications log
# ====================================================================

DELTA_A_MAIN_SD = 0.3
"""Counterfactual A universal HH shift (main; sensitivity [0.1, 0.5])."""

DELTA_B_MAIN_SD = 0.4
"""Counterfactual B targeted HH shift (main; sensitivity [0.2, 0.6])."""

EFFECT_C_MAIN = 0.20
"""Counterfactual C structural reduction (main; sensitivity [0.10, 0.30])."""

CLUSTER_B_TARGETS_PRIMARY = (0,)
"""Cluster 0 (Self-Oriented Independent profile) — primary target."""

CLUSTER_B_TARGETS_FULL = (0, 4, 6)
"""Cluster {0, 4, 6} — full main-analysis target set per v2.0 Section 5.7."""

H7_LEVEL = 0.05
"""m7: one-sided 5% lower-bound test (Berger & Hsu 1996 IUT)."""

# Index of HH (Honesty-Humility) in the HEXACO matrix
HH_INDEX = HEXACO_DOMAINS.index("H")


# ====================================================================
# do-operator implementations (Pearl 2009 n3 form per Methods Clarifications Log)
# ====================================================================


def apply_counterfactual_a(
    hexaco_matrix: np.ndarray, delta_a_sd: float, hh_sd: float
) -> np.ndarray:
    """Counterfactual A: do(HH := HH + δ_A × SD(HH)) for ALL individuals.

    Returns a NEW matrix (does not mutate input).
    """
    new_matrix = hexaco_matrix.copy()
    new_matrix[:, HH_INDEX] += delta_a_sd * hh_sd
    return new_matrix


def apply_counterfactual_b(
    hexaco_matrix: np.ndarray,
    cluster_assignment: np.ndarray,
    delta_b_sd: float,
    hh_sd: float,
    target_clusters: tuple[int, ...],
) -> np.ndarray:
    """Counterfactual B: do(HH := HH + δ_B × SD(HH)) for individuals in target clusters."""
    new_matrix = hexaco_matrix.copy()
    mask = np.isin(cluster_assignment, target_clusters)
    new_matrix[mask, HH_INDEX] += delta_b_sd * hh_sd
    return new_matrix


def apply_counterfactual_c(
    cell_propensities: np.ndarray, effect_c: float
) -> np.ndarray:
    """Counterfactual C: do(p_c := p_c × (1 − effect_C)) for all 14 cells."""
    return np.asarray(cell_propensities, dtype=float) * (1.0 - effect_c)


# ====================================================================
# National prevalence after intervention
# ====================================================================


def reclassify_clusters(
    hexaco_matrix: np.ndarray, centroids: np.ndarray
) -> np.ndarray:
    """Re-assign each individual to nearest centroid after HEXACO intervention."""
    distances = euclidean_distances_to_centroids(hexaco_matrix, centroids)
    return hard_assign(distances)


def compute_cell_propensities_from_assignment(
    cluster_assignment: np.ndarray, gender: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """Compute 14-cell binary outcome propensities from individual-level data."""
    cell_p = np.zeros(N_CELLS_MAIN, dtype=float)
    for c in range(N_CELLS_MAIN):
        type_idx = c // 2
        gender_idx = c % 2
        mask = (cluster_assignment == type_idx) & (gender == gender_idx)
        if mask.sum() > 0:
            cell_p[c] = float(y[mask].mean())
    return cell_p


def aggregate_national(cell_p: np.ndarray, cell_weights: np.ndarray) -> float:
    """Weighted national prevalence."""
    p = np.asarray(cell_p, dtype=float)
    w = np.asarray(cell_weights, dtype=float)
    total_w = float(w.sum())
    return float(np.sum(p * w) / total_w) if total_w > 0 else float("nan")


# ====================================================================
# Positivity diagnostic (m5)
# ====================================================================


def compute_positivity_a(n_cells: int = N_CELLS_MAIN) -> np.ndarray:
    """ρ ≡ 1 for Counterfactual A (universal; no extrapolation per m5)."""
    return np.ones(n_cells, dtype=float)


def compute_positivity_b(
    cluster_assignment_pre: np.ndarray,
    cluster_assignment_post: np.ndarray,
    gender: np.ndarray,
    target_clusters: tuple[int, ...],
) -> np.ndarray:
    """ρ_{c,B} = (observed in target Cluster within cell) / (expected post-intervention).

    Per Methods Clarifications Log Section 4.5 (m5).

    For each cell c (= type × gender):
    - Numerator: count of individuals who were ORIGINALLY in target Clusters
      AND end up in cell c after intervention
    - Denominator: total count of individuals in cell c after intervention

    A near-zero ratio means cell c is dominated by extrapolated (post-shift)
    individuals with little observational support.
    """
    rho = np.zeros(N_CELLS_MAIN, dtype=float)
    in_target = np.isin(cluster_assignment_pre, target_clusters)
    for c in range(N_CELLS_MAIN):
        type_idx = c // 2
        gender_idx = c % 2
        post_mask = (cluster_assignment_post == type_idx) & (gender == gender_idx)
        n_post = int(post_mask.sum())
        if n_post == 0:
            rho[c] = 0.0  # empty cell → flagged
            continue
        # Of individuals in cell c post-intervention, how many were originally in target?
        observed_target_in_cell = int((post_mask & in_target).sum())
        rho[c] = observed_target_in_cell / n_post
    return rho


def compute_positivity_c(n_cells: int = N_CELLS_MAIN) -> np.ndarray:
    """ρ ≡ 1 for Counterfactual C (cell-level multiplier; no extrapolation per m5)."""
    return np.ones(n_cells, dtype=float)


def evaluate_positivity_downgrade(
    rho: np.ndarray, cell_weights: np.ndarray
) -> tuple[float, bool]:
    """Apply m5 downgrade rule.

    Returns (flagged_weight_share, downgrade_to_exploratory).
    """
    flagged = rho < POSITIVITY_RATIO_THRESHOLD
    total_w = float(cell_weights.sum())
    if total_w <= 0:
        return 0.0, False
    flagged_share = float(cell_weights[flagged].sum() / total_w)
    downgrade = flagged_share >= POSITIVITY_FLAGGED_WEIGHT_MAX
    return flagged_share, downgrade


# ====================================================================
# H7 Intersection-Union Test (m7)
# ====================================================================


def h7_iut(
    boot_delta_a: np.ndarray,
    boot_delta_b: np.ndarray,
    boot_delta_c: np.ndarray,
    point_delta_a: float,
    point_delta_b: float,
    point_delta_c: float,
    level: float = H7_LEVEL,
) -> tuple[float, float, str]:
    """H7 IUT per m7 (Berger & Hsu 1996).

    H7: ΔP_B > ΔP_A AND ΔP_B > ΔP_C (positive ΔP_x = reduction).

    Computes:
        Δ_BA^{(b)} = ΔP_B^{(b)} − ΔP_A^{(b)}
        Δ_BC^{(b)} = ΔP_B^{(b)} − ΔP_C^{(b)}
        L_BA = lower 5th percentile of {Δ_BA^{(b)}}
        L_BC = lower 5th percentile of {Δ_BC^{(b)}}

    Classification:
        REVERSAL  : point ΔP_B < ΔP_A OR point ΔP_B < ΔP_C
        CONFIRMED : L_BA > 0 AND L_BC > 0
        PARTIAL   : exactly one of {L_BA, L_BC} > 0
        AMBIGUOUS : neither lower bound > 0 (CIs allow zero/reversal)

    Returns (L_BA, L_BC, classification_str).
    """
    delta_ba = boot_delta_b - boot_delta_a
    delta_bc = boot_delta_b - boot_delta_c

    valid_ba = ~np.isnan(delta_ba)
    valid_bc = ~np.isnan(delta_bc)
    L_BA = float(np.percentile(delta_ba[valid_ba], level * 100)) if valid_ba.any() else float("nan")
    L_BC = float(np.percentile(delta_bc[valid_bc], level * 100)) if valid_bc.any() else float("nan")

    # Classification per m7 priority
    if point_delta_b < point_delta_a or point_delta_b < point_delta_c:
        classification = "REVERSAL"
    elif L_BA > 0 and L_BC > 0:
        classification = "CONFIRMED"
    elif L_BA > 0 or L_BC > 0:
        classification = "PARTIAL"
    else:
        classification = "AMBIGUOUS"

    return L_BA, L_BC, classification


# ====================================================================
# Pipeline
# ====================================================================


def run(
    type_assignment_path: str | Path,
    cell_propensity_path: str | Path,
    aggregation_path: str | Path,
    target_trial_path: str | Path,
    output_path: str | Path,
    n_bootstrap: int = BOOTSTRAP_PER_CELL,
    sd_offset: float = 0.5,
    delta_a_sd: float = DELTA_A_MAIN_SD,
    delta_b_sd: float = DELTA_B_MAIN_SD,
    effect_c: float = EFFECT_C_MAIN,
    target_clusters: tuple[int, ...] = CLUSTER_B_TARGETS_FULL,
) -> None:
    """Estimate ΔP_A, ΔP_B, ΔP_C with positivity (m5) and H7 IUT (m7)."""

    # Load Stage 0 type assignment + raw harassment data + centroids
    type_arrays, _ = load_artifacts(type_assignment_path)
    hexaco_matrix = type_arrays["hexaco_matrix"].astype(float)
    cluster_baseline = type_arrays["hard_assignment"].astype(np.int8)
    centroids = type_arrays["centroids"].astype(float)

    # Load harassment data for binary outcome + gender
    harassment = load_harassment()
    df = harassment.df
    gender = df[GENDER_COL].to_numpy().astype(np.int8)
    y = binarize_outcome(df[POWER_HARASSMENT_COL].to_numpy(), sd_offset).astype(int)

    # Load Stage 1 cell weights (placeholder until Stage 1 actual implementation)
    agg_arrays, _ = load_artifacts(aggregation_path)
    cell_weights = agg_arrays["cell_weights"].astype(float)

    # Load Stage 0 cell propensities (baseline 14-cell)
    cell_arrays, _ = load_artifacts(cell_propensity_path)
    cell_p_baseline = cell_arrays["point_power"].astype(float)

    # SD(HH) for delta scaling
    hh_sd = float(np.nanstd(hexaco_matrix[:, HH_INDEX], ddof=1))

    rng = make_rng(extra_offset=90_000)

    # =================================================================
    # Point estimates (full-sample, no bootstrap)
    # =================================================================

    p_baseline = aggregate_national(cell_p_baseline, cell_weights)

    # Counterfactual A
    hexaco_a = apply_counterfactual_a(hexaco_matrix, delta_a_sd, hh_sd)
    cluster_a = reclassify_clusters(hexaco_a, centroids)
    cell_p_a = compute_cell_propensities_from_assignment(cluster_a, gender, y)
    p_a = aggregate_national(cell_p_a, cell_weights)
    delta_p_a = p_baseline - p_a

    # Counterfactual B
    hexaco_b = apply_counterfactual_b(
        hexaco_matrix, cluster_baseline, delta_b_sd, hh_sd, target_clusters
    )
    cluster_b = reclassify_clusters(hexaco_b, centroids)
    cell_p_b = compute_cell_propensities_from_assignment(cluster_b, gender, y)
    p_b = aggregate_national(cell_p_b, cell_weights)
    delta_p_b = p_baseline - p_b

    # Counterfactual C
    cell_p_c = apply_counterfactual_c(cell_p_baseline, effect_c)
    p_c = aggregate_national(cell_p_c, cell_weights)
    delta_p_c = p_baseline - p_c

    # =================================================================
    # Positivity diagnostics (m5)
    # =================================================================

    rho_a = compute_positivity_a()
    rho_b = compute_positivity_b(cluster_baseline, cluster_b, gender, target_clusters)
    rho_c = compute_positivity_c()

    flagged_a, downgrade_a = evaluate_positivity_downgrade(rho_a, cell_weights)
    flagged_b, downgrade_b = evaluate_positivity_downgrade(rho_b, cell_weights)
    flagged_c, downgrade_c = evaluate_positivity_downgrade(rho_c, cell_weights)

    # =================================================================
    # Bootstrap propagation (cell-stratified per M3)
    # =================================================================

    cell_idx_pre = {
        c: np.where((cluster_baseline == c // 2) & (gender == c % 2))[0]
        for c in range(N_CELLS_MAIN)
    }

    boot_delta_a = np.empty(n_bootstrap, dtype=float)
    boot_delta_b = np.empty(n_bootstrap, dtype=float)
    boot_delta_c = np.empty(n_bootstrap, dtype=float)

    for b in range(n_bootstrap):
        # Cell-stratified resample
        parts = [
            rng.choice(idx, size=len(idx), replace=True)
            for idx in cell_idx_pre.values()
            if len(idx) > 0
        ]
        r_idx = np.concatenate(parts)

        h_b = hexaco_matrix[r_idx]
        c_b = cluster_baseline[r_idx]
        g_b = gender[r_idx]
        y_b = y[r_idx]

        # Baseline on resampled data
        cell_p_base_b = compute_cell_propensities_from_assignment(c_b, g_b, y_b)
        p_base_b = aggregate_national(cell_p_base_b, cell_weights)

        # A
        h_a_b = apply_counterfactual_a(h_b, delta_a_sd, hh_sd)
        c_a_b = reclassify_clusters(h_a_b, centroids)
        cell_p_a_b = compute_cell_propensities_from_assignment(c_a_b, g_b, y_b)
        p_a_b = aggregate_national(cell_p_a_b, cell_weights)
        boot_delta_a[b] = p_base_b - p_a_b

        # B
        h_b_b = apply_counterfactual_b(h_b, c_b, delta_b_sd, hh_sd, target_clusters)
        c_b_b = reclassify_clusters(h_b_b, centroids)
        cell_p_b_b = compute_cell_propensities_from_assignment(c_b_b, g_b, y_b)
        p_b_b = aggregate_national(cell_p_b_b, cell_weights)
        boot_delta_b[b] = p_base_b - p_b_b

        # C (cell-level multiplier doesn't need re-classification)
        cell_p_c_b = apply_counterfactual_c(cell_p_base_b, effect_c)
        p_c_b = aggregate_national(cell_p_c_b, cell_weights)
        boot_delta_c[b] = p_base_b - p_c_b

    # CIs via percentile (BCa upgrade is utility-level future work)
    ci_lo_a, ci_hi_a = float(np.percentile(boot_delta_a, 2.5)), float(np.percentile(boot_delta_a, 97.5))
    ci_lo_b, ci_hi_b = float(np.percentile(boot_delta_b, 2.5)), float(np.percentile(boot_delta_b, 97.5))
    ci_lo_c, ci_hi_c = float(np.percentile(boot_delta_c, 2.5)), float(np.percentile(boot_delta_c, 97.5))

    # =================================================================
    # H7 IUT (m7)
    # =================================================================

    L_BA, L_BC, h7_classification = h7_iut(
        boot_delta_a, boot_delta_b, boot_delta_c,
        delta_p_a, delta_p_b, delta_p_c,
    )

    # =================================================================
    # Persist
    # =================================================================

    arrays = {
        "p_baseline": np.array([p_baseline]),
        "delta_p_a_point": np.array([delta_p_a]),
        "delta_p_a_ci": np.array([ci_lo_a, ci_hi_a]),
        "delta_p_b_point": np.array([delta_p_b]),
        "delta_p_b_ci": np.array([ci_lo_b, ci_hi_b]),
        "delta_p_c_point": np.array([delta_p_c]),
        "delta_p_c_ci": np.array([ci_lo_c, ci_hi_c]),
        "boot_delta_a": boot_delta_a.astype(np.float32),
        "boot_delta_b": boot_delta_b.astype(np.float32),
        "boot_delta_c": boot_delta_c.astype(np.float32),
        "rho_a": rho_a,
        "rho_b": rho_b,
        "rho_c": rho_c,
        "h7_iut": np.array([L_BA, L_BC]),
    }

    metadata = standard_metadata(
        stage="stage7_counterfactual",
        extra={
            "delta_a_main_sd": float(delta_a_sd),
            "delta_b_main_sd": float(delta_b_sd),
            "effect_c_main": float(effect_c),
            "cluster_b_targets": ",".join(map(str, target_clusters)),
            "hh_sd_used": float(hh_sd),
            "n_bootstrap": int(n_bootstrap),
            "positivity_flagged_share_a": float(flagged_a),
            "positivity_flagged_share_b": float(flagged_b),
            "positivity_flagged_share_c": float(flagged_c),
            "positivity_downgrade_a": bool(downgrade_a),
            "positivity_downgrade_b": bool(downgrade_b),
            "positivity_downgrade_c": bool(downgrade_c),
            "h7_L_BA": float(L_BA),
            "h7_L_BC": float(L_BC),
            "h7_classification": h7_classification,
            "ci_method": "percentile (BCa upgrade is future work)",
        },
    )

    save_artifacts(output_path, arrays=arrays, metadata=metadata)

    # Console summary
    print("[Stage 7] Counterfactual A/B/C results")
    print(f"  Baseline national prevalence: {p_baseline:.4f}")
    print(
        f"  ΔP_A (universal, +{delta_a_sd:.2f} SD HH): {delta_p_a:+.4f} "
        f"[{ci_lo_a:+.4f}, {ci_hi_a:+.4f}]"
    )
    print(
        f"  ΔP_B (targeted Clusters {target_clusters}, +{delta_b_sd:.2f} SD HH): "
        f"{delta_p_b:+.4f} [{ci_lo_b:+.4f}, {ci_hi_b:+.4f}]"
    )
    print(
        f"  ΔP_C (structural, ×{1.0 - effect_c:.2f}): "
        f"{delta_p_c:+.4f} [{ci_lo_c:+.4f}, {ci_hi_c:+.4f}]"
    )
    print()
    print("  Positivity (m5):")
    print(f"    A: flagged_weight = {flagged_a:.1%}, downgrade={downgrade_a}")
    print(f"    B: flagged_weight = {flagged_b:.1%}, downgrade={downgrade_b}")
    print(f"    C: flagged_weight = {flagged_c:.1%}, downgrade={downgrade_c}")
    print()
    print(f"  H7 IUT (m7): L_BA={L_BA:+.4f}, L_BC={L_BC:+.4f}")
    print(f"  H7 classification: {h7_classification}")


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
        "--aggregation",
        type=Path,
        default=Path("output/supplementary/stage1_population_aggregation.h5"),
    )
    parser.add_argument(
        "--target-trial",
        type=Path,
        default=Path("output/supplementary/stage6_target_trial.h5"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/supplementary/stage7_counterfactual.h5"),
    )
    parser.add_argument("--n-bootstrap", type=int, default=BOOTSTRAP_PER_CELL)
    parser.add_argument("--sd-offset", type=float, default=0.5)
    parser.add_argument("--delta-a-sd", type=float, default=DELTA_A_MAIN_SD)
    parser.add_argument("--delta-b-sd", type=float, default=DELTA_B_MAIN_SD)
    parser.add_argument("--effect-c", type=float, default=EFFECT_C_MAIN)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    if args.seed is not None and args.seed != 20260429:
        import warnings

        warnings.warn(
            "Seed override; v2.0 fixes seed=20260429.", stacklevel=2
        )
    run(
        type_assignment_path=args.type_assignment,
        cell_propensity_path=args.cell_propensity,
        aggregation_path=args.aggregation,
        target_trial_path=args.target_trial,
        output_path=args.output,
        n_bootstrap=args.n_bootstrap,
        sd_offset=args.sd_offset,
        delta_a_sd=args.delta_a_sd,
        delta_b_sd=args.delta_b_sd,
        effect_c=args.effect_c,
    )


if __name__ == "__main__":
    main()
