"""Stage 4: B0-B4 baseline hierarchy + Bonferroni-Holm + Page's L (n4).

Specification:
- v2.0 master Section 5.5 (B0 uniform, B1 gender-only logistic, B2
  HEXACO 6-domain logistic, B3 7-type × gender cell-conditional
  [proposed], B4 = B3 + age + estimated industry + employment).
- v2.0 master Section 6.3 (Bonferroni-Holm correction for H2 ordinal
  hypothesis: MAPE_B0 ≥ MAPE_B1 ≥ MAPE_B2 ≥ MAPE_B3 ≥ MAPE_B4).
- Methods Clarifications Log Section 6.4 (n4): Page's L (1963)
  ordinal-trend test as auxiliary; isotonic regression goodness-of-fit
  as secondary auxiliary.

Inputs:
- ../harassment/raw.csv (N=354)
- output/supplementary/stage0_type_assignment.h5
- output/supplementary/stage0_cell_propensity.h5
- output/supplementary/stage1_population_aggregation.h5

Output:
- output/supplementary/stage4_baselines.h5

Random seed: 20260429.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression

from .stage0_cell_propensity import (
    GENDER_COL,
    GENDER_HARASSMENT_COL,
    POWER_HARASSMENT_COL,
    binarize_outcome,
)
from .utils_io import (
    BOOTSTRAP_PER_CELL,
    HEXACO_DOMAINS,
    MHLW_VALIDATION_TARGETS,
    N_CELLS_MAIN,
    load_artifacts,
    load_harassment,
    make_rng,
    save_artifacts,
    standard_metadata,
)


# ====================================================================
# Helpers
# ====================================================================


def absolute_percentage_error(predicted: float, observed: float) -> float:
    """APE per period: |pred - obs| / obs * 100."""
    if observed == 0:
        return float("nan")
    return abs(predicted - observed) / observed * 100.0


def aggregate_per_individual(
    pred_per_individual: np.ndarray,
    cluster_assignment: np.ndarray,
    gender: np.ndarray,
    cell_weights: np.ndarray,
) -> float:
    """Average per cell -> weight cells by W_c -> national prevalence."""
    cell_means = np.zeros(N_CELLS_MAIN, dtype=float)
    for c in range(N_CELLS_MAIN):
        type_idx = c // 2
        gender_idx = c % 2
        mask = (cluster_assignment == type_idx) & (gender == gender_idx)
        if mask.sum() > 0:
            cell_means[c] = float(pred_per_individual[mask].mean())
    total_w = float(cell_weights.sum())
    if total_w <= 0:
        return float("nan")
    return float(np.sum(cell_means * cell_weights) / total_w)


# ====================================================================
# B0-B4 baseline implementations
# ====================================================================


def baseline_b0_uniform(n: int, target_value: float) -> np.ndarray:
    """B0: predict the MHLW grand mean for everyone (uniform)."""
    return np.full(n, target_value, dtype=float)


def baseline_b1_gender_logistic(
    df: pd.DataFrame, y: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """B1: gender-only logistic regression."""
    X = df[[GENDER_COL]].to_numpy().astype(float)
    model = LogisticRegression(random_state=int(rng.integers(0, 2**31 - 1)), max_iter=2000)
    model.fit(X, y)
    return model.predict_proba(X)[:, 1]


def baseline_b2_hexaco_logistic(
    df: pd.DataFrame, y: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """B2: logistic regression on HEXACO 6 domains."""
    X = df[list(HEXACO_DOMAINS)].to_numpy().astype(float)
    model = LogisticRegression(random_state=int(rng.integers(0, 2**31 - 1)), max_iter=2000)
    model.fit(X, y)
    return model.predict_proba(X)[:, 1]


def baseline_b3_cell_conditional(
    cluster_assignment: np.ndarray,
    gender: np.ndarray,
    cell_propensities: np.ndarray,
) -> np.ndarray:
    """B3: 7-type × gender cell-conditional propensity (proposed model)."""
    pred = np.zeros(len(cluster_assignment), dtype=float)
    for c in range(N_CELLS_MAIN):
        type_idx = c // 2
        gender_idx = c % 2
        mask = (cluster_assignment == type_idx) & (gender == gender_idx)
        if mask.any():
            pred[mask] = cell_propensities[c]
    return pred


def baseline_b4_extended(
    df: pd.DataFrame,
    cluster_assignment: np.ndarray,
    gender: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """B4: B3 + age + age×cluster interactions (industry deferred to Stage 1).

    Per clarifications log m8 limitation: full industry estimation via
    MHLW Labor Force 2022 crosstabs is deferred. Here we use age +
    age × cluster interaction as a tractable proxy that still extends
    B3 with peripheral covariates.
    """
    n = len(y)
    cluster_oh = np.zeros((n, 7), dtype=float)
    cluster_oh[np.arange(n), cluster_assignment.astype(int)] = 1.0
    age = df["age"].to_numpy().astype(float).reshape(-1, 1)
    gender_arr = gender.astype(float).reshape(-1, 1)
    age_x_cluster = cluster_oh * age
    X = np.hstack([cluster_oh, gender_arr, age, age_x_cluster])
    model = LogisticRegression(
        random_state=int(rng.integers(0, 2**31 - 1)), max_iter=3000, C=1.0
    )
    model.fit(X, y)
    return model.predict_proba(X)[:, 1]


# ====================================================================
# Statistical tests for H2
# ====================================================================


def bonferroni_holm_pairwise(
    bootstrap_mapes: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Bonferroni-Holm test for ordinal monotonicity B0 ≥ B1 ≥ ... ≥ B4."""
    n_pairs = len(bootstrap_mapes) - 1
    raw_pvalues = np.zeros(n_pairs, dtype=float)
    for i in range(n_pairs):
        a = bootstrap_mapes[i]
        b = bootstrap_mapes[i + 1]
        valid = ~(np.isnan(a) | np.isnan(b))
        if valid.sum() == 0:
            raw_pvalues[i] = 1.0
        else:
            raw_pvalues[i] = float(np.mean(b[valid] >= a[valid]))

    # Step-down at family-wise α=0.05
    order = np.argsort(raw_pvalues)
    decisions = np.zeros(n_pairs, dtype=bool)
    for rank, idx in enumerate(order):
        threshold = 0.05 / (n_pairs - rank)
        if raw_pvalues[idx] <= threshold:
            decisions[idx] = True
        else:
            break
    return raw_pvalues, decisions


def pages_l_test(mapes: np.ndarray) -> tuple[float, float]:
    """Page's L (1963) for ordered alternatives; asymptotic z-test."""
    arr = np.asarray(mapes, dtype=float)
    K = len(arr)
    desc_rank = (-arr).argsort().argsort() + 1
    L = float(np.sum((np.arange(K) + 1) * desc_rank))
    mu_L = K * (K + 1) ** 2 / 4.0
    var_L = (K * (K**2) * (K + 1) * (K - 1)) / 144.0
    if var_L <= 0:
        return L, float("nan")
    z = (L - mu_L) / np.sqrt(var_L)
    p_value = float(1.0 - stats.norm.cdf(z))
    return L, p_value


# ====================================================================
# Pipeline
# ====================================================================


def run(
    type_assignment_path: str | Path,
    cell_propensity_path: str | Path,
    aggregation_path: str | Path,
    output_path: str | Path,
    n_bootstrap: int = BOOTSTRAP_PER_CELL,
    sd_offset: float = 0.5,
    primary_outcome: str = "power",
) -> None:
    """Compute B0-B4 MAPEs, Bonferroni-Holm, Page's L."""
    type_arrays, _ = load_artifacts(type_assignment_path)
    cluster_assignment = type_arrays["hard_assignment"]

    cell_arrays, _ = load_artifacts(cell_propensity_path)
    cell_propensities = (
        cell_arrays["point_power"] if primary_outcome == "power" else cell_arrays["point_gender"]
    )

    agg_arrays, _ = load_artifacts(aggregation_path)
    cell_weights = agg_arrays["cell_weights"]

    harassment = load_harassment()
    df = harassment.df
    gender = df[GENDER_COL].to_numpy().astype(np.int8)

    if primary_outcome == "power":
        y = binarize_outcome(df[POWER_HARASSMENT_COL].to_numpy(), sd_offset)
    else:
        y = binarize_outcome(df[GENDER_HARASSMENT_COL].to_numpy(), sd_offset)
    y = y.astype(int)

    target_value = MHLW_VALIDATION_TARGETS["FY2016"]["value"]

    rng = make_rng(extra_offset=60_000)
    n = len(df)

    # B0-B4 per-individual predictions
    baselines = {
        "B0": baseline_b0_uniform(n, target_value),
        "B1": baseline_b1_gender_logistic(df, y, rng),
        "B2": baseline_b2_hexaco_logistic(df, y, rng),
        "B3": baseline_b3_cell_conditional(cluster_assignment, gender, cell_propensities),
        "B4": baseline_b4_extended(df, cluster_assignment, gender, y, rng),
    }

    # Bootstrap MAPE for each baseline (cell-stratified resample)
    cell_idx = {
        c: np.where((cluster_assignment == c // 2) & (gender == c % 2))[0]
        for c in range(N_CELLS_MAIN)
    }

    points = {}
    ci_los = {}
    ci_his = {}
    boot_mapes_list = []

    for name, pred in baselines.items():
        point_pred = aggregate_per_individual(pred, cluster_assignment, gender, cell_weights)
        point_mape = absolute_percentage_error(point_pred, target_value)

        boot_mapes = np.empty(n_bootstrap, dtype=float)
        for b in range(n_bootstrap):
            parts = [
                rng.choice(idx, size=len(idx), replace=True)
                for idx in cell_idx.values()
                if len(idx) > 0
            ]
            r_idx = np.concatenate(parts)
            boot_pred = aggregate_per_individual(
                pred[r_idx], cluster_assignment[r_idx], gender[r_idx], cell_weights
            )
            boot_mapes[b] = absolute_percentage_error(boot_pred, target_value)

        boot_mapes_list.append(boot_mapes)
        points[name] = point_mape
        ci_los[name] = float(np.nanpercentile(boot_mapes, 2.5))
        ci_his[name] = float(np.nanpercentile(boot_mapes, 97.5))

    # H2 Bonferroni-Holm primary
    bh_pvalues, bh_decisions = bonferroni_holm_pairwise(boot_mapes_list)

    # Page's L auxiliary (n4)
    point_mapes_array = np.array([points[k] for k in ["B0", "B1", "B2", "B3", "B4"]])
    pages_L, pages_p = pages_l_test(point_mapes_array)

    # Decision summary
    n_pairs_confirmed = int(bh_decisions.sum())
    if n_pairs_confirmed >= 3:
        h2_decision = "monotonic_confirmed"
    elif n_pairs_confirmed >= 1:
        h2_decision = "partial_monotonicity"
    else:
        h2_decision = "ambiguous_or_reversed"

    arrays = {
        "mape_point": point_mapes_array,
        "mape_ci_lo": np.array([ci_los[k] for k in ["B0", "B1", "B2", "B3", "B4"]]),
        "mape_ci_hi": np.array([ci_his[k] for k in ["B0", "B1", "B2", "B3", "B4"]]),
        "bh_pvalues": bh_pvalues,
        "bh_decisions": bh_decisions.astype(np.int8),
        "pages_l_statistic": np.array([pages_L]),
        "pages_l_pvalue": np.array([pages_p]),
        "pred_b0": baselines["B0"].astype(np.float32),
        "pred_b1": baselines["B1"].astype(np.float32),
        "pred_b2": baselines["B2"].astype(np.float32),
        "pred_b3": baselines["B3"].astype(np.float32),
        "pred_b4": baselines["B4"].astype(np.float32),
    }

    metadata = standard_metadata(
        stage="stage4_baselines",
        extra={
            "primary_outcome": primary_outcome,
            "n_bootstrap": int(n_bootstrap),
            "h2_test_primary": "Bonferroni-Holm on 4 pairwise inequalities (B0-B1, B1-B2, B2-B3, B3-B4)",
            "h2_test_auxiliary": "Page's L (1963) ordinal trend",
            "h2_decision": h2_decision,
            "n_pairs_confirmed": int(n_pairs_confirmed),
            "industry_estimation_status": "B4 uses age + age x cluster as proxy (m8 limitation; full MHLW industry deferred to Stage 1 actual impl)",
        },
    )

    save_artifacts(output_path, arrays=arrays, metadata=metadata)

    print(
        f"[Stage 4] B0-B4 baseline MAPE (FY2016 target, primary outcome={primary_outcome}):"
    )
    for name in ["B0", "B1", "B2", "B3", "B4"]:
        print(
            f"  {name}: {points[name]:6.2f}% "
            f"[{ci_los[name]:6.2f}%, {ci_his[name]:6.2f}%]"
        )
    print(
        f"\n  H2 Bonferroni-Holm decisions: {bh_decisions} ({n_pairs_confirmed}/4 confirmed)"
    )
    print(f"  H2 decision: {h2_decision}")
    print(f"  Page's L (auxiliary): L={pages_L:.2f}, p={pages_p:.4f}")


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
        "--output",
        type=Path,
        default=Path("output/supplementary/stage4_baselines.h5"),
    )
    parser.add_argument("--n-bootstrap", type=int, default=BOOTSTRAP_PER_CELL)
    parser.add_argument("--sd-offset", type=float, default=0.5)
    parser.add_argument(
        "--primary-outcome", choices=["power", "gender"], default="power"
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    if args.seed is not None and args.seed != 20260429:
        import warnings

        warnings.warn("Seed override; v2.0 fixes seed=20260429.", stacklevel=2)
    run(
        type_assignment_path=args.type_assignment,
        cell_propensity_path=args.cell_propensity,
        aggregation_path=args.aggregation,
        output_path=args.output,
        n_bootstrap=args.n_bootstrap,
        sd_offset=args.sd_offset,
        primary_outcome=args.primary_outcome,
    )


if __name__ == "__main__":
    main()
