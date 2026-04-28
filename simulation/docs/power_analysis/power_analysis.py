"""D13 Power analysis for HEXACO 7-typology harassment simulation (Phase 1).

Inputs
------
- harassment/raw.csv          (N=354, HEXACO 6 + DT 3 + power/gender harassment + age/gender/area)
- clustering/csv/clstr_kmeans_7c.csv  (7 cluster centroids, HEXACO 6, with mixing ratios)
- clustering/csv/hexaco_domain.csv    (N=13,668, HEXACO 6 raw scores)

Outputs (written next to this script)
------
- cell_counts.csv             7-type x gender x role cell sizes
- cell_stats.csv              cell-level mean/SD/SE/CI for harassment outcomes
- mde_table.csv               minimum detectable effect (Cohen's d) per cell at alpha=.05, power=.80
- pairwise_mde.csv             pairwise MDE for type comparisons (within gender)
- power_analysis_summary.json structured headline numbers used by the markdown report
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[3]
HARASSMENT_CSV = REPO_ROOT / "harassment" / "raw.csv"
CENTROIDS_CSV = REPO_ROOT / "clustering" / "csv" / "clstr_kmeans_7c.csv"
POPULATION_CSV = REPO_ROOT / "clustering" / "csv" / "hexaco_domain.csv"
OUT_DIR = Path(__file__).resolve().parent

ALPHA = 0.05
POWER = 0.80

# Z values for two-sided alpha and one-sided beta
Z_ALPHA = stats.norm.ppf(1 - ALPHA / 2)   # ≈ 1.960
Z_BETA = stats.norm.ppf(POWER)            # ≈ 0.842
NCP_TARGET = Z_ALPHA + Z_BETA             # ≈ 2.802

HEXACO_DOMAINS_CENTROID = [
    "Honesty-Humility",
    "Emotionality",
    "Extraversion",
    "Agreeableness",
    "Conscientiousness",
    "Openness",
]

HEXACO_DOMAINS_RAW = [
    "hexaco_HH",
    "hexaco_E",
    "hexaco_X",
    "hexaco_A",
    "hexaco_C",
    "hexaco_O",
]

# Type labels from research plan (Clustering paper, IEEE).
# Order is the cluster index (0..6) in clstr_kmeans_7c.csv.
# The labels are assigned by inspecting the centroid profile (provisional;
# the canonical assignment lives in the Clustering paper).
TYPE_LABELS = {
    0: "T0",
    1: "T1",
    2: "T2",
    3: "T3",
    4: "T4",
    5: "T5",
    6: "T6",
}


def assign_types(harassment_df: pd.DataFrame, centroids: np.ndarray) -> np.ndarray:
    """Nearest-centroid assignment of N=354 individuals to 7 types in raw HEXACO space."""
    X = harassment_df[HEXACO_DOMAINS_RAW].to_numpy()
    # squared Euclidean
    diffs = X[:, None, :] - centroids[None, :, :]
    d2 = (diffs ** 2).sum(axis=2)
    return d2.argmin(axis=1)


def estimate_role(harassment_df: pd.DataFrame, manager_share: float = 0.15) -> np.ndarray:
    """Probabilistic role estimation: top-quantile composite (C+X) above threshold = manager.

    The plan (D1) specifies three candidates (linear / tree / literature). For a
    feasibility-focused power analysis we use a single transparent rule that biases
    high-C, high-X individuals toward "manager" status, matching the leadership
    emergence literature (Judge, Bono, Ilies & Gerhardt, 2002, JAP).

    Parameters
    ----------
    manager_share : float
        Marginal share of managers in the Japanese labor force. Public statistics
        (Labour Force Survey) place this around 12-15%; we use 15% as a slight
        upper bound to keep manager cells as well-populated as possible (a
        conservative-for-feasibility choice — if power fails here, it fails worse
        with smaller manager cells).
    """
    composite = (
        harassment_df["hexaco_C"].to_numpy()
        + 0.5 * harassment_df["hexaco_X"].to_numpy()
    )
    threshold = np.quantile(composite, 1 - manager_share)
    return (composite >= threshold).astype(int)  # 1 = manager, 0 = non-manager


def cell_descriptives(df: pd.DataFrame, group_cols: list[str], outcomes: list[str]) -> pd.DataFrame:
    """Per-cell descriptive stats for outcomes, plus n."""
    rows = []
    for keys, sub in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n"] = len(sub)
        for outcome in outcomes:
            vals = sub[outcome].dropna().to_numpy()
            n = len(vals)
            if n >= 2:
                m = vals.mean()
                sd = vals.std(ddof=1)
                se = sd / np.sqrt(n)
                # 95% CI using t distribution
                ci = stats.t.ppf(0.975, n - 1) * se
            elif n == 1:
                m, sd, se, ci = vals.mean(), np.nan, np.nan, np.nan
            else:
                m, sd, se, ci = np.nan, np.nan, np.nan, np.nan
            row[f"{outcome}_mean"] = m
            row[f"{outcome}_sd"] = sd
            row[f"{outcome}_se"] = se
            row[f"{outcome}_ci95"] = ci
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def mde_one_sample(n: int) -> float:
    """Smallest standardized effect (Cohen's d) detectable at alpha=.05, power=.80
    for a one-sample t-test (cell mean vs reference)."""
    if n < 2:
        return np.nan
    return NCP_TARGET / np.sqrt(n)


def mde_two_sample(n1: int, n2: int) -> float:
    """Smallest standardized effect (Cohen's d) detectable at alpha=.05, power=.80
    for an independent two-sample comparison."""
    if n1 < 2 or n2 < 2:
        return np.nan
    return NCP_TARGET * np.sqrt(1 / n1 + 1 / n2)


def detectable_proportion_difference(n: int, p_baseline: float = 0.10) -> float:
    """For binary outcomes (e.g., 'committed harassment yes/no') the MDE depends on
    the baseline rate. Returns the smallest detectable absolute proportion difference
    against a one-sample reference at alpha=.05, power=.80, normal approximation.
    """
    if n < 2:
        return np.nan
    se = np.sqrt(p_baseline * (1 - p_baseline) / n)
    return NCP_TARGET * se


def proportion_ci_halfwidth(n: int, p: float) -> float:
    """Half-width of the 95% Wilson-style normal-approximation CI on a proportion.
    Returns NaN for n<1. Uses sqrt(p(1-p)/n) * Z_{.975}."""
    if n < 1:
        return np.nan
    return Z_ALPHA * np.sqrt(p * (1 - p) / n)


def binarize(series: pd.Series, threshold: float) -> pd.Series:
    return (series >= threshold).astype(int)


def main() -> None:
    # ---- 1. Load
    harassment = pd.read_csv(HARASSMENT_CSV)
    centroids_df = pd.read_csv(CENTROIDS_CSV, index_col=0)
    centroids = centroids_df[HEXACO_DOMAINS_CENTROID].to_numpy()
    population = pd.read_csv(POPULATION_CSV)

    print(f"[load] harassment N={len(harassment)}, "
          f"population N={len(population)}, centroids shape={centroids.shape}")

    # ---- 2. Type assignment for N=354
    harassment["type"] = assign_types(harassment, centroids)
    harassment["type_label"] = harassment["type"].map(TYPE_LABELS)
    type_dist = harassment["type"].value_counts().sort_index()
    print("[type assignment] N=354 distribution by type:")
    print(type_dist.to_string())

    # ---- 3. Role estimation
    harassment["role"] = estimate_role(harassment, manager_share=0.15)

    # gender in raw.csv is encoded as 0/1 (n=133 / n=220 respectively, matching
    # res/sex_stratified_R2.csv from the Harassment paper). The exact mapping to
    # M/F is documented in the Harassment paper; for power-analysis purposes we
    # treat the two strata symbolically as G0 / G1 — the cell sizes drive MDE,
    # not the label semantics.
    harassment["gender_label"] = harassment["gender"].map({0: "G0", 1: "G1"}).fillna("?")

    # ---- 4. Cell-size matrices
    counts_2way = (
        harassment.groupby(["type", "gender_label"]).size().unstack(fill_value=0)
    )
    counts_3way = (
        harassment.groupby(["type", "gender_label", "role"]).size().unstack(fill_value=0)
    )
    counts_3way.columns = [f"role={c}" for c in counts_3way.columns]
    print("\n[cells] 7 type x gender (14 cells)")
    print(counts_2way.to_string())
    print("\n[cells] 7 type x gender x role (28 cells)")
    print(counts_3way.to_string())

    # ---- 5. Per-cell descriptives + MDE
    outcomes = ["power_harassment", "gender_harassment"]

    desc_2way = cell_descriptives(harassment, ["type", "gender_label"], outcomes)
    desc_3way = cell_descriptives(harassment, ["type", "gender_label", "role"], outcomes)

    desc_2way["mde_d_one_sample"] = desc_2way["n"].apply(mde_one_sample)
    desc_3way["mde_d_one_sample"] = desc_3way["n"].apply(mde_one_sample)
    desc_2way["mde_p_diff_p10"] = desc_2way["n"].apply(
        lambda n: detectable_proportion_difference(n, 0.10)
    )
    desc_3way["mde_p_diff_p10"] = desc_3way["n"].apply(
        lambda n: detectable_proportion_difference(n, 0.10)
    )

    # ---- 6. Pairwise MDE matrix (within gender, across types) — 14-cell scenario
    pairwise_rows = []
    for gender_label, sub in desc_2way.groupby("gender_label"):
        sub = sub.set_index("type")
        for t1 in sub.index:
            for t2 in sub.index:
                if t2 <= t1:
                    continue
                n1 = int(sub.loc[t1, "n"])
                n2 = int(sub.loc[t2, "n"])
                pairwise_rows.append({
                    "gender": gender_label,
                    "type_a": t1,
                    "type_b": t2,
                    "n_a": n1,
                    "n_b": n2,
                    "mde_d_two_sample": mde_two_sample(n1, n2),
                })
    pairwise = pd.DataFrame(pairwise_rows)

    # ---- 7. Reference: full-sample SD for each outcome (used to translate d into raw units)
    full_sd = {o: float(harassment[o].std(ddof=1)) for o in outcomes}

    # ---- 7b. Binary-outcome estimation precision
    # The Phase-1 simulation aggregates to perpetrator counts per cell, so the
    # operationally relevant question is: how tight is the per-cell rate estimate?
    # We binarize each continuous outcome at "mean + 0.5 SD" (a moderate-perpetrator
    # cutoff yielding ~15-30% positive rate, matching MHLW-reported harassment
    # exposure rates) and report cell-level rate + 95% CI.
    bin_thresholds = {o: harassment[o].mean() + 0.5 * harassment[o].std() for o in outcomes}
    for o in outcomes:
        harassment[f"{o}_bin"] = binarize(harassment[o], bin_thresholds[o])

    bin_outcomes = [f"{o}_bin" for o in outcomes]
    bin_rows = []
    for keys, sub in harassment.groupby(["type", "gender_label"]):
        if not isinstance(keys, tuple):
            keys = (keys,)
        n = len(sub)
        row = {"type": keys[0], "gender_label": keys[1], "n": n}
        for bo in bin_outcomes:
            p = sub[bo].mean() if n > 0 else np.nan
            row[f"{bo}_rate"] = p
            row[f"{bo}_ci_halfwidth"] = proportion_ci_halfwidth(n, p) if n > 0 else np.nan
        bin_rows.append(row)
    bin_precision = pd.DataFrame(bin_rows).sort_values(["type", "gender_label"]).reset_index(drop=True)

    # ---- 8. Save artifacts
    counts_2way.to_csv(OUT_DIR / "cell_counts_14.csv")
    counts_3way.to_csv(OUT_DIR / "cell_counts_28.csv")
    desc_2way.to_csv(OUT_DIR / "cell_stats_14.csv", index=False)
    desc_3way.to_csv(OUT_DIR / "cell_stats_28.csv", index=False)
    pairwise.to_csv(OUT_DIR / "pairwise_mde_14.csv", index=False)
    bin_precision.to_csv(OUT_DIR / "binary_rate_precision_14.csv", index=False)

    # ---- 9. Headline numbers
    cells_14 = counts_2way.values.flatten()
    cells_28 = counts_3way.values.flatten()
    summary = {
        "alpha": ALPHA,
        "power_target": POWER,
        "z_alpha_two_sided": float(Z_ALPHA),
        "z_beta": float(Z_BETA),
        "ncp_target": float(NCP_TARGET),
        "type_distribution": type_dist.to_dict(),
        "manager_share_assumed": 0.15,
        "outcome_full_sd": full_sd,
        "cells_14": {
            "n_cells": int(cells_14.size),
            "min": int(cells_14.min()),
            "max": int(cells_14.max()),
            "median": float(np.median(cells_14)),
            "n_lt_10": int((cells_14 < 10).sum()),
            "n_lt_20": int((cells_14 < 20).sum()),
        },
        "cells_28": {
            "n_cells": int(cells_28.size),
            "min": int(cells_28.min()),
            "max": int(cells_28.max()),
            "median": float(np.median(cells_28)),
            "n_lt_10": int((cells_28 < 10).sum()),
            "n_lt_20": int((cells_28 < 20).sum()),
        },
        "mde_d_one_sample": {
            "14_cells_median": float(np.nanmedian(desc_2way["mde_d_one_sample"])),
            "14_cells_max": float(np.nanmax(desc_2way["mde_d_one_sample"])),
            "28_cells_median": float(np.nanmedian(desc_3way["mde_d_one_sample"])),
            "28_cells_max": float(np.nanmax(desc_3way["mde_d_one_sample"])),
        },
        "pairwise_mde_d_two_sample": {
            "median": float(np.nanmedian(pairwise["mde_d_two_sample"])),
            "max": float(np.nanmax(pairwise["mde_d_two_sample"])),
            "n_lt_0p5": int((pairwise["mde_d_two_sample"] < 0.5).sum()),
            "n_total": int(len(pairwise)),
        },
        "binary_thresholds": {o: float(t) for o, t in bin_thresholds.items()},
        "binary_rate_precision_14": {
            "power_harassment_bin": {
                "rate_median": float(bin_precision["power_harassment_bin_rate"].median()),
                "rate_min": float(bin_precision["power_harassment_bin_rate"].min()),
                "rate_max": float(bin_precision["power_harassment_bin_rate"].max()),
                "ci_halfwidth_median": float(bin_precision["power_harassment_bin_ci_halfwidth"].median()),
                "ci_halfwidth_max": float(bin_precision["power_harassment_bin_ci_halfwidth"].max()),
            },
            "gender_harassment_bin": {
                "rate_median": float(bin_precision["gender_harassment_bin_rate"].median()),
                "rate_min": float(bin_precision["gender_harassment_bin_rate"].min()),
                "rate_max": float(bin_precision["gender_harassment_bin_rate"].max()),
                "ci_halfwidth_median": float(bin_precision["gender_harassment_bin_ci_halfwidth"].median()),
                "ci_halfwidth_max": float(bin_precision["gender_harassment_bin_ci_halfwidth"].max()),
            },
        },
    }

    (OUT_DIR / "power_analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )

    # ---- 10. Print headline summary to stdout
    print("\n[summary]")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
