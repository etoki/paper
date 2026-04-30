"""Stage 5: Common-method-variance diagnostic.

Specification:
- v2.0 master Section 5.6 (Harman's single-factor test + marker-variable
  correction with HEXACO Openness per Lindell & Whitney 2001).
- v2.0 Section 4.2 limitation L8 (CMV).
- v2.0 Section 6.4 sensitivity sweep includes CMV diagnostic.

Inputs:
- ../harassment/raw.csv (N=354 individual-level data)

Output:
- output/supplementary/stage5_cmv_diagnostic.h5 with:
    - first_factor_variance_pct: Harman's first-factor variance %
    - cmv_concern_flag: True if first_factor_variance >= 50%
    - eigenvalues: full eigenvalue spectrum
    - marker_correlation: r(Openness, harassment) as CMV estimate
    - raw_correlations / adjusted_correlations matrices

Limitation note (Level 1 / v2.0 m8 + new clarification):
The v2.0 master specifies Harman's test on "N=13,668 HEXACO items" but
in practice we have access to HEXACO domain scores (not individual
items) in both N=354 and N=13,668 (the latter only as 7-cluster
centroids). This stage runs Harman + marker correction on N=354 with
the 11 available variables (6 HEXACO domains + 3 Dark Triad + 2
harassment continuous scales). Results are interpreted as a CMV-screening
diagnostic on the harassment study dataset specifically.

Random seed: 20260429.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .stage0_cell_propensity import (
    GENDER_HARASSMENT_COL,
    POWER_HARASSMENT_COL,
)
from .utils_io import (
    HEXACO_DOMAINS,
    load_harassment,
    make_rng,
    save_artifacts,
    standard_metadata,
)

# ====================================================================
# Constants per v2.0 Section 5.6
# ====================================================================

HARMAN_THRESHOLD = 50.0
"""v2.0 Section 5.6 + Methods Clarifications Log Section 4.7 of v2.0
master: first-factor variance % < 50 → CMV concern is 'limited'."""

DARK_TRIAD_COLS = ("machiavellianism", "narcissism", "psychopathy")
"""Dark Triad 3 column names (post-rename via HARASSMENT_COLUMN_ALIASES)."""

MARKER_DOMAIN = "O"
"""HEXACO Openness as theoretical marker (Lindell & Whitney 2001).

Per v2.0 Section 5.6: theoretical-marker for harassment because Openness
has weaker theoretical association with harassment than HH/A/E. Used to
estimate CMV from r(O, harassment)."""


# ====================================================================
# Harman's single-factor test
# ====================================================================


def harman_single_factor_test(
    df: pd.DataFrame, columns: list[str], standardize: bool = True
) -> tuple[float, np.ndarray, int]:
    """Run Harman's single-factor test (Podsakoff et al. 2003 standard reference).

    Method: PCA on standardized variables (after listwise deletion of any
    rows with NaN in the target columns). The first principal component's
    variance explained is reported as a percentage of total variance.

    Per v2.0 Section 5.6 + Podsakoff 2003: if first-factor variance < 50%,
    common method variance concern is considered limited.

    Parameters
    ----------
    df : DataFrame
    columns : list of column names to include
    standardize : if True, z-score before PCA (recommended)

    Returns
    -------
    first_factor_pct : float
        Variance explained by first PC, as percent of total
    eigenvalues : ndarray
        Full eigenvalue spectrum sorted descending
    n_used : int
        Number of complete-case rows used after listwise deletion
    """
    sub = df[columns].dropna()
    n_used = len(sub)
    X = sub.to_numpy().astype(float)
    if standardize:
        X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=min(len(columns), X.shape[0] - 1))
    pca.fit(X)
    variance_ratio = pca.explained_variance_ratio_
    first_factor_pct = float(variance_ratio[0] * 100.0)
    return first_factor_pct, pca.explained_variance_, n_used


# ====================================================================
# Marker-variable correction (Lindell & Whitney 2001)
# ====================================================================


def marker_variable_correction(
    df: pd.DataFrame,
    marker_col: str,
    substantive_cols: list[str],
    outcome_col: str,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Lindell-Whitney 2001 marker-variable correction.

    Identifies CMV from r(marker, outcome) and applies the correction:
        r_adj = (r_raw - r_M) / (1 - r_M)

    Where r_M is the marker-outcome correlation (assumed to be CMV).

    Parameters
    ----------
    df : DataFrame
    marker_col : str
        Theoretical marker variable (e.g., HEXACO Openness)
    substantive_cols : list of str
        Variables for which correlations need adjustment
    outcome_col : str
        Outcome variable

    Returns
    -------
    r_marker : float
        r(marker, outcome) — the CMV estimate
    raw_correlations : ndarray
        Raw r between each substantive variable and outcome
    adjusted_correlations : ndarray
        Lindell-Whitney adjusted r values
    """
    # Listwise deletion across all relevant columns
    relevant = [marker_col, outcome_col, *substantive_cols]
    sub = df[relevant].dropna()
    marker_vals = sub[marker_col].to_numpy().astype(float)
    outcome_vals = sub[outcome_col].to_numpy().astype(float)
    r_marker = float(np.corrcoef(marker_vals, outcome_vals)[0, 1])

    raw_correlations = np.zeros(len(substantive_cols), dtype=float)
    for i, col in enumerate(substantive_cols):
        raw_correlations[i] = float(
            np.corrcoef(sub[col].to_numpy().astype(float), outcome_vals)[0, 1]
        )

    # Lindell-Whitney adjustment
    if abs(1.0 - r_marker) > 1e-10:
        adjusted_correlations = (raw_correlations - r_marker) / (1.0 - r_marker)
    else:
        adjusted_correlations = np.full_like(raw_correlations, np.nan)

    return r_marker, raw_correlations, adjusted_correlations


# ====================================================================
# Pipeline
# ====================================================================


def run(output_path: str | Path) -> None:
    """Run Harman + marker-variable correction; persist diagnostics."""
    _ = make_rng(extra_offset=70_000)

    harassment = load_harassment()
    df = harassment.df

    # Variables to include in Harman's test:
    # 6 HEXACO domains + 3 Dark Triad + 2 harassment continuous scales = 11
    harman_vars = list(HEXACO_DOMAINS) + list(DARK_TRIAD_COLS) + [
        POWER_HARASSMENT_COL,
        GENDER_HARASSMENT_COL,
    ]
    available_vars = [c for c in harman_vars if c in df.columns]
    missing_vars = [c for c in harman_vars if c not in df.columns]

    if len(available_vars) < 6:
        raise ValueError(
            f"Insufficient variables for Harman's test: {available_vars} "
            f"(missing {missing_vars}). Need at least 6."
        )

    # Run Harman's single-factor test (listwise deletion for NaN rows)
    first_factor_pct, eigenvalues, n_used_harman = harman_single_factor_test(
        df, available_vars
    )
    cmv_concern = first_factor_pct >= HARMAN_THRESHOLD

    # Marker-variable correction with HEXACO Openness as theoretical marker
    # Substantive variables: other HEXACO domains + Dark Triad
    substantive_cols = [c for c in available_vars if c != MARKER_DOMAIN and c not in (
        POWER_HARASSMENT_COL, GENDER_HARASSMENT_COL
    )]

    # Power harassment as outcome
    r_marker_power, raw_corr_power, adj_corr_power = marker_variable_correction(
        df, MARKER_DOMAIN, substantive_cols, POWER_HARASSMENT_COL
    )

    # Gender harassment as outcome
    r_marker_gender, raw_corr_gender, adj_corr_gender = marker_variable_correction(
        df, MARKER_DOMAIN, substantive_cols, GENDER_HARASSMENT_COL
    )

    # Persist
    save_artifacts(
        output_path,
        arrays={
            "eigenvalues": eigenvalues.astype(np.float32),
            "raw_corr_power": raw_corr_power.astype(np.float32),
            "adj_corr_power": adj_corr_power.astype(np.float32),
            "raw_corr_gender": raw_corr_gender.astype(np.float32),
            "adj_corr_gender": adj_corr_gender.astype(np.float32),
        },
        metadata=standard_metadata(
            stage="stage5_cmv_diagnostic",
            extra={
                "first_factor_variance_pct": float(first_factor_pct),
                "harman_threshold": float(HARMAN_THRESHOLD),
                "cmv_concern_flag": bool(cmv_concern),
                "n_variables_used": int(len(available_vars)),
                "variables_included": ",".join(available_vars),
                "variables_missing": ",".join(missing_vars) if missing_vars else "(none)",
                "n_observations_total": int(len(df)),
                "n_observations_used_harman": int(n_used_harman),
                "marker_variable": MARKER_DOMAIN,
                "marker_substantive_cols": ",".join(substantive_cols),
                "r_marker_power_harassment": float(r_marker_power),
                "r_marker_gender_harassment": float(r_marker_gender),
                "harman_data_source": (
                    "N=354 harassment-preprint dataset (level 1 deviation: "
                    "v2.0 spec 'N=13,668 HEXACO items' is implemented on "
                    "available N=354 with 6 HEXACO domains + 3 Dark Triad + "
                    "2 harassment scales = 11 variables, since N=13,668 "
                    "individual-level items are not in repository — only "
                    "centroids are. See Stage 5 module docstring.)"
                ),
            },
        ),
    )

    # Console summary
    print("[Stage 5] CMV diagnostic results")
    print(f"  N (total) = {len(df)}, N (used after listwise) = {n_used_harman}, K = {len(available_vars)} variables")
    print(f"  Harman's first-factor variance: {first_factor_pct:.2f}%")
    print(
        f"  CMV concern: {'YES' if cmv_concern else 'NO'} "
        f"(threshold = {HARMAN_THRESHOLD:.0f}%)"
    )
    print()
    print("  Marker-variable correction (Lindell & Whitney 2001)")
    print(f"  Marker variable: HEXACO {MARKER_DOMAIN} (Openness)")
    print(f"  r(O, power_harassment)   = {r_marker_power:+.3f}  (CMV estimate)")
    print(f"  r(O, gender_harassment)  = {r_marker_gender:+.3f}  (CMV estimate)")
    print()
    print("  Raw vs adjusted correlations with power_harassment:")
    for i, col in enumerate(substantive_cols):
        print(
            f"    {col:>20s}: r_raw={raw_corr_power[i]:+.3f}, "
            f"r_adj={adj_corr_power[i]:+.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/supplementary/stage5_cmv_diagnostic.h5"),
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    if args.seed is not None and args.seed != 20260429:
        import warnings
        warnings.warn(
            "Seed override; v2.0 fixes seed=20260429.", stacklevel=2
        )
    run(args.output)


if __name__ == "__main__":
    main()
