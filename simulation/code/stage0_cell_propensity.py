"""Stage 0 step 2: 14-cell binary harassment propensity + bootstrap CIs.

Specification:
- v2.0 master Section 5.1 (cell-level estimation, Bootstrap B=2,000 BCa).
- v2.0 master Section 4.1 + 4.2 (binarization at mean+0.5 SD per outcome,
  with sensitivity sweep at mean+0.25 / +1.0 SD).
- Methods Clarifications Log Section 3.4 (M4): 4-step CI priority chain
  (Clopper-Pearson → BCa → BC → percentile).
- Methods Clarifications Log Section 4.4 (m4): jackknife acceleration
  per Efron 1987 Eq. 6.6.

Inputs:
- output/supplementary/stage0_type_assignment.h5
- ../harassment/raw.csv (re-loaded for the harassment outcome columns)

Output:
- output/supplementary/stage0_cell_propensity.h5 with arrays:
    - cell_ids (14,) int: encoded as type * 2 + gender
    - cell_n (14,) int: cell sizes
    - cell_x_power (14,) int: count of binary power-harassment perpetrators
    - cell_x_gender (14,) int: count of binary gender-harassment perpetrators
    - point_power (14,) float, ci_lo_power, ci_hi_power
    - point_gender, ci_lo_gender, ci_hi_gender
    - methods_power (14,) bytes: which CI method was used per cell
    - methods_gender (14,) bytes
  Metadata:
    - binarization_threshold (str: "mean+0.5SD", main; configurable for
      sensitivity)
    - bootstrap_iterations
    - n_individuals_assigned

Random seed: 20260429.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .utils_bootstrap import cell_proportion_ci
from .utils_io import (
    BOOTSTRAP_PER_CELL,
    N_CELLS_MAIN,
    N_CLUSTERS,
    load_artifacts,
    load_harassment,
    make_rng,
    save_artifacts,
    standard_metadata,
)

# ====================================================================
# Constants
# ====================================================================

POWER_HARASSMENT_COL = "power_harassment"
"""Tou et al. (2017) Workplace Power Harassment Scale column name."""

GENDER_HARASSMENT_COL = "gender_harassment"
"""Kobayashi & Tanaka (2010) Gender Harassment Scale column name."""

BINARIZATION_THRESHOLDS = {
    "mean+0.25SD": 0.25,
    "mean+0.5SD": 0.5,  # main per v2.0
    "mean+1.0SD": 1.0,
}
"""Binarization sensitivity sweep (v2.0 Section 6.4)."""

GENDER_COL = "gender"
"""Self-reported binary gender (0/1)."""

# ====================================================================
# Binarization
# ====================================================================


def binarize_outcome(scores: np.ndarray, sd_offset: float = 0.5) -> np.ndarray:
    """Binarize a continuous harassment scale at mean + offset × SD.

    Per v2.0 Section 4.2.1: main threshold is mean + 0.5 × SD.
    """
    s = np.asarray(scores, dtype=float)
    mean = float(np.nanmean(s))
    sd = float(np.nanstd(s, ddof=1))
    threshold = mean + sd_offset * sd
    return (s > threshold).astype(np.int8)


def cell_id(type_idx: int, gender_idx: int) -> int:
    """Encode (type, gender) into 14-cell index ∈ {0..13}.

    Convention: cell = type * 2 + gender, so type 0 / gender 0 → cell 0,
    type 6 / gender 1 → cell 13.
    """
    if not (0 <= type_idx < N_CLUSTERS):
        raise ValueError(f"type_idx must be in [0, {N_CLUSTERS}); got {type_idx}.")
    if gender_idx not in (0, 1):
        raise ValueError(f"gender_idx must be 0 or 1; got {gender_idx}.")
    return type_idx * 2 + gender_idx


# ====================================================================
# Pipeline
# ====================================================================


def run(
    type_assignment_path: str | Path,
    output_path: str | Path,
    sd_offset: float = 0.5,
    n_bootstrap: int = BOOTSTRAP_PER_CELL,
) -> None:
    """Compute 14-cell propensity and CIs; persist to HDF5."""
    arrays, _meta = load_artifacts(type_assignment_path)
    hard_assignment = arrays["hard_assignment"]

    harassment = load_harassment()
    df = harassment.df

    if GENDER_COL not in df.columns:
        raise ValueError(f"Expected gender column '{GENDER_COL}' in harassment data.")
    gender = df[GENDER_COL].to_numpy().astype(np.int8)

    # Binarize both harassment outcomes at the specified threshold
    if POWER_HARASSMENT_COL not in df.columns:
        raise ValueError(f"Expected '{POWER_HARASSMENT_COL}' in harassment data.")
    if GENDER_HARASSMENT_COL not in df.columns:
        raise ValueError(f"Expected '{GENDER_HARASSMENT_COL}' in harassment data.")

    bin_power = binarize_outcome(df[POWER_HARASSMENT_COL].to_numpy(), sd_offset)
    bin_gender = binarize_outcome(df[GENDER_HARASSMENT_COL].to_numpy(), sd_offset)

    # Build per-cell counts and sizes
    cell_n = np.zeros(N_CELLS_MAIN, dtype=np.int32)
    cell_x_power = np.zeros(N_CELLS_MAIN, dtype=np.int32)
    cell_x_gender = np.zeros(N_CELLS_MAIN, dtype=np.int32)
    cell_ids = np.zeros(N_CELLS_MAIN, dtype=np.int8)

    for c in range(N_CELLS_MAIN):
        cell_ids[c] = c
        type_idx = c // 2
        gender_idx = c % 2
        mask = (hard_assignment == type_idx) & (gender == gender_idx)
        n_c = int(mask.sum())
        cell_n[c] = n_c
        cell_x_power[c] = int(bin_power[mask].sum()) if n_c > 0 else 0
        cell_x_gender[c] = int(bin_gender[mask].sum()) if n_c > 0 else 0

    # Compute CIs cell-by-cell using the M4 priority chain
    point_power = np.zeros(N_CELLS_MAIN, dtype=np.float32)
    ci_lo_power = np.zeros(N_CELLS_MAIN, dtype=np.float32)
    ci_hi_power = np.zeros(N_CELLS_MAIN, dtype=np.float32)
    methods_power = np.zeros(N_CELLS_MAIN, dtype="S20")

    point_gender = np.zeros(N_CELLS_MAIN, dtype=np.float32)
    ci_lo_gender = np.zeros(N_CELLS_MAIN, dtype=np.float32)
    ci_hi_gender = np.zeros(N_CELLS_MAIN, dtype=np.float32)
    methods_gender = np.zeros(N_CELLS_MAIN, dtype="S20")

    rng = make_rng(extra_offset=10_000)  # isolate Stage 0 Step 2 stream

    for c in range(N_CELLS_MAIN):
        n_c = int(cell_n[c])
        if n_c == 0:
            point_power[c] = np.nan
            ci_lo_power[c] = np.nan
            ci_hi_power[c] = np.nan
            methods_power[c] = b"empty"
            point_gender[c] = np.nan
            ci_lo_gender[c] = np.nan
            ci_hi_gender[c] = np.nan
            methods_gender[c] = b"empty"
            continue

        ci_p = cell_proportion_ci(
            successes=int(cell_x_power[c]),
            n=n_c,
            rng=rng,
            n_bootstrap=n_bootstrap,
        )
        point_power[c] = ci_p.point
        ci_lo_power[c] = ci_p.lower
        ci_hi_power[c] = ci_p.upper
        methods_power[c] = ci_p.method.encode("ascii")

        ci_g = cell_proportion_ci(
            successes=int(cell_x_gender[c]),
            n=n_c,
            rng=rng,
            n_bootstrap=n_bootstrap,
        )
        point_gender[c] = ci_g.point
        ci_lo_gender[c] = ci_g.lower
        ci_hi_gender[c] = ci_g.upper
        methods_gender[c] = ci_g.method.encode("ascii")

    arrays_out = {
        "cell_ids": cell_ids,
        "cell_n": cell_n,
        "cell_x_power": cell_x_power,
        "cell_x_gender": cell_x_gender,
        "point_power": point_power,
        "ci_lo_power": ci_lo_power,
        "ci_hi_power": ci_hi_power,
        "methods_power": methods_power,
        "point_gender": point_gender,
        "ci_lo_gender": ci_lo_gender,
        "ci_hi_gender": ci_hi_gender,
        "methods_gender": methods_gender,
    }

    metadata = standard_metadata(
        stage="stage0_cell_propensity",
        extra={
            "binarization_offset_sd": float(sd_offset),
            "binarization_threshold_label": next(
                (k for k, v in BINARIZATION_THRESHOLDS.items() if v == sd_offset),
                f"mean+{sd_offset}SD",
            ),
            "bootstrap_iterations_per_cell": int(n_bootstrap),
            "n_cells": int(N_CELLS_MAIN),
            "n_individuals_total": int(len(hard_assignment)),
            "n_individuals_assigned": int(cell_n.sum()),
            "ci_method_priority": "Clopper-Pearson(degenerate) -> BCa -> BC -> percentile",
        },
    )

    save_artifacts(output_path, arrays=arrays_out, metadata=metadata)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--type-assignment",
        type=Path,
        default=Path("output/supplementary/stage0_type_assignment.h5"),
        help="Stage 0 step 1 output path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/supplementary/stage0_cell_propensity.h5"),
        help="Stage 0 step 2 output path",
    )
    parser.add_argument(
        "--sd-offset",
        type=float,
        default=0.5,
        choices=list(BINARIZATION_THRESHOLDS.values()),
        help="Binarization threshold (mean + offset × SD); main = 0.5",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=BOOTSTRAP_PER_CELL,
        help="Per-cell bootstrap iterations (default 2,000 per v2.0 Section 5.1)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override seed (NOT recommended)")
    args = parser.parse_args()

    if args.seed is not None:
        import warnings

        warnings.warn(
            "Seed override detected; v2.0 fixes seed=20260429.", stacklevel=2
        )

    run(
        type_assignment_path=args.type_assignment,
        output_path=args.output,
        sd_offset=args.sd_offset,
        n_bootstrap=args.n_bootstrap,
    )


if __name__ == "__main__":
    main()
