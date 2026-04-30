"""Stage 1: Population aggregation via MHLW Labor Force weights.

Specification:
- v2.0 master Section 2.3 Stage 1 + Section 5.3.
- Methods Clarifications Log Section 3.3 (M3): centroids fixed; bootstrap
  CI conditional on centroids (does NOT propagate centroid uncertainty).
- Methods Clarifications Log Section 5.1 (m8): cluster proportions taken
  as-observed from the IEEE-published clustering paper; representativeness
  acknowledged as limitation L_m8 in Section 10.

Inputs:
- output/supplementary/stage0_cell_propensity.h5 (14-cell point + CIs)

Output:
- output/supplementary/stage1_population_aggregation.h5

Random seed: 20260429.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .utils_io import (
    MHLW_VALIDATION_TARGETS,
    MHLW_WEIGHTS_PATH,
    N_CELLS_MAIN,
    load_artifacts,
    load_mhlw_weights,
    make_rng,
    save_artifacts,
    standard_metadata,
)

DEFAULT_CLUSTER_PROPORTIONS = np.array(
    [0.117, 0.158, 0.168, 0.129, 0.154, 0.136, 0.139]
)
"""7-cluster proportions from N=13,668 IEEE-published clustering.

Per m8 limitation: cluster proportions remain M3-fixed at IEEE values
even after MHLW post-stratification (cluster membership is not in the
MHLW dataset). Only gender marginal is updated by MHLW reweight."""

DEFAULT_GENDER_PROPORTIONS = np.array([0.5, 0.5])
"""Placeholder gender proportions used when MHLW data is not available.

Replaced by MHLW Labor Force Survey 2022 gender marginal once the file
is acquired (see ``utils_io.load_mhlw_weights``)."""


def aggregate_national_prevalence(
    cell_propensities: np.ndarray, cell_weights: np.ndarray
) -> float:
    """P̂_t = Σ_c (p̂_c × W_c) / Σ_c W_c."""
    p = np.asarray(cell_propensities, dtype=float)
    w = np.asarray(cell_weights, dtype=float)
    total_w = float(w.sum())
    if total_w <= 0:
        return float("nan")
    return float(np.sum(p * w) / total_w)


def construct_population_weights(
    cluster_proportions: np.ndarray,
    gender_proportions: np.ndarray,
    age_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Construct W_c per v2.0 Section 5.3.

    TODO: Stage 0 implementation phase to wire up MHLW Labor Force
    Survey 2022 marginal corrections via age × gender × employment
    crosstab (per clarifications log m8 limitation note: cluster
    proportions taken as-observed from IEEE clustering paper).
    """
    # Skeleton: equal age weighting + cluster × gender Cartesian product.
    # Real implementation in Stage 0 actual code phase.
    cp = np.asarray(cluster_proportions, dtype=float)
    gp = np.asarray(gender_proportions, dtype=float)
    if cp.shape[0] != 7:
        raise ValueError(f"Expected 7 cluster proportions; got {cp.shape[0]}.")
    if gp.shape[0] != 2:
        raise ValueError(f"Expected 2 gender proportions; got {gp.shape[0]}.")
    weights = np.zeros(N_CELLS_MAIN, dtype=float)
    for type_idx in range(7):
        for gender_idx in range(2):
            cell = type_idx * 2 + gender_idx
            weights[cell] = cp[type_idx] * gp[gender_idx]
    return weights


def run(
    cell_propensity_path: str | Path,
    output_path: str | Path,
    mhlw_path: str | Path | None = None,
) -> None:
    """Aggregate cell propensities to national prevalence per validation period.

    Parameters
    ----------
    cell_propensity_path : path-like
        Stage 0 step 2 HDF5 (14-cell propensities).
    output_path : path-like
        Stage 1 HDF5 destination.
    mhlw_path : path-like, optional
        MHLW Labor Force Survey 2022 CSV; if None, attempts the default
        path at ``MHLW_WEIGHTS_PATH``. Falls back to placeholder gender
        proportions [0.5, 0.5] with explicit warning if the file is absent.
    """
    arrays, _meta = load_artifacts(cell_propensity_path)
    point_power = arrays["point_power"]

    # Try to load MHLW weights; fall back to placeholder with warning
    mhlw_p = Path(mhlw_path) if mhlw_path is not None else MHLW_WEIGHTS_PATH
    if mhlw_p.is_file():
        mhlw = load_mhlw_weights(mhlw_p)
        gender_props = mhlw.gender_proportions
        weight_provenance = (
            f"MHLW Labor Force Survey 2022 ({mhlw_p.name}, "
            f"N_total={mhlw.total_population:,}, n_records={mhlw.n_records})"
        )
    else:
        import warnings
        warnings.warn(
            f"MHLW Labor Force Survey 2022 file not found at {mhlw_p}; "
            "Stage 1 falling back to placeholder gender proportions [0.5, 0.5]. "
            "Acquire from e-Stat to unlock Phase 1 actual aggregation.",
            stacklevel=2,
        )
        gender_props = DEFAULT_GENDER_PROPORTIONS
        weight_provenance = "PLACEHOLDER (MHLW e-Stat data not yet provided)"

    cell_weights = construct_population_weights(
        DEFAULT_CLUSTER_PROPORTIONS, gender_props
    )

    national_prevalence = aggregate_national_prevalence(point_power, cell_weights)

    # MHLW validation targets and gaps (point estimates only; CI in Stage 2)
    target_labels = list(MHLW_VALIDATION_TARGETS.keys())
    target_values = np.array(
        [MHLW_VALIDATION_TARGETS[k]["value"] for k in target_labels]
    )
    gaps = target_values - national_prevalence

    save_artifacts(
        output_path,
        arrays={
            "national_prevalence_point": np.array([national_prevalence]),
            "cell_weights": cell_weights,
            "target_values": target_values,
            "gaps_vs_targets": gaps,
        },
        metadata=standard_metadata(
            stage="stage1_population_aggregation",
            extra={
                "weight_construction": weight_provenance,
                "gender_proportion_female": float(gender_props[0]),
                "gender_proportion_male": float(gender_props[1]),
                "cluster_proportion_source": "IEEE-published 7-cluster k-means (M3-fixed)",
                "primary_target": "MHLW H28 FY2016 (32.5%)",
            },
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cell-propensity",
        type=Path,
        default=Path("output/supplementary/stage0_cell_propensity.h5"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/supplementary/stage1_population_aggregation.h5"),
    )
    parser.add_argument(
        "--mhlw-data",
        type=Path,
        default=None,
        help=(
            "Path to MHLW Labor Force Survey 2022 CSV. If omitted, looks for "
            f"{MHLW_WEIGHTS_PATH}; if not present, falls back to placeholder "
            "gender proportions with a warning."
        ),
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    if args.seed is not None and args.seed != 20260429:
        import warnings
        warnings.warn(
            "Seed override; v2.0 fixes seed=20260429.", stacklevel=2
        )
    _ = make_rng(extra_offset=30_000)  # reserve stream slot for Stage 1
    run(args.cell_propensity, args.output, mhlw_path=args.mhlw_data)


if __name__ == "__main__":
    main()
