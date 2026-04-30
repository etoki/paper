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
    N_CELLS_MAIN,
    load_artifacts,
    make_rng,
    save_artifacts,
    standard_metadata,
)


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


def run(cell_propensity_path: str | Path, output_path: str | Path) -> None:
    """Aggregate cell propensities to national prevalence per validation period."""
    arrays, _meta = load_artifacts(cell_propensity_path)
    point_power = arrays["point_power"]

    # TODO (Stage 0 actual impl): load MHLW Labor Force Survey 2022 weights.
    # Skeleton uses N=13,668 cluster proportions placeholder.
    placeholder_cluster_props = np.array([0.117, 0.158, 0.168, 0.129, 0.154, 0.136, 0.139])
    placeholder_gender_props = np.array([0.5, 0.5])
    cell_weights = construct_population_weights(
        placeholder_cluster_props, placeholder_gender_props
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
                "weight_construction": "PLACEHOLDER (TODO: MHLW Labor Force 2022 reweight)",
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
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    if args.seed is not None:
        import warnings

        warnings.warn("Seed override; v2.0 fixes seed=20260429.", stacklevel=2)
    _ = make_rng(extra_offset=30_000)  # reserve stream slot for Stage 1
    run(args.cell_propensity, args.output)


if __name__ == "__main__":
    main()
