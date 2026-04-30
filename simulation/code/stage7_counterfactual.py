"""Stage 7: Counterfactual A/B/C ΔP_x estimation + positivity + H7 IUT.

Specification:
- v2.0 master Section 5.7 (counterfactuals A/B/C with do-operator).
- Methods Clarifications Log Section 4.5 (m5): positivity quantitative
  criterion (ρ_{c,x} < 0.10 → flag; flagged_weight ≥ 20% → confirmatory
  → exploratory downgrade).
- Methods Clarifications Log Section 4.7 (m7): H7 intersection-union
  test (Berger & Hsu 1996) with one-sided 5% lower bounds on Δ_BA, Δ_BC.

Inputs:
- output/supplementary/stage0_type_assignment.h5 (HEXACO matrix +
  centroids + hard assignment)
- output/supplementary/stage0_cell_propensity.h5 (cell propensities)
- output/supplementary/stage1_population_aggregation.h5 (cell weights)
- output/supplementary/stage6_target_trial.h5 (PICO metadata)

Output:
- output/supplementary/stage7_counterfactual.h5 with:
    - delta_p_a, delta_p_b, delta_p_c (point + bootstrap distributions)
    - positivity_diagnostic_a/b/c (ρ_{c,x} per cell + flagged mask +
      flagged_weight share)
    - h7_iut: L_BA, L_BC, classification (CONFIRMED / PARTIAL / REVERSAL
      / AMBIGUOUS) per m7

Random seed: 20260429.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .utils_diagnostics import (
    POSITIVITY_FLAGGED_WEIGHT_MAX,
    POSITIVITY_RATIO_THRESHOLD,
)
from .utils_io import (
    BOOTSTRAP_PER_CELL,
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


def run(
    type_assignment_path: str | Path,
    cell_propensity_path: str | Path,
    aggregation_path: str | Path,
    target_trial_path: str | Path,
    output_path: str | Path,
    n_bootstrap: int = BOOTSTRAP_PER_CELL,
) -> None:
    """Estimate ΔP_A, ΔP_B, ΔP_C with positivity (m5) and H7 IUT (m7).

    TODO (Stage 0 actual impl):
    1. Load all input artifacts.
    2. For each counterfactual x ∈ {A, B, C}:
       a. Apply do-operator per Pearl 2009 (n3 form).
       b. Re-run Stage 0 → Stage 1 to get P̂_x with bootstrap CI.
       c. Compute ρ_{c,x} per cell (m5):
          - A: ρ ≡ 1 (universal, no extrapolation)
          - B: ρ_{c,B} = (observed in target Cluster) / (expected post-intervention)
          - C: ρ ≡ 1 (cell-level multiplier, no extrapolation)
       d. Apply m5 downgrade: if Σ W_c (flagged) / Σ W_c ≥ 20%,
          downgrade ΔP_x from confirmatory to exploratory.
    3. H7 IUT (m7):
       Δ_BA^{(b)} = ΔP_B^{(b)} − ΔP_A^{(b)}
       Δ_BC^{(b)} = ΔP_B^{(b)} − ΔP_C^{(b)}
       L_BA = 5th percentile of {Δ_BA^{(b)}}
       L_BC = 5th percentile of {Δ_BC^{(b)}}
       CONFIRMED if L_BA > 0 AND L_BC > 0
       PARTIAL if exactly one > 0
       REVERSAL if point ΔP_B < ΔP_A or point ΔP_B < ΔP_C
       AMBIGUOUS otherwise (CIs allow zero/reversal)
    """
    _ = make_rng(extra_offset=90_000)
    save_artifacts(
        output_path,
        arrays={},
        metadata=standard_metadata(
            stage="stage7_counterfactual",
            extra={
                "status": "SKELETON",
                "delta_a_main_sd": float(DELTA_A_MAIN_SD),
                "delta_b_main_sd": float(DELTA_B_MAIN_SD),
                "effect_c_main": float(EFFECT_C_MAIN),
                "cluster_b_targets_main": ",".join(map(str, CLUSTER_B_TARGETS_FULL)),
                "positivity_threshold_rho": float(POSITIVITY_RATIO_THRESHOLD),
                "positivity_downgrade_weight_threshold": float(POSITIVITY_FLAGGED_WEIGHT_MAX),
                "h7_test_method": "Berger & Hsu 1996 IUT; one-sided 5% bootstrap CIs on Δ_BA and Δ_BC",
                "h7_n_bootstrap": int(n_bootstrap),
            },
        ),
    )


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
    )


if __name__ == "__main__":
    main()
