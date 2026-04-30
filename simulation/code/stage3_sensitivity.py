"""Stage 3: Sensitivity sweeps over V, f1, f2, EB scale, etc.

Specification:
- v2.0 master Section 6.4 (sensitivity master table).
- Methods Clarifications Log Section 3.2 (M2): soft-assignment τ sweep.
- Methods Clarifications Log Section 4.1, 4.2 (m1, m2): MoM rule
  diagnostics already covered upstream in Stage 0 step 3.

This stage orchestrates the full sweep:
- V (victim multiplier) ∈ {2, 3, 4, 5}
- f1 (turnover rate) ∈ {0.05, 0.10, 0.15, 0.20}
- f2 (mental disorder rate) ∈ {0.10, 0.20, 0.30}
- EB shrinkage scale ∈ {0.5×, 1.0×, 2.0×}
- Binarization threshold ∈ {mean+0.25 SD, +0.5 SD, +1.0 SD}
- Cluster K ∈ {4, 5, 6, 7, 8}
- Role-estimation models ∈ {linear, tree-based, literature}
- Soft-assignment τ ∈ {0.5, 1.0, 2.0} × median NN distance (M2)

Output:
- output/supplementary/stage3_sensitivity.h5 with a Cartesian-product
  table of (parameter combination → MAPE_FY2016 + tier classification).

Random seed: 20260429.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .utils_io import make_rng, save_artifacts, standard_metadata


def run(output_path: str | Path) -> None:
    """Run sensitivity sweeps and persist results.

    TODO (Stage 0 actual impl): orchestrate sweeps via sub-process
    invocations of stages 0-2 with the parameter combinations above.
    Each combination produces a (point MAPE, CI, tier) triple, written
    as one row in the output. Total combinations ≈ 4 × 4 × 3 × 3 × 3 ×
    5 × 3 × 4 = 25,920. With per-combination cost of ~30 sec via
    streamlined re-runs, this is ~3.5 hours; reduced via incremental
    output caching.

    Skeleton: write empty result table with metadata.
    """
    _ = make_rng(extra_offset=50_000)
    save_artifacts(
        output_path,
        arrays={},
        metadata=standard_metadata(
            stage="stage3_sensitivity",
            extra={
                "status": "SKELETON (TODO: full Cartesian-product sweep)",
                "v_range": "2,3,4,5",
                "f1_range": "0.05,0.10,0.15,0.20",
                "f2_range": "0.10,0.20,0.30",
                "eb_scale_range": "0.5,1.0,2.0",
                "binarization_range": "mean+0.25SD,mean+0.5SD,mean+1.0SD",
                "k_range": "4,5,6,7,8",
                "role_models": "linear,tree-based,literature",
                "soft_assignment_tau_factors": "0.5,1.0,2.0",
            },
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/supplementary/stage3_sensitivity.h5"),
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
