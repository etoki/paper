"""Stage 2: Validation triangulation + 4-tier H1 classification.

Specification:
- v2.0 master Section 5.4 (Bootstrap MAPE CI 6-step procedure +
  4-tier judgment hierarchy).
- Methods Clarifications Log Section 4.3 (m3): headline national MAPE
  uses B = 10,000; per-cell remains 2,000.
- Methods Clarifications Log Section 6.2 (n2): MAPE^(b) vs APE^(b)
  notation; primary H1 is APE_FY2016, MAPE-of-three-targets is
  reported as supplementary.

Inputs:
- output/supplementary/stage0_cell_propensity.h5 (raw cell counts)
- output/supplementary/stage1_population_aggregation.h5 (point P̂_t,
  weights)

Output:
- output/supplementary/stage2_validation.h5 with:
    - point_mape_FY2016, ci_lo_FY2016, ci_hi_FY2016 (per period)
    - tier_FY2016 (str), tier_FY2020, tier_FY2023
    - boot_mape_FY2016 (B,) array for downstream H7 IUT etc.
    - global_mape (mean over 3 targets) point + CI

Random seed: 20260429.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .utils_bootstrap import (
    BOOTSTRAP_HEADLINE_MAPE,
    cell_stratified_bootstrap,
    classify_h1_tier,
    percentile_interval,
)
from .utils_io import (
    MHLW_VALIDATION_TARGETS,
    load_artifacts,
    make_rng,
    save_artifacts,
    standard_metadata,
)


def absolute_percentage_error(predicted: float, observed: float) -> float:
    """APE = |predicted − observed| / observed × 100 (per n2 notation)."""
    if observed == 0:
        return float("nan")
    return abs(predicted - observed) / observed * 100.0


def run(
    cell_propensity_path: str | Path,
    aggregation_path: str | Path,
    output_path: str | Path,
    n_bootstrap: int = BOOTSTRAP_HEADLINE_MAPE,
) -> None:
    """Compute headline MAPE with 4-tier classification per v2.0 Section 5.4."""
    cell_arrays, _ = load_artifacts(cell_propensity_path)
    agg_arrays, _ = load_artifacts(aggregation_path)

    cell_n = cell_arrays["cell_n"]
    cell_x = cell_arrays["cell_x_power"]
    cell_weights = agg_arrays["cell_weights"]
    point_pred = float(agg_arrays["national_prevalence_point"][0])

    # TODO (Stage 0 actual impl): wire up cell_data per the M3 cell-stratified
    # bootstrap procedure documented in clarifications log Section 4.3 step (i)–(vi).
    # Skeleton: build per-cell binary indicator arrays from cell_n, cell_x.
    cell_data = []
    for c in range(len(cell_n)):
        n_c = int(cell_n[c])
        x_c = int(cell_x[c])
        if n_c > 0:
            arr = np.zeros(n_c, dtype=int)
            arr[:x_c] = 1
            cell_data.append(arr)
        else:
            cell_data.append(np.array([], dtype=int))

    rng = make_rng(extra_offset=40_000)

    def aggregate_statistic(resampled: list) -> float:
        """Compute national prevalence from resampled cell data."""
        cell_p_boot = np.array(
            [
                float(np.mean(arr)) if len(arr) > 0 else 0.0
                for arr in resampled
            ]
        )
        total_w = float(cell_weights.sum())
        if total_w <= 0:
            return 0.0
        return float(np.sum(cell_p_boot * cell_weights) / total_w)

    # Bootstrap distribution of national prevalence
    boot_prevalence = cell_stratified_bootstrap(
        cell_data=cell_data,
        statistic_fn=aggregate_statistic,
        n_bootstrap=n_bootstrap,
        rng=rng,
    )

    # Per-target APE (point + bootstrap CI)
    results = {}
    for target_label, target_info in MHLW_VALIDATION_TARGETS.items():
        observed = target_info["value"]
        point_ape = absolute_percentage_error(point_pred, observed)
        boot_ape = np.array(
            [absolute_percentage_error(p, observed) for p in boot_prevalence]
        )
        ci_lo, ci_hi = percentile_interval(boot_ape)
        # NOTE: per-period CI uses percentile here; BCa applied via
        # utils_bootstrap.bca_interval can be added in Stage 0 actual
        # implementation phase.
        results[target_label] = {
            "point_ape": float(point_ape),
            "ci_lo": float(ci_lo),
            "ci_hi": float(ci_hi),
            "boot_ape": boot_ape,
        }

    # 4-tier classification, primary on FY2016
    tier_classification = classify_h1_tier(
        point_mape=results["FY2016"]["point_ape"],
        ci_lower=results["FY2016"]["ci_lo"],
        ci_upper=results["FY2016"]["ci_hi"],
    )

    arrays_out = {
        "boot_prevalence": boot_prevalence.astype(np.float32),
    }
    for k in MHLW_VALIDATION_TARGETS:
        arrays_out[f"point_ape_{k}"] = np.array([results[k]["point_ape"]])
        arrays_out[f"ci_lo_{k}"] = np.array([results[k]["ci_lo"]])
        arrays_out[f"ci_hi_{k}"] = np.array([results[k]["ci_hi"]])
        arrays_out[f"boot_ape_{k}"] = results[k]["boot_ape"].astype(np.float32)

    metadata = standard_metadata(
        stage="stage2_validation",
        extra={
            "n_bootstrap": int(n_bootstrap),
            "primary_target": "MHLW H28 FY2016 (32.5%)",
            "tier_FY2016": tier_classification.tier,
            "tier_explanation_FY2016": tier_classification.explanation,
            "ci_method": "percentile (BCa upgrade in Stage 0 actual impl)",
        },
    )

    save_artifacts(output_path, arrays=arrays_out, metadata=metadata)

    # Print headline tier to stdout for `make reproduce` end-of-pipeline summary
    print(f"H1 PRIMARY TIER (FY2016): {tier_classification.tier}")
    print(f"  {tier_classification.explanation}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
        default=Path("output/supplementary/stage2_validation.h5"),
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=BOOTSTRAP_HEADLINE_MAPE,
        help="Headline MAPE iterations (m3 default = 10,000)",
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    if args.seed is not None and args.seed != 20260429:
        import warnings
        warnings.warn(
            "Seed override; v2.0 fixes seed=20260429.", stacklevel=2
        )
    run(
        cell_propensity_path=args.cell_propensity,
        aggregation_path=args.aggregation,
        output_path=args.output,
        n_bootstrap=args.n_bootstrap,
    )


if __name__ == "__main__":
    main()
