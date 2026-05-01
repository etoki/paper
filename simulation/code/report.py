"""Final report orchestrator: aggregates stage outputs into tables/figures.

Specification:
- v2.0 master Section 8.3 (preregistered reporting items).
- Methods Clarifications Log Section 8.2 (eventual paper Methods will
  reference both v2.0 master + clarifications log).

Inputs (HDF5 sentinels from each stage):
- stage2_validation.h5 (FY2016/2020/2023 APE point + CI + tier)
- stage3_sensitivity.h5 (Cartesian-product sweep table)
- stage4_baselines.h5 (B0-B4 MAPE + Page's L)
- stage5_cmv_diagnostic.h5 (Harman + marker)
- stage7_counterfactual.h5 (ΔP_x + positivity + H7 IUT)
- stage8_transportability.h5 (factor sweep)

Outputs:
- output/tables/h1_classification.txt (headline tier — printed to stdout
  at end of `make reproduce`)
- output/tables/cell_propensity_table.csv (14 cells × point/CI/method)
- output/tables/baseline_hierarchy.csv (B0-B4 MAPE comparison)
- output/tables/counterfactual_summary.csv (ΔP_A/B/C + H7 IUT)
- output/figures/calibration_plot.png
- output/figures/h7_iut_distribution.png
- output/supplementary/deviation_log.md (Section 6.5 deviation log
  populated from this run)

Random seed: 20260429.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .utils_io import (
    load_artifacts,
    standard_metadata,
)


def write_h1_classification(
    validation_path: str | Path, output_path: str | Path
) -> None:
    """Write the headline H1 4-tier classification to a text file."""
    arrays, metadata = load_artifacts(validation_path)
    tier = metadata.get("tier_FY2016", "UNKNOWN")
    explanation = metadata.get("tier_explanation_FY2016", "")
    point = float(arrays["point_ape_FY2016"][0]) if "point_ape_FY2016" in arrays else None
    ci_lo = float(arrays["ci_lo_FY2016"][0]) if "ci_lo_FY2016" in arrays else None
    ci_hi = float(arrays["ci_hi_FY2016"][0]) if "ci_hi_FY2016" in arrays else None

    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"H1 PRIMARY TIER: {tier}",
        "",
        f"  Validation target: MHLW H28 (FY2016, pre-law) past-3-year power",
        f"                     harassment victimization rate = 32.5%",
        "",
    ]
    if point is not None:
        lines.append(f"  Point MAPE: {point:.2f}%")
    if ci_lo is not None and ci_hi is not None:
        lines.append(f"  95% bootstrap CI: [{ci_lo:.2f}%, {ci_hi:.2f}%]")
    lines.extend([
        "",
        f"  {explanation}",
        "",
        "  v2.0 OSF DOI: 10.17605/OSF.IO/3Y54U",
        "  Methods Clarifications Log v1.0 (locked 2026-04-30)",
    ])
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def write_deviation_log(
    output_path: str | Path,
    stage_paths: list[str | Path],
) -> None:
    """Aggregate per-stage Level 1/2 deviations into a unified log.

    TODO (Stage 0 actual impl): inspect each stage's HDF5 metadata for
    deviation flags (e.g., MoM rejected, BCa fallback triggered, soft-
    assignment shift > 5pp) and emit a structured Markdown deviation
    log per Section 6.5 Level 1/2.
    """
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        "# Deviation log (Section 6.5)\n\n"
        "TODO (Stage 0 actual impl): aggregate deviations from "
        + ", ".join(str(s) for s in stage_paths)
        + ".\n",
        encoding="utf-8",
    )


def run(
    validation_path: str | Path,
    sensitivity_path: str | Path,
    baselines_path: str | Path,
    cmv_path: str | Path,
    counterfactual_path: str | Path,
    transportability_path: str | Path,
    output_tier_path: str | Path,
) -> None:
    """Orchestrate report generation."""
    write_h1_classification(validation_path, output_tier_path)

    # TODO (Stage 0 actual impl): generate CSV tables and PNG figures
    # by reading HDF5 sentinels and applying tabulate / matplotlib.
    deviation_log = Path(output_tier_path).parent.parent / "supplementary" / "deviation_log.md"
    write_deviation_log(
        deviation_log,
        stage_paths=[
            validation_path,
            sensitivity_path,
            baselines_path,
            cmv_path,
            counterfactual_path,
            transportability_path,
        ],
    )

    # Reporting metadata (for traceability)
    _ = standard_metadata(stage="report")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validation",
        type=Path,
        default=Path("output/supplementary/stage2_validation.h5"),
    )
    parser.add_argument(
        "--sensitivity",
        type=Path,
        default=Path("output/supplementary/stage3_sensitivity.h5"),
    )
    parser.add_argument(
        "--baselines",
        type=Path,
        default=Path("output/supplementary/stage4_baselines.h5"),
    )
    parser.add_argument(
        "--cmv",
        type=Path,
        default=Path("output/supplementary/stage5_cmv_diagnostic.h5"),
    )
    parser.add_argument(
        "--counterfactual",
        type=Path,
        default=Path("output/supplementary/stage7_counterfactual.h5"),
    )
    parser.add_argument(
        "--transportability",
        type=Path,
        default=Path("output/supplementary/stage8_transportability.h5"),
    )
    parser.add_argument(
        "--output-tier",
        type=Path,
        default=Path("output/tables/h1_classification.txt"),
        help="Headline tier classification text file (printed at make reproduce end)",
    )
    args = parser.parse_args()
    run(
        validation_path=args.validation,
        sensitivity_path=args.sensitivity,
        baselines_path=args.baselines,
        cmv_path=args.cmv,
        counterfactual_path=args.counterfactual,
        transportability_path=args.transportability,
        output_tier_path=args.output_tier,
    )


if __name__ == "__main__":
    main()
