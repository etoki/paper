"""Stage 4: B0–B4 baseline hierarchy + Page's L auxiliary.

Specification:
- v2.0 master Section 5.5 (B0 uniform, B1 gender-only logistic, B2
  HEXACO 6-domain logistic, B3 7-type × gender, B4 = B3 + age + industry
  + employment).
- Methods Clarifications Log Section 6.4 (n4): Page's L (1963)
  ordinal-trend test as auxiliary, alongside Bonferroni-Holm primary.

Inputs:
- output/supplementary/stage2_validation.h5 (FY2016 point + CI)
- harassment data (re-run baselines on N=354)

Output:
- output/supplementary/stage4_baselines.h5
    - mape_b0, mape_b1, mape_b2, mape_b3, mape_b4 (point + CI per baseline)
    - bonferroni_holm_pvals (4,) for pairwise inequalities
    - pages_l_statistic (auxiliary)
    - h2_decision (str): "monotonic_confirmed" / "ambiguous" / "reversed"

Random seed: 20260429.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .utils_io import make_rng, save_artifacts, standard_metadata


def run(output_path: str | Path) -> None:
    """Compute B0-B4 MAPEs + Page's L. TODO (Stage 0 actual impl)."""
    _ = make_rng(extra_offset=60_000)
    save_artifacts(
        output_path,
        arrays={},
        metadata=standard_metadata(
            stage="stage4_baselines",
            extra={
                "status": "SKELETON",
                "h2_test_primary": "Bonferroni-Holm on 4 pairwise inequalities (B0-B1, B1-B2, B2-B3, B3-B4)",
                "h2_test_auxiliary": "Page's L (1963) ordinal trend, isotonic regression goodness-of-fit (n4)",
            },
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/supplementary/stage4_baselines.h5"),
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    if args.seed is not None:
        import warnings

        warnings.warn("Seed override; v2.0 fixes seed=20260429.", stacklevel=2)
    run(args.output)


if __name__ == "__main__":
    main()
