"""Stage 5: Common-method-variance diagnostic.

Specification:
- v2.0 master Section 5.6 (Harman's single-factor test on N=13,668 +
  marker-variable correction with HEXACO Openness per Lindell &
  Whitney 2001).

Inputs:
- N=13,668 HEXACO data (or aggregated centroid table; Stage 0 actual
  impl decides which is required).

Output:
- output/supplementary/stage5_cmv_diagnostic.h5 with:
    - first_factor_variance_pct (float): Harman's first-factor variance %
    - cmv_concern_flag (bool): True if first_factor_variance >= 50%
    - marker_adjusted_correlations (matrix)

Random seed: 20260429.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .utils_io import make_rng, save_artifacts, standard_metadata


def run(output_path: str | Path) -> None:
    """TODO (Stage 0 actual impl): Harman + marker-variable diagnostics."""
    _ = make_rng(extra_offset=70_000)
    save_artifacts(
        output_path,
        arrays={},
        metadata=standard_metadata(
            stage="stage5_cmv_diagnostic",
            extra={
                "status": "SKELETON",
                "harman_threshold": "first_factor_variance < 50% -> concern limited",
                "marker_variable": "HEXACO Openness (Lindell & Whitney 2001)",
            },
        ),
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
    if args.seed is not None:
        import warnings

        warnings.warn("Seed override; v2.0 fixes seed=20260429.", stacklevel=2)
    run(args.output)


if __name__ == "__main__":
    main()
