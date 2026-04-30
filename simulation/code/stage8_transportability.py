"""Stage 8: Transportability factor sweep for Phase 2 counterfactuals.

Specification:
- v2.0 master Section 5.8 (cultural attenuation: Western anchor effect ×
  factor before applying to Japan; factor ∈ {0.3, 0.5, 0.7, 1.0}).
- v2.0 master Section 6.4 (transportability_factor sweep main = 1.0).
- Citation anchors: Sapouna 2010 (UK→Germany null worst case);
  Nielsen 2017 (Asia/Oceania attenuation).

Inputs:
- output/supplementary/stage7_counterfactual.h5 (ΔP_x point + bootstrap)

Output:
- output/supplementary/stage8_transportability.h5 with:
    - factors (4,) float = [0.3, 0.5, 0.7, 1.0]
    - delta_p_a_attenuated (4, ...) float per factor
    - delta_p_b_attenuated (4, ...) float per factor
    - delta_p_c_attenuated (4, ...) float per factor

Random seed: 20260429.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .utils_io import make_rng, save_artifacts, standard_metadata

TRANSPORTABILITY_FACTORS = np.array([0.3, 0.5, 0.7, 1.0])
"""Cross-cultural attenuation factors per v2.0 Section 5.8."""


def run(counterfactual_path: str | Path, output_path: str | Path) -> None:
    """Apply transportability factors to ΔP_x. TODO (Stage 0 actual impl)."""
    _ = make_rng(extra_offset=100_000)
    save_artifacts(
        output_path,
        arrays={"factors": TRANSPORTABILITY_FACTORS},
        metadata=standard_metadata(
            stage="stage8_transportability",
            extra={
                "status": "SKELETON",
                "factors": ",".join(f"{f:.1f}" for f in TRANSPORTABILITY_FACTORS),
                "anchors": "Sapouna 2010 (UK->Germany null); Nielsen 2017 (Asia/Oceania attenuation r=.16 vs Europe r=.33)",
            },
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--counterfactual",
        type=Path,
        default=Path("output/supplementary/stage7_counterfactual.h5"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/supplementary/stage8_transportability.h5"),
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    if args.seed is not None and args.seed != 20260429:
        import warnings
        warnings.warn(
            "Seed override; v2.0 fixes seed=20260429.", stacklevel=2
        )
    run(args.counterfactual, args.output)


if __name__ == "__main__":
    main()
