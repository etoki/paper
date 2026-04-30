"""Stage 6: Target trial emulation specification.

Specification:
- v2.0 master Section 5.7 (Hernán & Robins 2020 target trial emulation).
- Methods Clarifications Log Section 4.6 (m6): SUTVA / no-interference
  language strengthened with Hudgens & Halloran 2008.
- Methods Clarifications Log Section 6.3 (n3): Pearl 2009 do-operator
  notation form.

This stage produces a structured PICO + identifying-assumptions
document that downstream Stage 7 consumes for counterfactual estimation.

Output:
- output/supplementary/stage6_target_trial.h5 with metadata-only
  attributes encoding the target trial protocol.

Random seed: 20260429.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .utils_io import make_rng, save_artifacts, standard_metadata


def run(output_path: str | Path) -> None:
    """Emit target trial specification metadata for Stage 7 consumption."""
    _ = make_rng(extra_offset=80_000)
    save_artifacts(
        output_path,
        arrays={},
        metadata=standard_metadata(
            stage="stage6_target_trial",
            extra={
                "framework": "Hernán & Robins 2020 target trial emulation",
                "do_operator_notation": "Pearl 2009 (n3 cleaner form: define HH'_i first, then do(HH = HH'_i))",
                # PICO
                "P_population": "Japanese workers aged 20-64",
                "I_counterfactual_A": "do(HH = HH'_i^A) for all i; HH'_i^A = HH_i + delta_A * SD(HH); delta_A = +0.3 SD main",
                "I_counterfactual_B": "do(HH = HH'_i^B) for i in Cluster {0,4,6}; HH'_i^B = HH_i + delta_B * SD(HH); delta_B = +0.4 SD main",
                "I_counterfactual_C": "do(p_c = p'_c^C) for all 14 cells; p'_c^C = p_c * (1 - effect_C); effect_C = 0.20 main",
                "C_control": "Pre-intervention baseline (observed propensities)",
                "O_outcome": "National harassment prevalence; ΔP_x = P̂_baseline − P̂_x",
                "duration": "24 weeks (Roberts 2017 anchor)",
                # 4 identifying assumptions (Section 5.7.4 + m6 strengthening)
                "assumption_1_exchangeability": "Y^a ⊥⊥ A | L; mitigation: B4 baseline + sensitivity sweep",
                "assumption_2_positivity": "P(A=a|L=l)>0; m5 quantitative diagnostic in Stage 7",
                "assumption_3_consistency_sutva": "STRENGTHENED per m6: harassment is dyadic; SUTVA plausibly violated 10-30% (Christakis & Fowler 2007 anchor); ΔP_B underestimates if positive peer effects exist, overestimates if negative rebound; future work cites Hudgens & Halloran 2008, Aronow & Samii 2017, VanderWeele 2017",
                "assumption_4_transportability": "Western anchor → Japan; Stage 8 sensitivity sweep {0.3×, 0.5×, 0.7×, 1.0×}",
            },
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/supplementary/stage6_target_trial.h5"),
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
