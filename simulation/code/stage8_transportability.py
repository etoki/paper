"""Stage 8: Transportability factor sweep for Phase 2 counterfactuals.

Specification:
- v2.0 master Section 5.8 (cultural attenuation: Western anchor effect ×
  factor before applying to Japan; factor ∈ {0.3, 0.5, 0.7, 1.0}).
- v2.0 master Section 6.4 (transportability_factor sweep main = 1.0×).
- Citation anchors:
  - Sapouna 2010 (UK→Germany null worst case) → factor 0.3 conservative
  - Nielsen 2017 (Asia/Oceania Neuroticism r=.16 vs Europe r=.33) →
    factor ≈ 0.5 expected attenuation
  - Other intermediate factors (0.7, 1.0) for sensitivity range

Inputs:
- output/supplementary/stage7_counterfactual.h5 (ΔP_x point + bootstrap)

Output:
- output/supplementary/stage8_transportability.h5 with arrays:
    - factors (4,) float = [0.3, 0.5, 0.7, 1.0]
    - delta_p_a_attenuated (4,) point + (4, 2) CI
    - delta_p_b_attenuated (4,) point + (4, 2) CI
    - delta_p_c_attenuated (4,) point + (4, 2) CI
    - h7_classifications (4,) bytes per-factor: REVERSAL/CONFIRMED/
      PARTIAL/AMBIGUOUS

Random seed: 20260429.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .stage7_counterfactual import H7_LEVEL, h7_iut
from .utils_io import (
    load_artifacts,
    make_rng,
    save_artifacts,
    standard_metadata,
)

# ====================================================================
# Constants per v2.0 Section 5.8
# ====================================================================

TRANSPORTABILITY_FACTORS = np.array([0.3, 0.5, 0.7, 1.0])
"""Cross-cultural attenuation factors per v2.0 Section 5.8 sensitivity sweep."""

FACTOR_LABELS = {
    0.3: "Sapouna 2010 worst case (UK→Germany null)",
    0.5: "Nielsen 2017 expected (Asia/Oceania attenuation r=.16 vs Europe .33)",
    0.7: "Mild attenuation (intermediate)",
    1.0: "Reference (no attenuation; anchor effect = Japan effect)",
}


def attenuate_counterfactual_effect(
    boot_delta: np.ndarray, factor: float
) -> np.ndarray:
    """Apply transportability factor to bootstrap distribution of ΔP.

    Per v2.0 Section 5.8: anchor effect × factor before applying to Japan.

    Parameters
    ----------
    boot_delta : ndarray, shape (B,)
        Bootstrap distribution of ΔP_x (positive = reduction)
    factor : float
        Cultural attenuation factor in [0, 1]

    Returns
    -------
    attenuated : ndarray, shape (B,)
    """
    return np.asarray(boot_delta, dtype=float) * float(factor)


def run(counterfactual_path: str | Path, output_path: str | Path) -> None:
    """Apply transportability factor sweep to Stage 7 counterfactual outputs."""
    _ = make_rng(extra_offset=100_000)

    arrays, meta = load_artifacts(counterfactual_path)

    # Read Stage 7 outputs
    delta_p_a_point = float(arrays["delta_p_a_point"][0])
    delta_p_b_point = float(arrays["delta_p_b_point"][0])
    delta_p_c_point = float(arrays["delta_p_c_point"][0])
    boot_delta_a = arrays["boot_delta_a"].astype(float)
    boot_delta_b = arrays["boot_delta_b"].astype(float)
    boot_delta_c = arrays["boot_delta_c"].astype(float)

    n_factors = len(TRANSPORTABILITY_FACTORS)

    delta_p_a_attenuated = np.zeros(n_factors, dtype=float)
    delta_p_b_attenuated = np.zeros(n_factors, dtype=float)
    delta_p_c_attenuated = np.zeros(n_factors, dtype=float)

    ci_a = np.zeros((n_factors, 2), dtype=float)
    ci_b = np.zeros((n_factors, 2), dtype=float)
    ci_c = np.zeros((n_factors, 2), dtype=float)

    L_BA_arr = np.zeros(n_factors, dtype=float)
    L_BC_arr = np.zeros(n_factors, dtype=float)
    classifications = np.zeros(n_factors, dtype="S20")

    for i, factor in enumerate(TRANSPORTABILITY_FACTORS):
        # Attenuate point estimates
        delta_p_a_attenuated[i] = delta_p_a_point * factor
        delta_p_b_attenuated[i] = delta_p_b_point * factor
        delta_p_c_attenuated[i] = delta_p_c_point * factor

        # Attenuate bootstrap distributions
        attenuated_a = attenuate_counterfactual_effect(boot_delta_a, factor)
        attenuated_b = attenuate_counterfactual_effect(boot_delta_b, factor)
        attenuated_c = attenuate_counterfactual_effect(boot_delta_c, factor)

        # 95% percentile CIs on attenuated distributions
        ci_a[i, 0] = float(np.percentile(attenuated_a, 2.5))
        ci_a[i, 1] = float(np.percentile(attenuated_a, 97.5))
        ci_b[i, 0] = float(np.percentile(attenuated_b, 2.5))
        ci_b[i, 1] = float(np.percentile(attenuated_b, 97.5))
        ci_c[i, 0] = float(np.percentile(attenuated_c, 2.5))
        ci_c[i, 1] = float(np.percentile(attenuated_c, 97.5))

        # H7 IUT per factor (m7)
        L_BA, L_BC, classification = h7_iut(
            attenuated_a,
            attenuated_b,
            attenuated_c,
            delta_p_a_attenuated[i],
            delta_p_b_attenuated[i],
            delta_p_c_attenuated[i],
            level=H7_LEVEL,
        )
        L_BA_arr[i] = L_BA
        L_BC_arr[i] = L_BC
        classifications[i] = classification.encode("ascii")

    save_artifacts(
        output_path,
        arrays={
            "factors": TRANSPORTABILITY_FACTORS,
            "delta_p_a_attenuated": delta_p_a_attenuated,
            "delta_p_b_attenuated": delta_p_b_attenuated,
            "delta_p_c_attenuated": delta_p_c_attenuated,
            "ci_a": ci_a,
            "ci_b": ci_b,
            "ci_c": ci_c,
            "L_BA_per_factor": L_BA_arr,
            "L_BC_per_factor": L_BC_arr,
            "h7_classifications": classifications,
        },
        metadata=standard_metadata(
            stage="stage8_transportability",
            extra={
                "factors": ",".join(f"{f:.1f}" for f in TRANSPORTABILITY_FACTORS),
                "anchors": (
                    "Sapouna 2010 (UK->Germany null) for 0.3; "
                    "Nielsen 2017 (Asia/Oceania r=.16 vs Europe r=.33) for 0.5; "
                    "intermediate 0.7; reference 1.0"
                ),
                "main_factor": 1.0,
                "sensitivity_range": "0.3 to 1.0",
                "h7_iut_method": "m7 IUT applied per attenuation factor",
            },
        ),
    )

    # Console summary
    print("[Stage 8] Transportability factor sweep")
    print(f"  {'Factor':>8s}  {'ΔP_A':>20s}  {'ΔP_B':>20s}  {'ΔP_C':>20s}  H7")
    for i, factor in enumerate(TRANSPORTABILITY_FACTORS):
        print(
            f"  {factor:>8.1f}  "
            f"{delta_p_a_attenuated[i]:+.4f} [{ci_a[i, 0]:+.4f}, {ci_a[i, 1]:+.4f}]  "
            f"{delta_p_b_attenuated[i]:+.4f} [{ci_b[i, 0]:+.4f}, {ci_b[i, 1]:+.4f}]  "
            f"{delta_p_c_attenuated[i]:+.4f} [{ci_c[i, 0]:+.4f}, {ci_c[i, 1]:+.4f}]  "
            f"{classifications[i].decode('ascii')}"
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
