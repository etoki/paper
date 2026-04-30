"""Stage 0 step 1: Type assignment via 7-centroid nearest-neighbor.

Specification:
- v2.0 master Section 2.3 Stage 0 step 1 (hard nearest-neighbor by
  Euclidean distance over HEXACO 6 domains).
- Methods Clarifications Log Section 3.2 (M2): hard nearest-neighbor as
  primary; soft assignment τ ∈ {0.5, 1.0, 2.0} × median NN distance as
  pre-registered sensitivity.
- Methods Clarifications Log Section 3.3 (M3): centroids fixed
  parameters; not bootstrapped.

Inputs:
- ../harassment/raw.csv (N=354, HEXACO 6 + Dark Triad 3 + harassment
  scales + demographics)
- ../clustering/csv/clstr_kmeans_7c.csv (7 centroids, IEEE-published)

Output:
- output/supplementary/stage0_type_assignment.h5 with arrays:
    - hard_assignment (N=354,) int: nearest-cluster index ∈ {0..6}
    - distances (N=354, K=7) float: Euclidean distances to each centroid
    - soft_weights_tau05/tau10/tau20 (N=354, K=7) float: M2 sensitivity
    - hexaco_matrix (N=354, 6) float: standardized HEXACO score matrix
    - centroids (K=7, 6) float: input centroids (recorded for traceability)
  And metadata:
    - seed, stage, version, osf_doi, median_nn_distance

Random seed: 20260429.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .utils_diagnostics import median_nn_distance, soft_assign_weights
from .utils_io import (
    HEXACO_DOMAINS,
    N_CLUSTERS,
    load_centroids,
    load_harassment,
    save_artifacts,
    standard_metadata,
)

# ====================================================================
# Constants per clarifications log
# ====================================================================

SOFT_TEMPERATURE_FACTORS = (0.5, 1.0, 2.0)
"""M2: τ ∈ {0.5, 1.0, 2.0} × median nearest-neighbor distance."""


# ====================================================================
# Core computation
# ====================================================================


def euclidean_distances_to_centroids(
    hexaco_matrix: np.ndarray, centroids: np.ndarray
) -> np.ndarray:
    """Compute Euclidean distance from each individual to each centroid.

    Parameters
    ----------
    hexaco_matrix : ndarray, shape (N, 6)
    centroids : ndarray, shape (K, 6)

    Returns
    -------
    distances : ndarray, shape (N, K)
    """
    if hexaco_matrix.shape[1] != centroids.shape[1]:
        raise ValueError(
            f"HEXACO dim mismatch: data has {hexaco_matrix.shape[1]} domains, "
            f"centroids have {centroids.shape[1]}."
        )
    diff = hexaco_matrix[:, None, :] - centroids[None, :, :]
    return np.sqrt(np.sum(diff**2, axis=2))


def hard_assign(distances: np.ndarray) -> np.ndarray:
    """Argmin over centroids → integer cluster index per individual."""
    return np.argmin(distances, axis=1).astype(np.int8)


# ====================================================================
# Pipeline orchestration
# ====================================================================


def run(output_path: str | Path) -> None:
    """Execute Stage 0 step 1 and persist artifacts to HDF5."""
    harassment = load_harassment()
    centroid_data = load_centroids()

    # Validate domain columns present
    missing = [d for d in HEXACO_DOMAINS if d not in harassment.df.columns]
    if missing:
        raise ValueError(
            f"Harassment data missing HEXACO domain columns: {missing}. "
            f"Expected columns: {HEXACO_DOMAINS}"
        )

    hexaco = harassment.hexaco_matrix
    centroids = centroid_data.matrix
    if centroids.shape[0] != N_CLUSTERS:
        raise ValueError(
            f"Expected {N_CLUSTERS} centroids; got {centroids.shape[0]}."
        )

    # Per M3: centroids are fixed parameters; we do NOT re-fit them here.

    # Hard assignment (primary)
    distances = euclidean_distances_to_centroids(hexaco, centroids)
    hard = hard_assign(distances)

    # Median nearest-neighbor distance (anchor for M2 soft assignment)
    median_nn = median_nn_distance(distances)

    # Soft assignment sensitivity (M2): τ = factor × median NN distance
    soft_weights: dict[str, np.ndarray] = {}
    for factor in SOFT_TEMPERATURE_FACTORS:
        tau = factor * median_nn
        if tau <= 0:
            # Degenerate: data has zero spread (impossible in N=354 HEXACO).
            # Fall back to one-hot hard assignment to keep the pipeline running.
            w = np.zeros_like(distances, dtype=float)
            w[np.arange(len(hard)), hard] = 1.0
        else:
            w = soft_assign_weights(distances, tau)
        soft_weights[f"soft_weights_tau{int(factor * 10):02d}"] = w

    arrays = {
        "hard_assignment": hard,
        "distances": distances.astype(np.float32),
        "hexaco_matrix": hexaco.astype(np.float32),
        "centroids": centroids.astype(np.float32),
        **soft_weights,
    }

    metadata = standard_metadata(
        stage="stage0_type_assignment",
        extra={
            "median_nn_distance": float(median_nn),
            "n_individuals": int(len(hard)),
            "n_clusters": int(N_CLUSTERS),
            "soft_temperature_factors": ",".join(
                f"{f:.1f}" for f in SOFT_TEMPERATURE_FACTORS
            ),
        },
    )

    save_artifacts(output_path, arrays=arrays, metadata=metadata)


# ====================================================================
# CLI entry point (for `make stage0` Makefile rule)
# ====================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=None, help="Override seed (NOT recommended)")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/supplementary/stage0_type_assignment.h5"),
        help="Output HDF5 path",
    )
    args = parser.parse_args()
    if args.seed is not None:
        # Per v2.0 Section 2.4, seed is fixed by preregistration.
        # CLI override is provided only for ad-hoc development; it
        # MUST NOT be used in reproduction runs.
        import warnings

        warnings.warn(
            "Seed override detected. v2.0 master fixes seed=20260429; "
            "results will not match reference outputs.",
            stacklevel=2,
        )
    run(args.output)


if __name__ == "__main__":
    main()
