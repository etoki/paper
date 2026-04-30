"""Data loading, seed management, and HDF5 artifact persistence.

Source specification:
- v2.0 master Section 4.2 (Measured variables) for input data structure
- v2.0 master Section 8 (Reproducibility) for seed and persistence
- Methods Clarifications Log (Section 6.5 Level 1 deviation) — no
  changes to data loading itself; persistence format documented here.

Random seed: 20260429 (hard-coded per v2.0 Section 2.4; NOT a tunable
parameter; reproducibility is contingent on this value).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from numpy.random import Generator, default_rng

# ====================================================================
# Constants from v2.0 master + clarifications log
# ====================================================================

SEED = 20260429
"""Random seed for all stochastic operations (v2.0 Section 2.4)."""

REPO_ROOT = Path(__file__).resolve().parents[2]
"""Path to repository root (where harassment/, clustering/, simulation/ live)."""

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"
"""Default location for simulation outputs."""

HARASSMENT_RAW_PATH = REPO_ROOT / "harassment" / "raw.csv"
"""Path to N=354 harassment data (Tokiwa harassment preprint, IRB-approved)."""

CENTROIDS_PATH = REPO_ROOT / "clustering" / "csv" / "clstr_kmeans_7c.csv"
"""Path to 7-cluster HEXACO centroids (Tokiwa clustering paper, IEEE-published)."""

# HEXACO 6-domain column names (Wakabayashi 2014 Japanese HEXACO-60)
HEXACO_DOMAINS = ["H", "E", "X", "A", "C", "O"]
"""HEXACO 6 domains: Honesty-Humility, Emotionality, eXtraversion, Agreeableness, Conscientiousness, Openness."""

N_CLUSTERS = 7
"""Number of HEXACO clusters per Tokiwa clustering paper (IEEE-published)."""

N_GENDERS = 2
"""Gender binary (0/1, n=133/220 in N=354)."""

N_CELLS_MAIN = 14
"""14 cells = 7 clusters × 2 genders (v2.0 Section 5.1 main analysis)."""

N_CELLS_SENSITIVITY = 28
"""28 cells = 7 clusters × 2 genders × 2 roles (v2.0 Section 5.2 EB sensitivity)."""

# MHLW past-3-year power harassment validation targets
# (v2.0 Section 1.4 H1 + Section 5.4)
MHLW_VALIDATION_TARGETS = {
    "FY2016": {
        "value": 0.325,
        "label": "MHLW H28 (FY2016, pre-law)",
        "role": "primary",
    },
    "FY2020": {
        "value": 0.314,
        "label": "MHLW R2 (FY2020, transition)",
        "role": "secondary",
    },
    "FY2023": {
        "value": 0.193,
        "label": "MHLW R5 (FY2023, post-law)",
        "role": "secondary",
    },
}
"""MHLW past-3-year power harassment victimization rates by validation period."""

# 4-tier H1 judgment hierarchy thresholds (v2.0 Section 5.4 + 6.1)
H1_THRESHOLDS = {
    "strict_success_max_mape": 30.0,  # both point + CI upper bound ≤ 30%
    "standard_success_max_mape": 30.0,  # point ≤ 30% (CI may overlap)
    "partial_success_max_mape": 60.0,  # 30 < point ≤ 60%
    # > 60% → FAILURE (publish per Section 7.3)
}

# Bootstrap iteration counts (v2.0 Section 5.1 + clarifications m3)
BOOTSTRAP_PER_CELL = 2_000
"""Per-cell bootstrap iterations (BCa CI, default; v2.0 Section 5.1)."""

BOOTSTRAP_HEADLINE_MAPE = 10_000
"""Headline national MAPE bootstrap iterations (clarifications m3)."""

# ====================================================================
# Random seed management
# ====================================================================


def make_rng(extra_offset: int = 0) -> Generator:
    """Create a deterministic NumPy ``Generator`` from the locked seed.

    Per v2.0 Section 2.4, all stochastic operations use the seed 20260429.
    Subsystems that need independent streams pass an ``extra_offset`` so
    that the parent seed remains the single source of truth.

    Parameters
    ----------
    extra_offset : int
        Optional additive offset (e.g., for stage-specific sub-streams).

    Returns
    -------
    numpy.random.Generator
        Deterministic Generator seeded with ``SEED + extra_offset``.

    Examples
    --------
    >>> rng = make_rng()
    >>> sample = rng.choice([0, 1, 2], size=5)
    >>> # Same code, same machine -> identical output every run.
    """
    return default_rng(SEED + extra_offset)


# ====================================================================
# Data loading
# ====================================================================


@dataclass(frozen=True)
class HarassmentData:
    """Container for N=354 harassment-preprint individual-level data."""

    df: pd.DataFrame
    """All individuals with HEXACO + Dark Triad + harassment scales + demographics."""

    hexaco_columns: tuple[str, ...] = tuple(HEXACO_DOMAINS)
    """Column names for HEXACO 6 domains."""

    @property
    def n(self) -> int:
        """Sample size."""
        return len(self.df)

    @property
    def hexaco_matrix(self) -> np.ndarray:
        """N x 6 HEXACO matrix (NaN handling per v2.0 Section 5 + clarifications M2/m5)."""
        return self.df.loc[:, list(self.hexaco_columns)].to_numpy()


@dataclass(frozen=True)
class CentroidData:
    """Container for 7-cluster HEXACO centroids from IEEE-published clustering."""

    df: pd.DataFrame
    """7 rows (clusters) × 6 columns (HEXACO domains) plus optional metadata."""

    hexaco_columns: tuple[str, ...] = tuple(HEXACO_DOMAINS)
    """HEXACO domain column names."""

    @property
    def n_clusters(self) -> int:
        """Number of clusters."""
        return len(self.df)

    @property
    def matrix(self) -> np.ndarray:
        """K x 6 centroid matrix in HEXACO space."""
        return self.df.loc[:, list(self.hexaco_columns)].to_numpy()


HARASSMENT_COLUMN_ALIASES = {
    "hexaco_HH": "H",
    "hexaco_E": "E",
    "hexaco_X": "X",
    "hexaco_A": "A",
    "hexaco_C": "C",
    "hexaco_O": "O",
    "Machiavellianism": "machiavellianism",
    "Narcissism": "narcissism",
    "Psychopathy": "psychopathy",
}
"""Canonicalize harassment/raw.csv column names to internal convention.

The raw CSV uses hexaco_HH/hexaco_E/... while v2.0 master + clarifications
log + utility code uses H/E/X/A/C/O. This alias map applied at load time
isolates the naming variability."""

CENTROID_COLUMN_ALIASES = {
    "Honesty-Humility": "H",
    "Emotionality": "E",
    "Extraversion": "X",
    "Agreeableness": "A",
    "Conscientiousness": "C",
    "Openness": "O",
}
"""Canonicalize centroid CSV column names to internal convention.

The IEEE-published centroid CSV uses full HEXACO domain names; this
alias map normalizes to internal H/E/X/A/C/O."""


def load_harassment(path: str | os.PathLike[str] | None = None) -> HarassmentData:
    """Load N=354 harassment-preprint individual-level data.

    Parameters
    ----------
    path : path-like, optional
        Override path. Defaults to ``HARASSMENT_RAW_PATH``.

    Returns
    -------
    HarassmentData
        Wrapper exposing the raw frame and the 6-domain HEXACO matrix.

    Notes
    -----
    Variables expected in the file (v2.0 Section 4.2.1):
    - HEXACO 6 domains (H, E, X, A, C, O), each Likert 1-5 mean
    - Dark Triad 3 (Machiavellianism, Narcissism, Psychopathy)
    - Power Harassment Scale (Tou 2017), continuous
    - Gender Harassment Scale (Kobayashi & Tanaka 2010), continuous
    - age (continuous), gender (binary 0/1), area (categorical)

    Column name canonicalization (Level 1 clarification, smoke-test
    discovered 2026-04-30): the raw CSV uses ``hexaco_HH``,
    ``hexaco_E``, etc. with the ``hexaco_`` prefix. ``HARASSMENT_COLUMN_ALIASES``
    renames these to internal H/E/X/A/C/O at load time. This keeps
    downstream code aligned with v2.0 master notation (Section 4.2.1).

    Missing-data handling per clarifications log Section 4.6 (m6 not
    present here; cluster reassignment behavior is implemented in
    ``code.stage0_type_assignment``).
    """
    p = Path(path) if path is not None else HARASSMENT_RAW_PATH
    if not p.is_file():
        raise FileNotFoundError(
            f"Harassment data not found at {p}. "
            "Verify ../harassment/raw.csv exists and is readable."
        )
    df = pd.read_csv(p)
    df = df.rename(columns=HARASSMENT_COLUMN_ALIASES)
    return HarassmentData(df=df)


def load_centroids(path: str | os.PathLike[str] | None = None) -> CentroidData:
    """Load 7-cluster HEXACO centroids (IEEE-published, FIXED parameters).

    Per v2.0 master Section 5.4 + clarifications log M3, these centroids
    are treated as fixed parameters; bootstrap CIs are conditional on
    them and do not propagate centroid-estimation uncertainty.

    Column name canonicalization (Level 1 clarification, smoke-test
    discovered 2026-04-30): the IEEE centroid CSV uses full domain
    names ``Honesty-Humility``, ``Emotionality``, etc. ``CENTROID_COLUMN_ALIASES``
    renames these to internal H/E/X/A/C/O at load time.

    Parameters
    ----------
    path : path-like, optional
        Override path. Defaults to ``CENTROIDS_PATH``.

    Returns
    -------
    CentroidData
        Wrapper exposing the raw frame and the K x 6 centroid matrix.
    """
    p = Path(path) if path is not None else CENTROIDS_PATH
    if not p.is_file():
        raise FileNotFoundError(
            f"Centroid file not found at {p}. "
            "Verify ../clustering/csv/clstr_kmeans_7c.csv exists."
        )
    df = pd.read_csv(p)
    df = df.rename(columns=CENTROID_COLUMN_ALIASES)
    expected_n = N_CLUSTERS
    if len(df) != expected_n:
        raise ValueError(
            f"Centroid file has {len(df)} rows; expected {expected_n}. "
            "Confirm the file is the 7-cluster k-means output from the "
            "Tokiwa clustering paper (IEEE-published)."
        )
    return CentroidData(df=df)


# ====================================================================
# HDF5 artifact persistence
# ====================================================================


def save_artifacts(
    path: str | os.PathLike[str],
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any] | None = None,
) -> None:
    """Persist NumPy arrays and metadata to HDF5 for reproducibility.

    Per v2.0 Section 8 (Reproducibility, D-NEW9), bootstrap resample
    states and stage outputs are stored in HDF5 for permanent, structured
    archival.

    Parameters
    ----------
    path : path-like
        Destination ``.h5`` file. Parent directories created if missing.
    arrays : dict[str, ndarray]
        Mapping of dataset names to NumPy arrays.
    metadata : dict, optional
        Scalar attributes (seed, version, stage, timestamp, etc.).
        Stored as root-level HDF5 attributes.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(p, "w") as f:
        for name, arr in arrays.items():
            f.create_dataset(name, data=arr, compression="gzip", compression_opts=4)
        if metadata is not None:
            for key, value in metadata.items():
                # HDF5 attributes accept scalars; coerce non-numerics to str
                if isinstance(value, (str, bytes, int, float, np.integer, np.floating)):
                    f.attrs[key] = value
                else:
                    f.attrs[key] = str(value)


def load_artifacts(path: str | os.PathLike[str]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load NumPy arrays and metadata from an HDF5 artifact file.

    Parameters
    ----------
    path : path-like
        ``.h5`` file produced by :func:`save_artifacts`.

    Returns
    -------
    arrays : dict[str, ndarray]
    metadata : dict[str, Any]
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Artifact not found: {p}")
    arrays: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {}
    with h5py.File(p, "r") as f:
        for name in f:
            arrays[name] = f[name][()]
        for key in f.attrs:
            metadata[key] = f.attrs[key]
    return arrays, metadata


def standard_metadata(stage: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Construct a standard metadata dict for HDF5 artifacts.

    Includes seed, stage label, and version markers required for
    reproducibility-audit traceability.
    """
    from . import __osf_doi__, __version__  # avoid circular at module load

    md: dict[str, Any] = {
        "seed": SEED,
        "stage": stage,
        "version": __version__,
        "osf_doi": __osf_doi__,
    }
    if extra:
        md.update(extra)
    return md
