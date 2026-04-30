"""Pytest fixtures for HEXACO Harassment Microsim tests."""

from __future__ import annotations

import sys
from pathlib import Path

# Add repo simulation/ root to sys.path so `from code.utils_io import ...`
# works during pytest runs.
SIMULATION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIMULATION_ROOT))

import numpy as np
import pytest


@pytest.fixture
def rng():
    """Deterministic RNG using the v2.0 locked seed."""
    return np.random.default_rng(20260429)


@pytest.fixture
def small_binary_cell():
    """A small (N=10) cell with mixed outcomes for CI tests."""
    return {"successes": 3, "n": 10}


@pytest.fixture
def degenerate_zero_cell():
    """X=0 cell that triggers M4 Clopper-Pearson fallback."""
    return {"successes": 0, "n": 10}


@pytest.fixture
def degenerate_full_cell():
    """X=N cell that triggers M4 Clopper-Pearson fallback."""
    return {"successes": 10, "n": 10}


@pytest.fixture
def synthetic_14_cell_propensities():
    """Synthetic 14-cell propensity vector for MoM tests.

    Mean ~0.30, modest variance — should NOT trigger m1 collapse.
    """
    rng = np.random.default_rng(20260429)
    return rng.uniform(0.15, 0.45, size=14)


@pytest.fixture
def synthetic_28_cell_sizes():
    """Synthetic 28-cell N_k array (mix of small + large cells)."""
    rng = np.random.default_rng(20260429)
    return rng.integers(low=1, high=70, size=28).astype(float)
