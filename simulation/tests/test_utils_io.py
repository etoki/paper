"""Tests for utils_io: seed, paths, HDF5 persistence."""

from __future__ import annotations

import numpy as np
import pytest

from code.utils_io import (
    HEXACO_DOMAINS,
    MHLW_VALIDATION_TARGETS,
    N_CELLS_MAIN,
    N_CELLS_SENSITIVITY,
    N_CLUSTERS,
    SEED,
    load_artifacts,
    make_rng,
    save_artifacts,
    standard_metadata,
)


class TestConstants:
    def test_seed_is_locked(self):
        assert SEED == 20260429

    def test_n_clusters(self):
        assert N_CLUSTERS == 7

    def test_cell_counts(self):
        assert N_CELLS_MAIN == 14
        assert N_CELLS_SENSITIVITY == 28

    def test_hexaco_domains(self):
        assert HEXACO_DOMAINS == ["H", "E", "X", "A", "C", "O"]

    def test_validation_targets_present(self):
        assert "FY2016" in MHLW_VALIDATION_TARGETS
        assert "FY2020" in MHLW_VALIDATION_TARGETS
        assert "FY2023" in MHLW_VALIDATION_TARGETS
        assert MHLW_VALIDATION_TARGETS["FY2016"]["value"] == 0.325
        assert MHLW_VALIDATION_TARGETS["FY2016"]["role"] == "primary"


class TestRNG:
    def test_make_rng_deterministic(self):
        a = make_rng().standard_normal(5)
        b = make_rng().standard_normal(5)
        assert np.allclose(a, b), "make_rng() must produce identical streams"

    def test_make_rng_offset_isolates(self):
        a = make_rng().standard_normal(5)
        b = make_rng(extra_offset=1).standard_normal(5)
        assert not np.allclose(a, b), "extra_offset must produce a different stream"


class TestArtifactPersistence:
    def test_roundtrip(self, tmp_path):
        path = tmp_path / "test.h5"
        arrays = {
            "x": np.array([1, 2, 3]),
            "y": np.array([[1.0, 2.0], [3.0, 4.0]]),
        }
        meta = {"stage": "test_io", "seed": 42}
        save_artifacts(path, arrays, meta)
        out_arrays, out_meta = load_artifacts(path)
        assert np.array_equal(arrays["x"], out_arrays["x"])
        assert np.allclose(arrays["y"], out_arrays["y"])
        assert int(out_meta["seed"]) == 42

    def test_load_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_artifacts(tmp_path / "nonexistent.h5")

    def test_standard_metadata_includes_seed(self):
        md = standard_metadata("stage_test")
        assert md["seed"] == SEED
        assert md["stage"] == "stage_test"
        assert "osf_doi" in md
