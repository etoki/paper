"""Tests for utils_io: seed, paths, HDF5 persistence."""

from __future__ import annotations

import numpy as np
import pytest

import pandas as pd

from code.utils_io import (
    HEXACO_DOMAINS,
    MHLW_VALIDATION_TARGETS,
    N_CELLS_MAIN,
    N_CELLS_SENSITIVITY,
    N_CLUSTERS,
    SEED,
    load_artifacts,
    load_mhlw_weights,
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


class TestMHLWLoader:
    def _write_mock(self, path, rows):
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)

    def test_canonical_numeric_gender(self, tmp_path):
        path = tmp_path / "mhlw.csv"
        self._write_mock(
            path,
            [
                {"age_group": "20-24", "gender": 0, "count": 100},
                {"age_group": "20-24", "gender": 1, "count": 150},
                {"age_group": "25-29", "gender": 0, "count": 200},
                {"age_group": "25-29", "gender": 1, "count": 250},
            ],
        )
        m = load_mhlw_weights(path)
        gp = m.gender_proportions
        assert np.isclose(gp.sum(), 1.0)
        # Female total = 300, male = 400, total = 700
        assert np.isclose(gp[0], 300 / 700)
        assert np.isclose(gp[1], 400 / 700)
        assert m.total_population == 700
        assert m.n_records == 4

    def test_japanese_gender_labels(self, tmp_path):
        path = tmp_path / "mhlw_jp.csv"
        self._write_mock(
            path,
            [
                {"age_group": "20-24", "gender": "女", "count": 50},
                {"age_group": "20-24", "gender": "男", "count": 50},
            ],
        )
        m = load_mhlw_weights(path)
        gp = m.gender_proportions
        assert np.allclose(gp, [0.5, 0.5])

    def test_western_gender_labels(self, tmp_path):
        path = tmp_path / "mhlw_en.csv"
        self._write_mock(
            path,
            [
                {"age_group": "20-24", "gender": "F", "count": 30},
                {"age_group": "20-24", "gender": "M", "count": 70},
            ],
        )
        m = load_mhlw_weights(path)
        gp = m.gender_proportions
        assert np.isclose(gp[0], 0.3)
        assert np.isclose(gp[1], 0.7)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_mhlw_weights(tmp_path / "nonexistent.csv")

    def test_missing_required_column_raises(self, tmp_path):
        path = tmp_path / "bad.csv"
        self._write_mock(
            path,
            [{"age_group": "20-24", "count": 100}],  # no gender col
        )
        with pytest.raises(ValueError, match="missing required columns"):
            load_mhlw_weights(path)

    def test_unrecognized_gender_raises(self, tmp_path):
        path = tmp_path / "bad_gender.csv"
        self._write_mock(
            path,
            [{"age_group": "20-24", "gender": "other", "count": 100}],
        )
        with pytest.raises(ValueError, match="Unrecognized MHLW gender"):
            load_mhlw_weights(path)

    def test_negative_count_raises(self, tmp_path):
        path = tmp_path / "neg.csv"
        self._write_mock(
            path,
            [
                {"age_group": "20-24", "gender": 0, "count": -10},
                {"age_group": "20-24", "gender": 1, "count": 100},
            ],
        )
        with pytest.raises(ValueError, match="negative counts"):
            load_mhlw_weights(path)

    def test_only_one_gender_raises(self, tmp_path):
        path = tmp_path / "one_gender.csv"
        self._write_mock(
            path,
            [{"age_group": "20-24", "gender": 0, "count": 100}],
        )
        m = load_mhlw_weights(path)
        with pytest.raises(ValueError, match="missing gender categories"):
            _ = m.gender_proportions

    def test_optional_employment_column(self, tmp_path):
        path = tmp_path / "with_emp.csv"
        self._write_mock(
            path,
            [
                {"age_group": "20-24", "gender": 0, "count": 50, "employment": "regular"},
                {"age_group": "20-24", "gender": 1, "count": 50, "employment": "regular"},
            ],
        )
        m = load_mhlw_weights(path)
        assert "employment" in m.df.columns
        assert (m.df["employment"] == "regular").all()
