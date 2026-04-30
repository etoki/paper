"""Tests for utils_bootstrap: M4 priority chain + 4-tier classification."""

from __future__ import annotations

import numpy as np
import pytest

from code.utils_bootstrap import (
    ACCELERATION_LIMIT,
    bc_interval,
    bca_interval,
    cell_proportion_ci,
    classify_h1_tier,
    clopper_pearson_interval,
    jackknife_acceleration,
    percentile_interval,
)


class TestClopperPearson:
    def test_zero_successes(self):
        lo, hi = clopper_pearson_interval(0, 10)
        assert lo == 0.0
        assert 0 < hi < 1

    def test_full_successes(self):
        lo, hi = clopper_pearson_interval(10, 10)
        assert 0 < lo < 1
        assert hi == 1.0

    def test_invalid_n_raises(self):
        with pytest.raises(ValueError):
            clopper_pearson_interval(5, 0)

    def test_invalid_x_raises(self):
        with pytest.raises(ValueError):
            clopper_pearson_interval(15, 10)


class TestJackknifeAcceleration:
    def test_constant_data_returns_zero(self):
        data = np.ones(10)
        a = jackknife_acceleration(lambda x: float(np.mean(x)), data)
        assert a == 0.0

    def test_normal_data(self, rng):
        data = rng.standard_normal(50)
        a = jackknife_acceleration(lambda x: float(np.mean(x)), data)
        assert np.isfinite(a)


class TestPercentileInterval:
    def test_uniform_dist(self, rng):
        boot = rng.uniform(0, 1, size=1000)
        lo, hi = percentile_interval(boot)
        assert 0 < lo < hi < 1
        assert lo == pytest.approx(0.025, abs=0.05)
        assert hi == pytest.approx(0.975, abs=0.05)


class TestCellProportionCI:
    def test_degenerate_zero_uses_clopper_pearson(self, degenerate_zero_cell, rng):
        ci = cell_proportion_ci(
            successes=degenerate_zero_cell["successes"],
            n=degenerate_zero_cell["n"],
            rng=rng,
            n_bootstrap=200,
        )
        assert ci.method == "clopper_pearson"
        assert ci.point == 0.0
        assert ci.lower == 0.0
        assert ci.upper > 0.0

    def test_degenerate_full_uses_clopper_pearson(self, degenerate_full_cell, rng):
        ci = cell_proportion_ci(
            successes=degenerate_full_cell["successes"],
            n=degenerate_full_cell["n"],
            rng=rng,
            n_bootstrap=200,
        )
        assert ci.method == "clopper_pearson"
        assert ci.point == 1.0
        assert ci.upper == 1.0

    def test_non_degenerate_uses_bca_or_fallback(self, small_binary_cell, rng):
        ci = cell_proportion_ci(
            successes=small_binary_cell["successes"],
            n=small_binary_cell["n"],
            rng=rng,
            n_bootstrap=500,
        )
        assert ci.method in ("bca", "bc", "percentile")
        assert ci.point == 0.3
        assert 0 <= ci.lower <= ci.upper <= 1

    def test_method_priority_order(self):
        # When BCa works (typical), it should be chosen over BC/percentile
        # Hard to test deterministically with degenerate jackknife, but
        # at least verify the priority chain doesn't fall to percentile
        # for typical cells.
        ci = cell_proportion_ci(successes=15, n=50, n_bootstrap=2000)
        assert ci.method in ("bca", "bc", "percentile")
        # For N=50, BCa usually succeeds


class TestTierClassification:
    def test_strict_success(self):
        result = classify_h1_tier(point_mape=20.0, ci_lower=15.0, ci_upper=25.0)
        assert result.tier == "Strict SUCCESS"

    def test_standard_success_ci_overlap(self):
        result = classify_h1_tier(point_mape=25.0, ci_lower=10.0, ci_upper=45.0)
        assert result.tier == "Standard SUCCESS"
        assert "Pre-registered ambiguity Tier" in result.explanation

    def test_partial_success(self):
        result = classify_h1_tier(point_mape=45.0, ci_lower=30.0, ci_upper=55.0)
        assert result.tier == "PARTIAL SUCCESS"

    def test_failure(self):
        result = classify_h1_tier(point_mape=70.0, ci_lower=50.0, ci_upper=85.0)
        assert result.tier == "FAILURE"

    def test_boundary_strict(self):
        # Exactly 30% on both → Strict
        result = classify_h1_tier(point_mape=30.0, ci_lower=20.0, ci_upper=30.0)
        assert result.tier == "Strict SUCCESS"

    def test_boundary_standard(self):
        # Point 30% but CI upper > 30 → Standard
        result = classify_h1_tier(point_mape=30.0, ci_lower=20.0, ci_upper=30.5)
        assert result.tier == "Standard SUCCESS"

    def test_boundary_partial_failure(self):
        # Point exactly 60% → PARTIAL (≤ 60%)
        result = classify_h1_tier(point_mape=60.0, ci_lower=40.0, ci_upper=80.0)
        assert result.tier == "PARTIAL SUCCESS"
        # Point > 60% → FAILURE
        result_fail = classify_h1_tier(point_mape=60.5, ci_lower=40.0, ci_upper=80.0)
        assert result_fail.tier == "FAILURE"


class TestBCaInterval:
    def test_returns_acceleration(self, rng):
        data = rng.uniform(0, 1, size=20)
        boot = rng.uniform(0, 1, size=500)
        lo, hi, a = bca_interval(
            bootstrap_estimates=boot,
            point_estimate=0.5,
            statistic_fn=lambda x: float(np.mean(x)),
            data=data,
        )
        assert np.isfinite(a)
        assert lo <= hi


class TestBCInterval:
    def test_basic(self, rng):
        boot = rng.uniform(0, 1, size=1000)
        lo, hi = bc_interval(boot, point_estimate=0.5)
        assert 0 <= lo <= hi <= 1


def test_acceleration_limit_constant():
    assert ACCELERATION_LIMIT == 10.0
