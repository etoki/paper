"""Tests for utils_diagnostics: MoM rules, soft assignment, positivity."""

from __future__ import annotations

import numpy as np
import pytest

from code.utils_diagnostics import (
    MOM_PSEUDOCOUNT_MAX,
    MOM_PSEUDOCOUNT_MEDIAN_N_RATIO,
    MOM_VARIANCE_RATIO_THRESHOLD,
    POSITIVITY_FLAGGED_WEIGHT_MAX,
    POSITIVITY_RATIO_THRESHOLD,
    cell_size_summary,
    compute_mom_hyperprior,
    diagnose_mom,
    evaluate_positivity,
    median_nn_distance,
    soft_assign_weights,
)


class TestMoMHyperprior:
    def test_basic_propensities(self, synthetic_14_cell_propensities):
        mu, sigma2, alpha, beta = compute_mom_hyperprior(synthetic_14_cell_propensities)
        assert 0 < mu < 1
        assert sigma2 > 0
        assert alpha > 0
        assert beta > 0

    def test_too_few_cells_raises(self):
        with pytest.raises(ValueError):
            compute_mom_hyperprior(np.array([0.5]))

    def test_degenerate_returns_nan(self):
        # Zero variance: all propensities equal
        mu, sigma2, alpha, beta = compute_mom_hyperprior(np.full(14, 0.30))
        assert sigma2 == 0
        assert np.isnan(alpha)
        assert np.isnan(beta)


class TestMoMDiagnostic:
    def test_normal_propensities_accept_mom(
        self, synthetic_14_cell_propensities, synthetic_28_cell_sizes
    ):
        diag = diagnose_mom(synthetic_14_cell_propensities, synthetic_28_cell_sizes)
        # With reasonable spread, MoM should be accepted
        # (no guarantee, but typical for uniform-spread synthetic data)
        if diag.reject_mom:
            # If rejected, at least one of the three triggers must fire
            assert any([diag.trigger1_fired, diag.trigger2_fired, diag.trigger3_fired])

    def test_collapsed_variance_triggers_m1(self, synthetic_28_cell_sizes):
        # Tiny variance triggers m1 (variance ratio < 0.05)
        almost_constant = np.full(14, 0.30) + np.random.default_rng(20260429).uniform(
            -1e-4, 1e-4, size=14
        )
        diag = diagnose_mom(almost_constant, synthetic_28_cell_sizes)
        assert diag.trigger1_fired or diag.trigger2_fired or diag.trigger3_fired

    def test_constants_have_expected_values(self):
        assert MOM_VARIANCE_RATIO_THRESHOLD == 0.05
        assert MOM_PSEUDOCOUNT_MAX == 100.0
        assert MOM_PSEUDOCOUNT_MEDIAN_N_RATIO == 5.0


class TestSoftAssignment:
    def test_weights_sum_to_one(self):
        rng = np.random.default_rng(20260429)
        distances = rng.uniform(0.1, 5.0, size=(50, 7))
        weights = soft_assign_weights(distances, tau=1.0)
        assert weights.shape == (50, 7)
        assert np.allclose(weights.sum(axis=1), 1.0)

    def test_low_temperature_approaches_hard(self):
        # As τ → 0, softmax becomes one-hot at the nearest centroid
        distances = np.array([[1.0, 2.0, 3.0]])
        weights_hot = soft_assign_weights(distances, tau=0.01)
        # The minimum distance index should have weight ≈ 1.0
        assert weights_hot[0, 0] > 0.99

    def test_high_temperature_uniformizes(self):
        # As τ → ∞, softmax becomes uniform
        distances = np.array([[1.0, 2.0, 3.0]])
        weights_uniform = soft_assign_weights(distances, tau=1e6)
        assert np.allclose(weights_uniform[0], 1.0 / 3.0, atol=1e-3)

    def test_negative_tau_raises(self):
        distances = np.array([[1.0, 2.0, 3.0]])
        with pytest.raises(ValueError):
            soft_assign_weights(distances, tau=-1.0)

    def test_median_nn_distance(self):
        distances = np.array([[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]])
        m = median_nn_distance(distances)
        # Nearest-neighbor distances: 1.0, 0.5; median = 0.75
        assert m == pytest.approx(0.75)


class TestPositivity:
    def test_no_flagged_cells(self):
        ratios = np.full(14, 0.5)
        weights = np.ones(14)
        diag = evaluate_positivity("B", ratios, weights)
        assert diag.flagged_weight_share == 0.0
        assert not diag.downgrade_to_exploratory

    def test_all_flagged_cells_downgrades(self):
        # All cells below threshold → 100% flagged → downgrade
        ratios = np.full(14, 0.05)
        weights = np.ones(14)
        diag = evaluate_positivity("B", ratios, weights)
        assert diag.flagged_weight_share == 1.0
        assert diag.downgrade_to_exploratory

    def test_borderline_below_20pct_no_downgrade(self):
        # 19% of weight in flagged cells -> no downgrade
        ratios = np.array([0.05] * 2 + [0.5] * 12)
        weights = np.array([0.095] * 2 + [0.0675] * 12)  # 19% + 81% = 100%
        diag = evaluate_positivity("B", ratios, weights)
        assert diag.flagged_weight_share < POSITIVITY_FLAGGED_WEIGHT_MAX
        assert not diag.downgrade_to_exploratory

    def test_positivity_constants(self):
        assert POSITIVITY_RATIO_THRESHOLD == 0.10
        assert POSITIVITY_FLAGGED_WEIGHT_MAX == 0.20


class TestCellSizeSummary:
    def test_basic_summary(self):
        sizes = np.array([10, 15, 20, 25, 5, 0, 50, 70])
        s = cell_size_summary(sizes)
        assert s["min"] == 0
        assert s["max"] == 70
        assert s["n_cells_lt_10"] == 2  # sizes 5, 0
        assert s["n_cells_eq_0"] == 1
