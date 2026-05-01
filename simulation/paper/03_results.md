# 03. Results

All results are reported using the locked seed 20260429. Numerical values are reproduced verbatim from the HDF5 artifacts under `output/supplementary/`. The full numerical record (~ 600 KB of HDF5 artifacts across 9 pipeline stages, plus ~ 2.2 MB of figures in PNG/PDF/SVG, totaling ~ 2.8 MB) is publicly archived at OSF DOI 10.17605/OSF.IO/3Y54U; SHA-256 hashes for byte-identical reproduction verification are committed in `simulation/output/reference_hashes.json`.

## 3.1 Sample characteristics and 14-cell propensity (Stage 0)

The N = 354 sample was assigned to 14 cells (7 HEXACO clusters × 2 genders) via Euclidean nearest-centroid hard-assignment, using the Tokiwa (2026, *IEEE Access*) centroids as fixed parameters. Cell sizes ranged from 10 to 70 individuals (median = 18), with no degenerate (N = 0) cells. Cell-level past-three-year power-harassment victimization rates ranged from 0.000 to 0.400 (mean across cells = 0.181).

Per-cell 95% confidence intervals followed the four-step priority cascade (Methods §2.2.1). Of the 14 cells, **13 were resolved at the BCa stage** and **1 cell (cluster × gender index 11; N = 12, X = 0) was resolved at the Clopper-Pearson stage** as the M4 cascade prescribes for degenerate cells (X_c ∈ {0, N_c}). No cells required the BC or Percentile fallbacks, confirming that the v2.0 master Section 5.1 priority cascade operates as designed for the realized data structure.

## 3.2 National prevalence and H1 four-tier classification (Stages 1–2)

After re-weighting with Statistics Bureau (MIC) Labor Force Survey 2022 marginals (F = 0.4498, M = 0.5502; total = 67.23 million employed persons), the model-based national past-three-year power-harassment prevalence was P̂ = 0.1744 (point estimate), corresponding to 17.44%.

Per the v2.0 master Section 5.4 four-tier judgment hierarchy, the H1 results across three MHLW validation periods are summarized in Table 1.

**Table 1.** Mean Absolute Percentage Error (MAPE) of model-based national prevalence against MHLW survey targets, with bootstrap 95% confidence intervals (B = 10,000 cell-stratified). Primary outcome: power harassment.

| Period (MHLW) | Observed | Predicted | MAPE | 95% CI | Tier |
|---|---:|---:|---:|---|---|
| **FY2016** (32.5%, primary) | 0.325 | 0.1744 | 46.34% | [32.36, 59.30] | **PARTIAL SUCCESS** |
| FY2020 (31.4%, secondary) | 0.314 | 0.1744 | 44.46% | [30.00, 57.88] | PARTIAL SUCCESS |
| **FY2023** (19.3%, secondary) | 0.193 | 0.1744 | **9.65%** | [0.52, 31.52] | **Standard SUCCESS** |

The FY2016 primary classification is **PARTIAL SUCCESS** (point MAPE > 30%, ≤ 60%). However, the post-Power-Harassment-Prevention-Law FY2023 result attains **Standard SUCCESS** with a point MAPE of 9.65% (well below the 30% threshold for both Strict and Standard tiers); the 95% CI upper bound of 31.52% narrowly exceeds the Strict criterion (≤ 30%) by 1.52 pp, placing the classification at Standard rather than Strict. Interpretation of this temporal heterogeneity is deferred to the Discussion (Section 4.1).

## 3.3 Sensitivity sweep (Stage 3)

The pre-registered one-at-a-time sweep over four parameter families produced 13 rows. Results are presented in Table 2.

**Table 2.** OAT sensitivity sweep around main configuration. B = 2,000 per row, percentile CI on FY2016 MAPE. The baseline row uses the locked v2.0 main configuration; all other rows hold remaining parameters at main while varying the indicated family.

| Family | Value | MAPE | 95% CI | Tier |
|---|---|---:|---|---|
| Baseline | main_v2.0 | 45.51% | [31.5, 59.5] | PARTIAL |
| Binarization | mean+0.25 SD | **29.58%** | [13.7, 45.2] | **Standard SUCCESS** |
| Binarization | mean+0.5 SD (main) | 45.51% | [30.5, 59.2] | PARTIAL |
| Binarization | mean+1.0 SD | 48.95% | [34.0, 62.4] | PARTIAL |
| Soft τ × median NN | 0.5× | 45.04% | [43.8, 46.3] | PARTIAL |
| Soft τ × median NN | 1.0× | 45.22% | [43.9, 46.5] | PARTIAL |
| Soft τ × median NN | 2.0× | 45.24% | [43.9, 46.4] | PARTIAL |
| EB scale | 0.5× pseudocount | 45.90% | [32.1, 58.7] | PARTIAL |
| EB scale | 1.0× (main) | 45.51% | [31.4, 59.0] | PARTIAL |
| EB scale | 2.0× pseudocount | 45.80% | [32.3, 58.9] | PARTIAL |
| Cluster mass shift δ | −0.10 high−low | 56.47% | [42.8, 68.8] | PARTIAL |
| Cluster mass shift δ | 0.0 (main) | 45.51% | [31.4, 59.2] | PARTIAL |
| Cluster mass shift δ | +0.10 high−low | 34.55% | [16.6, 51.8] | PARTIAL |

Two parameter families show substantial sensitivity:

- **Binarization threshold** (range: 29.58% to 48.95%): a more lenient threshold (+0.25 SD) yields Standard SUCCESS classification, suggesting that the MHLW survey may operationalize harassment more inclusively than the +0.5 SD main threshold;
- **Cluster proportion mass shift** (range: 34.55% to 56.47%): underscores the centrality of the Tokiwa (2026, *IEEE Access*) cluster proportions to the conclusion (m8 limitation per pre-registration v2.0).

Two parameter families show negligible sensitivity:

- **Soft-assignment τ** (range: 45.04% to 45.24%; 0.20 percentage-point spread): cluster-membership-fuzziness is essentially irrelevant to the conclusion;
- **Empirical Bayes shrinkage scale** (range: 45.51% to 45.90%; 0.40 percentage-point spread): the empirical Bayes prior-pseudocount magnitude is negligible.

The two stable axes provide affirmative robustness evidence; the two sensitive axes circumscribe the assumptions on which the conclusion explicitly depends.

## 3.4 Baseline comparison and H2 ordinal trend (Stage 4)

The pre-registered B0–B4 baseline ladder yielded the following MAPE_FY2016 estimates (Table 3).

**Table 3.** Baseline comparison for H2. B = 2,000 cell-stratified bootstrap. Note that B0 (uniform-MHLW-grand-mean prediction) trivially attains MAPE = 0 by construction and is included only as the lower-bound reference.

| Baseline | Description | MAPE | 95% CI |
|---|---|---:|---|
| B0 | Uniform = MHLW grand mean | 0.00% | [0.00, 0.00] |
| B1 | Gender logistic | 46.02% | [46.02, 46.02] |
| B2 | HEXACO 6-domain logistic | 43.98% | [40.33, 47.35] |
| B3 | 14-cell conditional | 46.34% | [46.34, 46.34] |
| B4 | Extended (age + age × cluster) | 47.04% | [45.02, 49.05] |

The H2 hypothesis (B0 > B1 > B2 > B3 > B4 in MAPE, monotonic improvement) was tested via Bonferroni-Holm-corrected pairwise inequalities; **0 of 4 pairs were confirmed** at one-sided α = 0.05. Page's L (1963) ordinal trend test produced p = 0.9757, far from significance.

The H2 decision is therefore **ambiguous_or_reversed**: the model sophistication ladder does not exhibit a monotonic improvement signature. Substantively, this indicates that the 14-cell conditional model (B3) does not dominate simpler alternatives (B1, B2) in terms of predictive accuracy. We interpret this as evidence that the 14-cell partition is *not over-fit* to the training subsample (a moderately favorable finding), but the benefit of the additional cluster × gender stratification is not detectable at this sample size with the present binarization.

## 3.5 Common-method-variance diagnostic (Stage 5)

Harman's single-factor test on 11 standardized variables (6 HEXACO + 3 Dark Triad + 2 harassment continuous scales; N = 353 used after listwise deletion) yielded:

- **First-factor variance**: 24.08% (well below the 50% concern threshold);
- **CMV concern flag**: False;
- **Marker-variable correction** (Lindell & Whitney, 2001) using HEXACO Openness as theoretical marker:
  - r(O, power harassment) = +0.068 (CMV estimate);
  - r(O, gender harassment) = −0.181;
  - Honesty-Humility–power harassment association: r_raw = −0.265 → r_adjusted = −0.356 (slightly strengthened after CMV correction);
  - Psychopathy–power harassment association: r_raw = +0.391 → r_adjusted = +0.347.

Common-method bias does not pose substantial concern for the present data, and the post-correction associations remain in the expected directions.

## 3.6 Counterfactual contrasts and H7 IUT (Stage 7)

The three pre-registered counterfactual interventions produced the following ΔP_x estimates (Table 4).

**Table 4.** Counterfactual ΔP_x = (P̂_counterfactual − P̂_baseline) at the national level. B = 2,000 cell-stratified bootstrap. Positive ΔP indicates *reduction* in prevalence (per pre-registration v2.0 sign convention).

| Intervention | Operationalization | ΔP point | 95% CI | Decision (CI excludes 0?) |
|---|---|---:|---|---|
| **Counterfactual A** | do(HH ← HH + 0.3σ; A ← A + 0.3σ; E ← E + 0.3σ) | −0.0061 | [−0.0207, +0.0062] | **No (null)** |
| **Counterfactual B** | do(HH := HH + 0.40σ_HH) for individuals whose baseline cluster ∈ {0, 4, 6} | −0.0059 | [−0.0207, +0.0066] | **No (null)** |
| **Counterfactual C** | do(P_{c, x} ← 0.80 × P_{c, x}) | **+0.0349** | [+0.0264, +0.0435] | **Yes (positive)** |

The intersection-union test (Berger & Hsu, 1996) produced one-sided 95% lower bounds L_BA = −0.0011 and L_BC = −0.0544. Because point ΔP_B = −0.0059 is less than point ΔP_C = +0.0349, the m7 priority-1 condition is satisfied and H7 is classified as **REVERSAL** (consistent with L_BC < 0 in the CI-based cascade; L_BA also lies just below zero, ruling out the CONFIRMED branch).

Substantively, this means that the cell-level structural intervention (Counterfactual C) outperforms both the universal personality intervention (Counterfactual A) and the targeted personality intervention on the low-prevalence subgroup (Counterfactual B) by a large margin. Numerically, |ΔP_C| / |ΔP_A| ≈ 5.7.

**Positivity diagnostic.** Counterfactual A and Counterfactual C trivially preserve positivity (ρ ≡ 1 by construction; per Methods Clarification m5). Counterfactual B's cell positivity ratio ρ_{c, B} was 44.5% flagged (i.e., 44.5% of population weight in cells where the post-intervention expected count is < 10% of observed), triggering the m5 downgrade rule (flagged_weight ≥ 20%). We therefore interpret Counterfactual B as **exploratory** rather than confirmatory; the IUT reading is unaffected (ΔP_B remains null), but conclusions about B's null finding cannot be made with the same confidence as A.

## 3.7 Transportability sensitivity (Stage 8)

Per the v2.0 master Section 5.8 cultural-attenuation sweep, all three ΔP_x bootstrap distributions were multiplied by transportability factor F ∈ {0.3, 0.5, 0.7, 1.0}, and the H7 IUT was re-classified per factor (Table 5).

**Table 5.** H7 classification under cultural attenuation factors. Each row uses the same H7 IUT procedure on attenuated bootstrap distributions.

| Factor F | Anchor | ΔP_A (attenuated) | ΔP_C (attenuated) | H7 |
|---|---|---:|---:|---|
| **0.3** | Conservative cross-cultural worst case | −0.0018 | +0.0105 | **REVERSAL** |
| **0.5** | Nielsen, Glasø, & Einarsen 2017 expected | −0.0031 | +0.0174 | **REVERSAL** |
| **0.7** | Mild attenuation | −0.0043 | +0.0244 | **REVERSAL** |
| **1.0** | Reference (no attenuation) | −0.0061 | +0.0349 | **REVERSAL** |

The H7 = REVERSAL classification is robust across all four pre-registered cultural attenuation factors. Even under the conservative cross-cultural worst-case scenario (F = 0.3), the structural intervention's effect remains substantially larger than the personality intervention's, preserving the qualitative ordering.

## 3.8 Summary of pre-registered hypothesis outcomes

**Table 6.** Pre-registered hypothesis outcomes summary.

| Hypothesis | Pre-registered prediction | Outcome | Tier |
|---|---|---|---|
| H1 | MAPE_FY2016 ≤ 30% (Strict / Standard SUCCESS) | MAPE = 46.34% | **PARTIAL SUCCESS** |
| H2 | Monotonic B0 > B1 > B2 > B3 > B4 | 0 of 4 pairs confirmed (Page's L p = 0.976) | **REJECTED** |
| H3 | Centroid concordance N=354 vs N=13,668 | (Reported in Tokiwa, 2026 *IEEE Access*) | Confirmed |
| H4 | Per-cell propensity stability across bootstrap procedures | 13/14 cells resolved at BCa; 1 degenerate cell (X=0) at Clopper-Pearson per M4 cascade | **CONFIRMED** |
| H5 | Gender invariance | High concordance of cluster-level power-harassment rankings across genders (Spearman ρ = 0.87, p = 0.01 at 7-cluster level); 14-cell vs 28-cell sensitivity stable | **CONFIRMED** |
| H6 | Power × gender harassment cluster ranking concordance | Weak concordance (Spearman ρ = 0.04 at 7-cluster level, p = 0.94; ρ = 0.14 at 14-cell level, p = 0.62) | **REJECTED** |
| H7 | IUT classification | **REVERSAL** (robust across F = 0.3–1.0) | **REVERSAL** (per pre-registered classification) |

The headline empirical finding is the H7 = REVERSAL classification with cross-cultural robustness, which we develop substantively in the Discussion.
