# 02. Methods

All procedures were pre-registered at OSF prior to data analysis (DOI 10.17605/OSF.IO/3Y54U; v2.0 registered/locked 2026-04-30). Code, intermediate artifacts (HDF5), random seeds, and Docker containerization are publicly archived at <https://github.com/etoki/paper>. The analysis is fully reproducible via `make reproduce` (~30 minutes on a single core). All stochastic operations use the locked seed 20260429.

## 2.1 Data sources

### 2.1.1 Individual-level harassment data (N = 354)

We use the Tokiwa harassment preprint dataset (Tokiwa, 2025, Research Square preprint, DOI 10.21203/rs.3.rs-7756124/v1; IRB-approved), comprising N = 354 Japanese workplace participants who completed the following self-report measures:

- **HEXACO-60** (Wakabayashi, 2014; Japanese adaptation of Ashton & Lee, 2009): 60-item Likert 1-5 scale yielding six domain scores (H, E, X, A, C, O).
- **Dark Triad** scales (Machiavellianism, Narcissism, Psychopathy): used in the common-method-variance diagnostic (Section 2.5) but not in the primary analysis.
- **Workplace Power Harassment Scale** (Tou et al., 2017): continuous severity scale measuring past-three-year power-harassment victimization frequency in the participant's workplace.
- **Gender Harassment Scale** (Kobayashi & Tanaka, 2010): continuous severity scale; used as triangulation outcome (H6).
- **Demographics**: binary self-reported gender (0 = female, 1 = male), age (continuous), prefecture-level region (categorical).

### 2.1.2 Cluster centroids (N = 13,668; Tokiwa 2026 IEEE Access, M3-fixed)

Seven HEXACO cluster centroids were taken from the previously published Tokiwa (2026, *IEEE Access*; DOI 10.1109/ACCESS.2026.3651324) clustering of N = 13,668 Japanese respondents. Per pre-registration v2.0 Methods Clarification M3, these centroids are treated as *fixed parameters*; bootstrap confidence intervals are computed conditional on the centroids and do not propagate centroid-estimation uncertainty. This decision aligns with the inferential target of the present study (causal contrasts within a fixed taxonomy) rather than the joint estimation of the taxonomy itself.

### 2.1.3 Population reweighting (Statistics Bureau Labor Force Survey 2022)

Marginal gender-by-age proportions for the Japanese working population were obtained from the Labor Force Survey Basic Tabulation 2022 Annual Average, published by the Statistics Bureau, Ministry of Internal Affairs and Communications (Statistics Bureau, 2023; PDF retrieved from e-Stat on 2026-04-30 by the investigator and archived at `simulation/data/`). We use Table 3 (employed persons by age group) restricted to the employed-persons category, yielding a total of 67.23 million persons with age × gender breakdown {15–64, 65+} × {male, female}. The resulting marginals (F = 0.4498, M = 0.5502) replace the v2.0 Stage 1 placeholder (F = 0.5, M = 0.5).

### 2.1.4 MHLW power-harassment victimization targets

Past-three-year power-harassment victimization rates from three MHLW Workplace Harassment Surveys (MHLW, 2017, 2021, 2024) serve as external validation targets for H1:

- FY2016 (32.5%) — pre-law, primary target;
- FY2020 (31.4%) — transition period, secondary target;
- FY2023 (19.3%) — post-law, secondary target.

## 2.2 Pre-registered analytic pipeline

The pipeline comprises nine stages (0 through 8), each producing a versioned HDF5 artifact under `output/supplementary/`. Stages 0–2 implement the H1 confirmatory chain; Stages 3–6 implement sensitivity analyses, baseline comparisons, common-method-variance diagnostics, and target trial documentation; Stages 7–8 implement the H7 counterfactual contrasts and transportability sensitivity. Random number streams across stages are isolated via additive seed offsets to ensure independence (see `code/utils_io.py:make_rng`).

### 2.2.1 Stage 0: Type assignment, cell propensity, and 28-cell empirical Bayes

Each individual is assigned to the closest cluster (Euclidean nearest-centroid, primary; Methods Clarification M2 also computed soft-assignment with τ ∈ {0.5, 1.0, 2.0} × median nearest-neighbor distance for sensitivity). Outcomes are binarized at the participant-level mean + 0.5 SD (main; sensitivity at + 0.25 and + 1.0 SD).

Per-cell propensity p̂_c = X_c / N_c is computed for each of the 14 cells (7 clusters × 2 genders) for the main analysis, and 28 cells (× 2 roles, classified via the literature-based top 15% on the C + 0.5 × X composite) for the empirical Bayes sensitivity. Per-cell 95% confidence intervals follow the four-step priority cascade (Methods Clarification M4):

1. **Clopper-Pearson** exact binomial interval for degenerate cells (X_c ∈ {0, N_c});
2. **BCa** (Bias-Corrected accelerated; Efron, 1987) with jackknife acceleration parameter â (Methods Clarification m4) when |a| ≤ 10;
3. **BC** (bias-corrected only) when BCa fails or |a| > 10;
4. **Percentile** as a final fallback.

Empirical Bayes shrinkage on 28 cells uses the method-of-moments hyperprior (μ̂, σ̂², α̂, β̂) with three rejection triggers: (i) bootstrap-stabilized variance ratio σ̂² / [μ̂(1 − μ̂)] < 0.05 (Methods Clarification m1); (ii) max(α̂, β̂) > 100 (v2.0 master); (iii) (α̂ + β̂) / median(N_k) > 5 (Methods Clarification m2). Per-iteration (α̂, β̂) re-estimation in bootstrap follows Carlin and Louis (2000; Methods Clarification M1).

### 2.2.2 Stage 1: National prevalence aggregation

The national prevalence is computed as the weighted sum P̂_t = Σ_c (p̂_c × W_c) / Σ_c W_c, where W_c is the cluster × gender population weight. Cluster proportions remain at the Tokiwa (2026, *IEEE Access*) values (M3-fixed; m8 limitation: cluster membership is not in the MHLW data); only the gender marginal is updated to the 2022 working-population value.

### 2.2.3 Stage 2: H1 four-tier classification

Per the v2.0 master Section 5.4, the four-tier H1 hierarchy is:

- **Strict SUCCESS**: point MAPE_FY2016 ≤ 30% AND CI upper bound ≤ 30%
- **Standard SUCCESS**: point MAPE ≤ 30% (CI upper bound > 30% permitted)
- **PARTIAL SUCCESS**: 30% < point MAPE ≤ 60%
- **FAILURE**: point MAPE > 60%

Per Methods Clarification m3, the headline national MAPE bootstrap uses B = 10,000; per-cell bootstrap remains at B = 2,000.

### 2.2.4 Stage 3: Sensitivity sweep

A one-at-a-time (OAT) sensitivity sweep around the locked main configuration is computed over four parameter families with B = 2,000 per row:

1. Binarization threshold ∈ {mean + 0.25 SD, mean + 0.5 SD (main), mean + 1.0 SD};
2. Soft-assignment τ factor ∈ {0.5, 1.0, 2.0} × median NN distance (Methods Clarification M2);
3. Empirical Bayes shrinkage scale ∈ {0.5×, 1.0× (main), 2.0×} prior pseudocount;
4. Cluster-proportion mass shift ∈ {−0.10, 0.0 (main), +0.10} (proxy for V/f1/f2 uncertainty per v2.0 Section 6.4; full Cartesian sweep deferred to follow-up work due to combinatorial cost).

### 2.2.5 Stage 4: Baseline comparison and H2 ordinal trend

Five baselines (B0–B4) of increasing sophistication are compared via cell-stratified bootstrap MAPE_FY2016:

- **B0**: predicts MHLW grand mean (uniform);
- **B1**: gender-only logistic regression;
- **B2**: HEXACO 6-domain logistic regression;
- **B3**: 14-cell conditional propensity (the main pipeline);
- **B4**: extended with age and age × cluster interactions (industry breakdown deferred to follow-up; m8 limitation).

H2 is tested via the Bonferroni-Holm-corrected family of pairwise inequalities (B0–B1, B1–B2, B2–B3, B3–B4) with a one-sided α = 0.05; Page's L (1963) ordinal trend test is reported as auxiliary (Methods Clarification n4).

### 2.2.6 Stage 5: Common-method-variance diagnostic

CMV is assessed via Harman's single-factor test (Podsakoff et al., 2003) on 11 standardized variables (6 HEXACO domains + 3 Dark Triad + 2 harassment continuous scales) using PCA on standardized inputs after listwise deletion. The 50% first-factor variance threshold (Methods Clarifications m8) demarcates "limited concern" from "concern present." The Lindell and Whitney (2001) marker-variable correction uses HEXACO Openness as the theoretical marker, given its weak theoretical link to harassment outcomes relative to HH/A/E.

### 2.2.7 Stage 6: Target trial PICO documentation

Stage 6 produces a metadata-only artifact documenting the target trial (Hernán & Robins, 2016, 2020) PICO and four identifying assumptions: consistency, exchangeability, positivity, and SUTVA (strengthened with Hudgens & Halloran, 2008 peer-effect framework per Methods Clarification m6).

### 2.2.8 Stage 7: Counterfactual ΔP_A, ΔP_B, ΔP_C and H7 IUT

Counterfactual A applies a +0.3 SD shift to HEXACO H, A, and E for every individual, recomputes nearest-centroid cluster membership, recomputes per-cell propensity, re-aggregates via Stage 1 weights, and produces ΔP_A as the per-bootstrap difference between counterfactual and baseline national prevalence. Per Methods Clarification m5, positivity is quantitatively diagnosed via the cell-level ratio ρ_{c, x = A} = (counterfactual cell N) / (observed cell N); cells with ρ < 0.10 are flagged, and intervention is downgraded from confirmatory to exploratory if flagged_weight ≥ 20%.

Counterfactual B applies a +0.40 SD shift in Honesty-Humility (`code/stage7_counterfactual.py:DELTA_B_MAIN_SD`) **only to individuals whose baseline nearest-centroid cluster is already in the pre-registered low-prevalence target set {0, 4, 6}**, leaving all other individuals' HEXACO scores unchanged. Following the targeted shift, all individuals are re-classified by nearest-centroid on the post-intervention HEXACO matrix, per-cell propensities are re-computed from the realized binary outcomes within the new cell assignments, and ΔP_B is reported per-bootstrap as the difference between counterfactual and baseline national prevalence. This operationalization matches pre-registration v2.0 Section 5.7 ("HH := HH + δ_B × SD(HH) for individuals in Cluster ∈ {0, 4, 6}") and Methods Clarifications Log n3 (Pearl 2009 do-operator notation form, which formalizes the targeted indicator "if i ∈ Cluster {0, 4, 6}; HH_i otherwise"). Positivity is computed via ρ_{c, B} = (count of individuals reassigned into target Cluster c after the shift) / (count of individuals in target Cluster c at baseline) and is downgraded by the same m5 rule as Counterfactual A.

Counterfactual C multiplies cell-level propensity by 0.80 uniformly across all cells while leaving HEXACO scores unchanged. The 20% reduction is calibrated against three meta-analyses (Escartín, 2016; Hodgins et al., 2014; Salin, 2021) and the MHLW FY2016–FY2023 natural experiment (32.5% → 19.3%; −13.2pp ≈ −40.6% relative). Positivity for C is trivially preserved (ρ ≡ 1) per pre-registration v2.0 Section 5.7.

The H7 intersection-union test (Berger & Hsu, 1996; Methods Clarification m7) computes one-sided 95% lower bounds L_BA and L_BC for ΔP_B − ΔP_A and ΔP_B − ΔP_C, respectively, with bootstrap-resampled cell data (B = 2,000). The configuration is classified per the m7 priority cascade implemented in `code/stage7_counterfactual.py:h7_iut`:

- **REVERSAL** (priority 1): point ΔP_B < point ΔP_A OR point ΔP_B < point ΔP_C (B is dominated by A or C in point estimate, indicating that the pre-registered hypothesis ordering ΔP_B > {ΔP_A, ΔP_C} is reversed);
- **CONFIRMED**: L_BA > 0 AND L_BC > 0 (B dominates both A and C in CI lower bound);
- **PARTIAL**: exactly one of L_BA, L_BC > 0;
- **AMBIGUOUS**: neither lower bound > 0 (CIs allow zero or reversal).

### 2.2.9 Stage 8: Transportability factor sweep

Per v2.0 Section 5.8 (cultural attenuation), the bootstrap distribution of each ΔP_x is multiplied by a transportability factor F ∈ {0.3, 0.5, 0.7, 1.0}, anchored as:

- F = 0.3: Conservative cross-cultural attenuation (worst-case anchor; supported by Power et al. 2013 cross-continent variability findings);
- F = 0.5: Nielsen et al. (2017) Asia/Oceania attenuation expected (r = .16 vs European r = .33);
- F = 0.7: Mild attenuation (intermediate);
- F = 1.0: Reference (no attenuation).

H7 is re-classified per attenuated factor to verify robustness.

## 2.3 Reproducibility

All stochastic operations use seed 20260429 (Methods Clarification M3). HDF5 artifacts at each stage include the seed, version, OSF DOI, and stage-specific provenance metadata as root-level attributes, enabling full audit-trail reconstruction. Verification hashes for all artifacts are computed via `simulation/code/verify_reproduction.py` and compared against `reference_hashes.json`. The Docker container `Dockerfile` and the `Makefile` `reproduce` target encapsulate the full pipeline. A 56-test pytest suite (including the MHLW loader tests activated under Phase 1) guards against regression in core utilities.

## 2.4 Software and statistical environment

Analysis was conducted in Python 3.11.15 using NumPy 2.x, SciPy 1.x, pandas 2.x, scikit-learn 1.x, h5py 3.x, and statsmodels 0.14.x (full pinned versions in `pyproject.toml`). All code is publicly available under MIT license at <https://github.com/etoki/paper/tree/main/simulation> and archived at OSF (DOI 10.17605/OSF.IO/3Y54U). The pre-registration is locked at OSF (DOI 10.17605/OSF.IO/3Y54U) with the v1.1 → v2.0 public diff document and v2.0 Methods Clarifications Log (Section 6.5 Level 1 deviation; SHA-256 fcaaf0d...) hosted alongside the registration.

## 2.5 Data availability

This study's data and reproducibility artifacts are deposited on a tiered basis:

- **Public tier** (no access request required): Stage 0–8 supplementary HDF5 artifacts, Figures 1–6 (PNG/PDF/SVG), the canonical numerical record (`canonical_numbers.md`), and SHA-256 reference hashes (`reference_hashes.json`) are openly available at the OSF v2.0 working project (osf.io/3hxz6) under `v2.0/v2.0_supplementary.tar.gz`, and reproducible from source via `make reproduce` on the GitHub repository.
- **Restricted tier** (IRB-approved access request): The N = 354 individual-level harassment dataset is hosted in a Private OSF component (`v2.0 IRB-Restricted Individual Data`) linked from the v2.0 working project. The parent project's public Wiki at https://osf.io/3hxz6/wiki/home/ documents the four data-use terms (research-purpose-bounded use; non-redistribution; required citation of Tokiwa 2025 Research Square preprint and the v2.0 OSF DOI; IRB acknowledgement) and the request procedure (click "Request access" on the Component page; provide institutional affiliation, IRB approval ID, and intended research use). Approved requesters are added as read-only contributors within 14 days. Methodological details of the underlying survey are reported in Tokiwa (2025, Research Square preprint; DOI 10.21203/rs.3.rs-7756124/v1).
- **External public sources**: The Statistics Bureau (MIC) Labour Force Survey 2022 Annual Average tabulations used for population reweighting are publicly archived in the repository at `simulation/data/mhlw_labor_force_2022.csv` (provenance: e-Stat retrieval 2026-04-30).
- **Pre-registration**: v2.0 master document, v1.1 → v2.0 public diff, and the Section 6.5 Level 1 Methods Clarifications Log are publicly accessible at the v2.0 registration page (osf.io/3y54u, DOI 10.17605/OSF.IO/3Y54U).
