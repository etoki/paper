# 02. Methods

All procedures were pre-registered prior to data analysis (v2.0 registered and locked 2026-04-30; deposit details and access conditions are given in the Data availability subsection below). The pre-registration consists of a master document together with an accompanying Methods Clarifications Log (locked alongside the v2.0 registration) that records minor specification clarifications introduced during the protocol-development phase; references to specific clarifications below are described inline rather than by their internal index codes. The analysis is fully reproducible from publicly available code and intermediate artifacts in approximately 30 minutes on a single core; all stochastic operations use a fixed random seed.

## Data sources

### Individual-level harassment data (N = 354)

We use the Tokiwa harassment preprint dataset (Tokiwa, 2025; IRB-approved), comprising N = 354 Japanese workplace participants who completed the following self-report measures:

- **HEXACO-60** (Wakabayashi, 2014; Japanese adaptation of Ashton & Lee, 2009): 60-item Likert 1-5 scale yielding six domain scores (H, E, X, A, C, O).
- **Dark Triad** scales (Machiavellianism, Narcissism, Psychopathy): used in the common-method-variance diagnostic (Stage 5, below) but not in the primary analysis.
- **Workplace Power Harassment Scale** (Tou et al., 2017): continuous severity scale measuring past-three-year power-harassment victimization frequency in the participant's workplace.
- **Gender Harassment Scale** (Kobayashi & Tanaka, 2010): continuous severity scale; used as triangulation outcome (H6).
- **Demographics**: binary self-reported gender (0 = female, 1 = male), age (continuous), prefecture-level region (categorical).

### Cluster centroids (N = 13,668; Tokiwa, 2026)

Seven HEXACO cluster centroids were taken from the previously published Tokiwa (2026) clustering of N = 13,668 Japanese respondents. Per pre-registration, these centroids are treated as *fixed parameters*; bootstrap confidence intervals are computed conditional on the centroids and do not propagate centroid-estimation uncertainty. This decision aligns with the inferential target of the present study (causal contrasts within a fixed taxonomy) rather than the joint estimation of the taxonomy itself.

### Population reweighting (Statistics Bureau Labor Force Survey 2022)

Marginal gender-by-age proportions for the Japanese working population were obtained from the Labor Force Survey Basic Tabulation 2022 Annual Average, published by the Statistics Bureau, Ministry of Internal Affairs and Communications (Statistics Bureau, 2023; PDF retrieved from e-Stat on 2026-04-30 by the investigator and archived at `simulation/data/`). We use Table 3 (employed persons by age group) restricted to the employed-persons category, yielding a total of 67.23 million persons with age × gender breakdown {15–64, 65+} × {male, female}. The resulting marginals (F = 0.4498, M = 0.5502) replace the v2.0 Stage 1 placeholder (F = 0.5, M = 0.5).

### MHLW power-harassment victimization targets

Past-three-year power-harassment victimization rates from three MHLW Workplace Harassment Surveys (MHLW, 2017, 2021, 2024) serve as external validation targets for H1:

- FY2016 (32.5%) — pre-law, primary target;
- FY2020 (31.4%) — transition period, secondary target;
- FY2023 (19.3%) — post-law, secondary target.

## Pre-registered analytic pipeline

The pipeline comprises nine stages (0 through 8), each producing a versioned HDF5 artifact under `output/supplementary/`. Stages 0–2 implement the H1 confirmatory chain; Stages 3–6 implement sensitivity analyses, baseline comparisons, common-method-variance diagnostics, and target trial documentation; Stages 7–8 implement the H7 counterfactual contrasts and transportability sensitivity. Random number streams across stages are isolated via additive seed offsets to ensure independence (see `code/utils_io.py:make_rng`).

### Stage 0: Type assignment, cell propensity, and 28-cell empirical Bayes

Each individual is assigned to the closest cluster (Euclidean nearest-centroid, primary; soft-assignment with τ ∈ {0.5, 1.0, 2.0} × median nearest-neighbor distance was also computed as a pre-registered sensitivity analysis). Outcomes are binarized at the participant-level mean + 0.5 SD (main; sensitivity at + 0.25 and + 1.0 SD).

Per-cell propensity p̂_c = X_c / N_c is computed for each of the 14 cells (7 clusters × 2 genders) for the main analysis, and 28 cells (× 2 roles) for the empirical Bayes sensitivity. The role split for the 28-cell sensitivity is operationalized as the top 15% on a Conscientiousness + 0.5 × eXtraversion composite of HEXACO domain scores (notation: "C + 0.5 × X" where C and X are the HEXACO Conscientiousness and eXtraversion domains, respectively); this composite is exploratory and is included as a robustness probe on the 14-cell main analysis rather than as a confirmatory role-classification step (no published validation of this exact composite for harassment perpetration vs victimization is invoked; the 14-cell results without role stratification remain the headline analysis). Per-cell 95% confidence intervals follow the pre-registered four-step priority cascade:

1. **Clopper-Pearson** exact binomial interval for degenerate cells (X_c ∈ {0, N_c});
2. **BCa** (Bias-Corrected accelerated; Efron, 1987) with the jackknife acceleration parameter â when |a| ≤ 10;
3. **BC** (bias-corrected only) when BCa fails or |a| > 10;
4. **Percentile** as a final fallback.

Empirical Bayes shrinkage on 28 cells uses the method-of-moments hyperprior (μ̂, σ̂², α̂, β̂) with three pre-registered rejection triggers: (i) bootstrap-stabilized variance ratio σ̂² / [μ̂(1 − μ̂)] < 0.05; (ii) max(α̂, β̂) > 100; (iii) (α̂ + β̂) / median(N_k) > 5. Per-iteration (α̂, β̂) re-estimation in bootstrap follows Carlin and Louis (2000).

### Stage 1: National prevalence aggregation

The national prevalence is computed as the weighted sum P̂_t = Σ_c (p̂_c × W_c) / Σ_c W_c, where W_c is the cluster × gender population weight. Cluster proportions remain at the Tokiwa (2026) values (cluster membership is not measured in the Labour Force Survey, and updating the cluster mass is therefore reserved for future work; this dependency is acknowledged as a pre-registered limitation); only the gender marginal is updated to the 2022 working-population value.

### Stage 2: H1 four-tier classification

Per the v2.0 master Section 5.4, the four-tier H1 hierarchy is:

- **Strict SUCCESS**: point MAPE_FY2016 ≤ 30% AND CI upper bound ≤ 30%
- **Standard SUCCESS**: point MAPE ≤ 30% (CI upper bound > 30% permitted)
- **PARTIAL SUCCESS**: 30% < point MAPE ≤ 60%
- **FAILURE**: point MAPE > 60%

Per the pre-registered headline-bootstrap rule, the national MAPE bootstrap uses B = 10,000; per-cell bootstrap remains at B = 2,000.

### Stage 3: Sensitivity sweep

A one-at-a-time (OAT) sensitivity sweep around the locked main configuration is computed over four parameter families with B = 2,000 per row:

1. Binarization threshold ∈ {mean + 0.25 SD, mean + 0.5 SD (main), mean + 1.0 SD};
2. Soft-assignment τ factor ∈ {0.5, 1.0, 2.0} × median NN distance;
3. Empirical Bayes shrinkage scale ∈ {0.5×, 1.0× (main), 2.0×} prior pseudocount;
4. Cluster-proportion mass shift ∈ {−0.10, 0.0 (main), +0.10} (proxy for cluster-mass uncertainty under the pre-registered sensitivity specification; the full Cartesian sweep is deferred to follow-up work due to combinatorial cost).

### Stage 4: Baseline comparison and H2 ordinal trend

Five baselines (B0–B4) of increasing sophistication are compared via cell-stratified bootstrap MAPE_FY2016:

- **B0**: predicts MHLW grand mean (uniform);
- **B1**: gender-only logistic regression;
- **B2**: HEXACO 6-domain logistic regression;
- **B3**: 14-cell conditional propensity (the main pipeline);
- **B4**: extended with age and age × cluster interactions (industry breakdown deferred to follow-up under the cluster-proportion limitation discussed in §Data availability).

H2 is tested via the Bonferroni-Holm-corrected family of pairwise inequalities (B0–B1, B1–B2, B2–B3, B3–B4) with a one-sided α = 0.05; Page's L (1963) ordinal trend test is reported as a pre-registered auxiliary statistic.

### Stage 5: Common-method-variance diagnostic

CMV is assessed via Harman's single-factor test (Podsakoff et al., 2003) on 11 standardized variables (6 HEXACO domains + 3 Dark Triad + 2 harassment continuous scales) using PCA on standardized inputs after listwise deletion. The 50% first-factor variance threshold demarcates "limited concern" from "concern present." The Lindell and Whitney (2001) marker-variable correction uses HEXACO Openness as the theoretical marker, given its weak theoretical link to harassment outcomes relative to HH/A/E.

### Stage 6: Target trial PICO documentation

Stage 6 produces a metadata-only artifact documenting the target trial (Hernán & Robins, 2016, 2020) PICO and four identifying assumptions: consistency, exchangeability, positivity, and SUTVA (strengthened with the Hudgens and Halloran (2008) peer-effect framework, per pre-registration).

### Stage 7: Counterfactual ΔP_A, ΔP_B, ΔP_C and H7 IUT

Counterfactual A applies a +0.3 SD shift to HEXACO H, A, and E for every individual, recomputes nearest-centroid cluster membership, recomputes per-cell propensity, re-aggregates via Stage 1 weights, and produces ΔP_A as the per-bootstrap difference between counterfactual and baseline national prevalence. Per the pre-registered positivity criterion, positivity is quantitatively diagnosed via the cell-level ratio ρ_{c, x = A} = (counterfactual cell N) / (observed cell N); cells with ρ < 0.10 are flagged, and the intervention is downgraded from confirmatory to exploratory if flagged_weight ≥ 20%.

Counterfactual B applies a +0.40 SD shift in Honesty-Humility **only to individuals whose baseline nearest-centroid cluster is already in the pre-registered low-prevalence target set {0, 4, 6}**, leaving all other individuals' HEXACO scores unchanged. Following the targeted shift, all individuals are re-classified by nearest-centroid on the post-intervention HEXACO matrix, per-cell propensities are re-computed from the realized binary outcomes within the new cell assignments, and ΔP_B is reported per-bootstrap as the difference between counterfactual and baseline national prevalence. This operationalization matches the pre-registered specification "HH := HH + δ_B × SD(HH) for individuals in Cluster ∈ {0, 4, 6}" and follows Pearl's (2009) do-operator notation, which formalizes the targeted indicator "if i ∈ Cluster {0, 4, 6}; HH_i otherwise". Positivity is computed via ρ_{c, B} = (count of individuals reassigned into target Cluster c after the shift) / (count of individuals in target Cluster c at baseline) and is downgraded by the same positivity rule as Counterfactual A.

Counterfactual C multiplies cell-level propensity by 0.80 uniformly across all cells while leaving HEXACO scores unchanged. The 20% reduction is calibrated against three meta-analyses (Escartín, 2016; Hodgins et al., 2014; Salin, 2021) and the MHLW FY2016–FY2023 natural experiment (32.5% → 19.3%; −13.2 pp ≈ −40.6% relative). Positivity for C is trivially preserved (ρ ≡ 1) per pre-registration.

The H7 intersection-union test (Berger & Hsu, 1996) computes one-sided 95% lower bounds L_BA and L_BC for ΔP_B − ΔP_A and ΔP_B − ΔP_C, respectively, with bootstrap-resampled cell data (B = 2,000). The configuration is classified per the pre-registered priority cascade:

- **REVERSAL** (priority 1): point ΔP_B < point ΔP_A OR point ΔP_B < point ΔP_C (B is dominated by A or C in point estimate, indicating that the pre-registered hypothesis ordering ΔP_B > {ΔP_A, ΔP_C} is reversed);
- **CONFIRMED**: L_BA > 0 AND L_BC > 0 (B dominates both A and C in CI lower bound);
- **PARTIAL**: exactly one of L_BA, L_BC > 0;
- **AMBIGUOUS**: neither lower bound > 0 (CIs allow zero or reversal).

### Stage 8: Transportability factor sweep

Per the pre-registered cultural-attenuation sweep, the bootstrap distribution of each ΔP_x is multiplied by a transportability factor F ∈ {0.3, 0.5, 0.7, 1.0}, anchored as:

- F = 0.3: Conservative cross-cultural attenuation (worst-case anchor; supported by Power et al. 2013 cross-continent variability findings);
- F = 0.5: Nielsen et al. (2017) Asia/Oceania attenuation expected (r = .16 vs European r = .33);
- F = 0.7: Mild attenuation (intermediate);
- F = 1.0: Reference (no attenuation).

H7 is re-classified per attenuated factor to verify robustness.

## Reproducibility

All stochastic operations use a fixed random seed. HDF5 artifacts at each stage include the seed, the pre-registration version identifier, the OSF DOI, and stage-specific provenance metadata as root-level attributes, enabling full audit-trail reconstruction. Verification hashes for all artifacts are computed via the project's verification script and compared against the committed reference-hash registry. A Docker container and a `reproduce` Makefile target encapsulate the full pipeline. A 56-test pytest suite (including the loader tests for the Statistics Bureau Labour Force Survey data activated under Phase 1) guards against regression in core utilities.

## Software and statistical environment

Analysis was conducted in Python 3.11.15 using NumPy 2.x, SciPy 1.x, pandas 2.x, scikit-learn 1.x, h5py 3.x, and statsmodels 0.14.x (full pinned versions in the project's `pyproject.toml`). All code is publicly available under MIT license at the project's GitHub repository and archived at OSF (DOI 10.17605/OSF.IO/3Y54U). The pre-registration is locked at OSF (DOI 10.17605/OSF.IO/3Y54U), and the v1.1 → v2.0 public diff document and v2.0 Methods Clarifications Log are hosted alongside the registration.

## Data availability

This study's data and reproducibility artifacts are deposited on a tiered basis:

- **Public tier** (no access request required): Stage 0–8 supplementary HDF5 artifacts, Figures 1–6 (PNG/PDF/SVG), the canonical numerical record (`canonical_numbers.md`), and SHA-256 reference hashes (`reference_hashes.json`) are openly available at the OSF v2.0 working project (osf.io/3hxz6) under `v2.0/v2.0_supplementary.tar.gz`, and reproducible from source via `make reproduce` on the GitHub repository.
- **Restricted tier** (IRB-approved access request): The N = 354 individual-level harassment dataset is hosted in a Private OSF component (`v2.0 IRB-Restricted Individual Data`) linked from the v2.0 working project. The parent project's public Wiki at https://osf.io/3hxz6/wiki/home/ documents the four data-use terms (research-purpose-bounded use; non-redistribution; required citation of Tokiwa, 2025, and the v2.0 OSF DOI; IRB acknowledgement) and the request procedure (click "Request access" on the Component page; provide institutional affiliation, IRB approval ID, and intended research use). Approved requesters are added as read-only contributors within 14 days. Methodological details of the underlying survey are reported in Tokiwa (2025).
- **External public sources**: The Statistics Bureau (MIC) Labour Force Survey 2022 Annual Average tabulations used for population reweighting are publicly archived in the repository at `simulation/data/mic_labor_force_2022.csv` (provenance: e-Stat retrieval 2026-04-30; the file was previously named `mhlw_labor_force_2022.csv` and renamed to reflect the correct MIC provenance).
- **Pre-registration**: v2.0 master document, v1.1 → v2.0 public diff, and the Section 6.5 Level 1 Methods Clarifications Log are publicly accessible at the v2.0 registration page (osf.io/3y54u, DOI 10.17605/OSF.IO/3Y54U).
