# D12 OSF Pre-Registration — HEXACO 7-Typology Workplace Harassment Microsimulation (Phase 1 + Phase 2)

**Document type**: OSF Standard Pre-Registration (English version, registered on OSF)
**Drafted**: 2026-04-29
**Last updated**: 2026-04-30 (v1.1 LOCKED — registered on OSF)
**Branch**: `claude/hexaco-harassment-simulation-69jZp`
**Author (corresponding)**: Eisuke Tokiwa (sole-authored)
**ORCID**: 0009-0009-7124-6669
**Affiliation**: SUNBLAZE Co., Ltd.
**Email**: eisuke.tokiwa@sunblaze.jp
**Status**: 🔒 **LOCKED v1.1** — registered on OSF; Stage 0 code execution **unlocked**
**OSF DOI**: 10.17605/OSF.IO/45QP9
**OSF Registration URL**: https://osf.io/45qp9
**OSF Associated Project**: https://osf.io/3hxz6 (HEXACO 7-Typology Workplace Harassment Microsimulation)
**Registered**: 2026-04-30
**Anchor template**: OSF Standard Pre-Registration (Bowman et al. 2020, https://osf.io/rh8jc) + Nosek et al. 2018 PNAS "preregistration revolution" 9-Challenge framework
**Companion document (Japanese master)**: `simulation/docs/pre_registration/D12_pre_registration_OSF.md`

---

## 0. Document Status (meta)

### 0.1 Purpose

This document is the English version of the master Pre-Registration draft for the HEXACO 7-typology workplace harassment microsimulation study. It is intended to be transcribed into the OSF Standard Pre-Registration web form and accompanied by the Japanese master as a supplementary PDF.

The Japanese master document (`D12_pre_registration_OSF.md`) is authoritative; this English version is a faithful translation prepared for OSF submission and international peer-review.

OSF submission workflow:
1. Internal review of this English version
2. Transcribe Sections 1–6 into the OSF Standard Pre-Registration web form
3. Attach the Japanese master as supplementary PDF
4. Obtain OSF DOI; record DOI in the Header of both documents
5. Lock both versions; subsequent edits follow Section 6.5 deviation policy

### 0.2 Compliance with Nosek et al. 2018 PNAS "preregistration revolution" 9 Challenges

| Challenge | Study situation | Response |
|---|---|---|
| **C1** Procedure changes during data collection | Simulation only; no new data collection | N/A |
| **C2** Discovery of assumption violations during analysis | 14-cell main analysis uses frequentist bootstrap with BCa (light assumptions). 28-cell EB uses Beta-Binomial conjugate (method of moments + sensitivity sweep) for robustness. | Section 6.5 specifies deviation policy |
| **C3** Data Are Preexisting | N = 354 (harassment) and N = 13,668 (clustering) are **prior IRB-approved data**. However, (a) the 7-type × gender 14-cell harassment cross-tabulation, (b) national-level aggregate predictions, and (c) Counterfactual A/B/C simulation outputs are **all unobserved**. | Section 3.1 specifies who has observed what; the partial-blinding state is honestly acknowledged |
| **C4** Longitudinal / large multivariate datasets | N = 13,668 clustering data is a large multivariate dataset (HEXACO 6 domains × 13,668 individuals). Although already used for cluster centroid extraction (IEEE-published), the 7-type × gender × harassment cross-tabulation in N = 354 has not been computed. | Section 2.3 enumerates Stages 0–8 in full upfront, fixing **all** analysis paths before any cell-level cross-tabulation is generated. Sensitivity sweeps (Section 6.4) are exhaustively pre-specified to remove garden-of-forking-paths risk. |
| **C5** Many experiments / many tests within one study | This study contains a single design with 7 hypotheses (H1–H7) tested at the population level. | Section 6.3 applies Bonferroni–Holm correction at family-wise α = .05 for H2 (4 ordinal pairwise tests) and H4–H7 (3 counterfactual tests). H1 is a single primary test (no correction). H7 is a composite single test. |
| **C6** Program-level null result reporting | This study commits to publication even if MAPE > 60% (D-NEW8) | Section 7 codifies the commitment in 5 explicit cases (success / partial / failure / H7 reversal / B3 < B2 overfitting) |
| **C7** Discovery research with weak prior expectations | Stages 5 (CMV diagnostic) and 4 (B0–B4 baseline hierarchy) have weaker prior expectations than the Phase 2 main hypotheses. | These analyses are still pre-registered with **specific decision rules** (Section 5.5 decision rule table; Section 5.6 Harman first-factor < 50% threshold). Outcomes that go either way are interpreted within the pre-specified rules; post-hoc reframing is prohibited. |
| **C8** Competing predictions | H7 (ΔP_B > ΔP_A AND ΔP_B > ΔP_C) is a strong-inference test that compares three competing intervention strategies. | Per Nosek 2018, competing predictions are well suited to preregistration and are turned to advantage by strong inference. Section 6.1 specifies criteria for confirmation, partial confirmation, and reversal. |
| **C9** Selectivity in narrative inference | All Stage 3 sensitivity sweeps (V, f1, f2, EB strength, threshold, K) and Stage 8 transportability sweeps are pre-registered. | Section 6.4 fixes the sweeps; Section 6.5 deviation policy further constrains post-hoc selectivity; Section 7 commits to publishing all pre-registered outcomes regardless of direction. |

### 0.3 Relationship between this preregistration and the research plan

- **Research plan v6/v7** (`research_plan_harassment_typology_simulation.md`): Internal document containing motivation, theory, and literature foundation (L1+L2+L3+L4 layers, where L3/L4 are normative claims about social responsibility)
- **This preregistration**: Restricted to **L1 descriptive/predictive commitments** only
- Normative claims (L3/L4) are deliberately excluded from this preregistration following the four-layer separation principle (research plan Part 0.4)

---

## 1. Study Information

### 1.1 Title

**HEXACO 7-Typology Workplace Harassment Microsimulation: Latent Prevalence Prediction and Target Trial Emulation of Personality-Based and Structural Counterfactuals in Japan**

Short form: "HEXACO Harassment Microsim (Phase 1 + Phase 2)"

### 1.2 Authors

- **Sole author**: Eisuke Tokiwa
- **ORCID**: 0009-0009-7124-6669
- **Affiliation**: SUNBLAZE Co., Ltd.
- **Corresponding email**: eisuke.tokiwa@sunblaze.jp
- Co-authors: None (sole-authored study)
- **Independent methodologist consultation**: After preregistration lock and prior to Stage 1 validation, an external methodologist with a mathematical biology background will be invited to review Section 5 (Analysis Plan) only. Per the methodologist's preference, the consultation will be acknowledged in **anonymous form** ("We thank an anonymous external methodologist with a mathematical biology background for review of the analysis plan") in this preregistration and in any future publication. The methodologist's identity is held privately by the author. This follows the lightweight variant of Munafò et al. (2017) Box 1 (CHDI Foundation independent statistical oversight).

### 1.3 Description

Workplace harassment is a multi-causal phenomenon. Organizational stressors (Bowling & Beehr 2006, ρ = .30–.53), subjective social status (Tsuno et al. 2015, OR = 4.21), industry composition, and legal/normative climate exert effects that are larger in magnitude than personality factors. The present study acknowledges these and instead **isolates the personality contribution**, asking how well a HEXACO 7-typology probabilistic model predicts Japanese workplace harassment prevalence.

**Phase 1 (descriptive simulation)**: Using existing N = 354 (harassment behavior) and N = 13,668 (HEXACO clustering) data, we estimate 14-cell (7 types × 2 genders) conditional harassment propensities, scale these via bootstrap to the Japanese workforce (~68 million), and triangulate the resulting **latent prevalence** against the Ministry of Health, Labour and Welfare (MHLW) national surveys' **expressed prevalence**.

**Phase 2 (intervention counterfactuals)**: Within the target trial emulation framework (Hernán & Robins 2020), we simulate three interventions: (A) universal HH (Honesty–Humility) intervention anchored in Kruse et al. (2014); (B) targeted high-risk-type intervention anchored in Hudson (2023), which is the **primary intervention of interest**; and (C) structural-only intervention anchored in Pruckner & Sausgruber (2013) plus Bezrukova et al. (2016), Roehling & Huang (2018), and Dobbin & Kalev (2018). For each, we estimate population-level harassment reduction.

No large language models are used. All mechanisms are transparent probability tables.

### 1.4 Hypotheses

#### H1 (Phase 1 main hypothesis)

**The aggregate national prediction obtained by scaling 14-cell (7 type × 2 gender) conditional harassment propensities to the population reproduces the MHLW H28 (FY2016, pre-Power Harassment Prevention Law, 32.5% past-3-year **power harassment** victimization rate) within MAPE ≤ 30%.**

- **Primary validation target**: **MHLW H28 (FY2016)** "Survey on Workplace Power Harassment" (past-3-year power harassment prevalence 32.5%, pre-law)
- **Secondary validation targets**: **MHLW R2 (FY2020)** "Survey on Workplace Harassment Report" (31.4%, transition); **MHLW R5 (FY2023, published March 2024)** "FY2023 MHLW-commissioned Survey on Workplace Harassment Report" (19.3%, post-law)
- ★ **Important scope clarification**: This study validates **power harassment (パワハラ) past-3-year prevalence only**. Sexual harassment, customer harassment (カスハラ), and pregnancy/childcare-related harassment are separate categories. MHLW R5 (FY2023) measures sexual harassment 6.3% and customer harassment 10.8% on the same N=8,000 sample, but this study's validation target is **restricted to power harassment**.
- **International baseline**: ILO (2022) Asia–Pacific lifetime prevalence 19.2%
- **Marginal-distribution check (1)**: **Pasona Research (2022)** N=28,135 nationwide workers aged 20–69, 5-year harassment prevalence 19.7% (lifetime 34.6%) — large-N harassment-specific domestic triangulation source
- **Marginal-distribution check (2)**: Tsuno et al. (2015) N=1,546 random sample, 30-day prevalence 6.1%

#### H2 (Phase 1 baseline hierarchy)

**The mean absolute percentage error (MAPE) is monotonically non-increasing across the baseline hierarchy: B0 (random) ≥ B1 (gender only) ≥ B2 (HEXACO 6-domain linear) ≥ B3 (7 typology) ≥ B4 (B3 + age + industry estimate + employment type).**

- Quantities of interest: MAPE difference B3 − B2 (typology incrementality) and B4 − B3 (personality slice incrementality)

#### H3 (Phase 1 latent vs expressed gap)

**The gap between MHLW H28 (FY2016, pre-law, past-3-year power harassment 32.5%) and our latent prediction is smaller than the gap between MHLW R5 (FY2023, post-law, past-3-year power harassment 19.3%) and our latent prediction.** That is, the pre-law condition is closer to the latent rate, while the post-law condition shows stronger environmental gating.

**The emergence of customer harassment (カスハラ) as a distinct category** (codified by the FY2022 amendment of the Power Harassment Prevention Law extending coverage to all employers including SMEs from April 2022, and first independently measured at 10.8% in MHLW R5) provides evidence that the apparent decline from 32.5% (pre-law) → 19.3% (post-law) reflects a **compound environmental gating effect**: (a) the legal/categorical boundary of what constitutes "harassment" itself shifted (customer harassment carved out as a new expressed category), and (b) reporting of the existing power harassment category was simultaneously suppressed. The Discussion will honestly acknowledge that the latent vs expressed gap reflects "category boundary shift × gating strength" rather than a simple environmental effect on a fixed construct.

#### H4 (Phase 2 Counterfactual A: Universal HH intervention)

**Under a +0.3 SD population-wide HH shift (a conservative discount of Kruse 2014's d = 0.71), the predicted national prevalence decreases by ΔP_A relative to baseline.**

- δ main = +0.3 SD; sensitivity range [0.1, 0.5] SD
- Predicted direction of ΔP_A: negative (reduction)

#### H5 (Phase 2 Counterfactual B: Targeted intervention — primary intervention of interest)

**Targeting high-risk types (primary: Cluster 0 = Self-Oriented Independent profile; secondary: Clusters 4 and 6) with a +0.4 SD HH shift (a conservative discount of Hudson 2023's self-selected effect) yields a reduction ΔP_B such that the cost-effectiveness ratio (ΔP_B / number treated) exceeds that of Counterfactual A.**

- δ main = +0.4 SD; sensitivity range [0.2, 0.6] SD
- Primary target: Cluster 0; secondary targets: Clusters 4 and 6 (the IEEE clustering paper's cluster numbering will be reconciled with the CSV numbering during implementation by matching centroids)

#### H6 (Phase 2 Counterfactual C: Structural-only intervention)

**Reducing all cell-conditional probabilities by 20% while leaving individual personality unchanged yields a reduction ΔP_C, but ΔP_C is smaller than ΔP_B from Counterfactual B.**

- effect_C main = 20%; sensitivity range [10%, 30%] (30% is the upper bound, justified by triangulation of Bezrukova 2016, Roehling 2018, Dobbin & Kalev 2018, and Pruckner 2013)
- Domain-transfer caveat (Pruckner = newspaper honor system; this study = workplace harassment) is explicitly acknowledged in the Discussion

#### H7 (Phase 2 main contrast — primary predictive commitment)

**ΔP_B > ΔP_A AND ΔP_B > ΔP_C** (targeted exceeds both universal and structural-only at the population level).

- If reversed (ΔP_B ≤ ΔP_A or ΔP_B ≤ ΔP_C), publication still occurs (Section 7 negative-result commitment)

#### Hypothesis hierarchy summary

| H# | Phase | Type | Inference metric |
|---|---|---|---|
| H1 | 1 | Confirmatory predictive | MAPE vs MHLW H28 (FY2016) past-3-year power harassment 32.5% |
| H2 | 1 | Confirmatory ordinal | Monotonic ordering of B0–B4 MAPE |
| **H2.industry** ★ | **1** | **Secondary confirmatory** | **B4 industry-stratified subgroup MAPE ≤ 50% against MHLW R5 (FY2023) industry-stratified power harassment (16-26.8% range)** |
| H3 | 1 | Exploratory descriptive | gap(MHLW H28 FY2016) < gap(MHLW R5 FY2023) |
| H4 | 2 | Conditional projection | Sign and magnitude of ΔP_A |
| H5 | 2 | Conditional projection | Sign of ΔP_B and cost-effectiveness ranking |
| H6 | 2 | Conditional projection | Sign and magnitude of ΔP_C |
| H7 | 2 | Confirmatory ordinal | ΔP_B > ΔP_A AND ΔP_B > ΔP_C |

### 1.5 What this study does NOT test (honest scope acknowledgment)

- Individual-level prediction (cell N = 10–30, pairwise MDE Cohen's d ≥ 0.92 from D13 power analysis preclude individual-level inference)
- Independent contributions of personality and SSS (no SSS measure in N = 354; research plan Part 1.5)
- Causal direction (cross-sectional design; reverse causation cannot be fully ruled out, mitigated by Roberts & DelVecchio 2000 plateau r = .74 evidence on adult personality stability)
- Long-term intervention effects (Roberts 2017 anchor measures effects only at 24 weeks)
- Cross-cultural transportability of Western anchors to Japan (Sapouna 2010 cultural moderator example; addressed by Section 5.8 sensitivity sweep)

---

## 2. Design Plan

### 2.1 Study Type

**Secondary analysis of preexisting data + microsimulation + counterfactual projection.**

- No new data collection
- No large language models or generative agents
- Mechanism: probability tables + Monte Carlo bootstrap + Empirical Bayes shrinkage (Beta-Binomial conjugate, method of moments)
- Causal framing: target trial emulation (Hernán & Robins 2020) + structural causal model (Pearl 2009 do-operator)

### 2.2 Blinding

The OSF Standard Pre-Registration concept of "blinding" applies to this study as follows:

| Item | Status |
|---|---|
| Individual HEXACO scores | **Already observed** (N = 354 / N = 13,668, prior IRB-approved) |
| Individual harassment self-reports (N = 354) | **Already observed** (aggregated in the Tokiwa harassment preprint) |
| 7 type × gender 14-cell harassment cross-tabulation | **Unobserved** (will be generated for the first time under the analysis specification fixed by this preregistration) |
| Stage 1 national aggregate predictions | **Unobserved** |
| Counterfactual A/B/C simulation outputs | **Unobserved** |
| MAPE against MHLW survey | **Unobserved** |

→ Under strict adherence to Nosek 2018 PNAS Challenge 3, **"pure" preregistration is achievable** for the unobserved analyses; for already-observed individual-level data, the preregistration is acknowledged as **partial blinding** (Section 3.1.3).

**Additional blinding-equivalent commitments**:
- This preregistration is registered on OSF prior to commencement of Stage 0 code execution
- Inference criteria (Section 6.6) and sensitivity sweeps (Section 6.4) are fixed before the 14-cell cross-tabulation is computed
- MAPE success/failure thresholds (30% / 60%) are fixed before any comparison with MHLW survey data is performed

### 2.3 Study Design

#### 2.3.1 Phase 1 Pipeline

```
Stage 0: Type assignment & probability table construction
  ├─ Input: harassment/raw.csv (N=354), clustering/csv/clstr_kmeans_7c.csv (centroids)
  ├─ Step 1: Assign each of N=354 individuals to nearest of 7 centroids by Euclidean distance
  ├─ Step 2: Compute 14-cell (7 type × 2 gender) crosstab for binary harassment outcome
  │            (binarization: mean + 0.5 SD per outcome [main]; sensitivity: +0.25 / +1.0 SD)
  ├─ Step 3: Bootstrap B = 2,000 iterations per cell with BCa CI (Efron 1987; DiCiccio & Efron 1996)
  └─ Output: 14-cell propensity table with 95% CIs

Stage 1: Population aggregation
  ├─ Input: Stage 0 output + N=13,668 type distribution + MHLW labor force composition
  ├─ Step 1: Obtain population proportions of the 7 types from N=13,668
  ├─ Step 2: Reweight by age × gender from the MHLW Labor Force Survey
  ├─ Step 3: Multiply cell-conditional probabilities by population weights → expected national latent prevalence
  └─ Output: National latent prevalence with bootstrap CI

Stage 2: Validation triangulation
  ├─ Compare national latent power-harassment prediction against MHLW H28 FY2016 (32.5%, primary), R2 FY2020 (31.4%), R5 FY2023 (19.3%)
  ├─ Metrics: MAPE (primary), Pearson r, Spearman ρ, KS distance, Wasserstein distance, calibration plot
  └─ Output: Validation report + cell-level prediction error map

Stage 3: Sensitivity sweeps
  ├─ V (victim multiplier) ∈ {2, 3, 4, 5}
  ├─ f1 (turnover rate) ∈ {0.05, 0.10, 0.15, 0.20}
  ├─ f2 (mental disorder rate) ∈ {0.10, 0.20, 0.30}
  ├─ EB shrinkage strength ∈ {0.5×, 1.0×, 2.0×}
  ├─ Binarization threshold ∈ {mean + 0.25 SD, +0.5 SD [main], +1.0 SD}
  ├─ Cluster K ∈ {4, 5, 6, 7 [main, IEEE published], 8}
  ├─ Role estimation models: (a) personality linear, (b) tree-based, (c) literature-based
  └─ Output: Robustness diagnostic table

Stage 4: Baseline hierarchy comparison (B0–B4)
  ├─ B0: Uniform random
  ├─ B1: Gender only
  ├─ B2: HEXACO 6-domain linear regression
  ├─ B3: 7 typology + gender (★ proposed method)
  ├─ B4: B3 + age + industry estimate + employment type
  └─ Output: MAPE for each baseline + monotonicity check

Stage 5: CMV diagnostic
  ├─ Harman's single-factor test on N=13,668 personality data (target < 50% variance)
  ├─ Marker variable correction (Lindell & Whitney 2001) using HEXACO Openness as theoretical marker
  └─ Output: CMV diagnostic supplementary
```

#### 2.3.2 Phase 2 Pipeline

```
Stage 6: Target trial emulation specification
  ├─ Specify target trial protocol (PICO + duration) for each counterfactual
  ├─ Eligibility: Japanese workers aged 20–64
  ├─ Treatment strategies: A (universal HH +δ SD), B (targeted HH +δ SD on Clusters 0/4/6),
  │                        C (cell probabilities × (1 − effect_C))
  ├─ Assignment: simulated random
  ├─ Outcome: population-level harassment prevalence
  ├─ Follow-up: 24 weeks (Roberts 2017 anchor)
  └─ Output: 4 identifying assumptions made explicit (exchangeability, positivity,
              consistency, transportability)

Stage 7: Counterfactual simulation
  ├─ A (Kruse 2014 anchor):  shift HH by +δ SD for all N, re-classify clusters,
  │                           re-estimate propensities, aggregate to national level
  ├─ B (Hudson 2023 anchor): shift HH by +δ SD only for individuals in Clusters 0/4/6
  ├─ C (Pruckner 2013 + Bezrukova 2016 + Dobbin & Kalev 2018 + Roehling 2018 triangulation):
  │     multiply cell-conditional probabilities by (1 − effect_C)
  └─ Output: ΔP_A, ΔP_B, ΔP_C with bootstrap CIs

Stage 8: Counterfactual sensitivity
  ├─ A: δ ∈ [0.1, 0.5] SD sweep
  ├─ B: δ ∈ [0.2, 0.6] SD sweep
  ├─ C: effect_C ∈ [0.10, 0.30] sweep
  ├─ Transportability: Western-anchor effect × {0.3, 0.5, 0.7, 1.0}
  │                    (Sapouna 2010 / Nielsen 2017 cultural moderator)
  └─ Output: Robustness table
```

### 2.4 Randomization

No physical randomization (observational + simulation). Bootstrap resamples and Monte Carlo runs are made deterministic via a **fixed random seed** (NumPy `default_rng(seed=20260429)`). The seed value is fixed by this preregistration.

---

## 3. Sampling Plan

### 3.1 Existing Data Statement (Nosek 2018 PNAS Challenge 3 strict adherence)

#### 3.1.1 Already collected and partially observed by the author

- **N = 354 harassment data** (`harassment/raw.csv`):
  - Aggregated in the Tokiwa harassment preprint via HC3-robust hierarchical regression of HEXACO + Dark Triad on harassment
  - **Observed by the author at the individual level**: HEXACO 6 domain scores, Dark Triad 3 scores, power harassment scale, gender harassment scale, age, gender, area
- **N = 13,668 clustering data**:
  - 7-type centroids and proportions are fixed in the Tokiwa clustering paper (IEEE-published)
  - **Observed by the author at the aggregate level**: 7 centroids (HEXACO 6 domains), cluster proportions, aggregate descriptive statistics

#### 3.1.2 Unobserved at the time of preregistration

- Distribution of 7-type memberships obtained by assigning N = 354 to the nearest of 7 centroids
- 14-cell (7 type × 2 gender) crosstabulated harassment binary outcomes
- Cell-conditional propensities with bootstrap BCa CIs
- 28-cell EB-shrunken estimates
- National-level aggregate latent prevalence
- MAPE against MHLW national surveys
- Counterfactual A/B/C ΔP estimates

#### 3.1.3 Honest acknowledgment of preregistration "purity"

The author has previously observed HEXACO ↔ harassment associations in the harassment preprint via HC3 regression. The **type-conditional, cell-level propensity table** and the **national-level aggregate predictions** fixed by this preregistration cannot be directly derived from that regression, but the author is **not fully blind to associations between HEXACO domains and harassment self-reports**.

→ Therefore, while Nosek 2018 Challenge 3's "pure preregistration" is the theoretical ideal, this study is in a state of **partial blinding**. In the interpretation of findings, we will distinguish between (a) the 7-type cell-cross-tabulation, which is preregistered, and (b) HEXACO domain-level associations, which are exploratory replications of prior work.

### 3.2 Sample Size

#### 3.2.1 Cell-level (Phase 1 main analysis)

From the D13 power analysis (`simulation/docs/power_analysis/D13_power_analysis.md`):

| Item | Value |
|---|---|
| Number of cells (main) | 14 (7 type × 2 gender) |
| Cell N: minimum / maximum / median | 10 / 70 / 18 |
| Cells with N < 10 | 0 (0%) |
| Cells with N < 20 | 7 (50%) |
| One-sample MDE (Cohen's d): median / maximum | 0.67 / 0.89 |
| Pairwise MDE (Cohen's d): median / maximum | 0.92 / 1.25 |
| Binary rate CI half-width: median | ±13 percentage points (power harassment) |
| Bootstrap iterations | 2,000 per cell |
| Bootstrap CI method | BCa (Efron 1987) |

#### 3.2.2 Cell-level (Phase 1 sensitivity)

| Item | Value |
|---|---|
| Number of cells (sensitivity) | 28 (7 type × 2 gender × 2 role) |
| Cells with N < 10 | 16 (57%) |
| Cells with N = 0 | 4 |
| Cells with N ≤ 3 | 9 |
| Required treatment | Empirical Bayes shrinkage (Beta-Binomial conjugate; Casella 1985 + Clayton & Kaldor 1987 + Efron 2014 + Greenland 2000) |

#### 3.2.3 National-level (Phase 1 + Phase 2)

Population: ~68 million (MHLW Labor Force Survey base). Cell-conditional probabilities are weighted by population shares; bootstrap is used to propagate cell-level uncertainty to national-level CIs.

### 3.3 Sample Size Rationale

- D13 confirms **N ≥ 250** (Funder & Ozer 2019's recommendation for stable r estimation) at the aggregate analysis level (N = 354 ≥ 250).
- The 14-cell main analysis satisfies **N ≥ 10 in every cell**, allowing bootstrap estimation without shrinkage.
- The pairwise MDE Cohen's d ≥ 0.92 corresponds to a "very large effect" (Cohen 1988) or, under Funder & Ozer 2019, an effect "rarely found in replication." Accordingly, **cell-level pairwise inference is avoided as a preregistered limitation**, and aggregate-level inference is the primary inferential target.

### 3.4 Stopping Rule

Not applicable: no new data collection.

**Sensitivity sweeps are restricted to the ranges fixed by this preregistration (Section 6.4).** Any post-registration extension of sensitivity ranges will be flagged as exploratory and excluded from confirmatory inference.

---

## 4. Variables

### 4.1 Manipulated Variables (counterfactual operators)

| Variable | Definition | Scope | Main value | Sensitivity range | Anchor |
|---|---|---|---|---|---|
| **δ_A** (universal HH shift) | Add δ × SD(HH) to HH score for all individuals | All N = 354 → re-classify cluster | **+0.3 SD** | [0.1, 0.5] SD | Conservative discount of Kruse 2014 d = 0.71 |
| **δ_B** (targeted HH shift) | Add δ × SD(HH) to HH score for individuals in Clusters 0/4/6 only | Primary: Cluster 0; secondary: Clusters 4, 6 | **+0.4 SD** | [0.2, 0.6] SD | Conservative discount of Hudson 2023 self-selected effect |
| **effect_C** (structural reduction) | Multiply cell-conditional probability by (1 − effect_C) | All 14 cells | **0.20** | [0.10, 0.30] | Triangulation of Pruckner 2013 + Bezrukova 2016 + Roehling 2018 + Dobbin & Kalev 2018 |
| **transportability_factor** | Phase 2 anchor effect × factor before applying to Japan | All counterfactuals | 1.0 (main) | [0.3, 0.5, 0.7, 1.0] | Sapouna 2010 / Nielsen 2017 cultural moderator evidence |

### 4.2 Measured Variables (existing observed data)

#### 4.2.1 Individual-level (N = 354 harassment data)

| Variable | Type | Operationalization | Source |
|---|---|---|---|
| **HEXACO 6 domains** | Continuous (Likert 1–5 mean) | Wakabayashi 2014 Japanese HEXACO-60 | `harassment/raw.csv` |
| **Dark Triad 3** | Continuous | Shimotsukasa & Oshio 2017 SD3-J | `harassment/raw.csv` |
| **Power harassment** | Continuous (item mean) → binarized at mean + 0.5 SD | Tou et al. 2017 Workplace Power Harassment Scale | `harassment/raw.csv` |
| **Gender harassment** | Continuous → binarized at mean + 0.5 SD | Kobayashi & Tanaka 2010 | `harassment/raw.csv` |
| **Age** | Continuous (years) | Self-report | `harassment/raw.csv` |
| **Gender** | Binary (0/1, n = 133/220) | Self-report | `harassment/raw.csv` |
| **Area** | Categorical | Self-report | `harassment/raw.csv` |

#### 4.2.2 Individual-level (N = 13,668 clustering data)

| Variable | Type | Use |
|---|---|---|
| **HEXACO 6 domains** | Continuous | Centroid extraction (already aggregated; `clustering/csv/clstr_kmeans_7c.csv`) |
| **Cluster proportion** | Categorical (7 types) | Population scaling weight |

#### 4.2.3 Population-level (MHLW + large-N domestic surveys; external validation targets)

| Variable | Source | Role |
|---|---|---|
| **Past-3-year power harassment victimization rate** | **MHLW H28 (FY2016, pre-law) 32.5% ★ primary**, MHLW R2 (FY2020, transition) 31.4%, **MHLW R5 (FY2023, post-law, published March 2024) 19.3%** | National validation target |
| **Industry-stratified past-3-year power harassment** | **MHLW R5 (FY2023) industry-stratified: Construction 26.8%, Compound services 22.5%, Education / Health & welfare 20.6%, …, Education 16.9% (industry range 16-26.8%, N=8,000 general sample)** | **B4 baseline subgroup validation (H2.industry)** |
| **Other harassment categories (for framing)** | MHLW R5 sexual harassment 6.3% (women 8.9% / men 3.9%), customer harassment 10.8% (for environmental shift framing) | Latent vs expressed gap framing (H3) |
| **5-year harassment prevalence (industry survey)** | **Pasona Research (2022) "Quantitative Survey on Workplace Harassment" N=28,135, 5-year prevalence 19.7% (lifetime 34.6%), industry-stratified 16.9-22.9%** | Marginal-distribution check (large-N harassment-specific) |
| **30-day prevalence (national-rep)** | Tsuno et al. 2015 N=1,546 (6.1%) | Marginal-distribution check (national-representative) |
| **International baseline** | ILO 2022 Asia–Pacific lifetime 19.2% | Comparative reference |
| **f1 anchor (harassment-victim turnover rate)** | **Pasona (2022) 5-year turnover among harassment victims: overall 10.3% (industry range 6.3-13.3%) ★ primary**; **MHLW R4 (FY2022) Employment Trend Survey** turnover-among-job-changers' previous-job exit reason "workplace interpersonal relations": men 8.3% / women 9.4% (upper bound) | f1 sensitivity sweep [0.05, 0.10, 0.15, 0.20] empirical anchor |
| **f1 macro cross-check** | Pasona (2022) annual harassment-induced turnover macro estimate: 865,000 persons/year (with 573,000 = 66% unreported / 暗数化), comprising 12.1% of total annual turnover (865,000 / 7,173,000) | Stage 2 chain output sanity check (Section 5.4) |
| **Mental disorder incidence** | MHLW Industrial Safety and Health Survey + Tsuno & Tabuchi 2022 PR = 3.20 | f2 anchor |

### 4.3 Indices (derived)

| Index | Definition | Stage |
|---|---|---|
| **7-type membership** | Each individual in N = 354 assigned to nearest centroid (Euclidean, HEXACO 6 domains) | Stage 0 |
| **Cell ID (14-cell)** | type ∈ {0..6} × gender ∈ {0, 1} | Stage 0 |
| **Cell ID (28-cell)** | type × gender × role ∈ {0, 1} | Stage 0 (sensitivity) |
| **Role probability** | Continuous, predicted from C + 0.5·X composite (top 15% → manager) | Stage 0; D1 sensitivity compares 3 alternative models |
| **National latent prevalence** | Σ_cell (cell propensity × cell population weight) | Stage 1 |
| **MAPE** | mean(|predicted − observed| / observed × 100) | Stage 2 (primary metric) |
| **ΔP_x** (counterfactual reduction) | predicted_baseline − predicted_counterfactual_x | Stage 7 |
| **Cost-effectiveness ratio** | ΔP_x / N_treated_x | Stage 7 (Phase 2) |

---

## 5. Analysis Plan (statistical models)

### 5.1 Phase 1 Stage 0: Cell-level propensity estimation

For each cell c ∈ {1..14} (7 type × 2 gender):
- X_c = number of "harassment perpetrators" (binarized at mean + 0.5 SD per outcome) in cell c
- N_c = total N in cell c
- Observed propensity: p̂_c = X_c / N_c
- Bootstrap distribution: B = 2,000 BCa resamples per cell (Efron 1987 *J Am Stat Assoc*; DiCiccio & Efron 1996 *Stat Sci*)
- BCa correction: bias correction z₀ + acceleration a from jackknife

**BCa numerical stability fallback chain** (★ added v2, anonymous methodologist consultation):
- BCa is the primary CI method.
- If acceleration parameter |a| > 10 (jackknife instability, typically caused by binary outcome with rate near 0/N or N/N at small N) OR BCa computation raises a numerical error (division by zero, NaN propagation), automatically fallback to **Bias-Corrected (BC) bootstrap** (z₀ correction only, no acceleration).
- If BC bootstrap also fails (rare; sample distribution too degenerate), fallback to **percentile bootstrap** (no correction).
- All cells record which CI method was actually used; the supplementary table reports per-cell method (BCa / BC / percentile).
- Aggregate-level (Stage 1, Stage 2) bootstrap propagation inherits the cell-level method choice automatically.

**Output**: 14-cell table with point estimates p̂_c and 95% CIs [p̂_c^lo, p̂_c^hi] (method = BCa / BC / percentile recorded per cell).

### 5.2 Phase 1 Stage 0 (sensitivity): 28-cell EB shrinkage

Beta-Binomial conjugate shrinkage (Casella 1985 + Clayton & Kaldor 1987 + Efron 2014 *Stat Sci* + Greenland 2000 *IJE*):

1. **Method-of-moments hyperprior estimation** (from 14-cell main):
   - μ̂ = mean(p̂_j), σ̂² = var(p̂_j) for j = 1..14
   - α̂ = μ̂ × [μ̂(1−μ̂)/σ̂² − 1]
   - β̂ = (1−μ̂) × [μ̂(1−μ̂)/σ̂² − 1]
2. **28-cell posterior** (each cell k):
   - E[p_k | X_k, N_k] = (α̂ + X_k) / (α̂ + β̂ + N_k)
   - 95% credible interval from Beta(α̂ + X_k, β̂ + N_k − X_k) quantiles
3. **Strength sensitivity sweep** (★ preregistered):
   - scale ∈ {0.5×, 1.0× (main), 2.0×}: hyperparameters (α̂, β̂) are multiplied by scale to produce weak / medium / strong shrinkage variants

**MoM rejection decision rule** (★ added v2, pre-specified, anonymous methodologist consultation):

The preregistration fixes the following hard rule for choosing between MoM and Stan hierarchical Bayes as the **primary** 28-cell estimator:

- **REJECT MoM, switch to Stan** if EITHER of:
  - σ̂² / [μ̂(1−μ̂)] < 0.05 (variance too small relative to maximum-possible binomial variance; MoM produces extreme α̂, β̂ with overshrink risk)
  - max(α̂, β̂) > 100 (pseudo-count exceeds 7× the median cell N=14; MoM dominates the likelihood signal)
- **ACCEPT MoM as primary** if both conditions clear.
- The threshold values 0.05 and 100 are FIXED by this v2 preregistration; no post-hoc adjustment is permitted.

**In ALL cases (regardless of acceptance / rejection), THREE estimators are computed and reported** (★ added v2 triangulation requirement):

| Estimator | Status | Reporting |
|---|---|---|
| MoM (Beta-Binomial conjugate) | Primary IF accepted | Main text + supplementary |
| Marginal MLE (numerical maximization) | Always run | Supplementary (triangulation) |
| Stan hierarchical Bayes (HMC, brms) | Primary IF MoM rejected | Main text (if primary) + supplementary always |

**Cross-check diagnostic** (★ added v2, methodologist suggestion): in addition to the σ̂² / [μ̂(1−μ̂)] threshold, the supplementary reports the alternative diagnostic **(α̂ + β̂) / median cell N** as a second indicator of overshrink concern. Readers can verify that the threshold-based switch does not produce qualitatively different conclusions.

**MoM threshold sensitivity caveat** (★ added v2, methodologist note): σ̂² is itself a sample estimate from N=14 cells with sampling variability. To address this, the supplementary reports MoM and Stan results in parallel even when MoM is "accepted" by the rule, demonstrating robustness across the threshold boundary.

**Diagnostic plots** (★ existing, retained): μ̂ ± SE plot; shrunken vs raw scatter; MoM-vs-Stan agreement scatter (★ added v2).

### 5.3 Phase 1 Stage 1: Population aggregation

For each validation period t ∈ {2016, 2020, 2024}:
- W_c (cell weight) = MHLW labor-force population × cluster proportion (from N = 13,668) × gender proportion × age weight
- National latent prevalence: P̂_t = Σ_c (p̂_c × W_c) / Σ_c W_c
- Bootstrap CI for P̂_t: 2,000 iterations, BCa

**Demographic reweighting source**: MHLW Labor Force Survey (age × gender × employment type).

### 5.4 Phase 1 Stage 2: Validation triangulation

**Bootstrap MAPE CI computation procedure** (★ added v2, anonymous methodologist consultation, pre-specified):

For each bootstrap iteration b ∈ {1..2,000}:

1. Resample N=354 with replacement, **stratified by 14 cell** (preserving cell membership marginals; this is cell-level bootstrap, NOT independent resampling of population weights).
2. Re-classify each resampled individual to the nearest of 7 centroids (centroids fixed from N=13,668; not bootstrapped).
3. Re-compute each cell's binary harassment outcome propensity p̂_c^(b) on the resampled data, applying the BCa numerical stability fallback chain (Section 5.1) at the cell level.
4. Apply population weights W_c (from MHLW Labor Force Survey, FIXED across iterations — not bootstrapped) to compute the resampled national latent prevalence:
   P̂_t^(b) = Σ_c (p̂_c^(b) × W_c) / Σ_c W_c
5. Compute the resampled MAPE:
   MAPE^(b) = |P̂_t^(b) − MHLW_observed_t| / MHLW_observed_t × 100
   where MHLW_observed_t is the FIXED point estimate from the MHLW survey for period t (no bootstrap of MHLW data).
6. Record MAPE^(b).

After all B = 2,000 iterations:
- **Point estimate MAPE** is computed on the original (non-bootstrapped) data, NOT the mean of MAPE^(b). The bootstrap distribution informs only the CI, not the point estimate.
- **95% BCa CI for MAPE** is constructed from the empirical distribution of {MAPE^(b)} using bias correction z₀ + acceleration a from jackknife (BCa fallback chain applies if numerical instability is detected at the MAPE level).

Both point estimate and 95% BCa CI are reported for each validation period t ∈ {FY2016, FY2020, FY2023}. Identical procedure applies to all baseline models B0–B4.

**4-tier judgment hierarchy** (★ added v2, primary inference criterion, pre-registered):

| Tier | Condition | Interpretation |
|---|---|---|
| **Strict SUCCESS** | Point MAPE ≤ 30% **AND** 95% bootstrap CI upper bound ≤ 30% | Strong confirmatory result; CI rules out PARTIAL or FAILURE regions |
| **Standard SUCCESS** | Point MAPE ≤ 30% (CI permitted to overlap the 30–60% PARTIAL region) | Weak confirmatory; Discussion explicitly notes the "Pre-registered ambiguity Tier" |
| **PARTIAL SUCCESS** | 30% < point MAPE ≤ 60% | Mixed evidence; CI flagged in Discussion |
| **FAILURE** | Point MAPE > 60% | Publish as failure-mode discovery (Section 7.3) |

Tier assignment for **H1 primary** is based on FY2016 validation (against MHLW H28 32.5%). FY2020 and FY2023 receive independent tier classifications and are reported as secondary.

**Tier-specific reporting requirements** (★ added v2, pre-registered):

- **Strict SUCCESS**: "Strict SUCCESS achieved" allowed in the headline / abstract without further qualification.
- **Standard SUCCESS**: must be qualified with an explicit CI ambiguity statement, e.g., "Standard SUCCESS achieved (point MAPE = X%, 95% bootstrap CI = [Y%, Z%]). The CI overlapped the PARTIAL region [30%, 60%], so the inferential strength is limited; the result falls in the **Pre-registered ambiguity Tier**, and confirmatory claims are correspondingly weakened."
- **PARTIAL SUCCESS**: standard PARTIAL framing with CI reported.
- **FAILURE**: standard FAILURE framing per Section 7.3.

All Tiers continue to satisfy the Section 7 negative-result publication commitment (D-NEW8). Post-hoc tier-threshold revision is PROHIBITED; a v3 OSF registration with a public diff is required for any change.

**Secondary metrics** (descriptive):
- Pearson r between cell-level p̂_c and MHLW R5 (FY2023) subgroup rates
- Spearman ρ (rank correlation)
- KS distance (distribution shape)
- Wasserstein distance (earth mover's distance)
- Calibration plot (cell-level predicted vs observed)

**Subgroup MAPE** (★ preregistered):
- Gender × age band → subgroup MAPE to localize failure modes

**Triangulation against marginal-distribution surveys** (★ preregistered, descriptive):
- **Pasona (2022) N=28,135, 5-year prevalence 19.7%**: report Pearson r between simulation cell-level prediction and Pasona industry-stratified rates (16.9-22.9%); descriptive only, no SUCCESS / FAILURE threshold.
- Tsuno et al. (2015) N=1,546 30-day prevalence 6.1%: reference period correction discussed in Methods (cannot be directly compared to past-3-year prediction).
- ILO (2022) 19.2% Asia-Pacific lifetime: report difference (Δ between predicted past-3-year and reported lifetime) for context.

**Stage 2 chain output sanity check** (★ preregistered, soft criterion):
- Compute predicted annual harassment-induced turnover via the V × f1 chain (Stage 2): Σ_cell (predicted perpetrators × V × f1).
- Compare against **Pasona (2022) annual harassment-induced turnover macro estimate of 865,000 persons/year** (with 573,000 = 66% 暗数化 / unreported).
- Soft criterion: predicted annual turnover within 50–200% of the Pasona estimate (i.e., 430,000–1,730,000 persons/year). Outside this range is flagged for failure-mode discussion.
- This is a SECONDARY criterion (does not affect H1 SUCCESS / FAILURE).

### 5.5 Phase 1 Stage 4: Baseline hierarchy

For each baseline B ∈ {B0, B1, B2, B3 (proposed), B4}:
- Train: same N = 354
- Predict national prevalence
- Compute MAPE against MHLW H28 (FY2016) past-3-year power harassment 32.5%
- Preregistered ordinal hypothesis (H2): MAPE_B0 ≥ MAPE_B1 ≥ MAPE_B2 ≥ MAPE_B3 ≥ MAPE_B4

**Models**:
- **B0**: uniform p_random = MHLW H28 (FY2016) grand mean
- **B1**: gender-only logistic (P̂(harassment | gender))
- **B2**: HEXACO 6-domain linear (logistic regression on all six domains)
- **B3** (proposed): 7 type × gender cell-conditional (this study's main model)
- **B4**: 7 type + age + **industry (estimated)** + employment type cell-conditional

**Industry estimation specification for B4** (★ preregistered, ★ Item 5):
- N = 354 does **not contain direct industry data**. B4's "industry" is **probabilistically estimated** from the joint (age × gender × employment-type) distribution against MHLW Labor Force Survey industry × demographic crosstabs.
- Estimation method (★ preregistered): for each individual i with (age_i, gender_i, employment_i), assign a 16-bucket industry probability vector P(industry_j | age_i, gender_i, employment_i) using MHLW Labor Force 2022 data. The cell prediction in B4 is then weighted by this probability vector.
- 16 industry buckets follow MHLW 日本標準産業分類 main divisions (matching the structure of the MHLW R5 industry-stratified harassment data).
- This estimation is **inherently noisy** (individual industry is not observed); B4's industry-level predictions are therefore **expected to have wide CIs**.

**Decision rule** (overall MAPE):
- B3 > B2 → "typology adds information beyond linear domain effects"
- B3 ≈ B2 → "typology does not exceed linear" (reported as a finding)
- B3 < B2 → "typology overfits" (reported as a critical finding)
- B4 ≫ B3 → "personality slice alone is insufficient; peripheral covariates needed"
- B4 ≈ B3 → "personality typology informationally subsumes peripheral covariates"

**Industry-stratified validation (H2.industry)** (★ preregistered, ★ Item 5):
- B4 industry-stratified predictions (16 buckets) are compared to **MHLW R5 (FY2023) industry-stratified past-3-year power harassment rates** (16-26.8% range)
- **Subgroup MAPE threshold (relaxed)**: 50% (more lenient than overall H1 30%, given industry-estimation noise)
- Decision rule:
  - Industry-MAPE ≤ 50% → CONFIRMS that personality + estimated industry can predict industry-level patterns
  - Industry-MAPE > 50% → REPORT as honest limitation (industry-estimation noise dominates the personality signal at the industry level)
- This is a **secondary criterion** to H1; failure does NOT trigger H1 failure.

### 5.6 Phase 1 Stage 5: CMV diagnostic

**Harman's single-factor test** (Podsakoff et al. 2003 *J Appl Psychol*):
- Apply unrotated EFA constrained to one factor on the full HEXACO item set in N = 13,668 personality data
- First-factor variance explained < 50% → CMV concern is limited (preregistered threshold)

**Marker variable correction** (Lindell & Whitney 2001 *J Appl Psychol*):
- Use HEXACO Openness as a theoretical marker (theoretically weak association with harassment)
- Partial out marker–target correlations to report adjusted associations

### 5.7 Phase 2 Stages 6–8: Counterfactual estimation

#### 5.7.1 Target trial emulation specification

For each counterfactual, specify PICO + duration:

| Element | Counterfactual A | Counterfactual B (★ main) | Counterfactual C |
|---|---|---|---|
| **P** (population) | Japanese workers aged 20–64 | Japanese workers aged 20–64 | Japanese workers aged 20–64 |
| **I** (intervention) | Universal HH +δ_A SD | Targeted (Clusters 0/4/6) HH +δ_B SD | Cell probabilities × (1 − effect_C) |
| **C** (control) | Pre-intervention baseline | Pre-intervention baseline | Pre-intervention baseline |
| **O** (outcome) | National harassment prevalence | National harassment prevalence + cost-effectiveness | National harassment prevalence |
| **Duration** | 24 weeks (Roberts 2017) | 24 weeks | 24 weeks |
| **Anchor** | Kruse 2014 d = 0.71 | Hudson 2023 b = .03/week | Pruckner 2013 + Bezrukova 2016 + Roehling 2018 + Dobbin & Kalev 2018 |

#### 5.7.2 Pearl 2009 do-operator notation

- Counterfactual A: do(HH := HH + δ_A × SD(HH)) for all individuals
- Counterfactual B: do(HH := HH + δ_B × SD(HH)) for individuals in Cluster ∈ {0, 4, 6}
- Counterfactual C: do(p_c := p_c × (1 − effect_C)) for all cells c

#### 5.7.3 Estimation

For each counterfactual x ∈ {A, B, C}:
- Apply the do-operator to N = 354 / cell-probability table per specification
- Re-run Stage 0 → Stage 1 (Stage 2 validation is omitted; only prediction is required)
- ΔP_x = P̂_baseline − P̂_x
- Bootstrap CI for ΔP_x (2,000 iterations, propagating cell-level uncertainty)
- Cost-effectiveness for B: ΔP_B / |Cluster 0 ∪ 4 ∪ 6 in population|

#### 5.7.4 Identifying assumptions (Hernán & Robins 2020)

To be honestly assessed in the Discussion:

1. **Exchangeability**: Y^a ⊥⊥ A | L
   - Violation risk: unmeasured confounding by culture and organizational climate
   - Mitigation: B4 baseline adjusts for peripheral covariates; sensitivity sweep
2. **Positivity**: P(A = a | L = l) > 0 for all l
   - Violation risk: with Cluster 6 dominant at 32%, intervention coverage is uneven
   - Mitigation: Cluster 6 is not deliberately excluded from counterfactual analysis
3. **Consistency**: observed Y when A = a equals Y^a (no interference between agents)
   - Violation risk: workplace peer effects, displacement
   - Mitigation: Counterfactual C displacement is explicitly acknowledged in the Discussion
4. **Transportability**: anchor-study population effects transport to the Japanese workforce
   - Violation risk: Kruse / Hudson / Pruckner are all Western / U.S. samples
   - Mitigation: Section 5.8 transportability sensitivity sweep + Sapouna 2010 / Nielsen 2017 cultural moderator citations

### 5.8 Phase 2 Stage 8: Transportability sensitivity

| Factor | Range | Interpretation |
|---|---|---|
| 0.3× | Strong cultural attenuation (Sapouna 2010 UK→Germany null worst case) | Conservative lower bound |
| 0.5× | Moderate attenuation (Nielsen 2017: Asia/Oceania Neuroticism r = .16 vs Europe .33) | "Expected" attenuation |
| 0.7× | Mild attenuation | Optimistic |
| 1.0× (main) | No attenuation (anchor effect = Japan effect) | Reference |

→ Robustness of conclusions to transportability_factor will be reported.

---

## 6. Inference Criteria & Sensitivity Master Table

### 6.1 Preregistered inference criteria summary

| H# | Criterion | Threshold | Decision |
|---|---|---|---|
| **H1** | MAPE(P̂_FY2016, MHLW H28 FY2016 past-3-year power harassment 32.5%) | ≤ 30% | SUCCESS |
| H1 | MAPE | 30 < x ≤ 60% | PARTIAL SUCCESS |
| H1 | MAPE | > 60% | FAILURE (publish anyway) |
| **H2** | MAPE_B0 ≥ MAPE_B1 ≥ MAPE_B2 ≥ MAPE_B3 ≥ MAPE_B4 | Strict monotonicity | Direction confirmed if ≥ 3 of the 4 pairwise inequalities hold |
| **H2.industry** ★ | B4 industry-stratified (16 buckets) MAPE vs MHLW R5 (FY2023) industry-stratified power harassment 16-26.8% | ≤ 50% (relaxed; industry estimated) | CONFIRMED if ≤ 50%; honest LIMITATION REPORT if > 50%. Secondary criterion (does not affect H1 SUCCESS / FAILURE). |
| H3 | gap(MHLW H28 FY2016) < gap(MHLW R5 FY2023) | Direction | Confirmed if MAPE_FY2016 < MAPE_FY2023 |
| H4 | sign(ΔP_A) | Negative (reduction) | Confirmed if 95% CI excludes 0 in the negative direction |
| H5 | sign(ΔP_B) and ΔP_B / N_treated > ΔP_A / N_total | Cost-effectiveness | Confirmed if both conditions hold |
| H6 | sign(ΔP_C) | Negative | Confirmed if 95% CI excludes 0 |
| **H7** | ΔP_B > ΔP_A AND ΔP_B > ΔP_C | Magnitude ranking | Confirmed if both inequalities hold at point estimates; uncertain if 95% CIs overlap |

### 6.2 Failure-mode commitments

- **H1 failure** (MAPE > 60%): publish as a failure-mode discovery (Doc 1 strategy 2 + Nosek 2018 Challenge 6)
- **H2 reversal** (B3 < B2): publish as a critical finding (typology overfitting)
- **H7 reversal** (ΔP_B ≤ ΔP_A or ΔP_B ≤ ΔP_C): publish; revise the main thesis claim

### 6.3 Multiple-comparison correction

- Phase 1 H1: single primary test; no correction needed
- Phase 1 H2: 4 ordinal pairwise comparisons (B0–B1, B1–B2, B2–B3, B3–B4) → Bonferroni–Holm at family-wise α = .05
- Phase 2 H4–H7: 3 counterfactuals × 1 main test each = 3 tests → Bonferroni–Holm at α = .05
- H7 ranking: already a single composite test (no further correction)

### 6.4 Preregistered sensitivity-sweep master table (★ Stages 3 & 8)

**Phase 1 sensitivity** (all combinations to be reported in MAPE table):

| Parameter | Main | Sweep range | Stage |
|---|---|---|---|
| **V** (victim multiplier) | 3 | {2, 3, 4, 5} | 3 |
| **f1** (harassment-induced turnover rate) | 0.10 (close to Pasona 2022 empirical 10.3%) | {0.05, 0.10, 0.15, 0.20} (covers Pasona industry range 6.3-13.3% + MHLW R4 "interpersonal" turnover 8.3-9.4%) | 3 |
| **f2** (mental disorder rate) | 0.20 | {0.10, 0.20, 0.30} | 3 |
| **EB shrinkage scale** | 1.0× | {0.5×, 1.0×, 2.0×} | 0 (28-cell) |
| **Binarization threshold** | mean + 0.5 SD | {mean + 0.25 SD, +0.5 SD, +1.0 SD} | 0 |
| **Cluster K** | 7 (IEEE published, main) | {4, 5, 6, 7, 8} | 0 |
| **Role estimation** | C + 0.5X composite top 15% | {(a) linear, (b) tree-based, (c) literature-based} | 0 (28-cell) |
| **Bootstrap iterations** | 2,000 | (fixed) | All |

**Phase 2 sensitivity**:

| Parameter | Main | Sweep range | Stage |
|---|---|---|---|
| **δ_A** (universal HH shift) | +0.3 SD | [0.1, 0.5] SD step 0.1 | 8 |
| **δ_B** (targeted HH shift) | +0.4 SD | [0.2, 0.6] SD step 0.1 | 8 |
| **effect_C** (structural reduction) | 0.20 | [0.10, 0.30] step 0.05 | 8 |
| **transportability_factor** | 1.0× | {0.3×, 0.5×, 0.7×, 1.0×} | 8 |
| **Target type set** (B) | {Cluster 0, 4, 6} | {Cluster 0 only}, {0+4}, {0+4+6 (main)}, {0+4+6+others} | 8 |

**Reporting**: All sensitivity outputs will be published as a supplementary table with open data (Section 8 reproducibility). The main text will use only the main values for quantitative claims; sensitivity outputs will be used qualitatively to describe robustness.

### 6.5 Deviation policy (Nosek 2018 Challenges 1 & 2)

Deviations from this preregistration are classified and reported as follows:

1. **Level 0 (no deviation)**: implementation matches the specification.
2. **Level 1 (minor specification clarification)**: e.g., explicit lookup table for bootstrap seeds, locking of CI library version. Recorded in a single line in the Methods.
3. **Level 2 (data-driven adjustment)**: e.g., method-of-moments produces divergent α̂, β̂ → switch to MLE / hierarchical Bayes. Justified in a one-paragraph Discussion subsection.
4. **Level 3 (analysis-plan revision)**: e.g., introducing 14-cell pairwise inference or other substantive change → **register a v2 of this preregistration on OSF**; publish a diff against v1.

→ All Level 2 and Level 3 deviations will be reported comprehensively in a dedicated Discussion subsection titled "Deviations from Pre-Registration."

### 6.6 Inference data lock-in commitments

- **Comparison against MHLW survey** is performed only after this preregistration is registered on OSF.
- **MAPE is computed** with the binarization threshold (mean + 0.5 SD) and population weights fixed by this preregistration prior to comparison.
- **Post-hoc revision of MAPE thresholds (30% / 60%) is prohibited.**

---

## 7. Negative-Result and Failure-Mode Publication Commitment (D-NEW8)

Anchor: **Nosek et al. 2018 PNAS Challenge 6** (program-level null-result reporting) + Doc 1 strategy 2 (failure as discovery).

### 7.1 Publication commitment

The author commits to journal submission in all of the following cases:

1. **H1 SUCCESS** (MAPE ≤ 30%): submit as a standard article.
2. **H1 PARTIAL SUCCESS** (30 < MAPE ≤ 60%): submit as a "partial validation" article that thematizes failure modes.
3. **H1 FAILURE** (MAPE > 60%): submit as a "negative result" article that characterizes the failure modes as discoveries.
4. **H7 reversal** (ΔP_B ≤ ΔP_A or ΔP_C): submit as a "thesis revision" article that revises the original framing.
5. **B3 < B2** (typology overfitting): submit as a "critical finding" article that documents the limits of the typology approach.

### 7.2 Target journal path

> **Note**: This preregistration locks the **analysis plan** (hypotheses, models, thresholds, sensitivity sweeps, ethical commitments). The choice of target journal is not locked and may change without invoking the Section 6.5 deviation policy. This section records the **submission roadmap** for transparency only.

#### Primary target

**Royal Society Open Science (Registered Report track)**

- Rationale: (1) The RR track guarantees publication after In-Principle Acceptance (IPA), structurally fulfilling the D-NEW8 negative-result commitment regardless of outcome; (2) within budget (£1,800 ≈ JPY 350k); (3) Open Access by default, aligned with the D-NEW9 reproducibility commitment; (4) Stage 1 review typically ~6–8 weeks; (5) IF is not a constraint (per the author's explicit preference for low-IF venues if necessary).
- Submission unit: **Stage 1 (Introduction + Methods + this preregistration)** is submitted first → IPA → **Stage 2 (Results + Discussion)** is submitted later.

#### Fallback path (if IPA is refused or scope mismatches)

| Order | Journal | Track | APC (JPY) | Acceptance certainty | Notes |
|---|---|---|---|---|---|
| 1 | **Royal Society Open Science** | RR | 350k | High after IPA | Primary |
| 2 | **Frontiers in Psychology** (Personality and Social Psychology specialty) | RR (specialty-dependent) | 440k | Equivalent in RR specialty | Fallback 1 |
| 3 | **Scientific Reports** (Nature) | Standard with preregistration DOI | 340k | ~50% | Fallback 2; broad scope |
| 4 | **PLOS ONE** | Standard with preregistration DOI | 290k | ~50%; explicitly accepts negative results | Fallback 3 |
| 5 | **PAID** (Personality and Individual Differences) | Standard | OA: 520k / subscription: 0 | ~40% | Fallback 4; personality fit |
| 6 | **JBE** (Journal of Business Ethics) | Standard | OA: 580k / subscription: 0 | ~10–20% | Fallback 5; ethics scope is restrictive |
| 7 | **RIO Journal** | "Pre-Registration" article type or Research Article | 70–130k | High | Fallback 6; low-APC option |
| 8 | **PCI Registered Reports + PCI-friendly OA journal** | RR (decentralized) | 0 | Higher overhead in finding a recommender | Fallback 7; cost-zero option |

#### This preregistration is journal-agnostic

- The commitments fixed by this preregistration apply regardless of which journal is selected.
- A change of journal is treated as Section 6.5 Level 0 (no deviation) and is not recorded in the deviation log.
- Each fallback venue cites the OSF preregistration DOI as the authoritative record.

### 7.3 Action if all venues refuse publication

- If all fallbacks fail: publish as a preprint (OSF / SocArXiv / PsyArXiv).
- A technical report of the failure modes will be published as an OSF supplementary, permanently archived.
- This preregistration will be maintained as active on OSF for at least 10 years.

---

## 8. Reproducibility Infrastructure (D-NEW9)

Anchor: **Munafò et al. 2017 *Nature Human Behaviour*** "A manifesto for reproducible science": five themes (Methods / Reporting / Reproducibility / Evaluation / Incentives) + TOP guidelines.

### 8.0 Mapping to the Munafò et al. 2017 five themes

| Theme | This study's implementation | Section |
|---|---|---|
| **1. Methods** (cognitive bias prevention; methodology training; independent statistical oversight; team science) | (a) Blinding-equivalent: this preregistration is registered before any 14-cell cross-tabulation is computed (Section 2.2); (b) Independent statistical oversight: anonymous external methodologist with mathematical biology background reviews Section 5 prior to Stage 2 (Section 1.2 + Section 8.1); (c) Methodology training and team science N/A for sole-authored secondary analysis. | 1.2, 2.2, 8.1 |
| **2. Reporting / Dissemination** (preregistration; reporting checklists CONSORT/ARRIVE/PRISMA/STROBE/STARD; conflict-of-interest disclosure) | (a) OSF Standard Pre-Registration this document; (b) Reporting per STROBE (observational study) and STARD where applicable, integrated into the Methods (Section 8.1); (c) COI disclosure (Section 12.1). | 8.1, 8.3, 12.1 |
| **3. Reproducibility** (open data, materials, software; TOP guidelines; Registered Reports) | (a) Open code (MIT) on GitHub + OSF mirror; (b) Open aggregated data; restricted-access raw data only where re-identification risk exists (Section 9.5); (c) Random seed (20260429), environment pinning (uv lock or Docker), `make reproduce` 30-minute target; (d) **TOP guidelines**: this study targets the highest TOP tier (Tier 3 / "Verification") for Data Citation, Data Transparency, Analytic Methods Transparency, Research Materials Transparency, Design and Analysis Transparency, Preregistration of Studies, Preregistration of Analysis Plans, and Replication; (e) Registered Reports submission (Section 7.2). | 7.2, 8.1, 8.2, 9.5 |
| **4. Evaluation** (preprints; post-publication review; results-blind review) | (a) **Results-blind review** via Registered Report track (Section 7.2 RSOS RR primary); (b) Preprint commitment if all journals refuse (Section 7.3, OSF / SocArXiv / PsyArXiv); (c) **Post-publication review**: this preregistration and supplementary materials remain open and citable on OSF for ≥ 10 years (Section 9.6), enabling PubPeer / commentary if misuse is reported. | 7.2, 7.3, 9.6 |
| **5. Incentives** (badges; Registered Reports; TOP guidelines; replication funding; open practices in hiring) | (a) **OSF Open Practice Badges** (Open Data, Open Materials, Preregistered) — this study qualifies for all three: aggregated data are open, code is open, and analysis is preregistered; (b) Registered Report submission (Section 7.2); (c) TOP-aligned reporting (this Section 8.0 row 3); (d) Replication funding and hiring N/A for the present preregistration but acknowledged as systemic levers per Munafò 2017. | 7.2, 8.0 row 3, 8.1 |

### 8.1 Preregistered infrastructure commitments

| Item | Implementation |
|---|---|
| **Random seed** | NumPy `default_rng(seed=20260429)`, Python `random.seed(20260429)`, Stan `seed = 20260429`. Seed value fixed and recorded in this preregistration. Bootstrap resample state persisted to HDF5. |
| **Environment pinning** | Either (a) a `uv` lock file (Python 3.11+) or (b) a Dockerfile (`python:3.11-slim` base) for full pinning. |
| **Make reproduce** | A `make reproduce` target regenerates all figures, tables, and supplementary outputs within 30 minutes (D-NEW9 README requirement). |
| **Open data** | Aggregated cell-level statistics (14-cell, 28-cell EB) are released openly on OSF and GitHub. Cell-level raw data are restricted-access (Section 9.5 ethics). |
| **Open code** | GitHub + OSF mirror under MIT license. |
| **Reporting checklist** | OSF Pre-Registration template + STROBE (observational study) + STARD (where applicable), integrated into the Methods. |
| **Independent statistical oversight** | Co-author / external methodologist review will be performed **after Stage 1 completion and before Stage 2 begins** (Munafò 2017 Box 1, CHDI Foundation example, lightweight variant). |

### 8.2 Preregistered repository structure

```
simulation/
├── docs/
│   ├── pre_registration/
│   │   ├── D12_pre_registration_OSF.md           (Japanese master, internal)
│   │   ├── D12_pre_registration_OSF.en.md        (this English version, OSF submission)
│   │   └── osf_registration_metadata.json        (registration timestamp, DOI, version log)
│   ├── notes/
│   ├── literature_audit/
│   └── power_analysis/
├── code/
│   ├── stage0_type_assignment.py
│   ├── stage0_cell_propensity.py
│   ├── stage1_population_aggregation.py
│   ├── stage2_validation.py
│   ├── stage3_sensitivity.py
│   ├── stage4_baselines.py
│   ├── stage5_cmv_diagnostic.py
│   ├── stage6_target_trial.py
│   ├── stage7_counterfactual.py
│   └── stage8_transportability.py
├── tests/
│   └── test_*.py (per stage)
├── output/
│   ├── tables/
│   ├── figures/
│   └── supplementary/
├── Dockerfile
├── pyproject.toml + uv.lock
├── Makefile (with `reproduce` target)
└── README.md (with "How to reproduce in 30 minutes" section)
```

### 8.3 Preregistered reporting items

The Methods and supplementary materials of the eventual paper will include:

- [ ] Random seed value (20260429)
- [ ] Software versions: Python, NumPy, SciPy, scikit-learn, statsmodels, Stan / brms (where applicable), PyMC (where applicable)
- [ ] Hardware: CPU model, RAM, OS (to the extent that they affect reproducibility)
- [ ] Bootstrap iterations: 2,000 per cell
- [ ] BCa CI parameters: numerical values of bias z₀ and acceleration a (jackknife-based)
- [ ] EB shrinkage hyperparameters: point estimates of α̂, β̂ and values for each scale of the sensitivity sweep
- [ ] MoM diagnostic: σ̂² as a fraction of μ̂(1 − μ̂)
- [ ] CMV diagnostic: percentage variance of Harman's first factor, marker-variable correction adjustments

---

## 9. Ethical Commitments (D-NEW10; research plan v6 Part 7)

### 9.1 Preregistered triple-locking structure

Anchor: research plan v6 Part 7 (8 sections addressing the ethical sensitivity of harassment research).

| Lock location | Content |
|---|---|
| **(a) Within the Methods** | The "Individual application disavowal" statement (Section 9.2 in full) is placed at the end of the Methods. |
| **(b) Within the Discussion** | A dedicated "How this should NOT be used" paragraph; language choices that avoid labeling individuals in any cluster as perpetrators. |
| **(c) Within this preregistration (this Section 9)** | All Part 7 items are listed here as ethical commitments; this preregistration is publicly archived on OSF. |

### 9.2 Individual application disavowal (Anti-screening Statement)

Reproduced verbatim or substantively in the Methods, Discussion, and any non-academic dissemination:

> **Statement on individual application**: This study estimates **population-level statistical patterns**, not individual risk profiles. The cell-conditional probabilities reported here have **no validity for individual prediction, screening, hiring, promotion, or any other personnel decision**. Application of these findings to individual personnel decisions would constitute (a) statistical misuse, given the pairwise MDE of d ≥ 0.92 and the absence of individual-level predictive validation, and (b) ethically unacceptable discrimination based on personality profiles. The authors explicitly disavow such application.

In addition, the following four points will be stated:

- The harassment probability for Cluster 0 is < 100%.
- Harassment occurs in Clusters 1–6 as well.
- Personality is changeable (Roberts 2017) → cluster membership is not a fixed individual attribute.
- Use in hiring or promotion decisions constitutes a violation of research ethics.

### 9.3 Authorial framing guidelines (preregistered)

| To avoid | Recommended phrasing |
|---|---|
| "perpetrator personality" | "personality profile associated with elevated risk in our model" |
| "high-risk individuals" | "cells with elevated baseline rates (population-level patterns)" |
| "Cluster X should be screened" | "Cluster X represents a target group for **opt-in voluntary intervention**" |
| "predict perpetrators" | "predict population-level prevalence" |
| "effective intervention reduces harassment" | "anchor literature suggests intervention can reduce harassment under specified transportability assumptions" |

### 9.4 Voluntary, opt-in policy principle (mandatory conditions for the Counterfactual B primary intervention)

- **Voluntary participation**: not coerced training or counseling; participants opt in.
- **No employment consequence**: participation has no effect on employment, evaluation, or compensation.
- **Anonymity / Confidentiality**: a participant's personality profile is not identifiable within their organization.
- **Resource provision, not coercion**: society's role is to provide opportunities for change, not to impose them.

### 9.5 Data availability ethics (D17)

| Data tier | Public availability |
|---|---|
| **Aggregated statistics** (14-cell propensities, 28-cell EB, national prevalence, ΔP_x) | Fully open (OSF + GitHub) |
| **Cell-level raw data** (per-cell individual records) | Restricted access (research-purpose review, IRB confirmation, re-identification risk assessment); 28-cell sensitivity has cells with N < 10 that pose re-identification risk |
| **N = 354 / N = 13,668 individual data** | Released according to the original publications' policies (anonymized, request-based) |

### 9.6 Long-term ethical monitoring (10-year commitment)

- **Misuse-case monitoring**: if findings are misappropriated for discrimination or screening, the author commits to publishing corrections or commentary in response.
- **Curation of secondary-analysis collaborations**: requests for secondary analysis are reviewed against the ethical principles in Part 7.
- **Maintenance of author contact**: corresponding-author contact is maintained for at least 10 years.
- **OSF maintenance**: this preregistration and supplementary materials are maintained as active on OSF for at least 10 years.

---

## 10. Limitations Pre-Acknowledged (research plan Part 4.2: 11 limitations)

The following limitations are acknowledged in advance to avoid post-hoc rationalization in the Discussion (Nosek 2018 Challenge 9):

| # | Limitation | Supporting / mitigating literature |
|---|---|---|
| L1 | Self-report → self-report circularity | Berry et al. 2012; Anderson & Bushman 2002; Vazire 2010 SOKA (direction-of-bias is conservative) |
| L2 | Representativeness of N = 354 (crowd-sourced self-selection) | Triangulated against Tsuno et al. 2015 N = 1,546 random sample |
| L3 | Error in role estimation | Promoted to a continuous covariate in D13; D1 compares 3 alternative models |
| L4 | Cell-level statistical power (pairwise MDE Cohen's d ≥ 0.92) | D13 power analysis; emphasis on aggregate inference |
| L5 | Uncertainty in V (victim multiplier), f1 (turnover rate), f2 (mental disorder rate) | Stage 3 sensitivity sweep |
| L6 | Cross-sectional design precludes causal inference; reverse causation cannot be fully ruled out | Roberts & DelVecchio 2000 plateau r = .74; Specht 2011 mid-life stability |
| L7 | Omission of GAM situational variables | "Personality slice" framing; Bowling & Beehr 2006 effects acknowledged |
| L8 | Common method bias | Podsakoff 2003 standard reference; Stage 5 diagnostic |
| L9 | Phase 2 transportability (Western anchors → Japan) | Section 5.8 sensitivity sweep; Sapouna 2010 / Nielsen 2017 |
| L10 | Inseparability of latent vs expressed prevalence | Section 1.4 frames the boundary as continuous |
| **L11** | **Construct validity** (whether the harassment scale measures "latent propensity" is empirically untested) | Vazire 2010 SOKA; convergent/discriminant validity check (supplementary) |

---

## 11. Researcher Reflexivity (Positionality)

### 11.1 Preregistered positionality statement

The following statement will be reproduced verbatim or substantively in the Methods or Acknowledgments of the eventual paper:

> "The author has previously argued in non-peer-reviewed work for a systemic-causation framing of workplace harassment, emphasizing social systems' responsibility over individual self-responsibility (Tokiwa, *forthcoming*). This normative stance may influence which limitations are emphasized and how findings are framed in the Discussion. To mitigate, (a) the empirical analysis section is restricted to L1 descriptive/predictive claims, (b) all causal language is constrained by the target trial emulation framework (Hernán & Robins, 2020) with explicit identifying assumptions, (c) anti-screening and anti-discrimination statements are included regardless of findings, (d) limitations are presented in a structured 11-item framework derived from a 60+ paper literature audit (rather than selectively), (e) negative-result publication is committed in advance via Section 7 of this preregistration (D-NEW8), and (f) review by an independent methodologist is sought prior to Stage 2 validation (Section 8.1; D-NEW9)."

### 11.2 Bias-mitigation procedures (preregistered)

- A preregistered analysis plan (this document) limits the scope for post-hoc revision (Nosek 2018).
- Independent review of the Discussion's interpretation by a methodologist (Munafò 2017 Box 1).
- Advance commitment to failure-mode publication (Section 7).
- Awareness of self-report direction-of-bias: per Vazire 2010 SOKA, harassment self-reports are framed as conservative (lower-bound) estimates.

---

## 12. Other (Conflicts, Funding, Data Sources)

### 12.1 Conflicts of Interest

- The author derives no commercial benefit from the findings of this study.
- The author's affiliation (SUNBLAZE Co., Ltd.) has no commercial interest in the results.
- This study is a secondary analysis of preexisting IRB-approved data; no new data collection is performed.
- OSF preregistration is free; there is no author cost.

### 12.2 Funding

- **Simulation phase (this study)**: No external funding. The work is conducted within the author's regular working hours at SUNBLAZE Co., Ltd.
- **Original data collection**:
  - N = 354 (harassment data): inherits the IRB approval and funding documented in the Tokiwa harassment preprint.
  - N = 13,668 (clustering data): inherits the IRB approval and funding documented in the Tokiwa clustering paper (IEEE-published).
- **No new IRB**: This study performs simulation only; it does not involve any new research activity with human subjects, and is a secondary analysis of existing anonymized data.

### 12.3 Data Sources Summary

| Source | Type | Access |
|---|---|---|
| `harassment/raw.csv` (N = 354) | Primary | Author's prior IRB-approved collection (Tokiwa harassment preprint) |
| `clustering/csv/clstr_kmeans_7c.csv` | Derived | Tokiwa clustering paper (IEEE-published; centroid table) |
| MHLW H28 (FY2016) Survey on Workplace Power Harassment | Public | https://www.mhlw.go.jp/ (pre-law, 32.5%, ★ H1 primary validation target) |
| MHLW R2 (FY2020) Survey on Workplace Harassment Report | Public | https://www.mhlw.go.jp/ (transition, 31.4%) |
| **MHLW R5 (FY2023, published March 2024) "FY2023 MHLW-Commissioned Survey on Workplace Harassment Report"** (PwC Consulting, 385 pages) | Public | `simulation/prior_research/_text/令和５年度 厚生労働省委託事業 職場のハラスメントに関する実態調査報告書.pdf` (post-law, 19.3%, full report; industry-stratified data 16-26.8% used for B4 H2.industry validation) |
| **MHLW R4 (FY2022) Employment Trend Survey Summary** | Public | `simulation/prior_research/_text/厚生労働省_令和４年雇用動向調査結果の概況.pdf` (FY2022 turnover by reason "workplace interpersonal" 8.3-9.4%; f1 secondary anchor) |
| MHLW Industrial Safety and Health Survey | Public | https://www.mhlw.go.jp/ |
| MHLW Labor Force Survey | Public | https://www.stat.go.jp/data/roudou/ (industry × demographic crosstabs for B4 industry estimation, Section 5.5) |
| ILO 2022 Global survey | Public | https://www.ilo.org/ |
| Tsuno et al. 2015 N = 1,546 | Published | *PLOS ONE* (Tsuno et al. 2015; national-rep 30-day prevalence 6.1%) |
| Tsuno & Tabuchi 2022 | Published | (Tsuno & Tabuchi 2022; PR=3.20 for f2 anchor) |
| **Pasona Research (2022) "Quantitative Survey on Workplace Harassment"** (PERSOL Research and Consulting Co., Ltd., Think Tank, 152 pages) | Industry survey | `simulation/prior_research/_text/パーソル_職場のハラスメントについての定量調査.pdf` (★ N=28,135 large-N harassment-specific 5-year prevalence 19.7%, industry-stratified 16.9-22.9%; ★ primary f1 empirical anchor 10.3% [industry range 6.3-13.3%]; ★ macro turnover estimate 865,000/year with 66% unreported / 暗数化) |

---

## 13. Citation Anchors (Tier 1+2+3+4 literature foundation)

The commitments in this preregistration are anchored to the literature below. The full annotated list (40-paper deep reading) is in `simulation/docs/literature_audit/deep_reading_notes.md`.

### 13.1 Tier 4 metascience anchors (★ core anchors of this preregistration)

- **Nosek, Ebersole, DeHaven & Mellor (2018)**. The preregistration revolution. *PNAS, 115*(11), 2600–2606. https://doi.org/10.1073/pnas.1708274114 [→ Sections 0.2, 3.1, 7]
- **Munafò et al. (2017)**. A manifesto for reproducible science. *Nature Human Behaviour, 1*, 0021. https://doi.org/10.1038/s41562-016-0021 [→ Section 8]
- **Vazire (2010)**. Who knows what about a person? The Self–Other Knowledge Asymmetry (SOKA) Model. *JPSP, 98*, 281–300. [→ Section 10 L1, L11]
- **Funder & Ozer (2019)**. Evaluating effect size in psychological research: Sense and nonsense. *AMPPS, 2*, 156–168. [→ Section 3.3]

### 13.2 Causal inference framework

- **Hernán & Robins (2020)**. *Causal Inference: What If*. Chapman & Hall/CRC. [→ Section 5.7.4]
- **Pearl (2009)**. *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge. [→ Section 5.7.2]

### 13.3 Statistical methods anchors

- **Efron (1987)**. Better bootstrap confidence intervals. *J Am Stat Assoc*. [→ BCa]
- **DiCiccio & Efron (1996)**. Bootstrap confidence intervals. *Stat Sci*. [→ BCa]
- **Casella (1985)**. An introduction to empirical Bayes data analysis. *Am Stat*. [→ EB]
- **Clayton & Kaldor (1987)**. Empirical Bayes estimates of age-standardized relative risks. *Biometrics*. [→ EB epidemiology]
- **Efron (2014)**. Two modeling strategies for empirical Bayes estimation. *Stat Sci*. [→ modern EB]
- **Greenland (2000)**. Principles of multilevel modelling. *Int J Epidemiol*. [→ EB / multilevel]
- **Podsakoff, MacKenzie, Lee & Podsakoff (2003)**. Common method biases in behavioral research. *J Appl Psychol*. [→ CMV]
- **Lindell & Whitney (2001)**. Accounting for common method variance. *J Appl Psychol*. [→ marker variable]
- **Schofield et al. (2018)**. Health-related microsimulation. *Eur J Health Econ*. [→ MAPE in microsimulation]

### 13.4 Phase 2 intervention anchors

- **Kruse, Chancellor, Ruberton & Lyubomirsky (2014)**. An upward spiral between gratitude and humility. *Soc Psychol Personal Sci, 5*(7), 805–814. [→ Counterfactual A]
- **Hudson (2023)**. Lighten the darkness: Personality interventions targeting agreeableness. *J Personality, 91*(4). [→ Counterfactual B, primary]
- **Pruckner & Sausgruber (2013)**. Honesty on the streets. *J Eur Econ Assoc, 11*(3), 661–679. [→ Counterfactual C]
- **Bezrukova et al. (2016)**. Diversity training meta-analysis. [→ Counterfactual C triangulation]
- **Roehling & Huang (2018)**. Sexual-harassment training meta-analysis. [→ Counterfactual C triangulation]
- **Dobbin & Kalev (2018)**. 985-study meta-analysis. [→ Counterfactual C triangulation]
- **Roberts et al. (2017)**. Personality trait change through intervention: a systematic review. *Psychol Bull, 143*(2), 117–141. [→ 24-week duration anchor]

### 13.5 Personality and harassment anchors

- **Pletzer et al. (2019)**. HEXACO meta-analysis. [→ HH × CWB ρ ≈ −.20 to −.35]
- **Nielsen, Glasø & Einarsen (2017)**. FFM × harassment meta-analysis. [→ cultural moderator]
- **Bowling & Beehr (2006)**. Harassment from the victim's perspective: a meta-analysis. [→ environmental ρ]
- **Roberts & DelVecchio (2000)**. Rank-order consistency meta-analysis. [→ stability r = .74]
- **Specht et al. (2011)**. SOEP N = 14,718 stability study. [→ mid-life stability]
- **Ashton & Lee (2007)**. The HEXACO model. [→ trait taxonomy]
- **Wakabayashi (2014)**. Japanese HEXACO-60. [→ Japanese measurement]
- **Tou et al. (2017)**. Workplace Power Harassment Scale. [→ measurement]
- **Kobayashi & Tanaka (2010)**. Gender Harassment Scale. [→ measurement]
- **Shimotsukasa & Oshio (2017)**. SD3-J. [→ Dark Triad measurement]
- **Lee & Ashton (2005)**. HEXACO ↔ Dark Triad correlations. [→ trait inter-correlation]
- **Berry, Carpenter & Barratt (2012)**. CWB self-other meta-analysis. [→ self-report defense]
- **Anderson & Bushman (2002)**. General Aggression Model. [→ self-report → real behavior]

### 13.6 Personality upstream of socioeconomic status

- **Heckman, Stixrud & Urzua (2006)**. Noncognitive skills predict outcomes. [→ Section 1.5]
- **Grijalva et al. (2015)**. Narcissism → leadership emergence meta-analysis. [→ Section 1.5]
- **Roberts et al. (2007)**. The power of personality. [→ Section 1.5]

### 13.7 Japanese context

- **MHLW (H28, FY2016)** Heisei 28 Survey on Workplace Power Harassment. [→ ★ H1 primary validation target, 32.5% pre-law, past-3-year power harassment]
- **MHLW (R2, FY2020, published 2021)** Reiwa 2 Survey on Workplace Harassment Report. [→ secondary validation 31.4% transition]
- **MHLW (R5, FY2023, published March 2024)** Reiwa 5 MHLW-Commissioned Survey on Workplace Harassment Report (PwC Consulting, 385 pages). [→ ★ secondary validation 19.3% post-law; ★ industry-stratified data 16-26.8% used in H2.industry secondary criterion (Sections 5.5, 6.1); customer harassment 10.8% as new environmental category (Section 1.4 H3)]
- **MHLW (R4, FY2022, published August 2023)** Reiwa 4 Employment Trend Survey Summary. [→ ★ f1 secondary anchor: turnover by reason "workplace interpersonal" men 8.3% / women 9.4% upper bound]
- **Pasona Research (2022)** Quantitative Survey on Workplace Harassment (PERSOL Research and Consulting Co., Ltd., Think Tank, N=28,135 nationwide workers aged 20–69, 152 pages). [→ ★ Pre-reg validation triangulation (Section 5.4): 5-year harassment prevalence 19.7%, industry-stratified 16.9-22.9%; ★ f1 PRIMARY empirical anchor: harassment-victim turnover rate 10.3% (industry range 6.3-13.3%) (Sections 4.2.3, 6.4); ★ Stage 2 chain output sanity check (Section 5.4): macro estimate 865,000 annual turnover with 66% unreported]
- **Tsuno et al. (2010)** Japanese NAQ-R. [→ measurement]
- **Tsuno et al. (2015)** Socioeconomic determinants in a Japanese national-representative sample N=1,546 (PLOS ONE). [→ Sections 1.5, 4.2; 30-day prevalence 6.1% as reference]
- **Tsuno & Tabuchi (2022)** Bullying → SPD PR = 3.20. [→ f2 anchor]

### 13.8 Self-citation hub

- **Tokiwa et al.** Clustering paper (IEEE-published). [→ N = 13,668, 7-type centroids]
- **Tokiwa et al.** Harassment preprint. [→ N = 354 HEXACO + Dark Triad regression]

---

## 14. Version Log & Implementation Checklist

### 14.1 Version log

| Version | Date | Changes |
|---|---|---|
| **v1.0 draft** | 2026-04-29 | Initial draft based on research plan v6/v7 (1,458 lines), the D13 power analysis (209 lines), and the 40-paper deep reading. Pending OSF registration. |
| **v1.1 draft** | 2026-04-29 | Pre-OSF-lock revision integrating 3 new domestic surveys uploaded by the author: (1) MHLW R5 (FY2023, 385 pages, full report) — used for industry-stratified validation in the H2.industry secondary criterion (Sections 5.5, 6.1); (2) MHLW R4 (FY2022) Employment Trend Survey — used as f1 secondary anchor; (3) Pasona Research (2022) N=28,135 quantitative survey — used for Pasona triangulation (Section 5.4), as the f1 PRIMARY empirical anchor (Sections 4.2.3, 6.4 — value 10.3% with industry range 6.3-13.3%), and for the Stage 2 chain output sanity check (predicted annual harassment-induced turnover should fall within 50-200% of Pasona's 865,000/year estimate). All MHLW citations standardized: era code (H28/R2/R5) + fiscal year + scope (past-3-year power harassment) + policy phase (pre-law/transition/post-law). Added customer harassment (カスハラ) category emergence (10.8% in MHLW R5) as evidence of environmental moderation in the latent vs expressed framing (Section 1.4 H3). H1 main MAPE threshold (≤30%) and sensitivity sweep ranges UNCHANGED; only secondary criteria (H2.industry, Stage 2 chain sanity check) and citation precision improved. Pending OSF registration. |
| **v1.1 LOCKED** | 2026-04-30 | 🔒 Registered on OSF. **DOI: 10.17605/OSF.IO/45QP9** (https://osf.io/45qp9). Associated project: https://osf.io/3hxz6. Subsequent modifications must follow Section 6.5 Level 3 deviation procedure (v2 registration with public diff against v1.1). Stage 0 code execution unlocked. |

### 14.2 Pre-registration submission checklist

#### 14.2.1 OSF preregistration submission (independent track)

- [N/A] **Internal review**: co-author confirmation — **not applicable to a sole-authored study**
- [ ] **Self-review against Nosek 2018's nine challenges**: complete Section 0.2
- [ ] **Self-review against Munafò 2017's five themes**: complete Section 8
- [x] **English translation**: this `D12_pre_registration_OSF.en.md` is the English version
- [x] **OSF account**: existing (used for `metaanalysis/`); only new project creation is pending
- [x] **OSF Standard Pre-Registration template**: Sections 1–6 transcribed into the OSF web form (Part B paste sheet used)
- [x] **PDF supplementary**: JP/EN markdown + JP/EN PDFs attached to OSF
- [x] **DOI acquisition**: 10.17605/OSF.IO/45QP9 acquired and recorded in the Header of both the Japanese master and this English version
- [x] **GitHub mirror**: `simulation/docs/pre_registration/` is committed publicly (since commit `c5c591e`)
- [x] **Funding & affiliation**: Section 12.2 is filled in (SUNBLAZE Co., Ltd. / no external funding for the simulation phase)
- [ ] **Anti-screening triple-lock**: Section 9.1 prepared in all three locations
- [ ] **Independent methodologist contact**: confirm review request per Section 8.1 (mode B: anonymous, mathematical biology)

#### 14.2.2 Registered Report submission (track separate from OSF preregistration; after preregistration lock)

- [ ] Confirm RSOS Registered Report **Author Guidelines** (latest version, scope and format requirements)
- [ ] Prepare **Stage 1 manuscript** (an English manuscript integrating Introduction + Methods + this preregistration, within the Stage 1 word limit)
  - Introduction: translated from research plan v6/v7 Part 10.3 (5-paragraph structure)
  - Methods: translated from Sections 4–5 of this preregistration; includes computational reproducibility (D-NEW9)
- [ ] Prepare a **cover letter** (citing the existing OSF preregistration DOI; requesting In-Principle Acceptance)
- [ ] Submit via the **submission portal** (RSOS / Editorial Manager, etc.)
- [ ] Respond to **IPA review** (typically 6–8 weeks; revise per reviewer comments)
- [ ] **IPA acquired** → begin Stage 0 code implementation (Section 14.3)
- [ ] Prepare **Stage 2 manuscript** (Results + Discussion based on the IPA); submit Stage 2
- [ ] **Final acceptance** → publication

#### 14.2.3 Fallback path (if IPA is refused; see Section 7.2 fallback table)

- [ ] Resubmit to a Frontiers in Psychology RR specialty, OR
- [ ] Switch to standard track (after Stages 0–8 are complete, submit to Sci Reports / PLOS ONE / etc.)
- [ ] Always cite the OSF preregistration DOI; transparently document any deviations

### 14.3 Final checks before Stage 0 implementation

The preregistration is **locked** when all of the following are complete; only then may Stage 0 code execution begin:

- [x] OSF DOI acquired (10.17605/OSF.IO/45QP9)
- [x] DOI added to the Header of this document (and the Japanese master)
- [N/A] Co-author sign-off — not applicable (sole-authored)
- [ ] Section 5 (Analysis Plan) reviewed by an independent methodologist
- [ ] Repository structure (Section 8.2) initialized
- [ ] `make reproduce` skeleton in place
- [ ] Random seed (20260429) hard-coded across all stages

---

## 15. End-of-Document Statement

This English version is the OSF-submission-ready translation of the **internal master preregistration draft** (the Japanese `D12_pre_registration_OSF.md`). Upon successful OSF registration (DOI acquisition), this preregistration becomes **locked**, and Stage 0 code implementation may begin. Subsequent modifications follow the Section 6.5 Level 3 procedure (analysis-plan revision).

**Lock status**: ⏳ DRAFT (pending self-review and OSF submission)
**Next step**: proceed sequentially through the items in Section 14.2.

---

**End of D12 OSF Pre-Registration v1.0 (English version draft).**









