# D12 OSF Pre-Registration — HEXACO 7-Typology Workplace Harassment Microsimulation (Phase 1 + Phase 2)

**Document type**: OSF Standard Pre-Registration draft (English version for OSF submission)
**Drafted**: 2026-04-29
**Branch**: `claude/hexaco-harassment-simulation-69jZp`
**Author (corresponding)**: Eisuke Tokiwa
**ORCID**: 0009-0009-7124-6669
**Affiliation**: SUNBLAZE Co., Ltd.
**Email**: eisuke.tokiwa@sunblaze.jp
**Status**: ⏳ DRAFT — to be finalized and registered on OSF **prior to Stage 0 code execution**
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
| **C3: Data Are Preexisting** | N=354 (harassment) and N=13,668 (clustering) are **prior IRB-approved data**. However, (a) the 7-type × gender 14-cell harassment cross-tabulation, (b) national-level aggregate predictions, and (c) Counterfactual A/B/C simulation outputs are **all unobserved**. | **Pure preregistration is achievable** for these unobserved analyses. Section 3.1 specifies who has observed what. |
| C1: Procedure changes during data collection | Simulation only; no new data collection | N/A |
| C2: Discovery of assumption violations during analysis | 14-cell main analysis uses frequentist bootstrap with BCa (light assumptions). 28-cell EB uses Beta-Binomial conjugate (method of moments + sensitivity sweep) for robustness. | Section 6.5 specifies deviation policy |
| **C6: Program-level null result reporting** | This study commits to publication even if MAPE > 60% (D-NEW8) | Section 7.3 codifies the commitment |
| C9: Selectivity in narrative inference | All Stage 3 sensitivity sweeps (V, f1, f2, EB strength, threshold, K) are pre-registered | Section 6.4 fixes the sweeps |

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

**The aggregate national prediction obtained by scaling 14-cell (7 type × 2 gender) conditional harassment propensities to the population reproduces the MHLW 2016 (pre-Power Harassment Prevention Law, 32.5% past-3-year harassment victimization rate) within MAPE ≤ 30%.**

- **Primary validation target**: MHLW 2016 R2 "Survey on Workplace Harassment" (past-3-year prevalence 32.5%)
- **Secondary validation targets**: MHLW 2020 R2 (31.4%, transition period), MHLW 2024 R5 (19.3%, post-law)
- **International baseline**: ILO (2022) Asia–Pacific lifetime prevalence 19.2%
- **Marginal-distribution check**: Tsuno et al. (2015) N = 1,546 random sample, 30-day prevalence 6.1%

#### H2 (Phase 1 baseline hierarchy)

**The mean absolute percentage error (MAPE) is monotonically non-increasing across the baseline hierarchy: B0 (random) ≥ B1 (gender only) ≥ B2 (HEXACO 6-domain linear) ≥ B3 (7 typology) ≥ B4 (B3 + age + industry estimate + employment type).**

- Quantities of interest: MAPE difference B3 − B2 (typology incrementality) and B4 − B3 (personality slice incrementality)

#### H3 (Phase 1 latent vs expressed gap)

**The gap between MHLW 2016 (pre-law, 32.5%) and our latent prediction is smaller than the gap between MHLW 2024 (post-law, 19.3%) and our latent prediction.** That is, the pre-law condition is closer to the latent rate, while the post-law condition shows stronger environmental gating.

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
| H1 | 1 | Confirmatory predictive | MAPE vs MHLW 2016 |
| H2 | 1 | Confirmatory ordinal | Monotonic ordering of B0–B4 MAPE |
| H3 | 1 | Exploratory descriptive | gap(2016) < gap(2024) |
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

