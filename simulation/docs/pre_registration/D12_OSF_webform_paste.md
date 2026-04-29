# OSF Standard Pre-Registration Web Form — Paste Sheet

**Source**: `D12_pre_registration_OSF.en.md` (English version)
**Created**: 2026-04-29
**Branch**: `claude/hexaco-harassment-simulation-69jZp`
**Author**: Eisuke Tokiwa (sole-authored, ORCID: 0009-0009-7124-6669)

---

## How to use this sheet

1. Log in to OSF, create a new project (e.g., "HEXACO 7-Typology Workplace Harassment Microsimulation").
2. **Upload supplementary files first** to the project:
   - `D12_pre_registration_OSF.pdf` (Japanese master)
   - `D12_pre_registration_OSF.en.pdf` (English version)
   - `D12_pre_registration_OSF.md` (Japanese markdown source)
   - `D12_pre_registration_OSF.en.md` (English markdown source)
3. Click **"New Registration"** → select **"OSF Preregistration"** template (also called "OSF Standard Pre-Registration").
4. For each field below, copy the labeled content into the corresponding OSF web form field.
5. After registering, the OSF DOI is issued. Record the DOI in the Header of both the JP and EN markdown documents (Section 14.3 lock procedure).

> **Tip**: Most OSF fields accept Markdown formatting (tables, headers, bold). Paste blocks below are formatted accordingly.

---

## OSF Field 1 — Title

```
HEXACO 7-Typology Workplace Harassment Microsimulation: Latent Prevalence Prediction and Target Trial Emulation of Personality-Based and Structural Counterfactuals in Japan
```

(Source: EN Section 1.1)

---

## OSF Field 2 — Description / Abstract

```
Workplace harassment is a multi-causal phenomenon. Organizational stressors (Bowling & Beehr 2006, ρ = .30–.53), subjective social status (Tsuno et al. 2015, OR = 4.21), industry composition, and legal/normative climate exert effects that are larger in magnitude than personality factors. The present study acknowledges these and instead isolates the personality contribution, asking how well a HEXACO 7-typology probabilistic model predicts Japanese workplace harassment prevalence.

Phase 1 (descriptive simulation): Using existing N = 354 (harassment behavior) and N = 13,668 (HEXACO clustering) data, we estimate 14-cell (7 types × 2 genders) conditional harassment propensities, scale these via bootstrap to the Japanese workforce (~68 million), and triangulate the resulting latent prevalence against the Ministry of Health, Labour and Welfare (MHLW) national surveys' expressed prevalence. Primary success criterion: MAPE ≤ 30% against MHLW 2016 (32.5%).

Phase 2 (intervention counterfactuals): Within the target trial emulation framework (Hernán & Robins 2020), we simulate three interventions: (A) universal HH (Honesty–Humility) intervention anchored in Kruse et al. (2014); (B) targeted high-risk-type intervention anchored in Hudson (2023), the primary intervention of interest; and (C) structural-only intervention anchored in Pruckner & Sausgruber (2013). For each, we estimate population-level harassment reduction.

No large language models are used. All mechanisms are transparent probability tables. The full preregistration document (English and Japanese versions) is attached as supplementary; the present field summarizes Sections 1.1–1.3 of that document.

Negative-result publication is committed in advance (Section 7): the author commits to journal submission regardless of whether MAPE is within or beyond the 30% / 60% thresholds. Target submission venue: Royal Society Open Science (Registered Report track), with seven fallback venues specified (Section 7.2).
```

(Source: EN Section 1.3 + brief synthesis from Sections 7 and 14)

---

## OSF Field 3 — Hypotheses

```
This preregistration commits to seven hypotheses (H1–H7), summarized below. Full operationalization, decision rules, and inference criteria are in the attached preregistration document (Section 1.4 + Section 6).

H1 (Phase 1 main hypothesis):
The aggregate national prediction obtained by scaling 14-cell (7 type × 2 gender) conditional harassment propensities to the Japanese workforce reproduces the MHLW 2016 (pre-Power Harassment Prevention Law) past-3-year harassment victimization rate of 32.5% within MAPE ≤ 30%.
- Primary validation target: MHLW 2016 R2 (32.5%)
- Secondary validation targets: MHLW 2020 R2 (31.4%), MHLW 2024 R5 (19.3%)

H2 (Phase 1 baseline hierarchy):
The mean absolute percentage error (MAPE) is monotonically non-increasing across the baseline hierarchy:
B0 (random) ≥ B1 (gender only) ≥ B2 (HEXACO 6-domain linear) ≥ B3 (7 typology) ≥ B4 (B3 + age + industry estimate + employment type).

H3 (Phase 1 latent vs expressed gap):
The gap between MHLW 2016 (pre-law, 32.5%) and our latent prediction is smaller than the gap between MHLW 2024 (post-law, 19.3%) and our latent prediction.

H4 (Phase 2 Counterfactual A: Universal HH intervention):
Under a +0.3 SD population-wide HH shift (a conservative discount of Kruse 2014's d = 0.71), the predicted national prevalence decreases by ΔP_A relative to baseline.
- Direction: ΔP_A < 0 (reduction); sensitivity range δ ∈ [0.1, 0.5] SD

H5 (Phase 2 Counterfactual B: Targeted intervention — primary intervention of interest):
Targeting high-risk types (primary: Cluster 0 = Self-Oriented Independent profile; secondary: Clusters 4 and 6) with a +0.4 SD HH shift yields a reduction ΔP_B such that the cost-effectiveness ratio (ΔP_B / number treated) exceeds that of Counterfactual A.
- Direction: ΔP_B < 0 AND ΔP_B / N_treated > ΔP_A / N_total; sensitivity range δ ∈ [0.2, 0.6] SD

H6 (Phase 2 Counterfactual C: Structural-only intervention):
Reducing all cell-conditional probabilities by 20% while leaving individual personality unchanged yields a reduction ΔP_C, but ΔP_C is smaller in magnitude than ΔP_B from Counterfactual B.
- Direction: ΔP_C < 0; sensitivity range effect_C ∈ [0.10, 0.30]

H7 (Phase 2 main contrast — primary predictive commitment):
ΔP_B > ΔP_A AND ΔP_B > ΔP_C (targeted intervention exceeds both universal and structural-only interventions in population-level reduction).

The author commits in advance (Section 7) to publish the study even if H1 fails (MAPE > 60%), if the H2 monotonicity is reversed, or if H7 is reversed. Inference criteria, multiple-comparison correction (Bonferroni–Holm), and deviation policy are specified in Sections 6.1–6.5 of the attached document.
```

(Source: EN Sections 1.4 + 6.1)

---

## OSF Field 4 — Study type

```
Observational study (secondary analysis of preexisting IRB-approved data) combined with computational microsimulation and counterfactual projection. No new data collection. No experimental manipulation of human participants. No large language models or generative agents. All mechanisms use transparent probability tables, Monte Carlo bootstrap, and Empirical Bayes shrinkage (Beta-Binomial conjugate, method of moments). Causal framing follows the target trial emulation framework (Hernán & Robins 2020) and structural causal model do-operator notation (Pearl 2009).
```

(If OSF presents a multiple-choice control: select **"Other"** and paste the above into the "Please describe" field.)

(Source: EN Section 2.1)

---

## OSF Field 5 — Blinding

If the OSF form offers a multiple-choice control:
- Select: **"No blinding is involved in this study."**

Then paste the following clarification into any associated text field (or under "Other") to record the preregistration-equivalent blinding state:

```
Strict experimental blinding is not applicable to this secondary-analysis simulation study. However, a preregistration-equivalent blinding state is enforced at the analysis level:

- Already observed at registration time: individual-level HEXACO scores (N = 354, N = 13,668) and individual-level harassment self-reports (N = 354). These were used in the Tokiwa harassment preprint (HC3-robust hierarchical regression) and the Tokiwa clustering paper (IEEE-published, 7-type centroids).

- Unobserved at registration time, fixed by this preregistration: the 7 type × gender 14-cell harassment cross-tabulation; the national aggregate latent prevalence; counterfactual A / B / C ΔP estimates; MAPE values against MHLW surveys.

This corresponds to Nosek 2018 PNAS Challenge 3 ("Data Are Preexisting") with partial blinding, honestly acknowledged in Section 3.1.3 of the attached document. All inference criteria (Section 6.1), sensitivity sweeps (Section 6.4), and MAPE thresholds (30% / 60%) are fixed by this preregistration before any cross-tabulation, aggregation, or comparison with MHLW data is performed.
```

(Source: EN Sections 2.2 + 3.1)

---

## OSF Field 6 — Is there any additional blinding?

```
None beyond the preregistration-equivalent state described in Field 5. The author has no co-authors and no organizational embargoes that would impose additional blinding.
```

(Source: EN Section 1.2 + 2.2)

---

## OSF Field 7 — Study design

```
Two-phase computational microsimulation with target trial emulation.

Phase 1 (descriptive simulation, 5 stages):
- Stage 0: Type assignment & probability table construction. Each of N = 354 is assigned to the nearest of 7 centroids (Euclidean, HEXACO 6 domains; centroids from N = 13,668 IEEE-published clustering paper). 14-cell (7 type × 2 gender) crosstab is computed for binary harassment outcomes (binarized at mean + 0.5 SD per outcome). Bootstrap B = 2,000 iterations per cell with BCa CI (Efron 1987).
- Stage 1: Population aggregation via cell-conditional probabilities × MHLW Labor Force Survey weights (~68 million worker base).
- Stage 2: Validation triangulation against MHLW 2016 (32.5%, primary), 2020 (31.4%), 2024 (19.3%); metrics include MAPE (primary), Pearson r, Spearman ρ, KS distance, Wasserstein distance, calibration plot.
- Stage 3: Sensitivity sweeps (V, f1, f2, EB shrinkage strength, binarization threshold, cluster K, role estimation models — fully enumerated in Section 6.4).
- Stage 4: Baseline hierarchy comparison B0 (random) → B1 (gender) → B2 (HEXACO 6-domain linear) → B3 (7 typology, proposed) → B4 (B3 + age + industry + employment).
- Stage 5: CMV diagnostic (Harman's single-factor + marker-variable correction).

Phase 2 (intervention counterfactuals, 3 stages):
- Stage 6: Target trial emulation specification (PICO + 24-week duration following Roberts 2017) for each counterfactual.
- Stage 7: Counterfactual simulation:
  - A (universal): do(HH := HH + δ_A × SD(HH)) for all individuals; main δ = +0.3 SD.
  - B (targeted, primary): do(HH := HH + δ_B × SD(HH)) only for individuals in Clusters 0/4/6; main δ = +0.4 SD.
  - C (structural): do(p_c := p_c × (1 − effect_C)) for all cells; main effect_C = 0.20.
- Stage 8: Transportability sensitivity (Western anchor effect × {0.3×, 0.5×, 0.7×, 1.0×}).

The four identifying assumptions of target trial emulation (exchangeability, positivity, consistency, transportability) are made explicit in Section 5.7.4 and honestly assessed in the Discussion. Pearl (2009) do-operator notation is used.

Full pipeline diagrams: Section 2.3 of the attached preregistration. Statistical models: Section 5. Inference criteria and sensitivity master table: Section 6.
```

(Source: EN Section 2.3)

---

## OSF Field 8 — Randomization

```
No physical randomization, as this is an observational secondary analysis combined with simulation. Within the simulation:

- Bootstrap resampling and Monte Carlo runs are made deterministic via a fixed random seed (NumPy default_rng(seed=20260429), Python random.seed(20260429), Stan seed=20260429). Bootstrap resample states are persisted to HDF5.
- Counterfactual A / B / C are simulated as if randomly assigned within the target trial emulation framework; this is documented as "simulated random" assignment in Section 2.3 (Stage 6) of the attached document.

The seed value 20260429 is fixed by this preregistration and is not to be changed.
```

(Source: EN Section 2.4)

---

## OSF Field 9 — Existing data

If OSF presents a multiple-choice control with the standard five options, select:
- **"Registration prior to analysis of the data"** (closest match)

Rationale: individual-level HEXACO and harassment data have been observed by the author for prior publications (Tokiwa harassment preprint and Tokiwa clustering paper IEEE-published); however, the 7 type × gender × harassment cell cross-tabulation, the national aggregate prediction, and the counterfactual outputs have not been computed at the time of registration. This is the partial-blinding state of Nosek 2018 PNAS Challenge 3, described fully in Field 10.

---

## OSF Field 10 — Explanation of existing data

```
Two preexisting datasets are used:

(1) N = 354 harassment data (`harassment/raw.csv`). Collected and analyzed in the Tokiwa harassment preprint via HC3-robust hierarchical regression of HEXACO + Dark Triad on power and gender harassment. The author has observed: HEXACO 6 domain scores, Dark Triad 3 scores, the Tou et al. (2017) Workplace Power Harassment Scale, the Kobayashi & Tanaka (2010) Gender Harassment Scale, age, gender, and area at the individual level.

(2) N = 13,668 clustering data (Tokiwa clustering paper, IEEE-published). The 7-type centroids and cluster proportions are observed at the aggregate level (centroid table fixed in `clustering/csv/clstr_kmeans_7c.csv`).

Unobserved at registration time (the analyses fixed by this preregistration):
- Distribution of 7-type membership obtained by assigning N = 354 to the nearest of 7 centroids.
- 14-cell (7 type × 2 gender) crosstabulated harassment binary outcomes.
- Cell-conditional propensities with bootstrap BCa CIs.
- 28-cell EB-shrunken estimates (sensitivity).
- National-level aggregate latent prevalence.
- MAPE against MHLW national surveys (2016 / 2020 / 2024).
- Counterfactual A / B / C ΔP estimates.

Honest acknowledgment of partial blinding (Section 3.1.3 of the attached document): the author has prior knowledge of HEXACO-domain-level associations with harassment self-reports from the regression analyses in the harassment preprint, but the type-conditional cell-level propensity table and national-level aggregate predictions cannot be derived from those regression results. The interpretation distinguishes between (a) preregistered analyses (the 7-type cross-tabulation and aggregation pipeline) and (b) exploratory replications of prior HEXACO-domain associations.

No new IRB is required: this is a secondary analysis of anonymized, IRB-approved data.
```

(Source: EN Section 3.1)

---

## OSF Field 11 — Data collection procedures

```
Not applicable: no new data collection. The two preexisting datasets (N = 354 harassment data; N = 13,668 clustering data) were collected and IRB-approved for the prior Tokiwa publications cited in Field 10. This preregistration covers only the secondary analysis and simulation pipeline, not new collection.

Public statistics used as external validation targets are downloaded from MHLW (`https://www.mhlw.go.jp/`) and Statistics Bureau of Japan (`https://www.stat.go.jp/data/roudou/`):
- MHLW 2016 R2, 2020 R2, 2024 R5 Surveys on Workplace Harassment (primary validation)
- MHLW Employment Trend Survey (turnover by reason; f1 anchor)
- MHLW Industrial Safety and Health Survey (mental disorder incidence; f2 anchor)
- MHLW Labor Force Survey (population reweighting)
- ILO 2022 Global survey (international baseline)
```

(Source: EN Section 12.3)

---

## OSF Field 12 — Sample size

```
Phase 1 main analysis: N = 354 individuals partitioned into 14 cells (7 HEXACO types × 2 genders).
- Cell N: minimum 10, maximum 70, median 18.
- 0 cells with N < 10 (no shrinkage required for main analysis).
- 7 cells (50%) with N < 20.

Phase 1 sensitivity analysis: 28 cells (7 type × 2 gender × 2 role).
- 16 cells (57%) with N < 10; 9 cells with N ≤ 3; 4 cells with N = 0.
- Empirical Bayes shrinkage (Beta-Binomial conjugate, method of moments) is mandatory for this sensitivity tier.

Population aggregation: ~68 million Japanese workers aged 20–64 (MHLW Labor Force Survey base).

Phase 2 counterfactual targets: Cluster 0 (primary, ~6.5% of N = 354), Clusters 4 (~14.4%) and 6 (~32.2%) (secondary). Population-level targets are scaled accordingly.

Bootstrap iterations: B = 2,000 per cell (BCa CI). Random seed 20260429.
```

(Source: EN Section 3.2)

---

## OSF Field 13 — Sample size rationale

```
Drawn from the D13 power analysis (`simulation/docs/power_analysis/D13_power_analysis.md`, attached as supplementary):

- N = 354 satisfies Funder & Ozer (2019)'s recommendation of N ≥ 250 for stable r estimation at the aggregate level.
- The 14-cell main analysis satisfies N ≥ 10 in every cell, allowing bootstrap estimation without Empirical Bayes shrinkage.
- Pairwise minimum detectable effect (Cohen's d ≥ 0.92) is "very large" by Cohen (1988) and "rarely found in replication" per Funder & Ozer (2019); accordingly, cell-level pairwise inference is avoided as a preregistered limitation, and aggregate-level inference is the primary inferential target.
- The 28-cell sensitivity tier with 16 small cells (N < 10) requires Empirical Bayes shrinkage with the Beta-Binomial conjugate (method of moments), with a strength sweep at scale ∈ {0.5×, 1.0× main, 2.0×}.

Sample size is fixed by the existing data and is not adjustable.
```

(Source: EN Section 3.3 + power analysis report)

---

## OSF Field 14 — Stopping rule

```
Not applicable: no new data collection.

Sensitivity sweeps are restricted to the ranges fixed by Section 6.4 of the attached document. Any post-registration extension of sensitivity ranges will be flagged as exploratory and excluded from confirmatory inference.
```

(Source: EN Section 3.4)

---



