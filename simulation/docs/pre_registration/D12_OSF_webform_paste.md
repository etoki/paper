# OSF Standard Pre-Registration Web Form — Paste Sheet

**Source**: `D12_pre_registration_OSF.en.md` (English version, v1.1)
**Created**: 2026-04-29
**Last updated**: 2026-04-29 (synced with pre-reg v1.1: 3 new domestic surveys integrated — MHLW R5 full report, MHLW R4, Pasona N=28,135)
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
Workplace harassment is a multi-causal phenomenon. Organizational stressors (Bowling & Beehr 2006, ρ = .30–.53), subjective social status (Tsuno et al. 2015, OR = 4.21), industry composition, and legal/normative climate exert effects that are larger in magnitude than personality factors. The present study acknowledges these and instead isolates the personality contribution, asking how well a HEXACO 7-typology probabilistic model predicts Japanese workplace power harassment prevalence.

Phase 1 (descriptive simulation): Using existing N = 354 (harassment behavior) and N = 13,668 (HEXACO clustering) data, we estimate 14-cell (7 types × 2 genders) conditional harassment propensities, scale these via bootstrap to the Japanese workforce (~68 million), and triangulate the resulting latent prevalence against the Ministry of Health, Labour and Welfare (MHLW) national surveys' expressed prevalence. Primary success criterion: MAPE ≤ 30% against MHLW H28 (FY2016) past-3-year power harassment victimization rate of 32.5%. Secondary criterion (★ added v1.1): industry-stratified MAPE ≤ 50% against MHLW R5 (FY2023) industry-stratified power harassment rates (16-26.8% range). Triangulation against Pasona Research (2022) N=28,135 5-year prevalence 19.7%.

Phase 2 (intervention counterfactuals): Within the target trial emulation framework (Hernán & Robins 2020), we simulate three interventions: (A) universal HH (Honesty–Humility) intervention anchored in Kruse et al. (2014); (B) targeted high-risk-type intervention anchored in Hudson (2023), the primary intervention of interest; and (C) structural-only intervention anchored in Pruckner & Sausgruber (2013). For each, we estimate population-level harassment reduction.

No large language models are used. All mechanisms are transparent probability tables. The full preregistration document (English and Japanese versions) is attached as supplementary; the present field summarizes Sections 1.1–1.3 of that document.

Negative-result publication is committed in advance (Section 7): the author commits to journal submission regardless of whether MAPE is within or beyond the 30% / 60% thresholds. Target submission venue: Royal Society Open Science (Registered Report track), with seven fallback venues specified (Section 7.2).
```

(Source: EN Section 1.3 + brief synthesis from Sections 5.5, 7, 14)

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

## OSF Field 15 — Manipulated variables

```
This is a simulation study; "manipulated variables" refer to counterfactual operators applied to the probability tables (Pearl 2009 do-operator notation), not physical experimental manipulations.

Four manipulated variables are pre-specified:

1. δ_A (universal HH shift): Add δ × SD(HH) to HH score for all individuals, then re-classify clusters.
   - Main: +0.3 SD; sensitivity range: [0.1, 0.5] SD
   - Anchor: conservative discount of Kruse et al. (2014) d = 0.71

2. δ_B (targeted HH shift): Add δ × SD(HH) to HH score only for individuals in Clusters 0/4/6 (primary: Cluster 0; secondary: Clusters 4 and 6).
   - Main: +0.4 SD; sensitivity range: [0.2, 0.6] SD
   - Anchor: conservative discount of Hudson (2023) self-selected effect

3. effect_C (structural reduction): Multiply each cell-conditional probability by (1 − effect_C) for all 14 cells.
   - Main: 0.20; sensitivity range: [0.10, 0.30]
   - Anchor: triangulation of Pruckner & Sausgruber (2013) + Bezrukova et al. (2016) + Roehling & Huang (2018) + Dobbin & Kalev (2018)

4. transportability_factor: Multiply Phase 2 anchor effect by factor before applying to Japan.
   - Main: 1.0×; sensitivity sweep: {0.3×, 0.5×, 0.7×, 1.0×}
   - Anchor: Sapouna (2010) cultural moderator; Nielsen et al. (2017) Asia/Oceania attenuation

Full specification: Section 4.1 of the attached document.
```

(Source: EN Section 4.1)

---

## OSF Field 16 — Measured variables

```
Individual-level (N = 354 harassment data, harassment/raw.csv):
- HEXACO 6 domains (Wakabayashi 2014 Japanese HEXACO-60), continuous Likert 1–5 mean
- Dark Triad 3 (Shimotsukasa & Oshio 2017 SD3-J), continuous
- Power harassment (Tou et al. 2017), continuous → binarized at mean + 0.5 SD per outcome
- Gender harassment (Kobayashi & Tanaka 2010), continuous → binarized at mean + 0.5 SD
- Age (continuous, years), gender (binary, 0/1, n = 133/220), area (categorical) — all self-reported

Individual-level (N = 13,668 clustering data, clustering/csv/clstr_kmeans_7c.csv):
- HEXACO 6 domains (continuous; centroids already extracted in IEEE-published clustering paper)
- 7-cluster proportions (categorical; population scaling weight)

Population-level (MHLW + large-N domestic surveys, external validation):
- Past-3-year POWER HARASSMENT victimization rate: MHLW H28 FY2016 (32.5%, ★ primary), MHLW R2 FY2020 (31.4%), MHLW R5 FY2023 (19.3%, post-law, full report attached)
- Industry-stratified power harassment past-3-year rate: MHLW R5 (FY2023) 16 industries, range 16-26.8% (Construction 26.8% high, Education 16.9% low) — used for B4 H2.industry secondary validation
- Other harassment categories (for framing only): MHLW R5 sexual harassment 6.3% (women 8.9% / men 3.9%), customer harassment 10.8% (new category emerging post-FY2022 law amendment, used in latent vs expressed gap framing)
- 5-year harassment prevalence (industry survey): Pasona Research (2022) N=28,135, 5-year 19.7% (lifetime 34.6%), industry-stratified 16.9-22.9% — large-N harassment-specific marginal-distribution check
- 30-day prevalence (national-rep): Tsuno et al. 2015 N=1,546 random sample (6.1%)
- International baseline: ILO 2022 Asia–Pacific lifetime 19.2%
- f1 PRIMARY anchor (harassment-victim turnover rate): Pasona (2022) overall 10.3%, industry range 6.3-13.3%
- f1 SECONDARY anchor: MHLW R4 (FY2022) Employment Trend Survey, turnover by reason "workplace interpersonal" men 8.3% / women 9.4% (upper bound)
- f1 macro cross-check: Pasona (2022) annual harassment-induced turnover 865,000 persons/year (66% unreported / 暗数化)
- f2 anchor: MHLW Industrial Safety and Health Survey + Tsuno & Tabuchi 2022 PR = 3.20

Full specification: Section 4.2 of the attached document.
```

(Source: EN Section 4.2)

---

## OSF Field 17 — Indices

```
Derived indices:

- 7-type membership: each individual in N = 354 assigned to nearest centroid (Euclidean, HEXACO 6 domains).
- Cell ID (14-cell, main analysis): type ∈ {0..6} × gender ∈ {0, 1}.
- Cell ID (28-cell, sensitivity): type × gender × role ∈ {0, 1}.
- Role probability: continuous, predicted from C + 0.5·X composite (top 15% → manager); D1 sensitivity compares 3 alternative models — (a) personality linear, (b) tree-based, (c) literature-based.
- National latent prevalence: P̂ = Σ_cell (cell propensity × cell population weight).
- MAPE: mean(|predicted − observed| / observed × 100); primary metric in Stage 2.
- ΔP_x (counterfactual reduction): predicted_baseline − predicted_counterfactual_x.
- Cost-effectiveness ratio (Phase 2 only): ΔP_x / N_treated_x.

Full specification: Section 4.3 of the attached document.
```

(Source: EN Section 4.3)

---

## OSF Field 18 — Statistical models

```
Phase 1:

Stage 0 cell-level propensity (14-cell main):
- For each cell c ∈ {1..14}: observed propensity p̂_c = X_c / N_c, where X_c is the count of binary "harassment perpetrator" cases (binarized at mean + 0.5 SD per outcome).
- Bootstrap distribution: B = 2,000 BCa resamples per cell (Efron 1987 J Am Stat Assoc; DiCiccio & Efron 1996 Stat Sci); BCa correction uses bias z₀ + acceleration a from jackknife.

Stage 0 sensitivity (28-cell EB shrinkage):
- Beta-Binomial conjugate (Casella 1985 + Clayton & Kaldor 1987 + Efron 2014 + Greenland 2000).
- Hyperprior estimated by method of moments from the 14-cell main: α̂ = μ̂ × [μ̂(1−μ̂)/σ̂² − 1], β̂ = (1−μ̂) × [μ̂(1−μ̂)/σ̂² − 1].
- 28-cell posterior: E[p_k | X_k, N_k] = (α̂ + X_k) / (α̂ + β̂ + N_k); 95% CI from Beta(α̂ + X_k, β̂ + N_k − X_k) quantiles.
- Strength sensitivity sweep at scale ∈ {0.5×, 1.0× main, 2.0×}.
- MoM stability diagnostic: Marginal MLE and Stan / brms hierarchical Bayesian posteriors run as auxiliary triangulation.

Stage 1 population aggregation:
- For each validation period t ∈ {2016, 2020, 2024}: P̂_t = Σ_c (p̂_c × W_c) / Σ_c W_c, where W_c is the cell weight (MHLW labor-force population × cluster proportion × gender proportion × age weight).
- Bootstrap CI for P̂_t: 2,000 iterations, BCa.

Stage 4 baseline hierarchy: B0 uniform, B1 gender-only logistic, B2 HEXACO 6-domain logistic, B3 7 type × gender cell-conditional (proposed), B4 = B3 + age + industry (PROBABILISTICALLY ESTIMATED from MHLW Labor Force 2022 industry × age × gender × employment-type crosstabs into 16 industry buckets) + employment type cell-conditional.

Industry estimation specification (★ preregistered, ★ added v1.1): N=354 contains no direct industry data. For each individual i with (age_i, gender_i, employment_i), assign a 16-bucket industry probability vector from MHLW Labor Force 2022. Cell predictions in B4 are weighted by this probability vector. Industry-level predictions are inherently noisy (industry not directly observed); B4 industry-stratified MAPE is validated against MHLW R5 (FY2023) industry-stratified power harassment rates (16-26.8%) at a relaxed threshold of 50% (H2.industry secondary criterion).

Stage 5 CMV diagnostic: Harman's single-factor unrotated EFA (Podsakoff et al. 2003); marker variable correction with HEXACO Openness (Lindell & Whitney 2001).

Phase 2 counterfactual estimation:
- Apply do-operator to N = 354 / cell-probability table per Field 15 specification.
- Re-run Stage 0 → Stage 1 (Stage 2 validation omitted; only prediction).
- ΔP_x = P̂_baseline − P̂_x.
- Bootstrap CI for ΔP_x propagating cell-level uncertainty (2,000 iterations).
- Cost-effectiveness for B: ΔP_B / |Cluster 0 ∪ 4 ∪ 6 in population|.

Identifying assumptions of target trial emulation (exchangeability, positivity, consistency, transportability) are explicitly assessed in the Discussion (Section 5.7.4 of the attached document).

Full specification: Section 5 of the attached document.
```

(Source: EN Section 5.1–5.7)

---

## OSF Field 19 — Transformations

```
Outcome binarization: harassment scale scores are binarized at mean + 0.5 SD per outcome (main); sensitivity sweep at mean + 0.25 SD and mean + 1.0 SD.

Cluster assignment: each individual's HEXACO 6-domain score vector is assigned to the nearest of 7 centroids by Euclidean distance.

Population reweighting: cell-level estimates are scaled to the Japanese workforce by MHLW Labor Force Survey weights (age × gender × employment type).

No log transforms, square roots, or other nonlinear transformations are applied to the primary HEXACO scores. The HEXACO domain scores are used as the standard Likert 1–5 mean (research plan Part 11.1).

Random seed 20260429 governs all stochastic operations (bootstrap, Monte Carlo, simulated assignment).
```

(Source: EN Sections 2.3, 5.1, 4.1)

---

## OSF Field 20 — Inference criteria

```
H1 (primary):
- MAPE(P̂_FY2016, MHLW H28 FY2016 past-3-year POWER HARASSMENT 32.5%) ≤ 30% → SUCCESS
- 30% < MAPE ≤ 60% → PARTIAL SUCCESS
- MAPE > 60% → FAILURE (publish as failure-mode discovery; Section 7)

H2: MAPE_B0 ≥ MAPE_B1 ≥ MAPE_B2 ≥ MAPE_B3 ≥ MAPE_B4. Direction confirmed if at least 3 of the 4 pairwise inequalities hold. Bonferroni–Holm correction at family-wise α = .05.

H2.industry (★ added v1.1): B4 industry-stratified (16 buckets) MAPE vs MHLW R5 (FY2023) industry-stratified power harassment rates (16-26.8% range) ≤ 50% (relaxed threshold given industry estimation noise) → CONFIRMED. > 50% → REPORTED as honest limitation. Secondary criterion only; does not affect H1 SUCCESS / FAILURE.

Stage 2 chain output sanity check (★ added v1.1): Predicted annual harassment-induced turnover via V × f1 chain compared against Pasona (2022) macro estimate 865,000/year. Soft criterion: predicted within 50-200% range (430,000 - 1,730,000). Outside range flagged for failure-mode discussion. SECONDARY only.

H3: gap(MHLW H28 FY2016) < gap(MHLW R5 FY2023). Confirmed if MAPE_FY2016 < MAPE_FY2023.

H4: sign(ΔP_A) is negative. Confirmed if 95% CI excludes 0 in the negative direction.

H5: sign(ΔP_B) negative AND ΔP_B / N_treated > ΔP_A / N_total. Confirmed if both conditions hold.

H6: sign(ΔP_C) is negative. Confirmed if 95% CI excludes 0.

H7: ΔP_B > ΔP_A AND ΔP_B > ΔP_C. Confirmed if both inequalities hold at point estimates; flagged as uncertain if 95% CIs overlap.

Multiple-comparison correction: Bonferroni–Holm at family-wise α = .05 for H2 (4 ordinal pairwise tests) and H4–H7 (3 counterfactual main tests). H1 is a single primary test (no correction). H7 is a composite single test.

Failure-mode commitments:
- H1 failure (MAPE > 60%): publish as failure-mode discovery.
- H2 reversal (B3 < B2): publish as critical finding (typology overfitting).
- H7 reversal (ΔP_B ≤ ΔP_A or ΔP_B ≤ ΔP_C): publish; revise main thesis claim.

Post-hoc revision of MAPE thresholds (30% / 60%) is prohibited. Comparison against MHLW survey is performed only after this preregistration is registered on OSF.

Full specification: Section 6 of the attached document.
```

(Source: EN Section 6.1–6.6)

---

## OSF Field 21 — Data exclusion

```
None at the analytic level. The N = 354 harassment dataset is used in full as released in the Tokiwa harassment preprint. The N = 13,668 clustering dataset is used in full as released in the Tokiwa clustering paper (IEEE-published).

Cluster 6 (population-dominant ~32%) is intentionally NOT excluded from any counterfactual analysis, in order to preserve the positivity assumption of target trial emulation (Section 5.7.4).

If a participant is missing one or more HEXACO domain scores, see Field 22.
```

(Source: EN Sections 3.1, 5.7.4)

---

## OSF Field 22 — Missing data

```
Existing datasets are used as released. Any rows with missing HEXACO domain scores in N = 354 are handled as follows:
- Cluster assignment uses Euclidean distance over the available 6 HEXACO domain scores. Rows missing more than 1 of the 6 domains are flagged in a sensitivity supplementary; if their proportion exceeds 5% of N = 354, an additional sensitivity analysis is run that excludes them from the 14-cell crosstab.
- Bootstrap resampling treats missingness as observed (no imputation): a row missing one domain is left out of cluster reassignment in that resample.
- No multiple imputation is performed at the main analysis stage to avoid introducing a model-based covariance structure not specified in this preregistration.

For population aggregation (Stage 1) and for MHLW external validation (Stage 2), any missing cells in MHLW supplementary tables are reported as such; no synthetic values are inserted.

If empirical missingness patterns require deviation, the deviation is logged per Section 6.5 of the attached document.
```

(Source: EN Section 6.5; new specification consistent with Section 5)

---

## OSF Field 23 — Exploratory analysis

```
The following analyses are pre-specified but interpreted as exploratory rather than confirmatory:

1. 28-cell EB sensitivity (Section 5.2): treated as sensitivity rather than as a confirmatory test of role × type interactions, because pairwise MDE in 28-cell is too large for confirmatory inference (D13 power analysis).
2. Stage 5 CMV diagnostic (Section 5.6): exploratory; Harman's single-factor first-factor variance < 50% is the preregistered "concern is limited" threshold.
3. Stage 4 B4 baseline (Section 5.5): exploratory comparison of personality slice incrementality with peripheral covariates; treated as evidence quality rather than confirmatory ranking.
4. Subgroup MAPE by gender × age band (Section 5.4): exploratory failure-mode localization.
5. HEXACO-domain-level associations (Section 3.1.3): interpreted as exploratory replication of the Tokiwa harassment preprint; the preregistered analyses are the type-conditional, cell-level cross-tabulations and the national aggregate predictions.
6. Discriminant validity check on the harassment scale vs depression / job stress / general negative affect (Section 4.2 limitations L11): exploratory construct validity check; if the relevant correlation data are unavailable in N = 354, HEXACO Emotionality is used as a proxy.

Any analysis not enumerated in Sections 5, 6, or here is exploratory and will be reported as such.
```

(Source: EN Sections 5.2, 5.4–5.6, 3.1.3)

---

## OSF Field 24 — Other (anything else relevant)

```
1. Sole-authored study. Author: Eisuke Tokiwa, ORCID 0009-0009-7124-6669, SUNBLAZE Co., Ltd., eisuke.tokiwa@sunblaze.jp.

2. Independent methodologist consultation (Section 1.2 + Section 8.1): An external methodologist with a mathematical biology background will review Section 5 (Analysis Plan) prior to Stage 2 validation. Acknowledged in anonymous form per the methodologist's preference (Munafò et al. 2017 Box 1, lightweight variant).

3. Funding: No external funding for the simulation phase. Original data collection (N = 354 / N = 13,668) inherits the IRB approvals and funding documented in the prior Tokiwa publications (harassment preprint and clustering paper IEEE-published). No new IRB is required (secondary analysis of anonymized data).

4. Conflicts of interest: The author derives no commercial benefit from the findings. The author's affiliation has no commercial interest in the results.

5. Negative-result publication commitment (D-NEW8, Section 7): The author commits to journal submission in all five enumerated cases (success / partial / failure / H7 reversal / B3 < B2 overfitting). Target submission: Royal Society Open Science (Registered Report track, primary), with seven fallback venues specified (Section 7.2). Choice of journal is not locked by this preregistration; a change of journal does not constitute a deviation under Section 6.5.

6. Reproducibility commitments (D-NEW9, Section 8): Random seed 20260429 fixed; environment pinning via `uv` lock or Dockerfile; `make reproduce` 30-minute regeneration target; open code (MIT license) on GitHub and OSF mirror; aggregated open data; restricted-access raw data only where re-identification risk exists (Section 9.5). The study targets Tier 3 / "Verification" of the TOP guidelines and qualifies for OSF Open Practice Badges (Open Data, Open Materials, Preregistered).

7. Ethics commitments (D-NEW10, Section 9): Triple-locking of the anti-screening statement (Methods + Discussion + this preregistration). Voluntary, opt-in policy principle for Counterfactual B (no employment consequence; anonymity / confidentiality; resource provision rather than coercion). Long-term ethical monitoring: this preregistration and supplementary materials remain active on OSF for at least 10 years; the author will publish corrections or commentary if findings are misappropriated for discrimination or screening.

8. Limitations pre-acknowledged (Section 10): 11 limitations are enumerated in advance to prevent post-hoc rationalization (Nosek 2018 Challenge 9). Major items: cross-sectional design precludes causal inference (mitigated by Roberts & DelVecchio 2000 plateau r = .74); pairwise MDE Cohen's d ≥ 0.92 precludes individual-level inference (D13 power analysis); transportability of Western-anchor effects to Japan is sensitivity-tested at {0.3×, 0.5×, 0.7×, 1.0×}.

9. Reflexivity statement (Section 11): The author has previously argued in non-peer-reviewed work for a systemic-causation framing of workplace harassment. This normative stance is acknowledged as a potential bias source. To mitigate: the empirical analysis is restricted to L1 descriptive/predictive claims; all causal language is constrained by target trial emulation with explicit identifying assumptions; anti-screening and anti-discrimination statements are included regardless of findings.

10. Companion documents (attached as supplementary):
- `D12_pre_registration_OSF.md` (Japanese master, 1,004 lines)
- `D12_pre_registration_OSF.en.md` (English version, 1,021 lines)
- `D12_pre_registration_OSF.pdf` (Japanese PDF, 21 pages)
- `D12_pre_registration_OSF.en.pdf` (English PDF, 22 pages)
- `D12_pre_registration_OSF` GitHub mirror: in the `claude/hexaco-harassment-simulation-69jZp` branch of the repository

11. After registration: the OSF DOI is recorded in the Header of both the JP master and the EN version markdown documents, locking the preregistration. Subsequent modifications follow Section 6.5 deviation policy (Levels 0–3); Level 3 changes require a v2 registration on OSF with a public diff against v1.
```

(Source: EN Sections 1.2, 7, 8, 9, 10, 11, 12, 14)

---

## After-registration steps (post-form submission)

1. **Issue OSF DOI**: After clicking "Register" in OSF, the DOI is issued (typically within minutes, but may take up to 48 hours for embargoed registrations).

2. **Update Header in both markdown documents** (Section 14.3 lock procedure):
   - Edit `simulation/docs/pre_registration/D12_pre_registration_OSF.md`: add `**OSF DOI**: 10.17605/OSF.IO/<your-id>` and `**Registered**: <date>` to the Header.
   - Edit `simulation/docs/pre_registration/D12_pre_registration_OSF.en.md`: same.
   - Set `**Status**: 🔒 LOCKED` (replacing `⏳ DRAFT`).

3. **Regenerate PDFs** to reflect the locked Header:
   ```
   cd simulation/docs/pre_registration
   make
   ```

4. **Commit and push**:
   ```
   git add simulation/docs/pre_registration/
   git commit -m "D12 pre-reg: OSF registration locked, DOI <id>"
   git push
   ```

5. **Send the methodologist contact request** (Section 1.2 / 8.1; mode B anonymous): include the OSF DOI in the request so the reviewer can verify the locked analysis plan.

6. **Stage 0 implementation may begin** once Section 14.3 final checks are complete (DOI added, repository structure initialized, `make reproduce` skeleton in place, random seed hard-coded across all stages).

---

## Field-to-Section cross-reference quick view

| OSF Field | EN doc Section | Content type |
|---|---|---|
| 1 Title | 1.1 | Single-line title |
| 2 Description | 1.3 + 7 + 14 | Abstract |
| 3 Hypotheses | 1.4 + 6.1 | H1–H7 |
| 4 Study type | 2.1 | "Other" + clarification |
| 5 Blinding | 2.2 + 3.1 | "No blinding" + preregistration-equivalent state |
| 6 Additional blinding | 1.2 + 2.2 | None |
| 7 Study design | 2.3 | 5-stage Phase 1 + 3-stage Phase 2 pipeline |
| 8 Randomization | 2.4 | Fixed seed 20260429 |
| 9 Existing data | 3.1 | "Registration prior to analysis" + partial blinding |
| 10 Explanation of existing data | 3.1 | Two datasets + unobserved analyses |
| 11 Data collection procedures | 12.3 | Not applicable + public sources |
| 12 Sample size | 3.2 | 14-cell main + 28-cell sensitivity |
| 13 Sample size rationale | 3.3 | D13 power analysis |
| 14 Stopping rule | 3.4 | Not applicable |
| 15 Manipulated variables | 4.1 | δ_A, δ_B, effect_C, transportability_factor |
| 16 Measured variables | 4.2 | Individual + population-level |
| 17 Indices | 4.3 | 7-type membership, MAPE, ΔP_x, etc. |
| 18 Statistical models | 5.1–5.7 | Bootstrap, EB, baselines, counterfactuals |
| 19 Transformations | 2.3, 5.1, 4.1 | Binarization + reweighting |
| 20 Inference criteria | 6.1–6.6 | H1–H7 thresholds + Bonferroni–Holm |
| 21 Data exclusion | 3.1, 5.7.4 | None at analytic level |
| 22 Missing data | 6.5 | Listwise within cluster reassignment |
| 23 Exploratory analysis | 5.2, 5.4–5.6, 3.1.3 | 6 enumerated exploratory items |
| 24 Other | 1.2, 7, 8, 9, 10, 11, 12, 14 | 11-item summary |

---

**End of OSF Standard Pre-Registration paste sheet.**




