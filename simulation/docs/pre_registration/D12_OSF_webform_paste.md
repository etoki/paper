# OSF Standard Pre-Registration Web Form — Paste Sheet

**Source**: `D12_pre_registration_OSF.en.md` (English version, v1.1)
**Created**: 2026-04-29
**Last updated**: 2026-04-30 (Part B added: live OSF webform transcription drafts synced from chat session, covering newer OSF template fields not in original 24-field structure)
**Branch**: `claude/hexaco-harassment-simulation-69jZp`
**Author**: Eisuke Tokiwa (sole-authored, ORCID: 0009-0009-7124-6669)

> **Structure note**: This document has TWO parts:
> - **Part A (Sections 1-24)**: original OSF Standard Pre-Registration template (24 fields, older / classic OSF webform)
> - **Part B (Sections B0-B21)**: current OSF webform transcription drafts (newer template with Foreknowledge / Subjects / Causal interpretation / etc.)
>
> If your OSF webform shows the newer fields, use Part B. If it shows the classic 24-field structure, use Part A. Field content is consistent between parts; Part B includes additional newer fields not present in Part A.

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

**End of Part A (OSF Standard 24-field paste sheet).**

---

# Part B: Current OSF Webform Transcription (newer template)

> **Note**: This Part contains paste-ready content for OSF webform fields that appear in the newer (2024+) OSF Preregistration template but are not in the classic 24-field structure of Part A. Use these alongside Part A as needed; field content overlaps where both parts apply, with Part B versions tailored to the actual webform wording observed during transcription.

---

## OSF Field B1 — Foreknowledge of data and evidence (radio button)

**Selection**: **"Authors have observed the data, but have not performed the proposed analyses"**

Rationale (matches preregistration Section 3.1.3 honest acknowledgment):
- Author has observed individual-level HEXACO data and harassment self-reports in N=354 (Tokiwa harassment preprint, HC3 regression already published)
- Author has observed aggregate-level 7-cluster centroids in N=13,668 (Tokiwa clustering paper, IEEE-published)
- Author has NOT performed the proposed cell-level cross-tabulations, national aggregate predictions, or counterfactual analyses
- This corresponds to Nosek 2018 PNAS Challenge 3 partial-blinding state

---

## OSF Field B2 — Explanation of foreknowledge and managing unintended influences (Optional)

```
The author has accessed and observed individual-level data in N=354 (harassment study) and aggregate-level centroids in N=13,668 (clustering study, IEEE-published) prior to this preregistration. Specifically:
- The author has previously published HC3-robust hierarchical regression results showing HEXACO and Dark Triad domain-level associations with power harassment and gender harassment self-reports (Tokiwa harassment preprint).
- The author has published 7-cluster centroids in HEXACO 6-domain space (Tokiwa clustering paper, IEEE).

However, the specific analyses preregistered here have NOT been performed and will not be performed until after this plan is registered:
1. The 7-type × 2-gender 14-cell cross-tabulation of harassment binary outcomes (Stage 0 main analysis)
2. The 28-cell empirical Bayes shrunken sensitivity (with role; Stage 0 sensitivity)
3. The Stage 1 national aggregate latent prevalence with bootstrap CIs
4. MAPE comparison against MHLW H28 (FY2016) 32.5%, MHLW R2 (FY2020) 31.4%, and MHLW R5 (FY2023) 19.3% past-3-year power harassment rates
5. The B0-B4 baseline hierarchy MAPE comparison (H2)
6. The B4 industry-stratified MAPE against MHLW R5 industry-stratified data (H2.industry secondary criterion)
7. The CMV diagnostic (Harman's single-factor + marker-variable correction)
8. The Stage 7 counterfactual A/B/C ΔP estimates with target trial emulation
9. The Pasona (2022) macro turnover sanity check (Stage 2 chain output validation)

Actions taken to reduce the risk of unintended influences:

1. Preregistration locked at OSF before any cell-level cross-tabulation, aggregation, or comparison with MHLW data is computed.
2. All inference criteria fixed in advance: MAPE thresholds (≤30% / 30-60% / >60%), H2.industry threshold (≤50% relaxed), H7 magnitude ranking, decision rules.
3. Sensitivity sweeps exhaustively pre-specified: V {2,3,4,5}, f1 {0.05,0.10,0.15,0.20}, f2 {0.10,0.20,0.30}, EB scale {0.5×,1.0×,2.0×}, binarization threshold {mean+0.25, +0.5, +1.0 SD}, cluster K {4-8}, transportability {0.3×,0.5×,0.7×,1.0×}.
4. Random seed fixed: NumPy default_rng(seed=20260429), Python random.seed(20260429), Stan seed=20260429.
5. Bonferroni-Holm correction pre-specified: family-wise α=.05 for H2 and H4-H7.
6. Negative-result publication committed in advance (D-NEW8, Section 7).
7. Independent methodologist review (mode B: anonymous, mathematical biology background) prior to Stage 2 validation.
8. Honest distinction (Section 3.1.3): HEXACO domain-level associations are exploratory replications of prior work, distinct from the type-conditional cell-level analyses preregistered here.
9. Reproducibility infrastructure (D-NEW9): open code, environment pinning, make reproduce, OSF Open Practice Badges, TOP guidelines Tier 3.
10. Deviation policy (Section 6.5): Levels 0-3, with Level 3 requiring v2 registration.
```

---

## OSF Field B3 — Subjects (taxonomy, multi-select)

**Recommended (3 minimum)**:
1. Social and Behavioral Sciences > Psychology > **Industrial and Organizational Psychology**
2. Social and Behavioral Sciences > Psychology > **Personality and Social Contexts**
3. Social and Behavioral Sciences > Psychology > **Quantitative Psychology**

**Optional (up to 5 for broader visibility)**:
4. Social and Behavioral Sciences > Psychology > **Social Psychology**
5. Medicine and Health Sciences > Public Health > **Occupational Health and Industrial Hygiene**

---

## OSF Field B4 — Intention for causal interpretation (radio button)

**Selection**: **"Indirect inference on causal relationship(s)"**

Rationale: study uses Hernán & Robins 2020 target trial emulation framework + Pearl 2009 do-operator notation, positioned at Pearl's Rung 2 (intervention prediction), not direct experimental causal inference (Section 5.1.5 of preregistration).

---

## OSF Field B5 — Blinding of experimental treatments (radio + multi-select)

**Selection**: **"No blinding is involved"** (single)

Rationale: observational secondary analysis combined with computational simulation; no physical assignment of subjects to treatment arms. Other blinding options do not apply.

---

## OSF Field B6 — Additional blinding during research or analysis (Optional)

```
No experimental blinding (this is an observational secondary-analysis study with computational microsimulation). However, the following preregistration-equivalent procedures are employed to remove unintended influences:

1. ANALYSIS-LEVEL BLINDING via preregistration. The cell-level cross-tabulations (7 type × 2 gender 14 cells), national aggregate predictions, MAPE values against MHLW H28/R2/R5 surveys, B0-B4 baseline comparisons, B4 industry-stratified MAPE, CMV diagnostic, and counterfactual A/B/C outputs are NOT computed or observed at the time of OSF registration. The author has only observed (a) individual-level HEXACO scores and harassment self-reports in N=354 (aggregated in the Tokiwa harassment preprint via HC3-robust regression), and (b) aggregate-level 7-cluster centroids in N=13,668 (published in the Tokiwa clustering paper, IEEE). All preregistered analyses (Section 5 of the attached preregistration document) are performed only after this preregistration is registered on OSF.

2. RANDOM SEED FIXING. NumPy default_rng(seed=20260429), Python random.seed(20260429), and Stan seed=20260429 make all stochastic operations (bootstrap, Monte Carlo resample, simulated random assignment in counterfactuals) deterministic and reproducible regardless of analyst.

3. INDEPENDENT METHODOLOGIST REVIEW (mode B: anonymous, mathematical biology background). The methodologist will review Section 5 (Analysis Plan) prior to Stage 2 validation, providing an external check on inference decisions (per Munafò et al. 2017 Box 1, lightweight variant).

4. NEGATIVE-RESULT PUBLICATION COMMITMENT (Section 7 of the preregistration). The author commits to journal submission for all five enumerated cases (success / partial / failure / H7 reversal / B3 < B2 typology overfitting), removing the incentive for post-hoc framing toward positive results.

5. EXHAUSTIVE SENSITIVITY SWEEPS (Section 6.4 of the preregistration). All ranges (V, f1, f2, EB strength, binarization threshold, cluster K, role-estimation models, counterfactual main values δ_A/δ_B/effect_C, transportability factor) are pre-specified, removing garden-of-forking-paths risk (Nosek 2018 Challenge 9).

6. DEVIATION POLICY (Section 6.5 of the preregistration). All deviations are reported in a dedicated Discussion subsection, classified Levels 0-3, with Level 3 (analysis-plan revision) requiring registration of v2 with public diff against v1.

7. HONEST DISTINCTION between preregistered confirmatory analyses and exploratory replications of prior HEXACO domain-level associations from the Tokiwa harassment preprint (Section 3.1.3 of the preregistration document).
```

---

## OSF Field B7 — Study design

```
This study has no experimental design in the traditional RCT sense (no two-group, factorial, randomized block, or repeated-measures structure with assigned treatments). It is an OBSERVATIONAL SECONDARY ANALYSIS combined with COMPUTATIONAL MICROSIMULATION and COUNTERFACTUAL PROJECTION.

Two-phase computational pipeline:

Phase 1 (descriptive simulation, 5 stages):
- Stage 0: Type assignment and probability table construction. Each of N=354 individuals is assigned to the nearest of 7 centroids by Euclidean distance over HEXACO 6-domain scores (centroids from the IEEE-published Tokiwa clustering paper, N=13,668). The 14-cell (7 type × 2 gender) crosstab is computed for binary harassment outcomes (binarized at mean + 0.5 SD per outcome). Bootstrap B = 2,000 iterations per cell with BCa confidence intervals (Efron 1987).
- Stage 1: Population aggregation. Cell-conditional probabilities are scaled to the Japanese workforce (~68 million) via MHLW Labor Force Survey weights (age × gender × employment type).
- Stage 2: Validation triangulation. National latent prediction is compared against MHLW H28 (FY2016, pre-law) 32.5% (primary), MHLW R2 (FY2020, transition) 31.4%, and MHLW R5 (FY2023, post-law, published March 2024) 19.3% past-3-year power harassment rates. Metrics: MAPE (primary), Pearson r, Spearman rho, KS distance, Wasserstein distance, calibration plot.
- Stage 3: Sensitivity sweeps over V, f1, f2, EB shrinkage strength, binarization threshold, cluster K, role-estimation models. All ranges fixed by Section 6.4 of the preregistration.
- Stage 4: Baseline hierarchy comparison. B0 (uniform random), B1 (gender-only logistic), B2 (HEXACO 6-domain linear logistic), B3 (proposed: 7 type × gender cell-conditional), B4 (B3 + age + estimated industry + employment type). Pre-registered ordinal hypothesis: MAPE_B0 ≥ MAPE_B1 ≥ MAPE_B2 ≥ MAPE_B3 ≥ MAPE_B4.
- Stage 5: Common-method-variance diagnostic (Harman's single-factor test on N=13,668 personality data; marker-variable correction using HEXACO Openness as theoretical marker, per Lindell & Whitney 2001).

Phase 2 (intervention counterfactuals, 3 stages):
- Stage 6: Target trial emulation specification (Hernán & Robins 2020). PICO + 24-week duration (Roberts 2017 anchor) for each counterfactual. Four identifying assumptions (exchangeability, positivity, consistency, transportability) are explicitly assessed.
- Stage 7: Counterfactual simulation using Pearl 2009 do-operator notation:
  - A (universal HH intervention): do(HH := HH + δ_A × SD(HH)) for all individuals; main δ_A = +0.3 SD; sensitivity range [0.1, 0.5] SD; anchor: Kruse 2014 d=0.71 (conservatively discounted).
  - B (targeted intervention, primary intervention of interest): do(HH := HH + δ_B × SD(HH)) only for individuals in Cluster 0 (primary, Self-Oriented Independent profile) and Clusters 4, 6 (secondary); main δ_B = +0.4 SD; sensitivity range [0.2, 0.6] SD; anchor: Hudson 2023 self-selected effect (conservatively discounted).
  - C (structural-only intervention): do(p_c := p_c × (1 − effect_C)) for all 14 cells; main effect_C = 0.20; sensitivity range [0.10, 0.30]; anchor: triangulation of Pruckner & Sausgruber 2013, Bezrukova et al. 2016, Roehling & Huang 2018, Dobbin & Kalev 2018.
- Stage 8: Transportability sensitivity. Western anchor effects multiplied by {0.3×, 0.5×, 0.7×, 1.0×} to test cross-cultural attenuation per Sapouna 2010 / Nielsen 2017 cultural moderator evidence.

Design characteristics:
- BETWEEN-SUBJECT structure: the 14 cells (7 type × 2 gender) define between-subject groupings; each individual belongs to exactly one cell.
- WITHIN-SUBJECT structure: not applicable (single observation per person; cross-sectional data).
- COUNTERBALANCING: not applicable (no temporal order or sequence effects).
- BLINDING: see Blinding fields above.

Full pipeline diagrams in Section 2.3, statistical model specifications in Section 5, inference criteria in Section 6, of the attached preregistration document (D12_pre_registration_OSF.en.pdf).
```

---

## OSF Field B8 — Randomization (Optional)

```
No physical randomization (this is an observational secondary analysis combined with simulation; there is no real-time assignment of subjects to treatments).

Within the simulation, all stochastic operations are made deterministic via a fixed random seed:
- NumPy: default_rng(seed=20260429)
- Python: random.seed(20260429)
- Stan: seed=20260429

The seed value 20260429 is fixed by this preregistration and will not be changed. Bootstrap resample states (B = 2,000 iterations per cell, BCa CI) are persisted to HDF5 for full reproducibility regardless of analyst.

Counterfactuals A / B / C in Phase 2 are framed as "simulated random assignment" within the target trial emulation framework (Hernán & Robins 2020). In practice, the do-operator is applied uniformly to the relevant subgroup:
- Counterfactual A: do(HH := HH + δ_A × SD(HH)) applied to ALL individuals
- Counterfactual B: do(HH := HH + δ_B × SD(HH)) applied to individuals in Clusters 0, 4, 6
- Counterfactual C: do(p_c := p_c × (1 − effect_C)) applied to all 14 cells

This "as-if randomization" is theoretical (counterfactual projection under explicit identifying assumptions), not actual experimental randomization.
```

---

## OSF Field B9 — Data collection procedures (REQUIRED)

```
This study performs SECONDARY ANALYSIS of preexisting datasets and public statistics. NO NEW DATA COLLECTION is involved. The data sources are:

1. INDIVIDUAL-LEVEL DATA (preexisting, IRB-approved):

   (a) N=354 harassment data (Tokiwa harassment preprint).
   - Population: Japanese workers
   - Sampling frame: Japanese online panel platform
   - Recruitment: self-selected respondents, monetary incentive
   - Inclusion: working-age adults, Japanese language proficiency
   - Exclusion: failed lie-scale and attention checks
   - Variables: HEXACO 6 domains (Wakabayashi 2014 Japanese HEXACO-60), Dark Triad 3 (Shimotsukasa & Oshio 2017 SD3-J), Workplace Power Harassment Scale (Tou et al. 2017), Gender Harassment Scale (Kobayashi & Tanaka 2010), age, gender, area
   - Source file: harassment/raw.csv (anonymized)
   - IRB: documented in the Tokiwa harassment preprint

   (b) N=13,668 clustering data (Tokiwa clustering paper, IEEE-published).
   - Population: Japanese adults
   - Sampling frame: large-scale Japanese online panel
   - Recruitment: stratified online survey
   - Inclusion: aged 18+, Japanese language proficiency
   - Exclusion: incomplete responses, failed attention checks
   - Variables: HEXACO 6 domains (full assessment), basic demographics
   - Source file: clustering/csv/clstr_kmeans_7c.csv (centroids only, aggregated)
   - IRB: documented in the Tokiwa clustering paper, IEEE

2. POPULATION-LEVEL STATISTICS (preexisting public sources):
   - MHLW H28 (FY2016) Survey on Workplace Power Harassment: 32.5% past-3-year power harassment, pre-law (★ H1 primary validation target)
   - MHLW R2 (FY2020) Survey on Workplace Harassment: 31.4% (transition)
   - MHLW R5 (FY2023, published March 2024) Survey on Workplace Harassment: 19.3% post-law; industry-stratified 16-26.8% (★ H2.industry secondary criterion). Full report (385 pages) attached in simulation/prior_research/_text/.
   - MHLW R4 (FY2022) Employment Trend Survey: turnover by reason "workplace interpersonal" 8.3-9.4% (f1 secondary anchor). PDF attached.
   - MHLW Industrial Safety and Health Survey: mental disorder incidence (f2 anchor).
   - MHLW Labor Force Survey 2022: industry × demographic crosstabs for B4 industry probabilistic estimation.
   - ILO 2022 Global Survey: Asia-Pacific lifetime 19.2% (international baseline).

3. INDUSTRY-SURVEY DATA (preexisting third-party):
   - Pasona Research (2022) "Quantitative Survey on Workplace Harassment" by PERSOL Research and Consulting Co., Ltd.
   - Population: Japanese workers aged 20-69
   - Sampling frame: PERSOL Research online panel
   - N=28,135 total (3,000 5-year harassment victims, 1,000 witnesses, 1,000 non-experiencers, plus general workforce)
   - Conducted August 30 - September 5, 2022
   - Used for: 5-year harassment prevalence triangulation (19.7%), industry-stratified prevalence (16.9-22.9%), f1 PRIMARY empirical anchor (10.3% with industry range 6.3-13.3%), Stage 2 chain output sanity check (865,000 annual harassment-induced turnover; 66% unreported / 暗数化)
   - Full report (152 pages) attached in simulation/prior_research/_text/

4. OTHER PUBLISHED ANCHORS:
   - Tsuno et al. (2010) Japanese NAQ-R (measurement validity)
   - Tsuno et al. (2015) PLOS ONE - Japanese national-representative N=1,546, 30-day prevalence 6.1% (marginal-distribution check)
   - Tsuno & Tabuchi (2022) - Bullying → SPD PR=3.20 (f2 anchor)

DATA PREPARATION:
- N=354 harassment data: loaded as-is from harassment/raw.csv (anonymized).
- N=13,668 centroids: loaded from clustering/csv/clstr_kmeans_7c.csv (IEEE-published aggregated values).
- Public PDFs and tables: accessed from simulation/prior_research/_text/ (committed to the repository).
- All individual-level data have been previously anonymized; no personal identifiers are present.

EXPECTED DURATION:
- Data are already collected. The secondary analysis will be performed over approximately 2-3 months following preregistration lock and OSF DOI acquisition.

ETHICS:
- Secondary analysis of anonymized data; no new IRB approval is required. Original IRB approvals are documented in the Tokiwa harassment preprint and the Tokiwa clustering paper (IEEE-published).
```

---

## OSF Field B10 — Sample size (REQUIRED)

```
PHASE 1 MAIN ANALYSIS:
- Individual-level: N=354 (Japanese workers, aged 20-64)
- Cell-level: 14 cells (7 HEXACO types × 2 genders)
  - Cell N: minimum 10, maximum 70, median 18
  - 0 cells with N < 10
  - 7 cells (50%) with N < 20

PHASE 1 SENSITIVITY ANALYSIS:
- 28 cells (7 type × 2 gender × 2 role)
- 16 cells (57%) with N < 10
- 9 cells with N ≤ 3
- 4 cells with N = 0
- Empirical Bayes shrinkage (Beta-Binomial conjugate, method of moments) is mandatory at this tier.

POPULATION AGGREGATE:
- ~68 million Japanese workers aged 20-64 (MHLW Labor Force Survey base, used for Stage 1 weighting)

EXTERNAL VALIDATION REFERENCE SAMPLES (existing public statistics):
- MHLW H28 / R2 / R5 surveys: each N≈8,000 (general worker sample)
- Pasona Research 2022: N=28,135 (industry-survey panel)
- Tsuno et al. 2015 PLOS ONE: N=1,546 (national-representative random sample)

PHASE 2 COUNTERFACTUAL TARGETS:
- Cluster 0 (primary, Self-Oriented Independent profile): ~6.5% of N=354 (~23 local; scaled to ~4.4 million in population)
- Cluster 4 (secondary): ~14.4% (~51 local; ~9.8 million)
- Cluster 6 (secondary): ~32.2% (~114 local; ~21.9 million)

BOOTSTRAP ITERATIONS: B=2,000 per cell with BCa CI correction.
RANDOM SEED: 20260429 (fixed by preregistration; deterministic for full reproducibility).

NESTING:
- Level 1: individuals (N=354)
- Level 2: cells (14 cells main / 28 cells sensitivity)
- Level 3: population aggregate (single point estimate per validation period)
- 28-cell tier uses hierarchical Beta-Binomial framework with cell-level posteriors.

The sample size is fixed by the existing data and is not adjustable.
```

---

## OSF Field B11 — Sample size rationale (Optional)

```
Sample size is fixed by the existing data and is not adjustable. The rationale for proceeding with the available sample size, drawn from the D13 power analysis (simulation/docs/power_analysis/D13_power_analysis.md, attached as supplementary), is:

1. N=354 satisfies Funder & Ozer (2019)'s recommendation of N ≥ 250 for stable correlation estimation at the aggregate level.

2. The 14-cell main analysis satisfies N ≥ 10 in every cell, allowing bootstrap estimation WITHOUT empirical Bayes shrinkage. Cell-level binary-rate 95% CI half-widths are approximately ±13 percentage points (median) for power harassment outcomes.

3. The pairwise minimum detectable effect (Cohen's d ≥ 0.92, computed at α = .05 two-sided, 1-β = .80) is "very large" by Cohen 1988 and "rarely found in replication" per Funder & Ozer 2019. Consequently:
   - Cell-level pairwise inference is AVOIDED as a preregistered limitation.
   - Aggregate-level (population) inference is the primary inferential target.
   - This limitation is acknowledged in Section 10 of the preregistration (Limitation L4).

4. The 28-cell sensitivity tier with 16 small cells (N < 10) requires empirical Bayes shrinkage (Beta-Binomial conjugate, method of moments) with a strength sensitivity sweep at scale ∈ {0.5×, 1.0× main, 2.0×}. This is enforced by Section 5.2 of the preregistration.

5. Bootstrap iterations B=2,000 per cell are sufficient for stable BCa CI estimation given the cell-level sample sizes (Efron 1987; DiCiccio & Efron 1996).

6. Population aggregate (~68 million workers) inherits cell-level uncertainty via bootstrap propagation; population-level CI half-widths are substantially narrower than cell-level CIs due to weighted aggregation across all 14 cells.

7. External validation samples (MHLW N≈8,000 each survey; Pasona N=28,135) are sufficiently large that their reported point estimates (32.5%, 31.4%, 19.3%, 19.7%) have negligible measurement uncertainty relative to the simulation's prediction CIs.

The full power analysis report is at simulation/docs/power_analysis/D13_power_analysis.md (attached as supplementary).
```

---

## OSF Field B12 — Starting and stopping rules

```
NOT APPLICABLE for new data collection: no new data are collected in this study.

PILOT TESTING: not applicable (preexisting data; secondary analysis only).

STOPPING RULE FOR DATA COLLECTION: not applicable.

STARTING RULE FOR ANALYSIS:
- Stage 0 code execution begins ONLY after this preregistration is registered on OSF and the OSF DOI is recorded in the document headers (Section 14.3 of the preregistration document).
- Stage 1 (validation against MHLW H28 FY2016) begins ONLY after Stage 0 cross-tabulations are complete and inspected for sanity (e.g., no NaN cells, all bootstrap CIs converged).
- Stage 2 (validation triangulation) is preceded by independent methodologist review of Section 5 (Analysis Plan) per Section 8.1 of the preregistration.

STOPPING RULES FOR THE ANALYSIS PIPELINE:

1. Bootstrap iterations: B=2,000 per cell, FIXED by preregistration. No adaptive stopping (e.g., based on convergence diagnostics) is permitted.

2. Sensitivity sweep ranges: pre-specified in Section 6.4 of the preregistration:
   - V ∈ {2, 3, 4, 5}
   - f1 ∈ {0.05, 0.10, 0.15, 0.20}
   - f2 ∈ {0.10, 0.20, 0.30}
   - EB shrinkage scale ∈ {0.5×, 1.0×, 2.0×}
   - Binarization threshold ∈ {mean+0.25 SD, +0.5 SD, +1.0 SD}
   - Cluster K ∈ {4, 5, 6, 7, 8}
   - Role-estimation models ∈ {linear, tree-based, literature}
   - Phase 2 δ_A, δ_B, effect_C, transportability_factor: respective fixed ranges

   Any post-registration EXTENSION of these sensitivity ranges will be flagged as EXPLORATORY and EXCLUDED from confirmatory inference.

3. Counterfactual exploration: counterfactuals A / B / C are exhaustively defined in Stages 6-8 with fixed parameter ranges. No ADDITIONAL counterfactuals (e.g., D, E) may be added post-lock without registering a v2 amendment per Section 6.5 Level 3 deviation procedure.

4. Multiple-comparison correction: pre-specified Bonferroni-Holm at family-wise α=.05 for H2 (4 ordinal pairwise tests) and H4-H7 (3 counterfactual main tests). No alternative correction is permitted.

5. Inference threshold modification: post-hoc revision of MAPE thresholds (≤30% SUCCESS / >60% FAILURE) is PROHIBITED. Any modification requires registering a v2 amendment.

6. Optional stopping: NOT permitted. The analysis is run to completion under the preregistered plan regardless of intermediate results (e.g., the analysis does NOT stop early if H1 is rejected at the 14-cell stage; all preregistered downstream analyses including baseline hierarchy and counterfactuals are still executed).

DEVIATION POLICY (Section 6.5 of the preregistration):
- Level 0: no deviation
- Level 1: minor specification clarification (recorded in Methods)
- Level 2: data-driven adjustment (justified in Discussion subsection)
- Level 3: analysis-plan revision (requires v2 registration on OSF with public diff against v1)

All Level 2 and Level 3 deviations are reported in a dedicated Discussion subsection ("Deviations from Pre-Registration").
```

---

## OSF Field B13 — Manipulated variables (REQUIRED)

```
This is an observational secondary analysis combined with COMPUTATIONAL MICROSIMULATION. There are NO PHYSICAL MANIPULATIONS of subjects; subjects are not assigned to treatment arms.

However, the Phase 2 counterfactual analysis involves COMPUTATIONAL MANIPULATIONS (do-operators per Pearl 2009 structural causal model notation) applied to the probability tables and individual-level HEXACO scores:

1. δ_A (universal HH shift, Counterfactual A):
   - Operation: do(HH := HH + δ × SD(HH)) applied to ALL individuals (N=354), then re-classify clusters
   - Main value: δ_A = +0.3 SD
   - Sensitivity range: [0.1, 0.5] SD step 0.1
   - Anchor: Kruse 2014 d=0.71 (conservatively discounted)

2. δ_B (targeted HH shift, Counterfactual B — PRIMARY intervention of interest):
   - Operation: do(HH := HH + δ × SD(HH)) applied ONLY to individuals in Cluster 0 (primary, Self-Oriented Independent profile) and Clusters 4 and 6 (secondary)
   - Main value: δ_B = +0.4 SD
   - Sensitivity range: [0.2, 0.6] SD step 0.1
   - Anchor: Hudson 2023 b=.03/week (conservatively discounted)
   - Target-type-set sensitivity: {Cluster 0 only}, {Clusters 0+4}, {Clusters 0+4+6 (main)}, {Clusters 0+4+6+others}

3. effect_C (structural reduction, Counterfactual C):
   - Operation: do(p_c := p_c × (1 − effect_C)) applied to all 14 cell-conditional probabilities
   - Main value: effect_C = 0.20
   - Sensitivity range: [0.10, 0.30] step 0.05
   - Anchor: triangulation of Pruckner & Sausgruber 2013, Bezrukova et al. 2016, Roehling & Huang 2018, Dobbin & Kalev 2018

4. transportability_factor (Phase 2 cultural attenuation):
   - Operation: anchor effect (from points 1-3) is multiplied by factor before applying to Japan
   - Main value: 1.0× (no attenuation)
   - Sensitivity sweep: {0.3×, 0.5×, 0.7×, 1.0×}
   - Anchor: Sapouna 2010 UK→Germany null finding (worst case 0.3×); Nielsen 2017 Asia/Oceania attenuation ratio

These four manipulations correspond to the do-operator in Pearl (2009) structural causal model notation. They are framed within Hernán & Robins (2020) target trial emulation, with four identifying assumptions explicitly assessed in Section 5.7.4 of the preregistration (exchangeability, positivity, consistency, transportability).

Random seed (NumPy default_rng(seed=20260429), Python random.seed(20260429), Stan seed=20260429) ensures all stochastic aspects of the manipulation (cluster reassignment after δ shift, Monte Carlo runs in counterfactuals) are deterministic and reproducible.
```

---

## OSF Field B14 — Measured variables

```
Measured variables are organized by level of analysis. All variables are PREEXISTING; no new measurement is performed.

INDIVIDUAL-LEVEL (N=354 harassment data, harassment/raw.csv):

1. HEXACO 6 domains (Wakabayashi 2014 Japanese HEXACO-60):
   - H (Honesty-Humility), E (Emotionality), X (eXtraversion), A (Agreeableness), C (Conscientiousness), O (Openness)
   - Each: continuous, Likert 1-5 mean of facet items
   - Role: predictor (cluster assignment); counterfactual operand (δ_A, δ_B target HH only)

2. Dark Triad 3 (Shimotsukasa & Oshio 2017 SD3-J):
   - Machiavellianism, Narcissism, Psychopathy: each continuous
   - Role: covariate; convergent validity check with HEXACO HH (Lee & Ashton 2005 r ≈ -.53 to -.72)

3. Power Harassment Scale (Tou et al. 2017 Workplace Power Harassment Scale):
   - Continuous item mean → BINARIZED at mean + 0.5 SD (main; sensitivity {+0.25, +0.5, +1.0 SD})
   - Role: outcome (binary perpetrator status)

4. Gender Harassment Scale (Kobayashi & Tanaka 2010):
   - Continuous item mean → BINARIZED at mean + 0.5 SD
   - Role: outcome (binary perpetrator status)

5. Demographics (self-reported):
   - Age: continuous (years)
   - Gender: binary (0/1, n=133/220)
   - Area: categorical

INDIVIDUAL-LEVEL (N=13,668 clustering data, AGGREGATED ONLY, clustering/csv/clstr_kmeans_7c.csv):

6. HEXACO 6-domain centroids: continuous (7 clusters × 6 domains = 42 values)
7. Cluster proportions: categorical (7 types; used as population scaling weight)

POPULATION-LEVEL (external validation targets):

8. Past-3-year POWER HARASSMENT victimization rate:
   - MHLW H28 (FY2016) 32.5% ★ H1 PRIMARY validation target (pre-law)
   - MHLW R2 (FY2020) 31.4% (transition)
   - MHLW R5 (FY2023, published March 2024) 19.3% (post-law)

9. Industry-stratified past-3-year power harassment (MHLW R5 FY2023):
   - 16 industry buckets, range 16-26.8% (e.g., Construction 26.8%, Compound services 22.5%, Health/welfare 20.6%, Education 16.9%)
   - Role: H2.industry secondary criterion (B4 baseline subgroup MAPE ≤ 50%)

10. Other harassment categories (FRAMING ONLY, not validation target):
    - MHLW R5 sexual harassment 6.3% (women 8.9% / men 3.9%)
    - MHLW R5 customer harassment 10.8% (new category emerging post-FY2022 law amendment)
    - Role: latent vs expressed framing in Section 1.4 H3

11. 5-year harassment prevalence (Pasona Research 2022, N=28,135):
    - Overall: 19.7% (lifetime 34.6%)
    - Industry-stratified: 16.9-22.9%
    - Role: marginal-distribution check (large-N harassment-specific)

12. Harassment-victim turnover rate (Pasona 2022):
    - Overall: 10.3% (industry range 6.3-13.3%)
    - Role: f1 PRIMARY empirical anchor

13. Macro annual harassment-induced turnover (Pasona 2022):
    - Estimate: 865,000 persons/year (66% unreported / 暗数化)
    - 12.1% of all annual turnovers (865,000 / 7,173,000)
    - Role: Stage 2 chain output sanity check (Section 5.4)

14. Other anchors:
    - MHLW R4 (FY2022) Employment Trend Survey: turnover by reason "workplace interpersonal" men 8.3% / women 9.4% (f1 secondary anchor, upper bound)
    - MHLW Industrial Safety and Health Survey: mental disorder incidence (f2 anchor)
    - Tsuno & Tabuchi 2022: bullying → SPD PR=3.20 (f2 anchor)
    - Tsuno et al. 2015 PLOS ONE N=1,546 30-day prevalence 6.1% (marginal-distribution check, national-rep)
    - ILO 2022 Asia-Pacific lifetime 19.2% (international baseline)

PARTITIONING / SUBSET VARIABLES:
- Gender (binary): used for 14-cell partition
- Age band: used for subgroup MAPE (failure-mode localization)
- Cluster membership (1-7): used for counterfactual targeting (B targets Clusters 0/4/6)

EXCLUSION VARIABLES:
- No participant-level exclusion at the analytic stage; all N=354 / N=13,668 used as released.
- Cluster 6 (32% population-dominant) is intentionally NOT excluded from counterfactuals to preserve positivity assumption (Section 5.7.4).
- HEXACO domain missingness >1 of 6 domains: handled by listwise within-resample (Section 5).
```

---

## OSF Field B15 — Indices

```
Derived indices and their formulas:

1. 7-TYPE MEMBERSHIP (Stage 0):
   - Method: each individual i in N=354 is assigned to the nearest of 7 centroids by Euclidean distance over HEXACO 6 domains
   - Formula: type(i) = argmin_j sqrt(sum_d (HEXACO_d_i − centroid_d_j)^2)
     where d ∈ {H, E, X, A, C, O} and j ∈ {0..6}
   - Centroids: Tokiwa clustering paper IEEE-published (clustering/csv/clstr_kmeans_7c.csv)

2. CELL ID (14-cell, main):
   - Method: cell(i) = (type(i), gender(i)), where type ∈ {0..6} and gender ∈ {0, 1}
   - Number of cells: 14

3. CELL ID (28-cell, sensitivity):
   - Method: cell(i) = (type(i), gender(i), role(i)), where role ∈ {0=non-manager, 1=manager}
   - Number of cells: 28
   - Used only in EB-shrunken sensitivity (Section 5.2)

4. ROLE PROBABILITY (D1 sensitivity):
   - Main: top 15% of (C + 0.5 × X) composite → manager (matches MHLW Labor Force ~12-15% manager rate, and Judge, Bono, Ilies & Gerhardt 2002 leadership emergence findings)
   - D1 sensitivity sweep: 3 alternative models — (a) linear regression P(manager) ~ HEXACO + age, (b) tree-based classifier, (c) literature rule (main)

5. INDUSTRY PROBABILITY (B4 baseline, ★ added v1.1):
   - Method: P(industry_j | age_i, gender_i, employment_i) for j ∈ 16 buckets, derived from MHLW Labor Force 2022 industry × demographic crosstabs
   - 16 buckets: Mining/quarrying, Construction, Manufacturing, Electricity/gas/water, Information/communication, Transport/postal, Wholesale/retail, Finance/insurance, Real estate, Academic/professional, Accommodation/food, Lifestyle/entertainment, Education, Health/welfare, Compound services, Other services
   - Used as weighting vector for B4 cell predictions

6. CELL-CONDITIONAL PROPENSITY (Stage 0 main):
   - Formula: p̂_c = X_c / N_c
     where X_c = number of "harassment perpetrators" (binarized at mean + 0.5 SD per outcome) in cell c, N_c = total N in cell c
   - Bootstrap distribution: B = 2,000 BCa resamples per cell (Efron 1987)

7. EB-SHRUNKEN PROPENSITY (Stage 0 sensitivity, 28-cell):
   - Method: Beta-Binomial conjugate posterior
   - Hyperprior estimation by method of moments from 14-cell distribution:
     α̂ = μ̂ × [μ̂(1−μ̂)/σ̂² − 1]
     β̂ = (1−μ̂) × [μ̂(1−μ̂)/σ̂² − 1]
     where μ̂ = mean(p̂_j), σ̂² = var(p̂_j) for j = 1..14
   - Posterior for 28-cell k: E[p_k | X_k, N_k] = (α̂ + X_k) / (α̂ + β̂ + N_k)
   - 95% credible interval: from Beta(α̂ + X_k, β̂ + N_k − X_k) quantiles
   - Strength sensitivity: scale ∈ {0.5×, 1.0× main, 2.0×}

8. NATIONAL LATENT PREVALENCE (Stage 1):
   - Formula: P̂_t = Σ_c (p̂_c × W_c) / Σ_c W_c
     where W_c = MHLW labor-force population × cluster proportion (from N=13,668) × gender proportion × age weight
   - For each validation period t ∈ {FY2016, FY2020, FY2023}
   - Bootstrap CI: B = 2,000 iterations, BCa

9. MAPE (PRIMARY metric, Stage 2):
   - Formula: MAPE = mean(|predicted_i − observed_i| / observed_i) × 100
   - Computed against MHLW H28 (FY2016, primary), MHLW R2 (FY2020), MHLW R5 (FY2023) power-harassment past-3-year rates
   - SUCCESS threshold: ≤ 30%; PARTIAL: 30-60%; FAILURE: > 60%

10. SUBGROUP MAPE (Stage 2 secondary):
    - Gender × age band: failure-mode localization
    - Industry-stratified (B4, H2.industry): MAPE against MHLW R5 16-bucket data; threshold ≤ 50% (relaxed)

11. ΔP_x (counterfactual reduction, Phase 2 Stages 6-8):
    - Formula: ΔP_x = P̂_baseline − P̂_x for x ∈ {A, B, C}
    - Bootstrap CI propagating cell-level uncertainty

12. COST-EFFECTIVENESS RATIO (H5):
    - Formula: (ΔP_B) / |Cluster 0 ∪ 4 ∪ 6 in population|
    - Compared to ΔP_A / N_total

13. STAGE 2 CHAIN OUTPUT (Pasona sanity check, ★ added v1.1):
    - Predicted annual harassment-induced turnover = Σ_cell (predicted perpetrators × V × f1)
    - V (victim multiplier): main 3, sensitivity {2, 3, 4, 5}
    - f1 (turnover rate): main 0.10, sensitivity {0.05, 0.10, 0.15, 0.20}
    - Soft criterion: predicted within 50-200% of Pasona 2022 estimate (865,000/year), i.e., 430,000-1,730,000

14. CMV INDICES (Stage 5):
    - Harman's first-factor variance %: from unrotated 1-factor EFA on N=13,668 HEXACO items; threshold < 50%
    - Marker-variable adjusted correlations: HEXACO Openness as theoretical marker (Lindell & Whitney 2001)

All formulas, parameters, and sensitivity ranges are fixed by the preregistration (Sections 5 and 6.4 of the attached document).
```

---

## OSF Field B16 — Statistical models (REQUIRED, compact)

```
Full specifications in Section 5 of the attached preregistration PDF. Summary:

Stage 0 (cell-level): Bootstrap B=2,000 with BCa CI (Efron 1987) for each of 14 cells (7 type × 2 gender). 28-cell sensitivity uses Beta-Binomial conjugate EB shrinkage (Casella 1985, Clayton & Kaldor 1987) with method-of-moments hyperprior; strength sweep at scale ∈ {0.5×, 1.0×, 2.0×}.

Stage 1 (population aggregation): P̂_t = Σ_c (p̂_c × W_c) / Σ_c W_c, weights W_c from MHLW Labor Force Survey age × gender × employment crosstabs.

Stage 2 (validation): MAPE against MHLW H28 (FY2016) 32.5% past-3-year power harassment (primary). Secondary: Pearson r, Spearman ρ, KS distance, Wasserstein distance, calibration plot.

Stage 4 (baseline hierarchy): B0 uniform, B1 gender logistic, B2 HEXACO 6-domain logistic, B3 (proposed) 7 type × gender cell-conditional, B4 = B3 + age + estimated industry + employment cell-conditional. Pre-registered ordinal: MAPE_B0 ≥ ... ≥ MAPE_B4.

Stage 5 (CMV): Harman's single-factor EFA on N=13,668 HEXACO items (threshold first-factor < 50%); marker-variable correction (Lindell & Whitney 2001) using HEXACO Openness.

Phase 2 Stages 6-8 (counterfactuals): Pearl 2009 do-operator notation. A = do(HH := HH + δ_A · SD) all individuals; B = same on Clusters 0/4/6 only; C = do(p_c := p_c × (1 − effect_C)) all 14 cells. Bootstrap CIs for ΔP_x. Target trial emulation with 4 identifying assumptions explicit (Hernán & Robins 2020).

Convergence criteria: bootstrap CI half-width must stabilize within 5% across the last 200 of 2,000 resamples per cell. MoM hyperprior diagnostics: σ̂² as fraction of μ̂(1−μ̂); auxiliary MLE / Stan posterior for triangulation if MoM diverges.

Assumption tests: D13 power analysis (Cohen's d ≥ 0.92 pairwise) precludes cell-level confirmatory inference (preregistered limitation). CMV diagnostic gates causal interpretation (Section 4.2 L8).
```

---

## OSF Field B17 — Transformations

```
Outcome binarization: harassment scale scores binarized at mean + 0.5 SD per outcome (main); sensitivity sweep at mean + 0.25 SD and mean + 1.0 SD.

Cluster assignment: each individual's HEXACO 6-domain vector → nearest of 7 centroids by Euclidean distance.

Population reweighting: cell-level estimates scaled by MHLW Labor Force Survey weights (age × gender × employment type).

Categorical coding: gender 0/1, role 0/1 (non-manager/manager), cluster 0..6, industry 0..15 (16 buckets matching MHLW industry classification).

No log transforms, square roots, or other nonlinear transformations on HEXACO scores. HEXACO domains used as standard Likert 1-5 means.

Random seed 20260429 governs all stochastic operations.
```

---




