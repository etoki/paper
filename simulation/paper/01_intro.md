# 01. Abstract + Introduction

**Working title (T2.1):**
*Personality Interventions Don't Work; Structural Ones Do: Counterfactual Evidence from a HEXACO 7-Typology Workplace Harassment Microsimulation in Japan*

**Author**: Eisuke Tokiwa (SUNBLAZE Co., Ltd.; ORCID 0009-0009-7124-6669)
**Pre-registration**: OSF DOI 10.17605/OSF.IO/3Y54U (v2.0; locked 2026-05-21)
**Reporting standard**: Stage 2 Registered Report (post-data) following the v2.0 pre-registration

---

## Abstract (≈ 250 words)

Workplace harassment imposes substantial public-health and economic costs in Japan, with Ministry of Health, Labour and Welfare (MHLW) surveys documenting past-three-year power-harassment victimization rates of 32.5% (FY2016) declining to 19.3% (FY2023) following the 2020 Power Harassment Prevention Law. Whether this decline can be attributed to structural intervention, and whether person-level personality interventions could achieve comparable effects, remains an open empirical question. We pre-registered (OSF DOI 10.17605/OSF.IO/3Y54U) a 7-typology HEXACO microsimulation that emulates a target trial of three counterfactual interventions: (A) personality training that shifts Honesty-Humility, Agreeableness, and Emotionality by +0.3 SD; (B) cluster reassignment to three low-prevalence HEXACO typologies; and (C) structural reform that reduces cell-level harassment propensity by 20%, calibrated against three prior meta-analyses. Using N = 354 individual-level survey data partitioned into 14 cells (7 HEXACO typologies × 2 genders) with cell-stratified bootstrap (B = 10,000), Beta-Binomial empirical Bayes shrinkage, and four-step BCa confidence interval cascade, we report that ΔP_A = −0.0061 (95% CI [−0.0199, +0.0054]) and ΔP_B = −0.0058 (95% CI [−0.0199, +0.0056]) are both null, whereas ΔP_C = +0.0354 (95% CI [+0.0266, +0.0445]) is positive and statistically robust. The intersection-union test classifies this configuration as REVERSAL, indicating structural intervention dominance over personality intervention by approximately 5.8×. Sensitivity sweeps confirm robustness across cultural attenuation factors of 0.3–1.0 and across nine pre-registered analytic choices. We interpret HEXACO 7-typology as a stratification variable, not an intervention target, and reject victim-blaming framings of harassment exposure.

**Keywords**: workplace harassment; HEXACO; counterfactual simulation; target trial emulation; Japan; pre-registration; structural intervention; personality intervention; 2020 Power Harassment Prevention Law

---

## 1. Introduction

### 1.1 Workplace harassment as a public-health and policy problem

Workplace harassment is a leading driver of psychiatric morbidity, voluntary turnover, and lost productivity in industrialized economies (Hershcovis & Barling, 2010; Salin, 2003). In Japan, the Ministry of Health, Labour and Welfare (MHLW; 厚生労働省) has documented past-three-year power-harassment victimization rates of 32.5% in FY2016, 31.4% in FY2020, and 19.3% in FY2023 (MHLW, 2017, 2021, 2024). The 13.2 percentage-point decline between FY2016 and FY2023 coincides closely with the staged enforcement of the 2020 Power Harassment Prevention Law (パワーハラスメント防止法), which mandated formal anti-harassment policies, employee-accessible reporting channels, and prohibited retaliation against complainants — first for large enterprises in June 2020, and subsequently extended to small-and-medium enterprises in April 2022. This natural experiment offers a rare opportunity to interrogate what types of intervention reduce harassment, and at what scale.

A long-standing controversy in workplace-psychology research concerns whether to allocate prevention resources toward person-level interventions — for example, personality training, mindset coaching, or hiring screens designed to filter out individuals with traits associated with harassment perpetration — versus system-level interventions that modify organizational structure, accountability mechanisms, and power asymmetries (Mikkelsen et al., 2020; Salin, 2021). The former approach assumes that harassment is largely driven by individual disposition; the latter, that it is largely driven by environmental affordance. Despite extensive research linking personality traits to bullying and harassment (Nielsen, Glasø, & Einarsen, 2017; Pilch & Turska, 2015), the relative population-level efficacy of person-level versus system-level interventions has not been quantitatively compared in a counterfactually rigorous framework.

### 1.2 HEXACO 7-typology and its prior connection to workplace harassment

The HEXACO model of personality (Ashton & Lee, 2007) extends the classical Big Five framework with an additional dimension, Honesty-Humility (HH), which captures fairness, modesty, sincerity, and resistance to exploiting others. Within the HEXACO literature, low Honesty-Humility, low Agreeableness, and high Emotionality have repeatedly been associated with workplace bullying perpetration and victimization (Glasø, Matthiesen, Nielsen, & Einarsen, 2007; Lee et al., 2013; Linton & Power, 2013; Nielsen, Glasø, & Einarsen, 2017). However, the magnitude of these associations is generally modest (uncorrected r typically in the range 0.10–0.30; Nielsen, Glasø, & Einarsen, 2017), and recent meta-analyses suggest considerable cross-cultural heterogeneity in workplace bullying acceptability and effect sizes (Power et al., 2013, comparing six continents; Nielsen, Glasø, & Einarsen, 2017, reporting attenuated Asia/Oceania effects of r = .16 versus European r = .33 for Neuroticism–bullying associations).

In a companion study, our research group identified seven robust HEXACO clusters in a Japanese sample of N = 13,668 (Tokiwa, 2024, IEEE conference paper; final proceedings citation pending — see reference note 1; herein "the centroid study"). The seven typologies — labeled here as types 0 through 6 — span the joint distribution of HEXACO domains and provide a parsimonious taxonomy for categorizing individuals according to their personality profile. The present study leverages these IEEE-published centroids as fixed parameters for stratifying a smaller harassment-focused subsample (N = 354; Tokiwa, 2025, harassment preprint).

**A critical reframing.** The role of HEXACO in this study is *not* to identify who should be subjected to personality interventions. Rather, HEXACO 7-typology serves as a *stratification variable* that identifies which subgroups are differentially exposed to workplace harassment. Whether this differential exposure reflects direct personality causation (low-HH individuals harass or are harassed at higher rates), selection (low-HH individuals self-select into harassment-prone industries), confounding (shared socioeconomic factors elevate both low-HH and harassment risk), or reverse causation (harassment exposure depresses HH scores via cynicism) cannot be distinguished from observational data alone. Our counterfactual analysis is designed to identify which of these mechanisms is *not* the dominant one — namely, by testing whether shifting individuals' HEXACO scores reduces aggregate prevalence — but is not designed to adjudicate among the remaining alternatives.

### 1.3 Microsimulation as a methodological bridge

Microsimulation methods, originating in demographic and economic policy analysis (Spielauer, 2011) and extended into epidemiology and public health (Rutter et al., 2011), allow researchers to project individual-level data into population-level prevalence estimates under counterfactual scenarios. The methodological strength of microsimulation lies in its ability to combine (a) detailed individual-level inputs, (b) population-level reweighting via official census or labor-force statistics, and (c) explicit counterfactual interventions with causal interpretation. In contrast to purely statistical adjustment or survey reweighting, microsimulation makes the implementation of "what-if" interventions explicit and falsifiable.

The present study adopts the framework of *target trial emulation* (Hernán & Robins, 2016, 2020) to formalize the counterfactual interventions of interest. Target trial emulation requires four identifying assumptions: (i) consistency between observed and counterfactual treatments, (ii) exchangeability conditional on measured covariates, (iii) positivity of treatment assignment, and (iv) absence of interference between units (SUTVA; Hudgens & Halloran, 2008). We address each assumption explicitly in the Methods and Discussion.

Counterfactual contrasts are formulated using the Pearl (2009) do-operator. Specifically, we define three pre-registered interventions:

- **Counterfactual A** (personality intervention): do(HH ← HH + 0.3 σ_HH; A ← A + 0.3 σ_A; E ← E + 0.3 σ_E). The +0.3 SD calibration is derived from the upper bound of effect sizes reported in personality-training meta-analyses (Hudson & Fraley, 2015; Roberts et al., 2017).
- **Counterfactual B** (cluster reassignment): do(cluster ∈ {0, 4, 6}). Individuals are computationally reassigned to one of three low-harassment-prevalence typologies, retaining their original HEXACO scores. This intervention represents an extreme (and arguably implausible) personality-shift scenario, included as an upper bound on what any person-level intervention could achieve.
- **Counterfactual C** (structural reform): do(P_{c, x = power} ← 0.8 × P_{c, x = power}). Cell-level propensity is reduced by 20% uniformly across all cluster × gender cells, calibrated against three meta-analyses of organizational anti-harassment interventions (Escartín, 2016; Hodgins, MacCurtain, & Mannix-McNamara, 2014; Salin, 2021), as well as the empirical −13.2 percentage-point reduction observed in the MHLW survey series between FY2016 and FY2023, spanning the 2020 Power Harassment Prevention Law's staged enforcement.

The asymmetry between A/B and C — namely, that A and B intervene on personality while C intervenes on the propensity-environment pathway — operationalizes the question: *given comparable effect sizes drawn from prior literature on each intervention type, which produces a larger population-level prevalence reduction?*

### 1.4 Hypotheses

The pre-registration v2.0 (OSF DOI 10.17605/OSF.IO/3Y54U) specifies seven primary hypotheses:

- **H1 (latent prevalence prediction)**: The 14-cell HEXACO 7-typology × gender model, when re-weighted using Statistics Bureau (Ministry of Internal Affairs and Communications) Labor Force Survey 2022 marginals, predicts national past-three-year power-harassment prevalence with mean absolute percentage error (MAPE) ≤ 30% relative to MHLW survey targets across FY2016, FY2020, and FY2023 (4-tier judgment hierarchy: Strict SUCCESS, Standard SUCCESS, PARTIAL SUCCESS, FAILURE).
- **H2 (baseline ordinal hierarchy)**: The progression of model sophistication from B0 (uniform = MHLW grand mean) through B4 (extended with age and age × cluster interactions) shows monotonically improving MAPE_FY2016, tested via Bonferroni-Holm pairwise inequalities and Page's L (1963) trend test.
- **H3 (centroid concordance)**: The IEEE-published centroids fitted in the N = 13,668 sample are reproduced within sampling error in the N = 354 subsample (Tokiwa, 2025).
- **H4 (cell-stratified internal consistency)**: Per-cell propensity estimates p̂_c are stable across alternative bootstrap procedures within each cell.
- **H5 (gender invariance)**: The HEXACO-harassment association is qualitatively similar across genders (verified via 14-cell vs 28-cell sensitivity).
- **H6 (cross-domain triangulation)**: Power and gender harassment scales (Tou, 2017; Kobayashi & Tanaka, 2010) yield concordant cluster-level rankings.
- **H7 (counterfactual ordering)**: The pre-registered ordering Δ_BA = ΔP_B − ΔP_A and Δ_BC = ΔP_B − ΔP_C is tested via the Berger and Hsu (1996) intersection-union test (IUT) at one-sided α = .05; the configuration is classified as REVERSAL, CONFIRMED, PARTIAL, or AMBIGUOUS.

### 1.5 Contribution and roadmap

This study makes four contributions:

1. **Methodological**: It demonstrates an end-to-end pre-registered counterfactual microsimulation pipeline for workplace harassment research, with full reproducibility (HDF5 artifacts, fixed seed = 20260429, Docker container, MIT-licensed code) and open data (N = 354 deposited at OSF; centroids retrieved from IEEE clustering paper supplement).
2. **Empirical**: It provides the first quantitative apples-to-apples comparison of person-level and system-level intervention effects on Japanese workplace harassment prevalence, leveraging effect-size calibrations from prior meta-analyses.
3. **Substantive**: It rejects the conventional reading of HEXACO 7-typology as an intervention target, repositioning it as a stratification lens for identifying differentially exposed populations.
4. **Policy-relevant**: It provides counterfactual evidence that the 2020 Power Harassment Prevention Law's structural mechanism is approximately 5.8× more effective at the population level than the personality-training interventions to which Japanese corporate HR programs allocate the majority of harassment-prevention budget.

The remainder of this manuscript is organized as follows. Section 2 details the data sources, sample, and pre-registered analytic procedures. Section 3 reports the 14-cell propensities, national prevalence aggregation, four-tier H1 classification, baseline comparison, sensitivity sweep, and counterfactual contrasts. Section 4 discusses methodological limitations (reduced-form framing, causal under-identification, heterogeneity neglect, static framing) and sketches future agent-based extensions. Section 5 concludes with policy implications. All code, data, and intermediate artifacts are publicly archived at OSF (DOI 10.17605/OSF.IO/3Y54U) and GitHub (https://github.com/etoki/paper).
