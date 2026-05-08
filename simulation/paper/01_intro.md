# 01. Abstract + Introduction

**Title:**
*Person-Level versus System-Level Anti-Harassment Interventions: A HEXACO 7-Typology Counterfactual Microsimulation in Japanese Workplaces*

**Author**: Eisuke Tokiwa
**Affiliation**: SUNBLAZE Co., Ltd., Tokyo, Japan
**ORCID**: 0009-0009-7124-6669
**Correspondence**: eisuke.tokiwa@sunblaze.jp
**Pre-registration**: OSF DOI 10.17605/OSF.IO/3Y54U (v2.0; locked 2026-04-30)
**Preprint**: SocArXiv DOI 10.31235/osf.io/p2d8w_v1 (posted 2026-05-08)
**Reporting standard**: Stage 2 Registered Report (post-data) following the v2.0 pre-registration

<!-- Note (markdown source): Funding, competing interests, ethics, authors' contributions, data availability, and acknowledgments are rendered in the Author Note / Declarations section of the docx by build_docx.py / build_ieee_docx.py from constants in those scripts; they do not need to be repeated in this markdown file. -->


---

## Abstract

Workplace harassment imposes mental-health and labor-supply costs in Japan, where Ministry of Health, Labour and Welfare (MHLW) surveys document past-three-year power-harassment victimization declining from 32.5% (FY2016) to 19.3% (FY2023) across the enforcement period of the 2020 Power Harassment Prevention Law. The relative population-level magnitudes of plausible person-level and system-level interventions in this setting have not been compared in a pre-registered counterfactual framework. We pre-registered a HEXACO 7-typology microsimulation emulating a target trial of three interventions: (A) +0.3 SD on Honesty-Humility, Agreeableness, and Emotionality; (B) +0.40 SD on Honesty-Humility restricted to three low-prevalence typologies; and (C) a 20% cell-level propensity reduction, calibrated against three meta-analyses of organizational anti-harassment interventions and the −13.2-percentage-point MHLW FY2016–FY2023 decline. Using N = 354 observations partitioned into 14 cells (7 typologies × 2 genders) with cell-stratified bootstrap (B = 10,000), Beta-Binomial empirical Bayes shrinkage, and a BCa confidence-interval cascade, the personality contrasts are null (ΔP_A = −0.0061, 95% CI [−0.0207, +0.0062]; ΔP_B = −0.0059, 95% CI [−0.0207, +0.0066]), whereas the structural contrast is positive (ΔP_C = +0.0349, 95% CI [+0.0264, +0.0435]). The pre-registered intersection-union test classifies this as REVERSAL: in absolute magnitude, |ΔP_C|/|ΔP_A| ≈ 5.7. The classification is robust across four cultural-attenuation factors and nine pre-registered analytic choices. The findings are consistent with treating HEXACO 7-typology as a stratification variable rather than a personality-modification target, and we caution against reading HEXACO–harassment associations as endorsing person-focused remediation.

**Keywords**: workplace harassment; HEXACO; counterfactual simulation; target trial emulation; Japan; pre-registration; structural intervention; personality intervention; 2020 Power Harassment Prevention Law

---

## Introduction

### Workplace harassment as a public-health and policy problem

Workplace harassment has been associated with elevated risk of psychiatric morbidity, voluntary turnover, and reduced labor productivity in industrialized economies (Hershcovis & Barling, 2010; Salin, 2003). In Japan, the Ministry of Health, Labour and Welfare (MHLW) has documented past-three-year power-harassment victimization rates of 32.5% in FY2016, 31.4% in FY2020, and 19.3% in FY2023 (MHLW, 2017, 2021, 2024). The 13.2 percentage-point decline between FY2016 and FY2023 coincides with the staged enforcement of the 2020 Power Harassment Prevention Law, which mandated formal anti-harassment policies, employee-accessible reporting channels, and prohibited retaliation against complainants — first for large enterprises in June 2020, and subsequently extended to small-and-medium enterprises in April 2022. This temporal coincidence has been treated in the policy literature as a natural-experiment-like setting; it provides a substantive backdrop for the present pre-registered analysis, although the present study is observational and not a formal causal inference about the law itself.

A long-standing theoretical question in workplace-psychology research concerns the relative explanatory weight of person-level factors — individual personality dispositions associated with harassment perpetration or victimization — versus system-level factors that involve organizational structure, accountability mechanisms, and power asymmetries (Mikkelsen et al., 2020; Salin, 2021). The former framing treats harassment as primarily driven by individual disposition; the latter treats it as primarily driven by environmental affordance. Despite extensive research linking personality traits to bullying and harassment (Nielsen et al., 2017; Pilch & Turska, 2015), the relative population-level magnitude of plausible person-level and system-level intervention contrasts has not been quantitatively compared in a counterfactually formal framework calibrated against prior meta-analytic effect sizes.

### HEXACO 7-typology and its prior connection to workplace harassment

The HEXACO model of personality (Ashton & Lee, 2007) extends the classical Big Five framework with an additional dimension, Honesty-Humility (HH), which captures fairness, modesty, sincerity, and resistance to exploiting others. Within the HEXACO literature, low Honesty-Humility, low Agreeableness, and high Emotionality have repeatedly been associated with workplace bullying perpetration and victimization (Glasø et al., 2007; Lee et al., 2013; Linton & Power, 2013; Nielsen et al., 2017). However, the magnitude of these associations is generally modest (uncorrected r typically in the range 0.10–0.30; Nielsen et al., 2017), and recent meta-analyses suggest considerable cross-cultural heterogeneity in workplace bullying acceptability and effect sizes (Power et al., 2013, comparing six continents; Nielsen et al., 2017, reporting attenuated Asia/Oceania effects of r = .16 versus European r = .33 for Neuroticism–bullying associations).

In a companion study, the author identified seven robust HEXACO clusters in a Japanese sample of N = 13,668 (Tokiwa, 2026; herein "the centroid study"). The seven typologies — labeled here as types 0 through 6 — span the joint distribution of HEXACO domains and provide a parsimonious taxonomy for categorizing individuals according to their personality profile. The present study leverages these published centroids as fixed parameters for stratifying a smaller harassment-focused subsample (N = 354; Tokiwa, 2025).

**A critical reframing.** The role of HEXACO in this study is *not* to identify who should be subjected to personality interventions. Rather, HEXACO 7-typology serves as a *stratification variable* that identifies which subgroups are differentially exposed to workplace harassment. Whether this differential exposure reflects direct personality causation (low-HH individuals harass or are harassed at higher rates), selection (low-HH individuals self-select into harassment-prone industries), confounding (shared socioeconomic factors elevate both low-HH and harassment risk), or reverse causation (harassment exposure depresses HH scores via cynicism) cannot be distinguished from observational data alone. Our counterfactual analysis is designed to identify which of these mechanisms is *not* the dominant one — namely, by testing whether shifting individuals' HEXACO scores reduces aggregate prevalence — but is not designed to adjudicate among the remaining alternatives.

### Microsimulation as a methodological bridge

Microsimulation methods, originating in demographic and economic policy analysis (Spielauer, 2011) and extended into epidemiology and public health (Rutter et al., 2011), allow researchers to project individual-level data into population-level prevalence estimates under counterfactual scenarios. The methodological strength of microsimulation lies in its ability to combine (a) detailed individual-level inputs, (b) population-level reweighting via official census or labor-force statistics, and (c) explicit counterfactual interventions with causal interpretation. In contrast to purely statistical adjustment or survey reweighting, microsimulation makes the implementation of "what-if" interventions explicit and falsifiable.

The present study adopts the framework of *target trial emulation* (Hernán & Robins, 2016, 2020) to formalize the counterfactual interventions of interest. Target trial emulation requires four identifying assumptions: (i) consistency between observed and counterfactual treatments, (ii) exchangeability conditional on measured covariates, (iii) positivity of treatment assignment, and (iv) absence of interference between units (SUTVA; Hudgens & Halloran, 2008). We address each assumption explicitly in the Methods and Discussion.

Counterfactual contrasts are formulated using the Pearl (2009) do-operator. Specifically, we define three pre-registered interventions:

- **Counterfactual A** (personality intervention): do(HH ← HH + 0.3 σ_HH; A ← A + 0.3 σ_A; E ← E + 0.3 σ_E). The +0.3 SD calibration is derived from the upper bound of effect sizes reported in personality-training meta-analyses (Hudson & Fraley, 2015; Roberts et al., 2017).
- **Counterfactual B** (targeted personality intervention on low-prevalence subgroup): do(HH := HH + 0.40 σ_HH) restricted to individuals whose baseline cluster is already in the pre-registered low-prevalence target set {0, 4, 6}; HEXACO scores of all other individuals are unchanged. After the targeted shift, all individuals are re-classified by nearest-centroid on the post-intervention HEXACO matrix and per-cell propensities are re-computed. This intervention represents a subgroup-targeted personality-amplification scenario rather than a forced reassignment of the full sample, and serves as a falsification benchmark for any person-level intervention focused on the lowest-prevalence typologies.
- **Counterfactual C** (structural reform): do(P_{c, x = power} ← 0.8 × P_{c, x = power}). Cell-level propensity is reduced by 20% uniformly across all cluster × gender cells, calibrated against three meta-analyses of organizational anti-harassment interventions (Escartín, 2016; Hodgins et al., 2014; Salin, 2021), as well as the empirical −13.2 percentage-point reduction observed in the MHLW survey series between FY2016 and FY2023, spanning the 2020 Power Harassment Prevention Law's staged enforcement.

The asymmetry between A/B and C — namely, that A and B intervene on personality while C intervenes on the propensity-environment pathway — operationalizes the question: *given comparable effect sizes drawn from prior literature on each intervention type, which produces a larger population-level prevalence reduction?*

### Hypotheses

The pre-registration (v2.0) specifies seven primary hypotheses:

- **H1 (latent prevalence prediction)**: The 14-cell HEXACO 7-typology × gender model, when re-weighted using Statistics Bureau (Ministry of Internal Affairs and Communications) Labor Force Survey 2022 marginals, predicts national past-three-year power-harassment prevalence with mean absolute percentage error (MAPE) ≤ 30% relative to MHLW survey targets across FY2016, FY2020, and FY2023 (4-tier judgment hierarchy: Strict SUCCESS, Standard SUCCESS, PARTIAL SUCCESS, FAILURE).
- **H2 (baseline ordinal hierarchy)**: The progression of model sophistication from B0 (uniform = MHLW grand mean) through B4 (extended with age and age × cluster interactions) shows monotonically improving MAPE_FY2016, tested via Bonferroni-Holm pairwise inequalities and Page's L (1963) trend test.
- **H3 (centroid concordance)**: The published centroids (Tokiwa, 2026) fitted in the N = 13,668 sample are reproduced within sampling error in the N = 354 subsample (Tokiwa, 2025).
- **H4 (cell-stratified internal consistency)**: Per-cell propensity estimates p̂_c are stable across alternative bootstrap procedures within each cell.
- **H5 (gender invariance)**: The HEXACO-harassment association is qualitatively similar across genders (verified via 14-cell vs 28-cell sensitivity).
- **H6 (cross-domain triangulation)**: Power and gender harassment scales (Tou et al., 2017; Kobayashi & Tanaka, 2010) yield concordant cluster-level rankings.
- **H7 (counterfactual ordering)**: The pre-registered ordering Δ_BA = ΔP_B − ΔP_A and Δ_BC = ΔP_B − ΔP_C is tested via the Berger and Hsu (1996) intersection-union test (IUT) at one-sided α = .05; the configuration is classified as REVERSAL, CONFIRMED, PARTIAL, or AMBIGUOUS.

### Contribution and roadmap

This study makes four scientific contributions:

1. **Methodological**: It demonstrates an end-to-end pre-registered counterfactual microsimulation pipeline applicable to workplace-harassment research, with full reproducibility (HDF5 artifacts, fixed seed, Docker container, MIT-licensed code; tiered open data, with public aggregate artifacts and IRB-restricted individual-level access via a documented access-request mechanism). Underlying instruments and centroids are described, respectively, in Tokiwa (2025) and Tokiwa (2026); deposit details are provided in Methods (Data availability).
2. **Empirical**: It provides a quantitative apples-to-apples comparison of person-level and system-level intervention contrasts on Japanese workplace-harassment prevalence, with each counterfactual calibrated to effect sizes drawn from prior meta-analyses.
3. **Theoretical**: It tests the implication of treating HEXACO 7-typology as an intervention target versus as a stratification lens, and reports that the data are consistent with the latter reading. This contributes to ongoing discussions in personality and occupational-health psychology about the conditions under which trait-based prediction translates into trait-based intervention.
4. **Substantive**: It documents a quantitative ordering — the absolute magnitude of the structural-intervention contrast is approximately five-fold larger than that of the personality-intervention contrast — under realistic prior calibrations of each, and traces the robustness of that ordering across nine pre-registered analytic axes and four cultural-attenuation factors.

The remainder of this manuscript is organized as follows. Section 2 details the data sources, sample, and pre-registered analytic procedures, and reports the deposit locations and access conditions for code and data. Section 3 reports the 14-cell propensities, national prevalence aggregation, four-tier H1 classification, baseline comparison, sensitivity sweep, and counterfactual contrasts. Section 4 discusses methodological limitations (reduced-form framing, causal under-identification, heterogeneity neglect, static framing), sketches future agent-based extensions, and develops the research implications of the integrated findings.
