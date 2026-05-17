# Cover Letter — Journal of Occupational Health Psychology (JOHP)

**Submission Type:** Pre-registered empirical research article
**Manuscript:** *Person-Level versus System-Level Anti-Harassment Interventions: A HEXACO 7-Typology Counterfactual Microsimulation in Japanese Workplaces*
**Author:** Eisuke Tokiwa (sole-authored)
**Pre-registration:** OSF DOI [10.17605/OSF.IO/3Y54U](https://osf.io/3y54u) (v2.0; registered 2026-04-30)
**Preprint:** SocArXiv DOI [10.31235/osf.io/p2d8w_v1](https://osf.io/preprints/socarxiv/p2d8w_v1) (posted 2026-05-08)

---

[Date]

The Editors
*Journal of Occupational Health Psychology* (American Psychological Association)

Dear Editors,

I am pleased to submit the enclosed manuscript, *Person-Level versus System-Level Anti-Harassment Interventions: A HEXACO 7-Typology Counterfactual Microsimulation in Japanese Workplaces*, for consideration at the *Journal of Occupational Health Psychology*.

This submission implements, without deviation, the analysis pre-registered as v2.0 at the Open Science Framework (DOI [10.17605/OSF.IO/3Y54U](https://osf.io/3y54u); registered 2026-04-30). All seven primary hypotheses (H1–H7) are evaluated against pre-registered classification rules; all sensitivity sweeps and diagnostic procedures specified in the pre-registration are executed and reported. The Methods Clarifications Log accompanying the v2.0 registration documents 16 minor specification clarifications resolved during the Stage 1 protocol-development phase.

## Headline finding

Using N = 354 individual-level Japanese workplace harassment data, partitioned into 14 cells (7 HEXACO clusters × 2 genders) and re-weighted with Statistics Bureau (Ministry of Internal Affairs and Communications) Labour Force Survey 2022 marginals, we tested three pre-registered counterfactual interventions: (A) +0.3 SD on Honesty-Humility, Agreeableness, and Emotionality for all individuals; (B) +0.40 SD on Honesty-Humility restricted to individuals in three low-prevalence HEXACO typologies; and (C) a 20% cell-level propensity reduction, calibrated against three meta-analyses of organizational anti-harassment interventions and the MHLW FY2016–FY2023 −13.2-percentage-point decline.

The pre-registered intersection-union test (Berger & Hsu, 1996) classifies the configuration as **REVERSAL**: the personality-intervention contrasts are statistically null (ΔP_A = −0.0061, 95% CI [−0.0207, +0.0062]; ΔP_B = −0.0059, 95% CI [−0.0207, +0.0066]), whereas the structural-intervention contrast is positive (ΔP_C = +0.0349, 95% CI [+0.0264, +0.0435]). In absolute magnitude the structural-pathway contrast is approximately five-fold larger than the personality-pathway contrast (|ΔP_C|/|ΔP_A| ≈ 5.7). The classification is robust across four pre-registered cultural-attenuation factors and across nine pre-registered analytic choices. We interpret these contrasts conditional on the prior meta-analytic and natural-experiment effect sizes used to calibrate each counterfactual.

## Why *Journal of Occupational Health Psychology*

JOHP's I-O psychology and occupational-health readership is the primary audience this work targets:

1. **Workplace-harassment substantive scope.** The manuscript addresses past-three-year power-harassment victimization in Japan, where MHLW survey rates declined from 32.5% (FY2016) to 19.3% (FY2023) over the enforcement period of the 2020 Power Harassment Prevention Law. JOHP's coverage of workplace stressors, harassment, and organizational interventions is the natural fit.
2. **Workplace-bullying intervention literature.** Counterfactual C's calibration draws on three meta-analyses central to JOHP's reader base — Hodgins, MacCurtain, & Mannix-McNamara (2014) systematic review; Escartín (2016) on psychosocial drivers and effective interventions; and Salin (2021) on workplace bullying and gender. The present analysis adds a common-metric comparison between personality-pathway and structural-pathway intervention contrasts that has not, to our knowledge, been quantitatively reported in this literature.
3. **Theoretical contribution to person–environment debate.** Our findings are consistent with treating HEXACO 7-typology as a stratification variable identifying differentially exposed subpopulations rather than as a target for individual-level personality modification. This contributes to ongoing JOHP-adjacent discussion about the conditions under which trait-based prediction translates into trait-based intervention.

## Substantive contribution

Four contributions distinguish this work:

1. **Methodological.** An end-to-end pre-registered counterfactual microsimulation pipeline applicable to workplace-harassment research, with full reproducibility (HDF5 artifacts, fixed random seed, Docker container, MIT-licensed code) and tiered open data.
2. **Empirical.** A quantitative apples-to-apples comparison of person-level and system-level intervention contrasts on Japanese workplace-harassment prevalence, with each counterfactual calibrated to effect sizes drawn from prior meta-analyses.
3. **Theoretical.** A test of the implication of treating HEXACO 7-typology as an intervention target versus as a stratification lens, with the data consistent with the latter reading.
4. **Substantive.** A documented quantitative ordering — the absolute magnitude of the structural-intervention contrast is approximately five-fold larger than that of the personality-intervention contrast — under realistic prior calibrations of each, robust across nine pre-registered analytic axes and four cultural-attenuation factors.

The MHLW FY2016 → FY2023 −13.2-percentage-point decline that overlaps with the staged enforcement of the 2020 Power Harassment Prevention Law is treated in the manuscript as substantive context rather than as a formal causal estimand of the law's effect; the present cross-sectional design does not support a formal causal inference about the law itself.

## Compliance with pre-registration and reproducibility criteria

- **Pre-registration adherence.** All deviations from the v2.0 master are documented in the Methods Clarifications Log (locked alongside the v2.0 registration) and disclosed in the Methods (Pre-registered analytic pipeline) section of the manuscript. No undocumented deviations occurred.
- **Hypotheses tested.** All seven pre-registered hypotheses (H1–H7) are evaluated; outcomes are reported per pre-registered classification rules in Table 6.
- **Reproducibility.** All code is publicly archived under MIT license at [https://github.com/etoki/paper](https://github.com/etoki/paper) (directory: `simulation/`); intermediate HDF5 artifacts at OSF; SHA-256 hash-verified reproduction via `make reproduce`.
- **Data availability.** Public-tier supplementary artifacts (Stage 0–8 HDF5, Figures 1–6 in PNG/PDF/SVG, canonical numerical record, SHA-256 reference hashes) are openly available at the v2.0 OSF working project (osf.io/3hxz6). The N = 354 individual-level dataset is governed by the IRB-approved data-sharing protocol and hosted in a Private OSF component with a documented Request-Access mechanism; the parent project's public Wiki (osf.io/3hxz6/wiki/home/) lists the access criteria, IRB requirements, and four data-use terms. Statistics Bureau / MIC Labour Force Survey 2022 population-reweighting data are publicly archived in the GitHub repository.

## Limitations openly disclosed

The Discussion (Limitations subsection) discloses six limitations explicitly:

1. **Reduced-form framing of Counterfactual C** (population-level statistical translation of structural change; mechanistic deterrence dynamics not endogenously modeled);
2. **Causal under-identification** (selection, confounding, and reverse causation cannot be distinguished from M1 direct causation; the data are inconsistent with M1 as the dominant mechanism but cannot adjudicate among M2–M4);
3. **Heterogeneity in counterfactual response** (uniform −20% across cells; cell-specific response not estimated);
4. **Static framing** (no temporal dynamics);
5. **Cluster-proportion limitation** (cluster proportions remain fixed at the Tokiwa (2026) values for N = 13,668 because the Labour Force Survey does not capture HEXACO cluster membership);
6. **Sample size** (N = 354; cell-level CIs adequate but not extensively replicated).

We position these limitations as future-work targets rather than threats to the present substantive conclusion, which depends only on the apples-to-apples comparison of pre-registered effect-size calibrations.

## Author, funding, and competing-interests statement

This is a sole-authored submission. The author (Eisuke Tokiwa, ORCID [0009-0009-7124-6669](https://orcid.org/0009-0009-7124-6669)) discloses the following potential competing interest: the author is the representative of SUNBLAZE Co., Ltd. (Tokyo, Japan), which provides HEXACO-JP, a proprietary Japanese-language HEXACO-based personality assessment service. The present study does **not** use HEXACO-JP; it uses the HEXACO-PI-R Japanese 60-item adaptation published by Wakabayashi (2014) as a separately validated, openly cited instrument. The manuscript does not reference, evaluate, market, recommend, or otherwise promote HEXACO-JP or any other SUNBLAZE product or service. The research received no SUNBLAZE funding, no external grant funding, and was conducted as an independent academic project. The headline empirical findings (null personality-intervention contrast, positive structural-intervention contrast, and the substantive ordering reported in the manuscript) do not differentially favor commercial HEXACO-based assessment products.

## Suggested reviewers

Per *Journal of Occupational Health Psychology* guidelines, the following authors of works cited in the manuscript have publicly demonstrated expertise relevant to this submission and have no known conflict of interest with the author:

- **Denise Salin** (Hanken School of Economics, Finland) — author of Salin (2003) on enabling/motivating/precipitating processes in workplace bullying and Salin (2021) on bullying and gender; central calibration source for Counterfactual C.
- **Jordi Escartín** (University of Barcelona, Spain) — author of Escartín (2016) on psychosocial drivers and effective workplace-bullying interventions; second meta-analytic anchor for Counterfactual C.
- **Margaret Hodgins** (University of Galway, Ireland) — first author of Hodgins, MacCurtain, & Mannix-McNamara (2014) systematic review of workplace bullying and incivility interventions in the *International Journal of Workplace Health Management*; third anchor for Counterfactual C.
- **M. Sandy Hershcovis** (University of Calgary, Canada) — first author of Hershcovis & Barling (2010) meta-analytic review of workplace aggression outcomes; foundational citation for the public-health-cost framing.
- **Morten Birkeland Nielsen** (National Institute of Occupational Health, Norway) — first author of Nielsen et al. (2017) meta-analysis on workplace harassment and the Five Factor Model; anchor for the F = 0.5 transportability factor.

We respectfully request that reviewers familiar with workplace-bullying intervention literature, pre-registered methodology, and Japanese occupational-health contexts be considered.

## Closing

I confirm that this manuscript has not been previously published in any peer-reviewed venue and is not under consideration at any other journal. A non-peer-reviewed version has been posted as a preprint at SocArXiv (DOI 10.31235/osf.io/p2d8w_v1) per *Journal of Occupational Health Psychology* preprint policy. The submission is sole-authored. Code, data, and intermediate artifacts will remain publicly accessible at the indicated OSF DOI and GitHub repository for the duration of peer review and beyond.

I look forward to the editorial team's response and the peer-review process.

Sincerely,

**Eisuke Tokiwa**
SUNBLAZE Co., Ltd.
Tokyo, Japan
Email: eisuke.tokiwa@sunblaze.jp
ORCID: [0009-0009-7124-6669](https://orcid.org/0009-0009-7124-6669)
GitHub: [https://github.com/etoki/paper](https://github.com/etoki/paper)
OSF: [https://osf.io/3y54u](https://osf.io/3y54u)
