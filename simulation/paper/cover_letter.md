# Cover Letter

**Submission Type:** Stage 2 Registered Report
**Manuscript:** *Personality Interventions Don't Work; Structural Ones Do: Counterfactual Evidence from a HEXACO 7-Typology Workplace Harassment Microsimulation in Japan*
**Author:** Eisuke Tokiwa (sole-authored)
**Pre-registration:** OSF DOI [10.17605/OSF.IO/3Y54U](https://osf.io/3y54u) (v2.0; registered 2026-04-30)

---

[Date]

The Editors
*XXXX*

Dear Editors,

I am pleased to submit the enclosed manuscript, *Personality Interventions Don't Work; Structural Ones Do: Counterfactual Evidence from a HEXACO 7-Typology Workplace Harassment Microsimulation in Japan*, for consideration as a Stage 2 Registered Report at *XXXX*.

This submission implements, without deviation, the analysis pre-registered as v2.0 at the Open Science Framework (DOI [10.17605/OSF.IO/3Y54U](https://osf.io/3y54u); registered 2026-04-30). All seven primary hypotheses (H1–H7) are evaluated against pre-registered classification rules; all sensitivity sweeps and diagnostic procedures specified in the v2.0 master are executed and reported. The Methods Clarifications Log accompanying the v2.0 registration (Section 6.5 Level 1 deviation; SHA-256 fcaaf0d…) documents 16 minor specification clarifications resolved during the Stage 1 protocol-development phase.

## Headline finding

Using N = 354 individual-level Japanese workplace harassment data, partitioned into 14 cells (7 HEXACO clusters × 2 genders) and re-weighted with Statistics Bureau (MIC) Labor Force Survey 2022 marginals, we tested three pre-registered counterfactual interventions: (A) personality intervention shifting Honesty-Humility, Agreeableness, and Emotionality by +0.3 SD; (B) targeted personality intervention applying +0.40 SD on Honesty-Humility to individuals already in the low-prevalence HEXACO Cluster {0, 4, 6} subgroup; and (C) structural intervention reducing cell-level propensity by 20%, calibrated against three meta-analyses and the MHLW FY2016–FY2023 natural experiment spanning the staged enforcement of the 2020 Power Harassment Prevention Law.

The pre-registered intersection-union test (Berger & Hsu, 1996) classifies the configuration as **REVERSAL**: ΔP_A and ΔP_B are statistically null, whereas ΔP_C = +3.49 percentage points (95% CI [+2.64, +4.35] excluding zero). The classification is robust across all four pre-registered cultural attenuation factors (F = 0.3 to 1.0), including the conservative cross-cultural worst-case anchor. Numerically, the structural intervention's population-level effect is approximately 5.7× larger than the personality intervention's, given prior meta-analytic effect-size calibrations.

## Substantive contribution

Three contributions distinguish this work:

1. **Methodological**: The first end-to-end pre-registered counterfactual microsimulation of Japanese workplace harassment, with full reproducibility (HDF5 artifacts, fixed seed = 20260429, Docker container, MIT-licensed code) and open data.
2. **Empirical**: The first quantitative apples-to-apples comparison of person-level and system-level intervention effects on a population-level harassment-prevalence metric, leveraging effect-size calibrations from prior meta-analyses.
3. **Substantive reframing**: We reposition HEXACO 7-typology as a *stratification variable* identifying differentially exposed populations rather than as an *intervention target*. This framing rejects victim-blaming readings of harassment exposure and aligns with workplace ethics literature.

The convergence of our pre-registered counterfactual evidence with the MHLW FY2016–FY2023 natural experiment (−13.2 pp post-Power-Harassment-Prevention-Law decline) provides a rare quasi-validation of our policy-relevant claim.

## Compliance with Stage 2 RR review criteria

- **Pre-registration adherence**: All deviations from the v2.0 master are documented in the Methods Clarifications Log (locked 2026-04-30 alongside the v2.0 registration) and disclosed in §2.2 of the manuscript. No undocumented deviations occurred.
- **Hypotheses tested**: All seven pre-registered hypotheses (H1–H7) are evaluated; outcomes are reported per pre-registered classification rules in Table 6.
- **Reproducibility**: All code is publicly archived under MIT license at [https://github.com/etoki/paper](https://github.com/etoki/paper) (directory: `simulation/`); intermediate HDF5 artifacts at OSF; SHA-256 hash-verified reproduction via `make reproduce`.
- **Data availability**: Public-tier supplementary artifacts (Stage 0–8 HDF5, Figures 1–6 in PNG/PDF/SVG, canonical numerical record, SHA-256 reference hashes) are openly available at the v2.0 OSF working project (osf.io/3hxz6, `v2.0/v2.0_supplementary.tar.gz`). The N = 354 individual-level dataset is governed by the IRB-approved data-sharing protocol and hosted in a Private OSF component with a documented Request-Access mechanism (Wiki at the component page lists the access criteria, IRB requirements, and four data-use terms). Population reweighting data (Statistics Bureau / MIC Labor Force Survey 2022) are publicly archived in the GitHub repository.

## Limitations openly disclosed

The Discussion section (§4.4) discloses six limitations explicitly:

1. **Reduced-form framing of Counterfactual C** (population-level statistical translation of structural change; mechanistic deterrence dynamics not endogenously modeled);
2. **Causal under-identification** (selection, confounding, and reverse causation cannot be distinguished from M1 direct causation, only the latter rejected);
3. **Heterogeneity in counterfactual response** (uniform −20% across cells; cell-specific response not estimated);
4. **Static framing** (no temporal dynamics);
5. **m8 limitation** (cluster proportions M3-fixed at Tokiwa 2026 *IEEE Access* values);
6. **Sample size** (N = 354; cell-level CIs adequate but not extensively replicated).

We position these limitations as future-work targets rather than threats to the Stage 2 conclusion, which depends only on the apples-to-apples comparison of pre-registered effect-size calibrations.

## Author and conflict-of-interest statement

This is a sole-authored submission. The author (Eisuke Tokiwa, ORCID [0009-0009-7124-6669](https://orcid.org/0009-0009-7124-6669)) declares no competing interests. The research was self-funded by SUNBLAZE Co., Ltd. (Tokyo, Japan); no external funding was received.

## Suggested reviewers

Per *XXXX* guidelines, the following experts have publicly stated expertise relevant to this manuscript and have no known conflict of interest with the author:

- *[Reviewer 1]* — workplace bullying / structural-intervention meta-analyses
- *[Reviewer 2]* — HEXACO model and cross-cultural personality differences
- *[Reviewer 3]* — target trial emulation and counterfactual epidemiology
- *[Reviewer 4]* — Japanese occupational health psychology

We respectfully request that reviewers familiar with both the HEXACO model and the Stage 2 Registered Report process be considered.

## Closing

I confirm that this manuscript has not been previously published, is not under consideration elsewhere, and that all authors (sole-authored) have agreed to the submission. Code, data, and intermediate artifacts will remain publicly accessible at the indicated OSF DOI and GitHub repository for the duration of peer review and beyond.

I look forward to the editorial team's response and the peer-review process.

Sincerely,

**Eisuke Tokiwa**
SUNBLAZE Co., Ltd.
Tokyo, Japan
Email: eisuke.tokiwa@sunblaze.jp
ORCID: [0009-0009-7124-6669](https://orcid.org/0009-0009-7124-6669)
GitHub: [https://github.com/etoki/paper](https://github.com/etoki/paper)
OSF: [https://osf.io/3y54u](https://osf.io/3y54u)
