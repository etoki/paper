# Cover Letter — IEEE Access

**Submission Category:** Regular Article (Original Research)
**Manuscript:** *Person-Level versus System-Level Anti-Harassment Interventions: A HEXACO 7-Typology Counterfactual Microsimulation in Japanese Workplaces*
**Author:** Eisuke Tokiwa (sole-authored)
**Pre-registration:** OSF DOI [10.17605/OSF.IO/3Y54U](https://osf.io/3y54u) (v2.0; registered 2026-04-30)
**Companion paper:** Tokiwa, *IEEE Access*, 2026, doi: 10.1109/ACCESS.2026.3651324

---

[Date]

The Editor-in-Chief
*IEEE Access*
Institute of Electrical and Electronics Engineers (IEEE)

Dear Editor-in-Chief,

I am pleased to submit the enclosed manuscript, *Person-Level versus System-Level Anti-Harassment Interventions: A HEXACO 7-Typology Counterfactual Microsimulation in Japanese Workplaces*, for consideration as a regular article in *IEEE Access*. This work continues the computational-psychology research program initiated in my prior IEEE Access publication on HEXACO cluster typology in a non-WEIRD Japanese sample (doi: 10.1109/ACCESS.2026.3651324) and applies the same emphasis on methodological transparency and reproducibility that *IEEE Access* requires.

## Why *IEEE Access*

*IEEE Access* is the natural home for this work for three reasons:

1. **Methodological / computational density.** The manuscript implements an end-to-end nine-stage pre-registered pipeline combining (i) Bias-Corrected accelerated (BCa) bootstrap with a four-step priority-cascade fallback for degenerate cells, (ii) Beta-Binomial empirical-Bayes shrinkage with Method-of-Moments hyperprior diagnostics and three rejection triggers, (iii) the Berger–Hsu intersection-union test, (iv) target-trial-emulation counterfactual contrasts under explicit positivity diagnostics, and (v) a transportability factor sweep stress-testing classification robustness. This level of computational machinery — verified by SHA-256 hash-locked outputs (`make verify` PASS) on a Docker-pinned canonical platform — fits *IEEE Access*'s emphasis on technically rigorous, fully reproducible work.

2. **Open Access mandate alignment.** *IEEE Access*'s gold-OA model + CC-BY 4.0 license aligns with this study's pre-registration commitments (full GitHub source, OSF artifact deposit, public Wiki access mechanism for IRB-restricted individual data). Code (MIT-licensed) and HDF5 supplementary artifacts are already publicly archived; the readership best served by free, immediate access to such artifacts (engineering practitioners, occupational-health researchers, data scientists in HR analytics) is *IEEE Access*'s readership.

3. **Companion-paper continuity.** The seven-cluster HEXACO typology used as the fixed parameter set in this study was published in *IEEE Access* (Tokiwa, 2026, doi: 10.1109/ACCESS.2026.3651324). Submitting the present microsimulation extension to the same venue preserves bibliographic continuity for citing readers and integrates two stages of the same research program in one journal's permanent archive.

## Headline finding

Using N = 354 individual-level Japanese workplace-harassment data, partitioned into 14 cells (7 HEXACO clusters × 2 genders) and re-weighted with Statistics Bureau (Ministry of Internal Affairs and Communications) Labour Force Survey 2022 marginals, we tested three pre-registered counterfactual interventions:

- **(A)** universal personality intervention shifting Honesty-Humility, Agreeableness, and Emotionality by +0.3 SD;
- **(B)** targeted personality intervention applying +0.40 SD on Honesty-Humility to individuals already in the low-prevalence HEXACO Cluster {0, 4, 6} subgroup;
- **(C)** structural cell-level propensity reduction of 20 %, calibrated against three meta-analyses and the MHLW FY2016–FY2023 natural experiment spanning the staged enforcement of the 2020 Power Harassment Prevention Law.

The pre-registered intersection-union test classifies the configuration as **REVERSAL**: ΔP_A = −0.0061 ([−0.0207, +0.0062]) and ΔP_B = −0.0059 ([−0.0207, +0.0066]) are statistically null, whereas ΔP_C = +0.0349 ([+0.0264, +0.0435]) is positive and robust. The classification is preserved across all four pre-registered cultural-attenuation factors (F = 0.3 to 1.0). Numerically, the structural intervention's population-level effect is approximately 5.7× larger than the personality intervention's, given prior meta-analytic effect-size calibrations.

## Substantive contribution

Three contributions distinguish this work, each speaking directly to *IEEE Access*'s methodological-breadth criteria:

1. **Methodological:** the first end-to-end pre-registered counterfactual microsimulation of Japanese workplace harassment, with full reproducibility (HDF5 artifacts, fixed seed = 20260429, Docker container, MIT-licensed code, SHA-256 hash-locked outputs, 56-test pytest suite all passing).
2. **Empirical:** the first quantitative apples-to-apples comparison of person-level and system-level intervention effects on a population-level harassment-prevalence metric, leveraging effect-size calibrations from prior meta-analyses.
3. **Substantive reframing:** the manuscript repositions HEXACO 7-typology as a *stratification variable* identifying differentially exposed populations rather than as an *intervention target*, and cautions against reading observational HEXACO–harassment associations as endorsing person-focused remediation.

The temporal coincidence between the MHLW FY2016 → FY2023 −13.2-percentage-point decline and the staged enforcement of the 2020 Power Harassment Prevention Law is treated in the manuscript as substantive context rather than as a formal causal estimand of the law's effect.

## IEEE Access policy compliance

- **Originality:** The manuscript has not been previously published, in whole or in part, and is not under consideration at any other journal. The companion *IEEE Access* publication (Tokiwa 2026) is a distinct paper covering the cluster-typology derivation; the present submission is a microsimulation analysis using the previously-published cluster centroids as fixed parameters.
- **Reproducibility:** Source code is publicly archived under MIT license at <https://github.com/etoki/paper> (directory: `simulation/`); supplementary artifacts at OSF DOI 10.17605/OSF.IO/3Y54U; SHA-256 hash-verified reproduction via `make verify`. The Docker container plus pinned `uv.lock` ensure byte-identical regeneration on the canonical platform.
- **Ethics:** The N = 354 individual-level data re-analyzed in this study were originally collected under an IRB-approved protocol (Tokiwa, 2025, Research Square preprint, doi: 10.21203/rs.3.rs-7756124/v1). Secondary analysis of de-identified records does not require additional ethics review.
- **Competing Interests:** The author discloses the following potential competing interest: the author is the founder and representative of SUNBLAZE Co., Ltd. (Tokyo, Japan), which provides HEXACO-JP, a proprietary Japanese-language HEXACO-based personality assessment service. Three points qualify the disclosure. (i) The present study does **not** use HEXACO-JP; the HEXACO instrument used here is the HEXACO-PI-R Japanese 60-item adaptation published by Wakabayashi (2014), a separately validated and openly cited instrument that predates and is independent of HEXACO-JP. (ii) The manuscript does not reference, evaluate, market, recommend, or otherwise promote HEXACO-JP or any other SUNBLAZE product or service. (iii) The headline empirical findings (a null personality-intervention contrast, a positive structural-intervention contrast, and the reframing of HEXACO 7-typology as a stratification variable rather than an intervention or screening target) do not differentially favor commercial HEXACO-based assessment products.
- **Funding:** No funding was received for this research. SUNBLAZE Co., Ltd. did not provide any funding, equipment, or compensation, and no external grant funding from public, commercial, or not-for-profit agencies was received. The author conducted the research as an independent academic project at no monetary cost, using publicly available analytical software and previously collected data.
- **Open Access fee acknowledgment:** The author acknowledges the *IEEE Access* article processing charge (APC) and is prepared to comply upon acceptance.
- **Generative AI disclosure:** Generative AI (Anthropic Claude) was used as a drafting and code-review assistant; all final claims, derivations, and statistical conclusions are the author's responsibility. This is disclosed in the manuscript ACKNOWLEDGMENT section per the IEEE 2025 AI use policy.

## Suggested reviewers

The following authors of methods or substantive works cited in the manuscript have publicly demonstrated relevant *IEEE Access*-aligned expertise (computational rigor + applied workplace / personality science) and have no known conflict of interest with the author:

- **Miguel A. Hernán** (Harvard T. H. Chan School of Public Health, USA) — co-author of the target-trial-emulation framework operationalized in Stages 6–7. Frequent reviewer for methodologically demanding venues.
- **Bradley Efron** (Stanford University, USA) — author of the BCa bootstrap [Efron, 1987], the central CI method used in our four-step cascade. Foundational statistical-computing expertise.
- **Roger L. Berger** (North Carolina State University, USA) — co-author of the Berger–Hsu intersection-union test [Berger & Hsu, 1996] used for H7 evaluation. IEEE-Access-friendly methodological depth.
- **Joon Sung Park** (Stanford University, USA) — first author of the generative-agents work [Park et al., 2023] cited in our Phase 2 follow-up plan. Brings computational-engineering perspective on agent-based extensions.
- **Morten Birkeland Nielsen** (National Institute of Occupational Health, Norway) — first author of the cross-cultural meta-analysis [Nielsen et al., 2017] anchoring our F = 0.5 transportability factor. Substantive workplace-harassment expertise.

I respectfully request reviewers familiar with statistical-computing methodology (bootstrap, empirical Bayes, IUT) and the Stage 2 Registered Report process; *IEEE Access* readers most directly served by this manuscript are at the intersection of computational psychology, occupational-health analytics, and pre-registered causal inference.

## Closing

I confirm that this manuscript has not been previously published, is not under consideration elsewhere, and that all author tasks (sole-authored) have been completed by me. Code, data, and intermediate artifacts will remain publicly accessible at the indicated OSF DOI and GitHub repository for the duration of peer review and beyond.

I look forward to the editorial team's response and the peer-review process.

Sincerely,

**Eisuke Tokiwa**
Founder, SUNBLAZE Co., Ltd.
Nishi-Shinjuku 3-3-13, Shinjuku-ku, Tokyo 160-0023, Japan
Email: eisuke.tokiwa@sunblaze.jp
ORCID: [0009-0009-7124-6669](https://orcid.org/0009-0009-7124-6669)
GitHub: [https://github.com/etoki/paper](https://github.com/etoki/paper)
OSF: [https://osf.io/3y54u](https://osf.io/3y54u)
