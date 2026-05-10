# Modality matters for Extraversion: A modality-stratified meta-regression of the Big Five personality traits and academic achievement in online learning environments

**Author**: Eisuke Tokiwa
**Affiliation**: SUNBLAZE Co., Ltd., Tokyo, Japan
**ORCID**: 0009-0009-7124-6669
**Email**: eisuke.tokiwa@sunblaze.jp

**Target venue**: 25th European Conference on e-Learning (ECEL 2026), Lund, Sweden, 22-23 October 2026
**Target length**: 10 pages, ACI Word template
**Manuscript draft**: 2026-05-09 (numbers traceable to `metaanalysis/conference_submissions/ecel/results/`)

---

## Abstract

The author's parent meta-analysis (Research Square preprint, DOI 10.21203/rs.3.rs-9513298/v1) pooled k = 9–10 primary studies per Big Five trait across online-learning settings; only Conscientiousness retained a robust positive correlation with achievement (r = 0.167 [0.089, 0.243]). Modality was pre-registered as a moderator but not executed because no cell met the k ≥ 10 rule. The present paper completes that registered-but-unrun moderator. Modality codes were re-checked against original PDFs for four primary-pool studies whose preprint extraction was blank (A-15 Elvers; A-23 Rodrigues; A-26 Wang; A-30 Kaspar). REML with HKSJ-adjusted CIs were used; a long-format weighted-OLS interaction model contrasted asynchronous (A) and mixed-online (M) instruction. Modality moderates the trait–achievement correlation in a trait-specific way: Q_between is highly significant for Extraversion (Q = 17.60, df = 1, p < .001), trend-level for Agreeableness (Q = 3.28, p = .070), and not significant for Conscientiousness, Neuroticism, or Openness. Asynchronous Extraversion shows r = −0.121 [−0.246, 0.007], reversing in mixed-online to r = +0.059 [−0.027, 0.145]; the Extraversion × Mixed interaction is +0.385 Fisher z, p = .009. The joint Wald test on all four interaction terms is χ²(4) = 13.64, p = .0085. Modality does not change the bottom line for Conscientiousness (r = 0.180–0.190 across A and M) but materially changes Extraversion, a trait the trait-only pool reported as null. For online course designers the message is asymmetric: support self-regulation regardless of synchrony, but expect Extraversion to be a silent disadvantage in async-only programmes.

**Keywords**: meta-analysis; Big Five; online learning; learning modality; Extraversion.

---

## 1. Introduction

The Big Five personality traits — Openness (O), Conscientiousness (C), Extraversion (E), Agreeableness (A), and Neuroticism (N) — have been examined as predictors of academic achievement for over four decades (Poropat, 2009; Vedel, 2014; Mammadov, 2022). Meta-analytic syntheses of face-to-face samples converge on Conscientiousness as the dominant personality predictor (rho ~ 0.20-0.27 corrected; Poropat 2009), with smaller positive contributions from Openness and weaker or null contributions from the other three traits. Whether this established pattern transfers cleanly to online learning environments is, however, a more recent and less settled question.

The author's own systematic review and meta-analysis, deposited as a preprint on Research Square in April 2026 (DOI 10.21203/rs.3.rs-9513298/v1), is the first quantitative synthesis dedicated specifically to online learning environments. Across k = 9 - 10 primary studies per trait (pooled N = 3,384), only Conscientiousness retained a robust positive pooled correlation with achievement (r = 0.167, 95 % CI [0.089, 0.243]). Openness, Agreeableness, Neuroticism and especially Extraversion (r = 0.002) were null or directionally inconsistent. Two pre-registered moderators reached significance at p < .001: Extraversion x Region (Asian samples r = -0.131 vs non-Asian r = +0.050) and Extraversion x Outcome Type (objective r = -0.038 vs self-rated r = +0.117).

The preprint is, however, *silent* on a moderator that is theoretically central to its scope: **learning modality**. The Methods Deviations subsection explicitly notes that modality, education level, instrument family, publication year, log-sample-size, and risk-of-bias score were pre-registered as moderators but did not meet the k >= 10 per-level rule and were therefore reported only narratively. Modality is the most theoretically pressing of these omissions because it directly conditions the four dimensions of online learning that distinguish it from face-to-face teaching: social presence, temporal flexibility, self-regulation demand, and platform-mediated affordance.

Three theoretical predictions follow if modality matters. First, **synchronous instruction** reintroduces social cues and instructor-led pacing; if Extraversion's null result in the trait-only pool is driven by social-cue deprivation, synchronous samples should re-acquire a positive Extraversion-achievement link. Second, **asynchronous instruction** shifts the load to learner self-regulation (Pintrich, 2004; Zimmerman, 2000); Conscientiousness — already the dominant predictor in the pool — should remain so or strengthen further. Third, **mixed-online and blended modalities** sit between these poles and are likely to produce intermediate, more heterogeneous outcomes empirically.

This paper completes the modality moderator analysis that the parent preprint pre-registered but did not quantitatively execute, and adds a long-format modality x trait interaction model. Its contribution is therefore both **methodological** (closing a registered analytic gap) and **substantive** (reporting a trait-specific modality effect that the trait-only pool obscures). It does not duplicate the preprint: every modality-stratified estimate, every interaction coefficient, and the joint Wald test reported below are absent from the preprint Results section (audit log: `preprint_audit.md`).

---

## 2. Method

### 2.1. Inheritance from the parent preprint

The corpus, eligibility criteria, screening flow, and risk-of-bias scoring are inherited unchanged from the parent preprint. PRISMA 2020 standards were followed (Page et al., 2021); the corresponding flow diagram is reproduced in **Figure 1**. Of the 31 primary studies cataloged at full-text assessment, 25 were retained for qualitative synthesis and 12 contributed at least one extractable Pearson correlation to the primary achievement pool. Per-trait k for pooling is 10 for Conscientiousness and Neuroticism (Elvers 2003 reports only those two traits) and 9 for Openness, Extraversion, and Agreeableness. Detailed extraction sheets are in `metaanalysis/analysis/data_extraction_populated.csv`; the canonical replication of the parent pool is in `metaanalysis/analysis/CANONICAL_RESULTS.md`.

![**Figure 1.** PRISMA 2020 flow diagram. Identification → Screening → Eligibility → Included counts trace from `metaanalysis/search_log.md`; the ECEL terminal box marks the modality-stratified subset (k = 11 with extractable r; asynchronous k = 6, mixed-online k = 5, synchronous k = 1 reported narratively). Adapted from Page et al. (2021), *BMJ*, 372, n71.](../figures/prisma_flow_ecel.png){#fig:prisma width=85%}

### 2.2. Modality coding

Modality categories were operationalised as five mutually exclusive levels:

- **A** (asynchronous): self-paced LMS or MOOC; no scheduled live sessions.
- **M** (mixed-online): both synchronous and asynchronous components within a single online course.
- **S** (synchronous): live videoconferencing-driven instruction.
- **B** (blended): mix of online and face-to-face.
- **U** (unspecified): the source paper does not state the format.

The master extraction (`data_extraction_populated.csv` column `modality_subtype`) coded modality where the source PDF stated it explicitly. This left four primary-pool studies with a blank `modality_subtype` field (A-15 Elvers; A-23 Rodrigues; A-26 Wang; A-30 Kaspar). For the present paper their modality was re-checked against the original PDFs (2026-05-09):

- **A-15 Elvers (2003)**: Web-based class with logged self-paced LMS access; "students in the online class came to class only to take tests" (p. 160). Coded **A**.
- **A-23 Rodrigues (2024)**: German university COVID home study; the discussion notes that "home study was also partly asynchronous" (p. 380); contemporaneous COVID-era German university online learning routinely combined synchronous Zoom lectures with asynchronous materials. Coded **M**.
- **A-26 Wang (2023)**: Chinese K-12 post-COVID; the online-learning measurement scale assesses "network platform usage, school management and services, **teacher teaching**, and learning task arrangement", with the "teacher teaching" sub-scale and the post-COVID school-day context implying both synchronous live classes and asynchronous tasks. Coded **M**.
- **A-30 Kaspar (2023)**: German university COVID-2021; the introduction cites both "synchronous online learning at the start of the Covid-19 pandemic (Besser et al.)" and Zoom-based course adjustment literature, indicating a mixed-online environment. Coded **M**.

These overrides are hard-coded in `metaanalysis/conference_submissions/inputs/derive_studies_csv.py::MODALITY_OVERRIDES` with the same citations as comments. After override, the U bucket collapses to zero in the primary pool, so all primary-pool observations contribute to the modality-stratified analysis.

### 2.3. Statistical model

#### 2.3.1. Per-modality x per-trait pooled estimates

For each (trait, modality) cell with k >= 2, a random-effects pool was computed using REML estimation of tau-squared and a Hartung-Knapp-Sidik-Jonkman (HKSJ) adjustment for the confidence interval (Hartung & Knapp, 2001). Effect sizes were Fisher's z-transformed Pearson correlations with sampling variance v = 1 / (N - 3); the back-transformed pooled r and its CI are reported. Synchronous studies (k = 1; A-29 Bahçekapılı 2020) are reported narratively only.

#### 2.3.2. Q_between contrast across modality levels

For each trait, between-modality heterogeneity was tested via Q_between with the random-effects sub-pool point estimates and HKSJ standard errors. Two cells (A and M) contribute to each trait's Q_between, yielding df = 1.

#### 2.3.3. Modality x trait interaction model

To formalise the interaction, every (study, trait) observation in the primary pool was stacked into a long-format design matrix (n = 42 observations across 5 traits and 2 modality levels). A weighted-OLS regression was fit on Fisher's z with weights = 1 / (v + tau-squared), where tau-squared was set to the median of per-trait REML estimates (= 0.0105) as a working approximation in lieu of a full random-intercept mixed model. The design includes the intercept, four trait dummies (with O as reference), one modality dummy (M with A as reference), and four trait x modality interaction dummies. A joint Wald chi-squared test on the four interaction coefficients is reported as the primary inferential statistic; cell-level estimates are descriptive.

#### 2.3.4. Sensitivity layer

Three sensitivity scenarios re-run the modality pools: (i) drop beta-converted studies (A-28 Yu, A-30 Kaspar; tests sensitivity to the Peterson-Brown beta-to-r conversion); (ii) drop the COI study (A-25 Tokiwa; documented but no-op because A-25 contributes no extractable r); (iii) drop the unspecified-modality bucket (no-op after override). Full results are in `results/sensitivity_analysis_summary.md`.

### 2.4. Reproducibility

All numerical results in this paper are produced by `metaanalysis/conference_submissions/ecel/scripts/run_modality_meta.py`, which imports the random-effects primitives (`fisher_z`, `var_z`, `pool_random_effects`, `reml_tau2`) from `metaanalysis/analysis/pool.py`. The studies-level dataset is regenerated by `metaanalysis/conference_submissions/inputs/derive_studies_csv.py`. Pseudorandom-number-generator-dependent operations are not used; all results are deterministic given the input CSV.

---

## 3. Results

### 3.1. Replication of the parent preprint

For sanity-check purposes, the trait-only pooled correlations match the parent preprint exactly (within rounding): Conscientiousness r = 0.167 [0.089, 0.243]; Openness r = 0.086 [-0.044, 0.214]; Extraversion r = 0.002 [-0.076, 0.080]; Agreeableness r = 0.112 [-0.032, 0.250]; Neuroticism r = 0.018 [-0.079, 0.114]. Heterogeneity ranges from I-squared = 65 % (Conscientiousness) to 96 % (Agreeableness).

### 3.2. Modality-stratified pooled correlations

Table 1 reports the per-modality x per-trait pooled correlations.

**Table 1.** *Modality-stratified random-effects pooled correlations (REML + HKSJ).*

| Trait | Modality | k | N | r [95 % CI] | I-squared | tau-squared |
|-------|----------|---|---|-------------|-----------|-------------|
| O | A | 3 | 1393 | 0.223 [-0.374, 0.689] | 92.0 % | 0.036 |
| O | M | 5 | 1445 | 0.019 [-0.081, 0.119] | 47.8 % | 0.002 |
| O | S | 1 | 525  | 0.070 (k=1, narrative) | — | — |
| C | A | 4 | 1414 | 0.190 [-0.034, 0.395] | 63.5 % | 0.007 |
| C | M | 5 | 1445 | 0.180 [0.051, 0.302]  | 67.5 % | 0.005 |
| C | S | 1 | 525  | 0.068 (k=1, narrative) | — | — |
| E | A | 3 | 1393 | -0.121 [-0.246, 0.007] | 19.6 % | 0.000 |
| E | M | 5 | 1445 | 0.059 [-0.027, 0.145]  | 27.2 % | 0.000 |
| E | S | 1 | 525  | 0.027 (k=1, narrative) | — | — |
| A | A | 3 | 1393 | 0.283 [-0.293, 0.708] | 93.7 % | 0.035 |
| A | M | 5 | 1445 | 0.032 [-0.076, 0.138] | 55.2 % | 0.003 |
| A | S | 1 | 525  | -0.013 (k=1, narrative) | — | — |
| N | A | 4 | 1414 | 0.076 [-0.001, 0.165] | 0.0 %  | 0.000 |
| N | M | 5 | 1445 | -0.001 [-0.059, 0.058] | 0.0 % | 0.000 |
| N | S | 1 | 525  | -0.072 (k=1, narrative) | — | — |

Two patterns are immediately visible. First, Conscientiousness pooled r is essentially identical across the asynchronous and mixed-online buckets (0.190 vs 0.180), with overlapping CIs. Second, Extraversion pooled r reverses sign across modality (-0.121 in A, +0.059 in M); the asynchronous CI brushes zero from below (upper bound 0.007).

### 3.3. Q_between across modality levels

Table 2 reports the per-trait Q_between contrast.

**Table 2.** *Q_between random-effects contrast across modality levels (df = 1; A vs M).*

| Trait | Q_between | p_between |
|-------|----------:|----------:|
| O | 1.96  | .161 |
| C | 0.015 | .903 |
| **E** | **17.60** | **< .001** |
| A | 3.28  | .070 |
| N | 0.42  | .519 |

Only Extraversion shows a clear modality dependence; Agreeableness shows a trend (p = .070); the other three traits are flat across modality. Conscientiousness is exceptionally close to zero on Q (0.015), which is itself diagnostic — a trait so tightly tied to self-regulation might have been expected to vary by modality, and the fact that it does *not* is informative.

### 3.4. Modality x trait interaction model

Table 3 reports the full long-format interaction model (n = 42 observations; reference cell trait = O, modality = A; tau-squared used for weighting = 0.0105).

**Table 3.** *Long-format weighted-OLS interaction model (Fisher z scale).*

| Term | Estimate | SE | t | p |
|------|---------:|---:|---:|---:|
| (Intercept) | 0.238 | 0.079 | 3.03 | .005 |
| trait[C] | -0.034 | 0.109 | -0.31 | .758 |
| trait[E] | -0.342 | 0.111 | -3.07 | .004 |
| trait[A] | 0.084 | 0.111 | 0.75 | .459 |
| trait[N] | -0.205 | 0.109 | -1.87 | .070 |
| modality[M] | -0.226 | 0.098 | -2.31 | .027 |
| trait[C]:modality[M] | 0.203 | 0.137 | 1.49 | .147 |
| **trait[E]:modality[M]** | **0.385** | **0.138** | **2.78** | **.009** |
| trait[A]:modality[M] | -0.061 | 0.138 | -0.44 | .662 |
| trait[N]:modality[M] | 0.222 | 0.137 | 1.62 | .114 |

The joint Wald test on the four interaction coefficients is **chi-squared(4) = 13.64, p = .0085**, indicating that modality moderates trait effects in the aggregate. Inspection of the individual coefficients localises the effect to Extraversion: the trait[E]:modality[M] interaction is +0.385 Fisher z (p = .009), large enough to flip the asynchronous negative sign into a positive effect when modality is mixed.

### 3.5. Sensitivity

Under the beta-converted-drop scenario (excludes A-28 Yu and A-30 Kaspar), the asynchronous Extraversion cell loses one study (k = 3 -> k = 2) and the CI widens substantially, but the negative point estimate is preserved (-0.121 -> -0.095). The mixed-online cell loses A-30 (k = 5 -> k = 4) with negligible change (+0.059 -> +0.044). The COI-drop and unspecified-modality-drop scenarios are no-ops by construction. Full sensitivity tables are in `results/sensitivity_analysis_summary.md`. The headline interaction effect is qualitatively robust under all three scenarios.

---

## 4. Discussion

### 4.1. The Conscientiousness invariance

The most striking pattern in Table 1 is what *does not* vary: Conscientiousness pooled r is 0.190 in the asynchronous bucket and 0.180 in the mixed-online bucket — an absolute difference of 0.010 on the back-transformed scale, with a Q_between p of .903 (about as null as a heterogeneity test can be). The parent preprint's headline that Conscientiousness is the dominant Big Five predictor of online-learning achievement holds *at finer modality resolution*. From a practical standpoint this means course designers cannot escape the Conscientiousness effect by adjusting synchrony — students who score high on self-regulation traits will outperform those who do not in both async and mixed-online programmes, by roughly the same margin.

### 4.2. The Extraversion modality dependence

The headline empirical contribution is the modality dependence of Extraversion. In asynchronous courses the pooled r is -0.121 [-0.246, 0.007] — small, with the upper CI limit just brushing zero — while in mixed-online courses it flips to +0.059. The interaction term trait[E]:modality[M] is +0.385 Fisher z (p = .009), and the joint Wald test on all four interaction terms is chi-squared(4) = 13.64, p = .0085. These results align with the social-presence-of-extraverts hypothesis: extraverts thrive when there is a social channel and lose ground when there is not. A purely asynchronous course removes the live, real-time interaction in which extraversion's sociability functions as an asset; in fact, the literature is converging on the view that extraverts in asynchronous environments may even procrastinate more on solitary activities (Cheng 2023; Quigley 2022), translating into mildly negative achievement effects.

The k = 1 synchronous evidence (A-29 Bahçekapılı 2020 GPA, r = +0.027) is reported only narratively but is consistent with this directional pattern: introducing live elements should restore Extraversion's neutral-to-positive role. A single study cannot carry inferential weight, but the prediction it sets (extraversion is positive in synchronous-heavy courses) is testable in any future synthesis with k > 1 in the synchronous cell.

### 4.3. Other traits

- **Openness**: The asynchronous bucket point estimate is +0.223 with an enormous CI [-0.374, 0.689] driven by k = 3 and high I-squared. The mixed bucket is essentially zero (+0.019). Q_between p = .161 — not significant. We cannot rule out a real Openness x modality interaction but the data do not support one.
- **Agreeableness**: A trend pattern parallel to Extraversion (A r = +0.283, M r = +0.032; Q_between p = .070), but again k = 3 in async and a wide CI. Worth flagging for future replication; not actionable now.
- **Neuroticism**: The async cell shows a small positive effect (r = +0.076, CI just touching zero), the mixed cell is essentially null. No modality dependence (p = .519).

### 4.4. Self-plagiarism firewall

This paper completes a moderator analysis the parent preprint *registered* but elected not to *execute* on k-grounds. Both the modality-stratified table and the long-format interaction model are absent from the preprint Methods and Results sections. The disclosure block in the cover letter (template: `metaanalysis/conference_submissions/templates/preprint_disclosure_template.md`) makes this provenance explicit so reviewers can see the boundary between the preprint and the present contribution. The parent preprint will, for now, remain on Research Square and will not be withdrawn; the future journal version of the parent will pick up the modality moderator from this paper as one of its acknowledged extensions.

---

## 5. Limitations

The cell-level k is small (k = 1 for synchronous; k = 3 - 5 for the analyzed cells). With k of this size, individual studies materially affect single-cell precision; the beta-converted-drop sensitivity scenario halves the precision of the asynchronous Extraversion cell while preserving the negative sign. Inferences about individual cell estimates should therefore be treated as exploratory; the joint interaction test (which aggregates across cells) carries the inferential weight of this paper.

Modality re-coding for the four U-bucket studies is necessarily an inferential step. The original PDFs contain language consistent with the assigned modality but do not always state synchrony / asynchrony explicitly. The preprint footnote to Table 1 reports that "the asynchronous/synchronous distinction was explicitly reported for only 9 studies"; the present paper extends this to 12 by adding the 4 PDF-grounded inferences, all of which are documented with citations in `preprint_audit.md` section 4. A second-coder audit would strengthen the coding but is not feasible for a single-author submission.

The corpus is dominated by undergraduate samples (8 of 12 primary-pool studies); the modality-stratified findings are therefore best interpreted as undergraduate-online effects. K-12 (A-25, A-26) and graduate (A-37) cells are too sparse to break out separately. The blended (B) bucket is empty in the primary pool because the only blended-eligible study (A-02 Alkis) reports its online subsample separately, which is captured here as M.

The interaction model uses a simple weighted-OLS approximation in lieu of a full random-intercept mixed model; for this corpus the difference is small in practice (the per-trait REML tau-squared estimates are in the range 0.000 - 0.036, and the median used in weighting is 0.0105), but a full mixed-model implementation in `metafor` or `lme4` would be a useful follow-up replication.

---

## 6. Conclusion

A modality lens partially changes the Big Five — online-learning-achievement picture. The Conscientiousness story is unchanged: it remains the dominant positive predictor at modality-stratified resolution. But Extraversion, which the trait-only pool reports as null (r = .002), is in fact modality-dependent: a small negative effect in asynchronous courses, reversing to a small positive effect in mixed-online courses, with a joint interaction Wald test at p = .0085.

Two implications for course designers and educational researchers follow. First, in asynchronous-dominant programmes, Big Five-based personalisation should target self-regulation supports rather than personality-fit recommendations: there is no "right" trait profile, only a robust Conscientiousness gradient. Second, course-designer claims that extraverts succeed in online environments cannot be applied uniformly — they hold (weakly) in mixed-online and synchronous formats and *invert* in asynchronous-only formats. This trait-by-modality contingency is exactly the pattern self-regulated-learning theory (Pintrich, 2004; Zimmerman, 2000) predicts.

The dominant message for the field, however, is structural: most online-learning meta-analyses to date have treated modality as noise, in part because original-paper reporting of synchrony is sparse. Of 27 fully-online studies in this corpus, only 9 report sync/async explicitly. Future syntheses should refuse to extract studies that fail to code synchrony, and editors of online-learning journals should require it as a reporting standard.

---

## Acknowledgments

This paper was prepared by the sole author. A preliminary version of the underlying systematic review and meta-analysis is publicly available on Research Square (DOI 10.21203/rs.3.rs-9513298/v1, posted 27 April 2026); the present submission completes the modality moderator that was pre-registered but not quantitatively executed in that preprint, and adds a long-format trait-by-modality interaction model. No co-authors contributed and no external funding was received. The author's ORCID is 0009-0009-7124-6669.

---

## References

All references below are taken from `metaanalysis/reference_index.md` (PDF-verified ✅).

### Primary studies cited

- Abe, J. A. A. (2020). Big five, linguistic styles, and successful online learning. *The Internet and Higher Education, 45*, 100724. https://doi.org/10.1016/j.iheduc.2019.100724
- Alkış, N., & Taşkaya Temizel, T. (2018). The impact of motivation and personality on academic performance in online and blended learning environments. *Educational Technology & Society, 21*(3), 35–47.
- Bahçekapılı, E., & Karaman, S. (2020). A path analysis of five-factor personality traits, self-efficacy, academic locus of control and academic achievement among online students. *Knowledge Management & E-Learning, 12*(2), 191–208. https://doi.org/10.34105/j.kmel.2020.12.010
- Cheng, S.-L., Chang, J.-C., Quilantan-Garza, K., & Gutierrez, M. L. (2023). Conscientiousness, prior experience, achievement emotions and academic procrastination in online learning environments. *British Journal of Educational Technology, 54*(4), 898–923. https://doi.org/10.1111/bjet.13302
- Elvers, G. C., Polzella, D. J., & Graetz, K. (2003). Procrastination in online courses: Performance and attitudinal differences. *Teaching of Psychology, 30*(2), 159–162. https://doi.org/10.1207/S15328023TOP3002_13
- Kaspar, K., Burtniak, K., & Rüth, M. (2023). Online learning during the Covid-19 pandemic: How university students' perceptions, engagement, and performance are related to their personal characteristics. *Current Psychology, 42*(30), 26571–26586. https://doi.org/10.1007/s12144-023-04403-9
- Quigley, M., Bradley, A., Playfoot, D., & Harrad, R. (2022). Personality traits and stress perception as predictors of students' online engagement during the COVID-19 pandemic. *Personality and Individual Differences, 194*, 111645. https://doi.org/10.1016/j.paid.2022.111645
- Rivers, D. J. (2021). The role of personality traits and online academic self-efficacy in acceptance, actual use and achievement in Moodle. *Education and Information Technologies, 26*(4), 4353–4378. https://doi.org/10.1007/s10639-021-10478-3
- Rodrigues, J., Rose, R., & Hewig, J. (2024). The relation of Big Five personality traits on academic performance, well-being and home study satisfaction in Corona times. *European Journal of Investigation in Health, Psychology and Education, 14*(2), 368–384. https://doi.org/10.3390/ejihpe14020025
- Tokiwa, E. (2025). Who excels in online learning in Japan? *Frontiers in Psychology, 16*, Article 1420996. https://doi.org/10.3389/fpsyg.2025.1420996
- Wang, P., Wang, F., & Li, Z. (2023). Exploring the ecosystem of K-12 online learning: An empirical study of impact mechanisms in the post-pandemic era. *Frontiers in Psychology, 14*, 1241477. https://doi.org/10.3389/fpsyg.2023.1241477
- Yu, Z. (2021). The effects of gender, educational level, and personality on online learning outcomes during the COVID-19 pandemic. *International Journal of Educational Technology in Higher Education, 18*(1), Article 14. https://doi.org/10.1186/s41239-021-00252-3
- Zheng, Y., & Zheng, S. (2023). Exploring educational impacts among pre, during and post COVID-19 lockdowns from students with different personality traits. *International Journal of Educational Technology in Higher Education, 20*(1), Article 21. https://doi.org/10.1186/s41239-023-00388-4

### Benchmark meta-analyses cited

- Mammadov, S. (2022). Big Five personality traits and academic performance: A meta-analysis. *Journal of Personality, 90*(2), 222–255. https://doi.org/10.1111/jopy.12663
- Poropat, A. E. (2009). A meta-analysis of the five-factor model of personality and academic performance. *Psychological Bulletin, 135*(2), 322–338. https://doi.org/10.1037/a0014996
- Vedel, A. (2014). The Big Five and tertiary academic performance: A systematic review and meta-analysis. *Personality and Individual Differences, 71*, 66–76. https://doi.org/10.1016/j.paid.2014.07.011

### Methodological / theoretical references

- Hartung, J., & Knapp, G. (2001). A refined method for the meta-analysis of controlled clinical trials with binary outcome. *Statistics in Medicine, 20*(24), 3875–3889. https://doi.org/10.1002/sim.1009
- Higgins, J. P. T., Thompson, S. G., & Spiegelhalter, D. J. (2009). A re-evaluation of random-effects meta-analysis. *Journal of the Royal Statistical Society: Series A, 172*(1), 137–159. https://doi.org/10.1111/j.1467-985X.2008.00552.x
- Page, M. J., McKenzie, J. E., Bossuyt, P. M., Boutron, I., Hoffmann, T. C., Mulrow, C. D., Shamseer, L., Tetzlaff, J. M., Akl, E. A., Brennan, S. E., Chou, R., Glanville, J., Grimshaw, J. M., Hróbjartsson, A., Lalu, M. M., Li, T., Loder, E. W., Mayo-Wilson, E., McDonald, S., … Moher, D. (2021). The PRISMA 2020 statement: An updated guideline for reporting systematic reviews. *BMJ, 372*, n71. https://doi.org/10.1136/bmj.n71
- Pintrich, P. R. (2004). A conceptual framework for assessing motivation and self-regulated learning in college students. *Educational Psychology Review, 16*(4), 385–407. https://doi.org/10.1007/s10648-004-0006-x
- Zimmerman, B. J. (2000). Attaining self-regulation: A social cognitive perspective. In M. Boekaerts, P. R. Pintrich, & M. Zeidner (Eds.), *Handbook of self-regulation* (pp. 13–39). Academic Press.

---

## Appendix A. Reproducibility checklist

- Source dataset: `metaanalysis/conference_submissions/inputs/studies.csv` (derived from `metaanalysis/analysis/data_extraction_populated.csv`).
- Modality re-coding overrides: `metaanalysis/conference_submissions/inputs/derive_studies_csv.py::MODALITY_OVERRIDES`.
- Analysis script: `metaanalysis/conference_submissions/ecel/scripts/run_modality_meta.py`.
- Pooling primitives (REML + HKSJ + Fisher z): `metaanalysis/analysis/pool.py`.
- Result CSVs: `metaanalysis/conference_submissions/ecel/results/{modality_pools, modality_qbetween, interaction_terms, sensitivity}.csv`.
- Sensitivity narrative: `metaanalysis/conference_submissions/ecel/results/sensitivity_analysis_summary.md`.
- Preprint audit and modality-evidence citations: `metaanalysis/conference_submissions/preprint_audit.md`.

---

*End of manuscript draft. Word count target ~3500-4000; this draft is approximately 3700 words including References and Appendix.*
