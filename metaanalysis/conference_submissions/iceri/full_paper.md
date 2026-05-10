# Where the Big Five evidence really lives: A cross-tabulated meta-analytic interaction of education level and academic discipline in online learning

**Author**: Eisuke Tokiwa
**Affiliation**: SUNBLAZE Co., Ltd., Tokyo, Japan
**ORCID**: 0009-0009-7124-6669
**Email**: eisuke.tokiwa@sunblaze.jp

**Target venue**: ICERI 2026 — 19th International Conference of Education, Research and Innovation, Seville, Spain
**Target length**: 10 pages, IATED Word template
**Manuscript draft**: 2026-05-09 (numbers traceable to `metaanalysis/conference_submissions/iceri/results/`)

---

## Abstract

The parent meta-analysis (Research Square preprint, DOI 10.21203/rs.3.rs-9513298/v1) pools k = 9 — 10 primary studies per Big Five trait across heterogeneous education levels (K-12 / Undergraduate / Graduate / Mixed) and disciplines (STEM / Humanities / Psychology / Mixed). Education-level was pre-registered as a moderator but was not quantitatively executed (k constraint); discipline was not pre-registered at all. Whether the pooled estimates mask interaction patterns between level and discipline remains untested. The present paper cross-tabulates the primary-pool studies by 4 education levels (K-12 / UG / Graduate / Mixed_UG_Grad) and 4 disciplines (STEM / Humanities / Psychology / Mixed), pools within each cell where k >= 2, and fits a long-format weighted-OLS interaction model on Fisher z-transformed correlations. Cross-tab cell counts confirm an undergraduate-dominated literature: of 12 candidate cells (4 levels x 3 non-empty disciplines), only 2 cells reach k >= 2: UG x Mixed (k = 3) and UG x Psychology (k = 3 for C / N, k = 2 for O / E / A). UG x Psychology shows the strongest Conscientiousness pooled correlation (r = 0.292 [0.123, 0.444]) and a positive UG x Psychology Agreeableness effect (r = 0.152 [0.098, 0.205]). Joint Wald test on all six level x discipline interactions: chi-squared(6) = 1.64, p = .95 — the interaction is null with the present evidence base. The empirical message is that where evidence is dense (UG x Psychology / UG x Mixed), Conscientiousness is the strongest positive predictor; everywhere else, k <= 1 prohibits inference. The structural message is sharper: K-12 STEM-online and Graduate-Humanities-online are evidence deserts. Pooled Big Five — achievement correlations from any current meta-analysis should not be applied uniformly across teaching contexts.

**Keywords**: meta-analysis, Big Five, online learning, education level, discipline, interaction model, evidence map, ICERI.

---

## 1. Introduction

The Big Five — academic-achievement literature has historically pooled across education levels and disciplines, treating these dimensions as background heterogeneity rather than as moderators worth modelling. Poropat (2009) included K-12 and tertiary samples in the same meta-analytic pool; Vedel (2014) restricted to tertiary but did not separate by discipline; Mammadov (2022) added domain-specificity moderators (language vs STEM) and reported that Openness varied substantially across domains while Conscientiousness was stable. None of these syntheses applied an *interaction* model that crosses education level with discipline.

The author's own preprint on online-learning specifically (Tokiwa, 2026, Research Square preprint DOI 10.21203/rs.3.rs-9513298/v1) inherits this limitation. Education level was pre-registered as a moderator but was not quantitatively executed because no level had k >= 10 (Methods Deviations subsection); discipline was not pre-registered at all. The primary pool is reported across all education levels and disciplines combined, with three other pre-registered moderators (Region, Era, Outcome Type) executed.

The present paper asks: **does the Big Five — online-learning achievement effect depend on the joint education-level x discipline cell, beyond what either main effect captures?**

This is not a confirmatory question for the present corpus. With 12 primary-pool studies distributed across (in principle) up to 16 cells (4 levels x 4 disciplines), most cells will be sparse or empty. The contribution of this paper is therefore *evidence mapping* combined with an interaction test: we report which cells have data, with what effect-size estimates; we report a joint Wald test on the level x discipline interaction terms that documents whether the present corpus supports any non-additive structure; and we use the result to make a structural argument about where the field's evidence is densest and where it is absent.

---

## 2. Related Work

### 2.1. Education-level moderators

Poropat (2009) reported broadly stable Conscientiousness — achievement correlations across primary, secondary, and tertiary samples, with slight strengthening at the tertiary level. Mammadov (2022) found a similar stability and noted that the small differences across levels disappeared when measurement-instrument variation was controlled. Online-learning-specific evidence is largely absent — most online-learning studies are undergraduate-only by design.

### 2.2. Discipline / domain moderators

Vedel (2014) reported domain effects in tertiary samples: Openness was substantially stronger for language students than for STEM students; Conscientiousness was stable across domains. Meyer, Jansen, Hübner, and Lüdtke (2023) — a K-12-focused meta-analysis (k = 110, N = 500,218) — extended this with a domain-specificity moderator and reported that the Conscientiousness — achievement link is largely domain-invariant while Openness is domain-sensitive (stronger for language than for STEM).

### 2.3. Online-learning-specific cross-tab evidence

Wang et al. (2023) report a K-12 online-learning study (China; full-mediation SEM with Conscientiousness as the strongest engagement predictor). Boonyapison et al. (2025) report a Thailand high-school sample. These K-12 online-learning studies are recent and methodologically diverse, and the present meta-analysis includes only one each (A-26 Wang) plus A-25 Tokiwa as a Japan K-12 case. The K-12 online evidence is sparse and the K-12 x specific discipline crossing has no representation in the corpus.

### 2.4. Why a cross-tab interaction matters

If education level and discipline both moderate the personality — achievement link, the *joint* level x discipline cell can in principle deviate from what each main effect predicts. For instance, the Conscientiousness — achievement effect might be stable across levels (per Poropat 2009) and stable across STEM disciplines (per Mammadov 2022) but show a *non-additive* boost in K-12 STEM-online specifically, where structural rigidity and clear performance criteria amplify self-regulation. Without a cross-tab model this hypothesis is untestable. With one — even at small k — we get an evidence-presence map and a directional first reading.

---

## 3. Method

### 3.1. Data

The primary-pool dataset is `metaanalysis/conference_submissions/inputs/studies.csv`, derived from the parent preprint extraction. We retain studies with `inclusion_status in {include, include_with_caveat, include_COI}` and `primary_achievement in {yes, partial}` and at least one extractable Pearson r per trait.

### 3.2. Education-level collapsing

Raw education-level codes were collapsed to four categories using the mapping in `metaanalysis/conference_submissions/iceri/scripts/run_cross_tab_meta.py::LEVEL_MAP`:

- **K-12** = K-12 + HS_Year3 + HS_Grade12;
- **UG** = Undergraduate + Mixed_secondary_postsecondary;
- **Graduate** = Graduate;
- **Mixed_UG_Grad** = Mixed_UG_Grad (kept as a separate level due to its analytic heterogeneity).

### 3.3. Discipline categorisation

Disciplines are derived in `metaanalysis/conference_submissions/inputs/derive_studies_csv.py::classify_discipline`:

- **STEM** = subject_domain in {IT, Engineering, Computer Science, Health-related disciplines with STEM emphasis};
- **Humanities** = subject_domain in {Linguistics, Language, Humanities, History};
- **Psychology** = subject_domain matches "psychology" exactly;
- **Mixed** = anything else, including All_5_subjects, blank, or generic samples.

### 3.4. Cell-level pooling

For each (level, discipline, trait) cell, if k >= 2, a random-effects pool is computed using REML estimation of tau-squared and Hartung-Knapp-Sidik-Jonkman (HKSJ) confidence intervals. Effect sizes are Fisher z-transformed Pearson r with v = 1 / (N - 3). The back-transformed pooled r and 95 % CI are reported. Single-study cells (k = 1) report the raw r as descriptive only.

### 3.5. Long-format interaction model

To formalise the level x discipline interaction, every (study, trait) observation is stacked into a long-format design matrix. A weighted-OLS regression is fit on Fisher z with weights = 1 / (v + tau-squared), tau-squared from the median of per-trait REML estimates. The design includes the intercept, four trait dummies (with O as reference), three level dummies (Mixed_UG_Grad / UG / K-12 with Graduate as reference), one discipline dummy (with Humanities as the reference; Psychology / Mixed as the represented levels in the present corpus), and the level x discipline interaction terms.

A joint Wald chi-squared test on the interaction coefficients is reported as the primary inferential statistic; the per-cell estimates from §3.4 are descriptive complements.

### 3.6. Reproducibility

All numerical results are produced by `metaanalysis/conference_submissions/iceri/scripts/run_cross_tab_meta.py`. Pooling primitives (Fisher z, REML, HKSJ-adjusted CI) are re-used from `metaanalysis/analysis/pool.py`.

---

## 4. Results

### 4.1. Cross-tab cell-count map

Table 1 shows the maximum k per (level, discipline) cell across the five Big Five traits.

**Table 1.** *Cell-count map: maximum k across all five traits per (level, discipline) cell.*

| level / discipline | Humanities | Mixed | Psychology | STEM |
|--------------------|-----------:|------:|-----------:|-----:|
| **Graduate** | 0 | 1 | 0 | 0 |
| **Mixed_UG_Grad** | 1 | 1 | 0 | 0 |
| **UG** | 0 | 3 | 3 | 1 |

(K-12 row is omitted because the only K-12 primary-pool studies — A-25 Tokiwa, A-26 Wang — do not contribute extractable r values.)

The picture is unambiguous: the corpus is **undergraduate-dominated** and **mixed-or-Psychology-discipline-dominated**. Of 16 nominal cells, only 2 reach k >= 2 (UG x Mixed, UG x Psychology); 5 cells have k = 1; 9 cells are empty.

### 4.2. Pooled correlations per (cell, trait) where k >= 2

Table 2 reports the per-cell pooled r for the two cells that meet the k >= 2 threshold.

**Table 2.** *Pooled correlations per (level, discipline, trait) where k >= 2 (REML + HKSJ).*

| Trait | Level | Discipline | k | r [95 % CI] | I-squared |
|-------|-------|------------|---|-------------|-----------|
| O | UG | Mixed | 3 | 0.029 [-0.126, 0.183] | 17.3 % |
| O | UG | Psychology | 2 | 0.160 [-0.973, 0.986] | 89.1 % |
| **C** | **UG** | **Psychology** | **3** | **0.292 [0.123, 0.444]** | **0.0 %** |
| C | UG | Mixed | 3 | 0.136 [-0.081, 0.340] | 57.4 % |
| E | UG | Mixed | 3 | -0.028 [-0.252, 0.199] | 57.0 % |
| E | UG | Psychology | 2 | 0.115 [-0.444, 0.610] | 0.0 % |
| **A** | **UG** | **Psychology** | **2** | **0.152 [0.098, 0.205]** | **0.0 %** |
| A | UG | Mixed | 3 | 0.011 [-0.130, 0.152] | 1.0 % |
| N | UG | Mixed | 3 | -0.055 [-0.324, 0.223] | 66.3 % |
| N | UG | Psychology | 3 | -0.037 [-0.268, 0.199] | 19.5 % |

The strongest cell-level effects are **UG x Psychology x Conscientiousness** (r = 0.292 [0.123, 0.444]) and **UG x Psychology x Agreeableness** (r = 0.152 [0.098, 0.205]). UG x Mixed cells are mostly small and CI-wide, consistent with the heterogeneous nature of "Mixed" as a discipline label.

### 4.3. Single-cell estimates (k = 1, descriptive)

Five further cells have k = 1 each. We list them for completeness but make no inferential claims:

- A-37 Zheng 2023: Graduate x Mixed — point r ranging from 0.000 (all five traits) (TIPI bounded reporting).
- A-28 Yu 2021: Mixed_UG_Grad x Humanities — point r O = +0.355, C = +0.107, E = -0.126, A = +0.492, N = +0.087 (β-converted).
- A-30 Kaspar 2023: Mixed_UG_Grad x Mixed — point r O = +0.130, C = +0.200, E = +0.100, A = -0.060, N = +0.250 (β-converted).
- A-02 Alkis 2018: UG x STEM — point r O = -0.092, C = +0.205, E = +0.051, A = +0.094, N = +0.030.

The A-02 STEM data is the only direct STEM-discipline data point in the corpus; it is consistent with the broader pattern (small Conscientiousness positive effect, small or null other traits) but cannot carry confirmatory weight on its own.

### 4.4. Long-format interaction model

The full long-format model is fit on n = 47 (study, trait) observations with weights = 1 / (v + tau-squared = 0.0103). The reference cell is trait = O, level = Graduate, discipline = Humanities. The joint Wald test on the six level x discipline interaction coefficients:

> **chi-squared(6) = 1.64, p = .95**

The interaction is null. With six interaction terms each estimated from a sparse design matrix, the test has very low power; the p = .95 result therefore should be read as "no detectable non-additive structure given the present coverage" rather than as "additive structure is confirmed".

The full coefficient table is in `results/interaction_terms.csv`.

---

## 5. Discussion

### 5.1. The empirical message: dense-cell findings

The two cells that have actual evidence converge on the same picture: in undergraduate Psychology samples, Conscientiousness is the dominant Big Five predictor (r = 0.292 [0.123, 0.444]), with Agreeableness providing a smaller but reliable positive contribution (r = 0.152 [0.098, 0.205]). Other traits are essentially null in this dense cell.

In undergraduate Mixed-discipline samples, the Conscientiousness pooled estimate is smaller (r = 0.136 [-0.081, 0.340]) and the CI brushes zero — the discipline heterogeneity within "Mixed" plausibly attenuates the average effect. None of the other traits is reliably non-null in UG x Mixed.

### 5.2. The structural message: evidence deserts

The cell-count map (Table 1) is itself a finding. The corpus has **zero K-12 STEM-online observations with extractable r**, **zero Graduate-Humanities-online observations**, and only k = 1 in the entire Mixed_UG_Grad row. Pooled Big Five — online-learning-achievement claims that aspire to generalise across education contexts cannot be justified from the current evidence base.

The corollary for educational-research policy is direct: targeted recruitment and reporting of K-12 STEM-online and Graduate-Humanities-online primary studies should be a field priority. This is not a critique of any individual study; it is a description of where the cumulative evidence happens to sit.

### 5.3. Why the interaction Wald test is null

The interaction Wald test (chi-squared(6) = 1.64, p = .95) is null because the design matrix is sparse and the present corpus does not have enough cell variation to drive any interaction term significantly away from zero. This is consistent with the *additive* interpretation of education level and discipline as independent main effects — but the test has very low power and cannot be read as confirmatory of additivity. It is most accurately read as "with the present coverage, no interaction effect can be detected".

### 5.4. Self-plagiarism firewall

Education-level was pre-registered as a moderator in the parent preprint but was not quantitatively executed (k constraint, Methods Deviations subsection). Discipline was not pre-registered at all. The cross-tab interaction is novel to this submission. Disclosure follows `templates/preprint_disclosure_template.md` ICERI block, which acknowledges that education-level pre-registration overlap with the preprint and positions the cross-tab interaction as the genuinely new analytic step.

---

## 6. Limitations

The cell-level k is too small for confirmatory cross-tab claims in any cell except the two with k >= 2; even in those, the I-squared can be high (UG x Psychology x Openness has I-squared = 89 %) and the CI can be very wide.

The discipline classification is heuristic (single coder, single source of truth). Studies whose subject domain is reported as "All_5_subjects" or blank are coded as "Mixed" without further granularity, which inflates the Mixed cell at the cost of finer discriminations.

The long-format interaction model uses a simple weighted-OLS approximation in lieu of a full random-intercept mixed model. With the present corpus the difference is small in practice.

The corpus is dominated by convenience samples, which introduces selection effects that no statistical method can correct for.

Single coder, single author throughout.

---

## 7. Conclusion

The cross-tab interaction of education level and discipline in the present online-learning Big Five corpus is, statistically, null (chi-squared(6) = 1.64, p = .95) — but the absence of detectable interaction is informative chiefly as a documentation of how thin the cell-level evidence is. Where evidence is dense (UG x Psychology), Conscientiousness is the strongest predictor (r = 0.292 [0.123, 0.444]) and Agreeableness contributes a smaller positive effect (r = 0.152). Outside that cell, k <= 1 prohibits inference.

The structural message is sharper than the inferential one: K-12 STEM-online and Graduate-Humanities-online are evidence deserts, and pooled Big Five — achievement claims should not be extrapolated to those contexts on the current evidence base. For ICERI's audience of educators and educational-research policy makers, this paper offers a coverage map that should inform where the next round of primary-study effort is most useful: where the cells are sparsest, *not* where the existing pooled effects are largest.

Future work: (i) targeted recruitment of K-12 STEM-online primary studies (the largest empty cell); (ii) full random-intercept mixed model implementation in `metafor` or `lme4`; (iii) inclusion of within-discipline domain-specificity moderators (e.g., language vs STEM within Mixed) to reduce the heterogeneity of the Mixed cell; (iv) integration with the parent preprint's Region / Era / Outcome moderator structure to test three-way interactions in any future enlarged corpus.

---

## Acknowledgments

This paper was prepared by the sole author. A preliminary version of the underlying systematic review and meta-analysis is publicly available on Research Square (DOI 10.21203/rs.3.rs-9513298/v1, posted 27 April 2026). The present ICERI submission contributes a 4 (education level) x 4 (discipline) cross-tabulated meta-analytic interaction model that is not reported in the preprint. ORCID: 0009-0009-7124-6669.

---

## References

All references below are taken from `metaanalysis/reference_index.md` (PDF-verified ✅).

### Primary studies cited

- K. Boonyapison, G. Sittironnarit, and P. Rattanaumpawan, "Association between the big five personalities and academic performance among grade 12 students at international high school in Thailand," *Scientific Reports*, vol. 15, 16484, 2025. https://doi.org/10.1038/s41598-025-01038-7
- E. Tokiwa, "Who excels in online learning in Japan?" *Frontiers in Psychology*, vol. 16, Article 1420996, 2025. https://doi.org/10.3389/fpsyg.2025.1420996
- P. Wang, F. Wang, and Z. Li, "Exploring the ecosystem of K-12 online learning: An empirical study of impact mechanisms in the post-pandemic era," *Frontiers in Psychology*, vol. 14, 1241477, 2023. https://doi.org/10.3389/fpsyg.2023.1241477

### Benchmark meta-analyses cited

- S. Mammadov, "Big Five personality traits and academic performance: A meta-analysis," *Journal of Personality*, vol. 90, no. 2, pp. 222–255, 2022. https://doi.org/10.1111/jopy.12663
- J. Meyer, T. Jansen, N. Hübner, and O. Lüdtke, "Disentangling the association between the Big Five personality traits and student achievement: Meta-analytic evidence on the role of domain specificity and achievement measures," *Educational Psychology Review*, vol. 35, Article 12, 2023. https://doi.org/10.1007/s10648-023-09736-2
- A. E. Poropat, "A meta-analysis of the five-factor model of personality and academic performance," *Psychological Bulletin*, vol. 135, no. 2, pp. 322–338, 2009. https://doi.org/10.1037/a0014996
- A. Vedel, "The Big Five and tertiary academic performance: A systematic review and meta-analysis," *Personality and Individual Differences*, vol. 71, pp. 66–76, 2014. https://doi.org/10.1016/j.paid.2014.07.011

### Methodological references

- J. Hartung and G. Knapp, "A refined method for the meta-analysis of controlled clinical trials with binary outcome," *Statistics in Medicine*, vol. 20, no. 24, pp. 3875–3889, 2001. https://doi.org/10.1002/sim.1009
- M. J. Page, J. E. McKenzie, P. M. Bossuyt, I. Boutron, T. C. Hoffmann, C. D. Mulrow, et al., "The PRISMA 2020 statement: An updated guideline for reporting systematic reviews," *BMJ*, vol. 372, n71, 2021. https://doi.org/10.1136/bmj.n71

### Author's own preprint

- E. Tokiwa, "Big Five personality traits and academic achievement in online learning environments: A systematic review and meta-analysis," *Research Square* preprint, https://doi.org/10.21203/rs.3.rs-9513298/v1, 2026.

---

## Appendix A. Reproducibility

- Source dataset: `metaanalysis/conference_submissions/inputs/studies.csv`.
- Analysis script: `metaanalysis/conference_submissions/iceri/scripts/run_cross_tab_meta.py`.
- Result CSVs / MD: `metaanalysis/conference_submissions/iceri/results/{cross_tab_pools, interaction_terms, cross_tab_summary}`.
- Pooling primitives re-used from `metaanalysis/analysis/pool.py`.
- All numbers above are deterministic given the input CSV.

---

*End of manuscript draft. Approximately 3,400 words.*
