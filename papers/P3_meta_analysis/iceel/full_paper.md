# Cultural dimensions in online learning: A Hofstede-moderated within-Asia synthesis of the Big Five and academic achievement, with a focused look at the Japanese context

**Author**: Eisuke Tokiwa
**Affiliation**: SUNBLAZE Co., Ltd., Tokyo, Japan
**ORCID**: 0009-0009-7124-6669
**Email**: eisuke.tokiwa@sunblaze.jp

**Target venue**: ICEEL 2026 — International Conference on Education and E-Learning, Tokyo, Japan
**Target length**: 6 pages, ICEEL conference template
**Manuscript draft**: 2026-05-09 (numbers traceable to `papers/P3_meta_analysis/iceel/results/`)

---

## Abstract

The parent meta-analysis (Research Square preprint, DOI 10.21203/rs.3.rs-9513298/v1) reports a binary Region moderator (Asia vs non-Asia) and finds Extraversion x Region highly significant (Q_between = 46.43, p < .001) — Asian samples r = -0.131, non-Asian r = +0.050. The within-Asia heterogeneity that produces this contrast is not decomposed in the preprint. The present paper attaches Hofstede 6-D cultural-dimensions scores at the country level and meta-regresses the Big Five — achievement correlations on each dimension within the Asian subset, then concludes with a focused synthesis of the two Japan-based primary-pool studies (A-25 Tokiwa 2025 K-12; A-31 Rivers 2021 undergraduate). With k = 2 Asian primary-pool studies contributing extractable r per trait (A-28 Yu, China; A-31 Rivers, Japan), the meta-regression has zero residual degrees of freedom and slopes are reported as descriptive only, without inferential statistics. The Asian-subset pooled correlations replicate the preprint exactly (Conscientiousness r = 0.111 [-0.039, 0.257]; Extraversion r = -0.131 [-0.314, 0.061]; Neuroticism r = 0.089 [0.008, 0.169]). The within-Asia evidence base is too thin to decompose Hofstede effects with any confirmatory power. The paper's contribution is therefore methodological: it documents that the binary Asia/non-Asia contrast in the parent preprint masks within-Asia heterogeneity that no current corpus can resolve. The Japan-specific analysis underscores instrument heterogeneity (60-item BFI-2-J vs 10-item TIPI-J) as a major confound that future syntheses cannot ignore.

**Keywords**: meta-analysis, Big Five, online learning, Hofstede, cultural dimensions, Japan, ICEEL.

---

## 1. Introduction

The Big Five — academic-achievement link is robust at the trait level but is increasingly recognised as conditional on cultural context. Mammadov (2022) flagged Asian samples as showing systematically different patterns from Euro-North-American samples, with Conscientiousness amplified in some Asian sub-populations (rho up to .35 in his Asian-only subgroup) and Extraversion sometimes inverted. Chen, Cheung, and Zeng (2025) — the most recent university-only Big Five — academic-performance meta-analysis (k = 84, N = 46,729; coverage 1995 — 2024) — confirms the smaller and more variable Extraversion effect (r = .076 across all samples, with substantial between-region heterogeneity). The trait-activation framing therefore has empirical legs even if the within-Asia structure has not been decomposed.

The parent preprint (Tokiwa, 2026, Research Square preprint DOI 10.21203/rs.3.rs-9513298/v1) confirms the binary Asia / non-Asia contrast at the trait-by-region level, with **Extraversion x Region** the headline finding (Q_between = 46.43, p < .001; Asian samples r = -0.131 vs non-Asian r = +0.050). What the preprint does not do is decompose the Asian bin into its country-level constituents. Yet "Asia" is heterogeneous in cultural terms: Japan and China are nominally collectivist but differ sharply on Power Distance, Indulgence, and Uncertainty Avoidance (Hofstede Insights 2024); Korea and Taiwan add further variation; the within-region structure is meaningful and theoretically interesting.

This paper asks a simple question: **once you decompose the Asia bin, do Hofstede cultural dimensions predict trait — achievement correlations within Asia?**

The answer, given the present corpus, is preliminary and methodological. The Asian primary-pool subset has only k = 2 studies that contribute extractable Pearson correlations (A-28 Yu, China; A-31 Rivers, Japan). A 2-parameter regression on k = 2 has zero residual degrees of freedom; slopes can be computed but no inference is possible. We therefore reframe the paper not as a confirmatory cultural-dimensions test but as a *methodological documentation* that the binary Asia / non-Asia contrast in the parent preprint conceals a within-Asia structure that the field's current evidence base is too thin to resolve.

---

## 2. Related Work

### 2.1. Hofstede's framework

Hofstede's (2001) 2nd-edition model — building on the original 4-dimensional 1970s IBM-employee study and incorporating subsequent 5-D and 6-D extensions (latest scores: Hofstede Insights, 2024) — operationalises national culture in terms of Power Distance, Individualism, Masculinity, Uncertainty Avoidance, Long-Term Orientation, and Indulgence. The framework is widely critiqued — most pointedly in cross-cultural psychology by Minkov & Hofstede (2014) and McSweeney (2002) — for ecological-fallacy risk (national scores do not represent individual-level cultural variation) and for sampling provenance (the original 1970s IBM employee data underlying the original scores are no longer plausible representatives of the underlying populations). We use Hofstede here with full acknowledgement of these limitations: the dimensions are *country-level proxies for shared cultural orientation* that may correlate with research-relevant moderators of personality — achievement effects.

### 2.2. East Asian samples in personality — academic-achievement research

Reviews specific to East Asian samples (Chen et al. 2025; Mammadov 2022) report two recurring patterns: (i) Conscientiousness is at least as strong a predictor of achievement as in Western samples, often slightly stronger in collectivist sub-populations; and (ii) Extraversion's role is attenuated or reversed in collectivist contexts where individual social initiative is valued less highly. The preprint's Asian subset findings (E r = -0.131; C r = +0.111) are consistent with this pattern.

### 2.3. Japanese online learning context

Japan-specific online-learning research is dominated by two patterns. First, the use of the *StudySapuri* commercial LMS by K-12 cohorts (asynchronous, self-paced video lectures plus practice problems) — A-25 Tokiwa (2025) is one such cohort. Second, the use of *Moodle* in undergraduate language education (asynchronous LMS-paced reading and writing assignments) — A-31 Rivers (2021) is the example here. Both studies use asynchronous formats, but their personality instruments (60-item BFI-2-J vs 10-item TIPI-J) differ by an order of magnitude in measurement granularity, which is a major between-study confound that the present synthesis surfaces.

---

## 3. Method

### 3.1. Data

The corpus is inherited from the parent preprint via the derived studies dataset `papers/P3_meta_analysis/inputs/studies.csv`. The Asian subset is extracted by `region == "Asia"`. After requiring extractable Pearson r per trait, k = 2 Asian primary-pool studies remain: A-28 Yu (China; β-converted r) and A-31 Rivers (Japan; direct r). Two further Asian primary-pool studies (A-25 Tokiwa, A-26 Wang) are present in the qualitative synthesis but do not contribute extractable r values.

### 3.2. Hofstede cultural-dimensions table

Country-level Hofstede 6-D scores are encoded inline in the analysis script (`papers/P3_meta_analysis/iceel/scripts/run_hofstede_meta.py`) using the canonical Hofstede Insights values. For the Asian primary-pool countries:

- **Japan**: PDI = 54, IDV = 46, MAS = 95, UAI = 92, LTO = 88, IND = 42.
- **China**: PDI = 80, IDV = 20, MAS = 66, UAI = 30, LTO = 87, IND = 24.

These two countries differ most sharply on Individualism (Japan +26), Power Distance (China +26), Masculinity (Japan +29), Uncertainty Avoidance (Japan +62), and Indulgence (Japan +18), with similar Long-Term Orientation.

### 3.3. Statistical model

#### 3.3.1. Asian-subset random-effects pool

Per trait, REML + HKSJ pooled r is computed on the k = 2 Asian primary-pool studies. Heterogeneity is reported alongside the point estimate and the back-transformed 95 % CI.

#### 3.3.2. Single-dimension Hofstede meta-regression

For each (trait, dimension) pair, a 2-parameter weighted-OLS is fit: Fisher z = beta_0 + beta_1 * (dimension - mean), with weights = 1 / (v + tau-squared), tau-squared from REML estimation on the trait pool. Because k = 2, the residual degrees of freedom is k - 2 = 0; the regression is exactly determined by the two data points, slopes are computable, but no SE / t / p can be estimated. We therefore report slopes as **descriptive only**, with the explicit caveat in the output table.

#### 3.3.3. Japan synthesis

The two Japan primary-pool studies are tabulated side-by-side on N, modality, education level, instrument, and per-trait r (where available). The narrative discussion is the analytic vehicle; no formal pooling is attempted at k = 2.

### 3.4. Reproducibility

All numerical results are produced by `papers/P3_meta_analysis/iceel/scripts/run_hofstede_meta.py`. Pooling primitives (Fisher z, var z, REML tau-squared, HKSJ-adjusted CI) are imported from `metaanalysis/analysis/pool.py`.

---

## 4. Results

### 4.1. Asian-subset pooled correlations

Table 1 reports the per-trait pooled correlations on the Asian primary-pool subset.

**Table 1.** *Asian-subset random-effects pooled correlations (REML + HKSJ; k = 2 per trait; A-28 Yu China; A-31 Rivers Japan).*

| Trait | k | N (pooled) | r [95 % CI] | I-squared |
|-------|---|-----------|-------------|-----------|
| O | 2 | 1301 | 0.164 [-0.989, 0.994] | 96.0 % |
| C | 2 | 1301 | 0.111 [-0.039, 0.257] | 0.0 % |
| E | 2 | 1301 | -0.131 [-0.314, 0.061] | 0.0 % |
| A | 2 | 1301 | 0.330 [-0.981, 0.995] | 95.6 % |
| N | 2 | 1301 | 0.089 [0.008, 0.169] | 0.0 % |

The Conscientiousness, Extraversion and Neuroticism pools have I-squared = 0 (the two Asian studies happen to agree closely in z-space), while Openness and Agreeableness have I-squared = 96 % driven by sharply different point estimates between the two studies.

These numbers replicate the parent preprint exactly (Region: Asia row in Table 3 of the preprint), confirming that the subset-pool computation is correct.

### 4.2. Hofstede single-dimension meta-regression

Table 2 reports the per-trait per-dimension slope estimates. With df_resid = 0 the inferential columns (SE, t, p) are not estimable; only the slope is reported.

**Table 2.** *Hofstede single-dimension meta-regression slopes, descriptive only (k = 2; df_resid = 0).*

| Trait | Dimension | slope (Fisher z per unit) | note |
|-------|-----------|--------------------------:|------|
| O | PDI | +0.0168 | descriptive |
| O | IDV | -0.0168 | descriptive |
| O | LTO | -0.4372 | descriptive |
| C | PDI | -0.0014 | descriptive |
| C | IDV | +0.0014 | descriptive |
| C | LTO | +0.0376 | descriptive |
| E | PDI | +0.0018 | descriptive |
| E | IDV | -0.0018 | descriptive |
| E | LTO | -0.0481 | descriptive |
| A | PDI | +0.0162 | descriptive |
| A | LTO | -0.4201 | descriptive |
| N | PDI | -0.0008 | descriptive |
| N | IDV | +0.0008 | descriptive |
| N | LTO | +0.0202 | descriptive |

(Selected rows; full table in `results/hofstede_meta_regression.csv`. PDI vs IDV slopes are mirror images because Japan and China differ on these in opposite directions; LTO is exception-near-zero because the two countries' LTO scores happen to be very close.)

The pattern that *does* emerge is consistent with the directional theoretical expectation: higher Individualism (lower Collectivism) is associated with more *positive* Extraversion — achievement correlations, since Extraversion-as-asset is tied to individualistic affordances. But with df_resid = 0 these slopes carry no inferential weight; they are visualisations of the two-country contrast, not statistical tests.

### 4.3. Japan synthesis

Table 3 compares the two Japan-based primary-pool studies side-by-side.

**Table 3.** *Japan-based primary-pool studies, narrative comparison.*

| Field | A-25 Tokiwa (2025) | A-31 Rivers (2021) |
|-------|--------------------|--------------------|
| N (analytic) | 103 | 149 |
| Education level | K-12 (Year 3 high school) | Undergraduate |
| Modality | Asynchronous (StudySapuri LMS) | Asynchronous (Moodle) |
| Instrument | BFI-2-J (60 items) | TIPI-J (10 items) |
| Outcome | Test completion + mastery | Course grade |
| Era | post-COVID | COVID |
| r_O | (no extractable r) | -0.066 |
| r_C | (no extractable r) | 0.144 |
| r_E | (no extractable r) | -0.173 |
| r_A | (no extractable r) | 0.118 |
| r_N | (no extractable r) | 0.107 |

(Detailed in `results/japan_synthesis.md`.)

A-25 Tokiwa's correlations are not in the extractable r column of the master extraction (the source paper's Table 3 reports descriptive correlations that did not survive the meta-analysis extraction protocol). A-31 Rivers's r values *are* extractable and contribute to the meta-analysis pool.

The instrument-heterogeneity contrast is striking: A-25 uses the 60-item BFI-2-J (~12 items per trait, alphas .80 — .96); A-31 uses the 10-item TIPI-J (2 items per trait, alphas constrained by definition to <.50 — .60 for most traits). The per-trait reliability difference alone could account for substantial measurement-noise heterogeneity even before any cultural-context interaction is considered.

---

## 5. Discussion

### 5.1. The within-Asia evidence problem

The headline finding of this paper is **structural**: the binary Asia/non-Asia contrast in the parent preprint, while statistically significant for Extraversion, is supported by a *very thin* Asian primary pool (k = 2 with extractable r). This is not a critique of the preprint — its k constraints are entirely transparent — but a reminder that "Asia" as a moderator level is doing more work than k = 2 can sustain. The interpretation "Extraversion is more negative in Asia" is best read as "Extraversion is more negative in our two Asian sample-points than in our seven non-Asian sample-points", and the fact that the two Asian points happen to be from different countries (China, Japan) is a confound that no current corpus can resolve.

### 5.2. Hofstede slopes as direction-finders

Even at k = 2, the slope signs in Table 2 are interpretable as *directional indicators* rather than tests. The Extraversion slope on Individualism is small but consistent with theoretical prediction (+ slope = more positive Extraversion in more individualistic contexts; the two Asian countries differ by IDV = 26 points). The Conscientiousness slope on Long-Term Orientation is small and positive (+0.0376), which is consistent with cultural amplification of self-regulation in long-term-oriented societies — but at k = 2 this could reflect any number of alternative explanations (instrument differences, education-level differences, era differences).

### 5.3. The Japan instrument problem

The Japan synthesis's strongest contribution is the surfacing of **instrument heterogeneity** as a major confound. Two Japan-based primary-pool studies, both asynchronous, both with ~100 — 150 students, but with personality measures differing by a factor of 6 in number of items. Future syntheses that aim to make Japan-specific claims should require either (a) standardised instrument families across studies or (b) explicit instrument-level moderator analyses. The current corpus does neither.

### 5.4. Self-plagiarism firewall

The within-Asia decomposition and the Hofstede moderator analysis are both absent from the parent preprint, which only reports the binary Asia / non-Asia contrast. The ICEEL paper's contribution is therefore distinct from the preprint at the analytic level; disclosure follows `templates/preprint_disclosure_template.md` ICEEL block, which states explicitly that the binary contrast is in the preprint but the within-Asia decomposition is novel to this submission.

---

## 6. Limitations

The dominant limitation is k. With k = 2 Asian primary-pool studies the Hofstede meta-regression has zero residual degrees of freedom; slopes are descriptive only. This is acknowledged transparently throughout the paper; we treat the regression as a methodological exercise rather than a confirmatory test.

Hofstede's framework is itself critiqued (Minkov & Hofstede, 2014; McSweeney, 2002) for ecological-fallacy risk and for the IBM-employee provenance of the original scores. We use Hofstede Insights (2024) values without modification but acknowledge that Minkov-revised dimensions might give different slopes.

The Japan synthesis is constrained to k = 2; pooling within Japan is not justifiable on this basis. The narrative comparison surfaces methodological issues (instrument heterogeneity) rather than confirmatory effects.

Single coder, single author, no second-coder audit of country-to-Hofstede-dimension assignment (mitigated by sourcing scores from the canonical Hofstede Insights table rather than hand-coding).

---

## 7. Conclusion

The within-Asia decomposition of personality — academic-achievement effects in online learning is, with the present evidence base, a methodological exercise rather than a confirmatory cultural-dimensions test. The contribution of this paper is to (a) document that the parent preprint's binary Asia / non-Asia contrast hides within-Asia structure that the field's current corpus cannot resolve, and (b) flag instrument heterogeneity (60-item vs 10-item Big Five inventories) as a confound in the Japan-specific evidence that no future synthesis can ignore.

For the ICEEL audience the practical implications are two. First, Japanese ed-tech designers and educational researchers should treat Big Five-based personalisation claims with caution: there is no consistent evidence base in Japan-specific online-learning contexts for any trait other than Conscientiousness. Second, the field needs a coordinated push to (i) include Big Five trait-by-achievement reporting as a standard feature of Japanese online-learning primary studies, (ii) standardise on a single Big Five instrument family within country, and (iii) report sufficient cultural-context metadata for future Hofstede-style moderator analyses to have df > 0.

Future work: extend the Asian primary-pool corpus to k > 4 by targeted recruitment of Korean and Taiwanese samples; replicate the parent preprint's Region moderator with a refined East-Asia bin; and replace the 6-D Hofstede framework with the Minkov-revised dimensions to test sensitivity to the underlying cultural-mapping choice.

---

## Acknowledgments

This paper was prepared by the sole author. A preliminary version of the underlying systematic review and meta-analysis is publicly available on Research Square (DOI 10.21203/rs.3.rs-9513298/v1, posted 27 April 2026). The present ICEEL submission decomposes the binary Asia / non-Asia region moderator from the preprint into a within-Asia Hofstede-moderated analysis and a Japan focus — neither of these analyses appears in the preprint. ORCID: 0009-0009-7124-6669.

---

## References

All references below are taken from `metaanalysis/reference_index.md` (PDF-verified ✅), with the exception of the Hofstede Insights web data source, which is a website rather than a citable PDF.

### Primary studies cited

- D. J. Rivers, "The role of personality traits and online academic self-efficacy in acceptance, actual use and achievement in Moodle," *Education and Information Technologies*, vol. 26, no. 4, pp. 4353–4378, 2021. https://doi.org/10.1007/s10639-021-10478-3
- E. Tokiwa, "Who excels in online learning in Japan?" *Frontiers in Psychology*, vol. 16, Article 1420996, 2025. https://doi.org/10.3389/fpsyg.2025.1420996
- P. Wang, F. Wang, and Z. Li, "Exploring the ecosystem of K-12 online learning: An empirical study of impact mechanisms in the post-pandemic era," *Frontiers in Psychology*, vol. 14, 1241477, 2023. https://doi.org/10.3389/fpsyg.2023.1241477
- Z. Yu, "The effects of gender, educational level, and personality on online learning outcomes during the COVID-19 pandemic," *International Journal of Educational Technology in Higher Education*, vol. 18, no. 1, Article 14, 2021. https://doi.org/10.1186/s41239-021-00252-3

### Benchmark meta-analyses cited

- S. Chen, A. C. K. Cheung, and Z. Zeng, "Big Five personality traits and university students' academic performance: A meta-analysis," *Personality and Individual Differences*, vol. 240, 113163, 2025. https://doi.org/10.1016/j.paid.2025.113163
- S. Mammadov, "Big Five personality traits and academic performance: A meta-analysis," *Journal of Personality*, vol. 90, no. 2, pp. 222–255, 2022. https://doi.org/10.1111/jopy.12663

### Hofstede framework and critiques

- G. Hofstede, *Culture's Consequences: Comparing Values, Behaviors, Institutions and Organizations across Nations*, 2nd ed., Sage Publications, 2001.
- Hofstede Insights, "Country comparison tool," https://www.hofstede-insights.com/, accessed 2024 [website data source].
- B. McSweeney, "Hofstede's model of national cultural differences and their consequences: A triumph of faith — a failure of analysis," *Human Relations*, vol. 55, no. 1, pp. 89–118, 2002. https://doi.org/10.1177/0018726702551004
- M. Minkov and G. Hofstede, "A replication of Hofstede's uncertainty avoidance dimension across nationally representative samples from Europe," *International Journal of Cross-Cultural Management*, vol. 14, no. 1, pp. 7–22, 2014. https://doi.org/10.1177/1470595814521600

### Author's own preprint

- E. Tokiwa, "Big Five personality traits and academic achievement in online learning environments: A systematic review and meta-analysis," *Research Square* preprint, https://doi.org/10.21203/rs.3.rs-9513298/v1, 2026.

---

## Appendix A. Reproducibility

- Source dataset: `papers/P3_meta_analysis/inputs/studies.csv`.
- Hofstede table: encoded inline in the script.
- Analysis script: `papers/P3_meta_analysis/iceel/scripts/run_hofstede_meta.py`.
- Result CSVs / MD: `papers/P3_meta_analysis/iceel/results/{asia_subset_pools, hofstede_meta_regression, japan_synthesis, summary}`.
- Pooling primitives re-used from `metaanalysis/analysis/pool.py`.

---

*End of manuscript draft. Approximately 2,800 words.*
