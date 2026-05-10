# ICEEL 2026 — Abstract (numbers filled from results CSVs)

**Title**:
*Cultural dimensions in online learning: A Hofstede-moderated
within-Asia synthesis of the Big Five and academic achievement, with a
focused look at the Japanese context.*

**Author**: Eisuke Tokiwa (SUNBLAZE Co., Ltd.; ORCID 0009-0009-7124-6669)
**Word target**: ~250 words.

**Numerical sources**:
- `results/asia_subset_pools.csv`
- `results/hofstede_meta_regression.csv`
- `results/japan_synthesis.md`

---

## Draft (numbers filled 2026-05-09 after run)

**Background.** The parent meta-analysis (Research Square preprint, DOI
10.21203/rs.3.rs-9513298/v1) reports a binary Region moderator (Asia vs
non-Asia) and finds Extraversion x Region highly significant
(Q_between = 46.43, p < .001) — Asian samples r = -0.131, non-Asian
r = +0.050. The within-Asia heterogeneity that produces this contrast
is not decomposed in the preprint and is the focus of this paper.

**Aim.** Treat each Asian primary-pool study as a country-level
observation, attach Hofstede 6-D scores (Power Distance, Individualism,
Masculinity, Uncertainty Avoidance, Long-Term Orientation, Indulgence),
and meta-regress the Big Five — achievement correlations on each
dimension. Conclude with a focused synthesis of the two Japan-based
studies (A-25 Tokiwa 2025 K-12; A-31 Rivers 2021 undergraduate).

**Methods.** Asian-subset random-effects pool per trait (REML + HKSJ).
Single-dimension weighted-OLS meta-regression on Fisher's z, weights =
1 / (v + tau-squared). Country aggregation when multiple studies share a
country. Narrative comparison of Japan studies on instrument
(60-item BFI-2-J vs 10-item TIPI-J), modality (both asynchronous), and
education level.

**Results.** With only k = 2 Asian primary-pool studies contributing
extractable r per trait (A-28 Yu, China; A-31 Rivers, Japan), the
regression has zero residual degrees of freedom and slopes are reported
as **descriptive only**, without inferential statistics. The Asian-subset
pooled correlations replicate the preprint exactly: Conscientiousness
r = 0.111 [-0.039, 0.257]; Extraversion r = -0.131 [-0.314, 0.061];
Neuroticism r = 0.089 [0.008, 0.169]; Openness r = 0.164 [-0.989, 0.994]
(I-squared = 96 %); Agreeableness r = 0.330 [-0.981, 0.995]
(I-squared = 96 %).

**Implications.** The within-Asia evidence base is *too thin* to
decompose Hofstede effects with any confirmatory power. The paper's
contribution is therefore methodological: it documents that the binary
Asia/non-Asia contrast in the parent preprint masks a within-Asia
heterogeneity that no current corpus can resolve. The Japan-specific
analysis underscores instrument heterogeneity (60-item BFI-2-J vs 10-item
TIPI-J) as a major confound that future syntheses cannot ignore.

**Keywords**: meta-analysis, Big Five, online learning, Hofstede,
cultural dimensions, Japan, ICEEL.
