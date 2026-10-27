# Correction to: Modality matters for Extraversion: A Big Five meta-regression in online learning environments

Eisuke Tokiwa, SUNBLAZE Co., Ltd., Tokyo, Japan (ORCID 0009-0009-7124-6669)

Original paper: Tokiwa, E. (2026). Modality matters for Extraversion: A Big Five meta-regression in online learning environments. In *Proceedings of the 25th European Conference on e-Learning (ECEL 2026)*, Lund, Sweden, 22–23 October 2026 (Paper EL-095, pp. [PAGES]). https://doi.org/[DOI]

## Summary

After publication, the author identified errors in the statistical analysis of this paper. The modality moderator test (Table 2) used an inappropriate procedure, the trait × modality interaction model (Table 3) treated non-independent observations as independent, the between-study variance (τ²) was under-estimated, and the Neuroticism rows of Table 1 were carried over from an earlier draft. The corrected analysis is given below. The Conscientiousness findings are unchanged. The Extraversion difference between asynchronous and mixed-online courses remains nominally significant, but it is weaker than reported and rests on two studies; it should be read as a hypothesis for future testing rather than an established effect. The author apologises for these errors.

## 1. Modality moderator test (Table 2)

The published Q_between statistic entered the Hartung–Knapp–Sidik–Jonkman (HKSJ) subgroup standard errors into a large-sample χ²(1) distribution. This is not a standard procedure and is inconsistent with the small-sample, t-based confidence intervals in Table 1. With three to five studies per subgroup, the test was recomputed as a mixed-effects meta-regression of Fisher z on a modality indicator (asynchronous = reference, mixed-online = 1), with REML residual τ² and the Knapp–Hartung adjustment, referred to F(1, k − 2) (Knapp & Hartung, 2003; Viechtbauer et al., 2015). Conventional large-sample subgroup tests would give smaller p-values, but they ignore the uncertainty in τ² and can have inflated Type I error rates when the number of studies is this small (Viechtbauer et al., 2015).

Table 2 (corrected). Modality moderator tests, asynchronous (A) vs mixed-online (M); Knapp–Hartung mixed-effects meta-regression on the Fisher z scale.

| Trait | k | Δz (M − A) [95% CI] | F (df) | p | Published Q (p) |
|---|---|---|---|---|---|
| Openness | 8 | −0.221 [−0.499, 0.058] | 3.75 (1, 6) | .101 | 1.96 (.161) |
| Conscientiousness | 9 | −0.019 [−0.218, 0.179] | 0.05 (1, 7) | .823 | 0.015 (.903) |
| Extraversion | 8 | +0.177 [0.057, 0.297] | 12.95 (1, 6) | .011 | 17.60 (< .001) |
| Agreeableness | 8 | −0.266 [−0.543, 0.011] | 5.53 (1, 6) | .057 | 3.28 (.070) |
| Neuroticism | 9 | +0.003 [−0.247, 0.254] | 0.00 (1, 7) | .975 | 0.42 (.519) |

## 2. Interaction model (Table 3) — withdrawn

The weighted-OLS model treated the 42 study × trait rows as independent, but they come from only nine studies, each contributing up to five traits measured in the same participants. Its standard errors, t-tests and the joint Wald test (χ²(4) = 13.64, p = .0085) are therefore not valid. In addition, because Openness was the reference trait, the trait[E]:modality[M] coefficient (+0.385) equals the Extraversion modality difference (+0.159) minus the Openness modality difference (−0.226); it does not measure the Extraversion difference alone. Table 3 and every statement based on it (Abstract; Sections 1, 2.3, 3.4, 4.2, 4.4, 5 and 6) are withdrawn. The per-trait tests in the corrected Table 2 replace it.

## 3. τ² estimation and Table 1

The REML routine iterated an update that under-estimated τ². τ² is now estimated by maximising the restricted likelihood directly (results verified against the PyMARE implementation). In addition, the published Neuroticism rows of Table 1 were carried over from a draft computed before the modality re-coding described in Section 2.2 (Figure 2 already showed the re-coded Neuroticism values).

Table 1 (corrected). Modality-stratified random-effects pooled correlations (REML + HKSJ).

| Trait | Modality | k | N | r [95% CI] | I² | τ² |
|---|---|---|---|---|---|---|
| O | A | 3 | 1393 | 0.222 [−0.376, 0.689] | 92.0% | 0.056 |
| O | M | 5 | 1445 | 0.017 [−0.083, 0.116] | 47.8% | 0.003 |
| O | S | 1 | 525 | 0.070 (k = 1, narrative) | — | — |
| C | A | 4 | 1414 | 0.209 [−0.029, 0.424] | 63.5% | 0.014 |
| C | M | 5 | 1445 | 0.180 [0.051, 0.302] | 67.5% | 0.007 |
| C | S | 1 | 525 | 0.068 (k = 1, narrative) | — | — |
| E | A | 3 | 1393 | −0.121 [−0.246, 0.007] | 19.6% | 0.000 |
| E | M | 5 | 1445 | 0.058 [−0.028, 0.143] | 27.2% | 0.001 |
| E | S | 1 | 525 | 0.027 (k = 1, narrative) | — | — |
| A | A | 3 | 1393 | 0.278 [−0.294, 0.703] | 93.7% | 0.053 |
| A | M | 5 | 1445 | 0.033 [−0.075, 0.139] | 55.2% | 0.004 |
| A | S | 1 | 525 | −0.013 (k = 1, narrative) | — | — |
| N | A | 4 | 1414 | 0.076 [−0.035, 0.186] | 42.2% | 0.000 |
| N | M | 5 | 1445 | 0.027 [−0.154, 0.207] | 86.2% | 0.019 |
| N | S | 1 | 525 | −0.072 (k = 1, narrative) | — | — |

Apart from the Neuroticism rows, pooled estimates changed by at most 0.019 (Conscientiousness, asynchronous: 0.190 → 0.209). The trait-only values in Section 3.1 change by at most 0.003 (e.g., Conscientiousness r = 0.169 [0.090, 0.246]). Figure 2 differs from the corrected Table 1 by at most 0.02 in point estimates and 0.03 in confidence limits.

## 4. Robustness of the Extraversion result (Section 3.5)

The asynchronous Extraversion cell contains three studies: A-01 Abe (2020; USA; r = +0.030), A-28 Yu (2021; China; r = −0.126) and A-31 Rivers (2021; Japan; r = −0.173). In a leave-one-out analysis, omitting A-28 gives F(1, 5) = 3.28, p = .130, and omitting A-31 gives F(1, 5) = 7.13, p = .044 (other single omissions: p = .005 to .036). Omitting both East Asian studies leaves Δz = +0.028, F(1, 4) = 0.06, p = .819. Under the published β-converted-drop scenario (A-28 and A-30 removed), asynchronous r = −0.083 [−0.880, 0.837] (k = 2), mixed-online r = 0.043 [−0.078, 0.163] (k = 4), and F(1, 4) = 2.25, p = .208. The published statements "−0.121 → −0.095" and "The headline interaction effect is qualitatively robust under all three scenarios" are therefore incorrect. Because both negative asynchronous estimates come from East Asian samples, modality and region cannot be separated in these data.

## 5. Other corrections

- **Section 1 (results cited from the parent preprint).** "Two pre-registered moderators reached significance at p < .001" should read: with the procedure of Section 1 of this correction, Extraversion × Region gives F(1, 7) = 29.52, p < .001, and Extraversion × Outcome Type gives F(1, 7) = 6.92, p = .034; in both cases the smaller subgroup has k = 2.
- **Section 2.1.** "12 contributed at least one extractable Pearson correlation" should read "12 met the primary-pool criteria, of which 10 contributed at least one extractable Pearson correlation".
- **Figure 1 (final box and caption).** "k = 11 … asynchronous k = 6, mixed-online k = 5, synchronous k = 1" should read "k = 10 … asynchronous k = 4, mixed-online k = 5, synchronous k = 1".
- **Section 2.3.** The sentences from "Between-modality heterogeneity was tested via Q_between" to "is reported as the primary inferential statistic" are replaced by: "For each trait, the difference between asynchronous and mixed-online studies was tested with a mixed-effects meta-regression of Fisher z on a modality indicator, with REML residual τ² and the Knapp–Hartung adjustment, F(1, k − 2) (Knapp & Hartung, 2003; Viechtbauer et al., 2015)."
- **Sections 3.2 and 4.2.** "(0.190 vs 0.180)" should read "(0.209 vs 0.180)", and the mixed-online Extraversion estimate "+0.059" should read "+0.058".
- **Section 3.3.** "Agreeableness shows a trend (p = .070)" should read "Agreeableness shows a trend (p = .057)", and "Conscientiousness is exceptionally close to zero on Q (0.015)" should read "Conscientiousness shows essentially no difference (F(1, 7) = 0.05, p = .823)".
- **Section 4.1.** "0.190 in the asynchronous bucket … an absolute difference of 0.010 … with a Q_between p of .903" should read "0.209 in the asynchronous bucket … an absolute difference of 0.029 … with a moderator-test p of .823".
- **Section 4.3.** Openness: "+0.223 with an enormous CI [-0.374, 0.689] … Q_between p = .161" should read "+0.222 with an enormous CI [−0.376, 0.689] … p = .101". Agreeableness: "(A r = +0.283, M r = +0.032; Q_between p = .070)" should read "(A r = +0.278, M r = +0.033; p = .057)". Neuroticism: "(r = +0.076, CI just touching zero)" should read "(r = +0.076 [−0.035, 0.186])", and "(p = .519)" should read "(p = .975)".
- **Section 5.** In the first paragraph, "the joint interaction test (which aggregates across cells) carries the inferential weight of this paper" is replaced by "the Extraversion contrast in particular depends on two East Asian studies (Section 4 of this correction)". The fourth paragraph ("The interaction model uses a simple weighted-OLS approximation …") is deleted.

## 6. Corrected text

**Abstract.** From "REML with HKSJ-adjusted CIs were used" to the end of the abstract, the text is replaced by: "REML with HKSJ-adjusted CIs were used, and each trait was tested for a difference between asynchronous (A) and mixed-online (M) instruction with a Knapp–Hartung-adjusted mixed-effects meta-regression. Extraversion differed by modality (asynchronous r = −0.121 [−0.246, 0.007], k = 3; mixed-online r = +0.058 [−0.028, 0.143], k = 5; F(1, 6) = 12.95, p = .011); Agreeableness showed a trend (p = .057); Conscientiousness, Neuroticism and Openness did not differ (p ≥ .10). The Extraversion difference is fragile: both negative asynchronous estimates come from East Asian samples, and without them the difference disappears (p = .82), so modality cannot be separated from region. Modality does not change the bottom line for Conscientiousness (r = 0.180–0.209 across A and M). For online course designers, the robust message is to support self-regulation regardless of synchrony; whether Extraversion is a disadvantage in asynchronous programmes is a hypothesis that requires studies from other regions."

**Section 4.2, first paragraph.** The sentences from "The interaction term trait[E]:modality[M]" to "extraverts thrive when there is a social channel and lose ground when there is not." are replaced by: "The modality difference is +0.177 on the Fisher z scale (F(1, 6) = 12.95, p = .011). It rests, however, on two asynchronous studies from East Asia (A-28, China; A-31, Japan); without them the difference disappears (p = .82). The pattern is consistent with the social-presence hypothesis — extraverts benefit from a live social channel and lose that advantage when it is absent — but the present data cannot distinguish a modality effect from a regional one."

**Section 6, first paragraph.** "is in fact modality-dependent: a small negative effect in asynchronous courses, reversing to a small positive effect in mixed-online courses, with a joint interaction Wald test at p = .0085" is replaced by "may be modality-dependent: a small negative estimate in asynchronous courses and a small positive one in mixed-online courses (p = .011), a difference that rests on two East Asian studies and needs replication".

**Section 6, second paragraph.** "they hold (weakly) in mixed-online and synchronous formats and invert in asynchronous-only formats. This trait-by-modality contingency is exactly the pattern self-regulated-learning theory (Pintrich, 2004; Zimmerman, 2000) predicts." is replaced by "they may hold (weakly) in mixed-online and synchronous formats but not in asynchronous-only formats; this contingency, which self-regulated-learning theory (Pintrich, 2004; Zimmerman, 2000) would predict, remains to be tested outside East Asia."

## 7. Cause of the errors and data availability

The errors were in the analysis scripts, which were prepared with the AI assistance declared in the paper. They were not detected when the author re-executed the scripts before submission; they were identified when the statistical procedures were re-examined after publication. The corrected analysis script and its complete output are openly available in the project repository (https://github.com/etoki/paper, folder metaanalysis/conference_submissions/ecel/erratum).

## Additional references

Knapp, G., & Hartung, J. (2003). Improved tests for a random effects meta-regression with a single covariate. *Statistics in Medicine, 22*(17), 2693–2710. https://doi.org/10.1002/sim.1482

Viechtbauer, W., López-López, J. A., Sánchez-Meca, J., & Marín-Martínez, F. (2015). A comparison of procedures to test for moderators in mixed-effects meta-regression models. *Psychological Methods, 20*(3), 360–374. https://doi.org/10.1037/met0000023
