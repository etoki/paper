# Correction to: Modality matters for Extraversion: A Big Five meta-regression in online learning environments

Eisuke Tokiwa, SUNBLAZE Co., Ltd., Tokyo, Japan (ORCID 0009-0009-7124-6669)

Original paper: Tokiwa, E. (2026). Modality matters for Extraversion: A Big Five meta-regression in online learning environments. In *Proceedings of the 25th European Conference on e-Learning (ECEL 2026)*, Lund, Sweden, 22–23 October 2026 (Paper EL-095, pp. [PAGES]). https://doi.org/[DOI]

## Summary

After publication, the author identified errors in the statistical analysis of this paper. The modality moderator test (Table 2) used an inappropriate procedure; the trait × modality interaction model (Table 3) treated non-independent observations as independent; standardised regression coefficients from two studies were converted to correlations with an incorrect formula; the between-study variance (τ²) was under-estimated; and the Neuroticism rows of Table 1 were carried over from an earlier draft. The corrected analysis is given below. The Conscientiousness findings are unchanged. The Extraversion difference between asynchronous and mixed-online courses remains nominally significant, but it is considerably weaker than reported and rests on two studies; it should be read as a hypothesis for future testing rather than an established effect. The author apologises for these errors.

## 1. Modality moderator test (Table 2)

The published Q_between statistic entered the Hartung–Knapp–Sidik–Jonkman (HKSJ) subgroup standard errors into a large-sample χ²(1) distribution. This is not a standard procedure and is inconsistent with the small-sample, t-based confidence intervals in Table 1. With three to five studies per subgroup, the test was recomputed as a mixed-effects meta-regression of Fisher z on a modality indicator (asynchronous = reference, mixed-online = 1), with REML residual τ² and the Knapp–Hartung adjustment, referred to F(1, k − 2) (Knapp & Hartung, 2003; Viechtbauer et al., 2015). Conventional large-sample subgroup tests would give smaller p-values, but they ignore the uncertainty in τ² and can have inflated Type I error rates when the number of studies is this small (Viechtbauer et al., 2015). The corrected values below also incorporate the conversion correction in Section 3.

Table 2 (corrected). Modality moderator tests, asynchronous (A) vs mixed-online (M); Knapp–Hartung mixed-effects meta-regression on the Fisher z scale.

| Trait | k | Δz (M − A) [95% CI] | F (df) | p | Published Q (p) |
|---|---|---|---|---|---|
| Openness | 8 | −0.218 [−0.494, 0.058] | 3.75 (1, 6) | .101 | 1.96 (.161) |
| Conscientiousness | 9 | −0.020 [−0.219, 0.179] | 0.05 (1, 7) | .821 | 0.015 (.903) |
| Extraversion | 8 | +0.139 [0.019, 0.258] | 8.07 (1, 6) | .030 | 17.60 (< .001) |
| Agreeableness | 8 | −0.254 [−0.517, 0.009] | 5.59 (1, 6) | .056 | 3.28 (.070) |
| Neuroticism | 9 | +0.002 [−0.246, 0.250] | 0.00 (1, 7) | .984 | 0.42 (.519) |

## 2. Interaction model (Table 3) — withdrawn

The weighted-OLS model treated the 42 study × trait rows as independent, but they come from only nine studies, each contributing up to five traits measured in the same participants. Its standard errors, t-tests and the joint Wald test (χ²(4) = 13.64, p = .0085) are therefore not valid. In addition, because Openness was the reference trait, the trait[E]:modality[M] coefficient (+0.385) equals the Extraversion modality difference (+0.159) minus the Openness modality difference (−0.226); it does not measure the Extraversion difference alone. Table 3 and every statement based on it (Abstract; Sections 1, 2.3, 3.4, 4.2, 4.4, 5 and 6) are withdrawn. The per-trait tests in the corrected Table 2 replace it.

## 3. Conversion of regression coefficients to correlations

Two studies reported standardised regression coefficients (β) rather than correlations: A-28 Yu (2021) and A-30 Kaspar et al. (2023). The analysis converted them as r = β + .05 for β ≥ 0 and r = β − .05 for β < 0. The published rule of Peterson and Brown (2005) is r = .98β + .05λ, with λ = 1 for β ≥ 0 and λ = 0 for β < 0; the erroneous rule pushed every negative β a further .05 away from zero. With the correct rule, A-28's Extraversion correlation is −.074 rather than −.126, and A-30's Agreeableness correlation is −.010 rather than −.060; the other converted values change by less than .01. All corrected values in this notice use the correct rule.

## 4. τ² estimation and Table 1

The REML routine iterated an update that under-estimated τ². τ² is now estimated by maximising the restricted likelihood directly (results verified against the PyMARE implementation). In addition, the published Neuroticism rows of Table 1 were carried over from a draft computed before the modality re-coding described in Section 2.2 (Figure 2 already showed the re-coded Neuroticism values).

Table 1 (corrected). Modality-stratified random-effects pooled correlations (REML + HKSJ).

| Trait | Modality | k | N | r [95% CI] | I² | τ² |
|---|---|---|---|---|---|---|
| O | A | 3 | 1393 | 0.219 [−0.374, 0.685] | 91.7% | 0.055 |
| O | M | 5 | 1445 | 0.017 [−0.082, 0.115] | 46.8% | 0.003 |
| O | S | 1 | 525 | 0.070 (k = 1, narrative) | — | — |
| C | A | 4 | 1414 | 0.209 [−0.030, 0.425] | 63.8% | 0.014 |
| C | M | 5 | 1445 | 0.179 [0.051, 0.301] | 67.3% | 0.007 |
| C | S | 1 | 525 | 0.068 (k = 1, narrative) | — | — |
| E | A | 3 | 1393 | −0.078 [−0.203, 0.049] | 17.3% | 0.000 |
| E | M | 5 | 1445 | 0.058 [−0.028, 0.143] | 26.8% | 0.001 |
| E | S | 1 | 525 | 0.027 (k = 1, narrative) | — | — |
| A | A | 3 | 1393 | 0.274 [−0.282, 0.693] | 93.4% | 0.050 |
| A | M | 5 | 1445 | 0.043 [−0.047, 0.133] | 34.4% | 0.002 |
| A | S | 1 | 525 | −0.013 (k = 1, narrative) | — | — |
| N | A | 4 | 1414 | 0.076 [−0.036, 0.185] | 42.0% | 0.000 |
| N | M | 5 | 1445 | 0.026 [−0.153, 0.204] | 85.9% | 0.018 |
| N | S | 1 | 525 | −0.072 (k = 1, narrative) | — | — |

The largest changes are the asynchronous Extraversion estimate (−0.121 → −0.078, from the conversion correction), the asynchronous Conscientiousness estimate (0.190 → 0.209, from the τ² correction) and the Neuroticism rows. The trait-only values in Section 3.1 change by at most 0.008 (e.g., Conscientiousness r = 0.168 [0.090, 0.245]; Extraversion r = 0.010 [−0.059, 0.080]). Figure 2 was drawn with the previous values; the corrected values are those in this table.

## 5. Robustness of the Extraversion result (Section 3.5)

The asynchronous Extraversion cell contains three studies: A-01 Abe (2020; USA; r = +0.030), A-28 Yu (2021; China; r = −0.074) and A-31 Rivers (2021; Japan; r = −0.173). In a leave-one-out analysis, omitting A-28 gives F(1, 5) = 3.28, p = .130, and omitting A-31 gives F(1, 5) = 5.26, p = .070 (other single omissions: p = .014 to .077). Omitting both East Asian studies leaves Δz = +0.028, F(1, 4) = 0.06, p = .820. Under the published β-converted-drop scenario (A-28 and A-30 removed), asynchronous r = −0.083 [−0.880, 0.837] (k = 2), mixed-online r = 0.043 [−0.078, 0.163] (k = 4), and F(1, 4) = 2.25, p = .208. The published statements "−0.121 → −0.095" and "The headline interaction effect is qualitatively robust under all three scenarios" are therefore incorrect. Because both negative asynchronous estimates come from East Asian samples, modality and region cannot be separated in these data.

## 6. Other corrections

- **Section 1 (results cited from the parent preprint).** "Two pre-registered moderators reached significance at p < .001" should read: with the procedures of Sections 1, 3 and 4 of this correction, Extraversion × Region gives F(1, 7) = 12.94, p = .009 (Asian r = −0.095 vs non-Asian r = +0.050), and Extraversion × Outcome Type gives F(1, 7) = 9.01, p = .020; in both cases the smaller subgroup has k = 2.
- **Section 2.1.** "12 contributed at least one extractable Pearson correlation" should read "12 met the primary-pool criteria, of which 10 contributed at least one extractable Pearson correlation".
- **Figure 1 (final box and caption).** "k = 11 … asynchronous k = 6, mixed-online k = 5, synchronous k = 1" should read "k = 10 … asynchronous k = 4, mixed-online k = 5, synchronous k = 1".
- **Section 2.3.** The sentences from "Between-modality heterogeneity was tested via Q_between" to "is reported as the primary inferential statistic" are replaced by: "For each trait, the difference between asynchronous and mixed-online studies was tested with a mixed-effects meta-regression of Fisher z on a modality indicator, with REML residual τ² and the Knapp–Hartung adjustment, F(1, k − 2) (Knapp & Hartung, 2003; Viechtbauer et al., 2015). Standardised regression coefficients were converted to correlations with the rule of Peterson and Brown (2005), r = .98β + .05λ."
- **Sections 3.2 and 4.2.** "(0.190 vs 0.180)" should read "(0.209 vs 0.179)"; the asynchronous Extraversion estimate "−0.121 [−0.246, 0.007]" should read "−0.078 [−0.203, 0.049]" (its confidence interval no longer "brushes zero"); and the mixed-online Extraversion estimate "+0.059" should read "+0.058".
- **Section 3.3.** "Agreeableness shows a trend (p = .070)" should read "Agreeableness shows a trend (p = .056)", and "Conscientiousness is exceptionally close to zero on Q (0.015)" should read "Conscientiousness shows essentially no difference (F(1, 7) = 0.05, p = .821)".
- **Section 4.1.** "0.190 in the asynchronous bucket and 0.180 in the mixed-online bucket — an absolute difference of 0.010 … with a Q_between p of .903" should read "0.209 in the asynchronous bucket and 0.179 in the mixed-online bucket — an absolute difference of 0.030 … with a moderator-test p of .821".
- **Section 4.3.** Openness: "+0.223 with an enormous CI [-0.374, 0.689] … Q_between p = .161" should read "+0.219 with an enormous CI [−0.374, 0.685] … p = .101". Agreeableness: "(A r = +0.283, M r = +0.032; Q_between p = .070)" should read "(A r = +0.274, M r = +0.043; p = .056)". Neuroticism: "(r = +0.076, CI just touching zero)" should read "(r = +0.076 [−0.036, 0.185])", and "(p = .519)" should read "(p = .984)".
- **Section 5.** In the first paragraph, "the joint interaction test (which aggregates across cells) carries the inferential weight of this paper" is replaced by "the Extraversion contrast in particular depends on two East Asian studies (Section 5 of this correction)". The fourth paragraph ("The interaction model uses a simple weighted-OLS approximation …") is deleted.

## 7. Corrected text

**Abstract.** From "REML with HKSJ-adjusted CIs were used" to the end of the abstract, the text is replaced by: "REML with HKSJ-adjusted CIs were used, and each trait was tested for a difference between asynchronous (A) and mixed-online (M) instruction with a Knapp–Hartung-adjusted mixed-effects meta-regression. Extraversion differed by modality (asynchronous r = −0.078 [−0.203, 0.049], k = 3; mixed-online r = +0.058 [−0.028, 0.143], k = 5; F(1, 6) = 8.07, p = .030); Agreeableness showed a trend (p = .056); Conscientiousness, Neuroticism and Openness did not differ (p ≥ .10). The Extraversion difference is fragile: both negative asynchronous estimates come from East Asian samples, and without them the difference disappears (p = .82), so modality cannot be separated from region. Modality does not change the bottom line for Conscientiousness (r = 0.179–0.209 across A and M). For online course designers, the robust message is to support self-regulation regardless of synchrony; whether Extraversion is a disadvantage in asynchronous programmes is a hypothesis that requires studies from other regions."

**Section 4.2, first paragraph.** The sentences from "The interaction term trait[E]:modality[M]" to "extraverts thrive when there is a social channel and lose ground when there is not." are replaced by: "The modality difference is +0.139 on the Fisher z scale (F(1, 6) = 8.07, p = .030). It rests, however, on two asynchronous studies from East Asia (A-28, China; A-31, Japan); without them the difference disappears (p = .82). The pattern is consistent with the social-presence hypothesis — extraverts benefit from a live social channel and lose that advantage when it is absent — but the present data cannot distinguish a modality effect from a regional one."

**Section 6, first paragraph.** "is in fact modality-dependent: a small negative effect in asynchronous courses, reversing to a small positive effect in mixed-online courses, with a joint interaction Wald test at p = .0085" is replaced by "may be modality-dependent: a small negative estimate in asynchronous courses and a small positive one in mixed-online courses (p = .030), a difference that rests on two East Asian studies and needs replication".

**Section 6, second paragraph.** "they hold (weakly) in mixed-online and synchronous formats and invert in asynchronous-only formats. This trait-by-modality contingency is exactly the pattern self-regulated-learning theory (Pintrich, 2004; Zimmerman, 2000) predicts." is replaced by "they may hold (weakly) in mixed-online and synchronous formats but not in asynchronous-only formats; this contingency, which self-regulated-learning theory (Pintrich, 2004; Zimmerman, 2000) would predict, remains to be tested outside East Asia."

## 8. Cause of the errors and data availability

The errors were in the analysis scripts, which were prepared with the AI assistance declared in the paper. They were not detected when the author re-executed the scripts before submission; they were identified when the statistical procedures were re-examined after publication. The corrected analysis script and its complete output are openly available in the project repository (https://github.com/etoki/paper, folder metaanalysis/conference_submissions/ecel/erratum).

## Additional references

Knapp, G., & Hartung, J. (2003). Improved tests for a random effects meta-regression with a single covariate. *Statistics in Medicine, 22*(17), 2693–2710. https://doi.org/10.1002/sim.1482

Peterson, R. A., & Brown, S. P. (2005). On the use of beta coefficients in meta-analysis. *Journal of Applied Psychology, 90*(1), 175–181. https://doi.org/10.1037/0021-9010.90.1.175

Viechtbauer, W., López-López, J. A., Sánchez-Meca, J., & Marín-Martínez, F. (2015). A comparison of procedures to test for moderators in mixed-effects meta-regression models. *Psychological Methods, 20*(3), 360–374. https://doi.org/10.1037/met0000023
