# Meta-Analysis Pooling Results

**Pre-registered protocol**: Random-effects meta-analysis, REML estimator, HKSJ confidence intervals, Fisher's z transformation.

**Input**: `data_extraction_populated.csv`
**Output**: `pooling_results.csv`, `moderator_results.csv`

## Per-trait pooled effects (primary analysis)

| Trait | k | N_total | r (95% CI) | 95% PI | I² | τ² | Q(df), p |
|-------|---|---------|-----------|--------|-----|-----|----------|
| **O** | 8 | 3081 | 0.096 [-0.052, 0.239] | [-0.319, 0.480] | 92.2% | 0.0265 | 89.58(7), p=0.000 |
| **C** | 9 | 3102 | 0.185 [0.112, 0.255] | [0.007, 0.351] | 59.0% | 0.0047 | 19.49(8), p=0.012 |
| **E** | 8 | 3081 | 0.012 [-0.070, 0.093] | [-0.191, 0.213] | 68.0% | 0.0058 | 21.90(7), p=0.003 |
| **A** | 8 | 3081 | 0.130 [-0.022, 0.277] | [-0.327, 0.538] | 96.0% | 0.0324 | 176.86(7), p=0.000 |
| **N** | 9 | 3102 | 0.017 [-0.095, 0.128] | [-0.275, 0.305] | 80.7% | 0.0138 | 41.45(8), p=0.000 |

## Moderator analyses (exploratory; k < 10 per level caveats apply)

Three pre-registered moderators are reported; the remaining six (instrument, publication year, sample size, RoB score, modality, education level) are reported narratively in the manuscript due to insufficient k per level.

Moderator test: Knapp–Hartung mixed-effects meta-regression on the levels with k ≥ 2 (REML residual τ²), F(df1, df2). p_Holm: Holm correction within each trait across the three moderators.

### Moderator: region

| Trait | Level | k | N | r [95% CI] | I² | F (df1, df2), p, p_Holm |
|-------|-------|---|---|-----------|-----|-------------------------|
| O | non-Asia | 6 | 1780 | 0.066 [-0.082, 0.211] | 69.9% | 0.43 (1, 6), p=0.536, p_Holm=1.000 |
|  | Asia | 2 | 1301 | 0.155 [-0.988, 0.994] | 95.8% |  |
| C | non-Asia | 7 | 1801 | 0.211 [0.118, 0.300] | 59.0% | 1.83 (1, 7), p=0.219, p_Holm=0.656 |
|  | Asia | 2 | 1301 | 0.110 [-0.045, 0.260] | 0.0% |  |
| E | non-Asia | 6 | 1780 | 0.058 [-0.004, 0.120] | 2.0% | 12.96 (1, 6), p=0.011, p_Holm=0.034 |
|  | Asia | 2 | 1301 | -0.095 [-0.541, 0.393] | 23.0% |  |
| A | non-Asia | 6 | 1780 | 0.047 [-0.036, 0.128] | 40.6% | 6.98 (1, 6), p=0.038, p_Holm=0.115 |
|  | Asia | 2 | 1301 | 0.319 [-0.979, 0.994] | 95.4% |  |
| N | non-Asia | 7 | 1801 | -0.013 [-0.163, 0.138] | 83.7% | 0.87 (1, 7), p=0.381, p_Holm=0.692 |
|  | Asia | 2 | 1301 | 0.089 [0.005, 0.171] | 0.0% |  |

### Moderator: era

| Trait | Level | k | N | r [95% CI] | I² | F (df1, df2), p, p_Holm |
|-------|-------|---|---|-----------|-----|-------------------------|
| O | pre-COVID | 3 | 806 | 0.104 [-0.427, 0.581] | 84.2% | 0.00 (1, 6), p=0.954, p_Holm=1.000 |
|  | COVID | 5 | 2275 | 0.094 [-0.122, 0.300] | 93.9% |  |
| C | pre-COVID | 4 | 827 | 0.221 [-0.033, 0.448] | 71.7% | 0.02 (1, 7), p=0.889, p_Holm=0.889 |
|  | COVID | 5 | 2275 | 0.180 [0.097, 0.261] | 53.9% |  |
| E | pre-COVID | 3 | 806 | 0.033 [0.002, 0.063] | 0.0% | 0.22 (1, 6), p=0.659, p_Holm=0.659 |
|  | COVID | 5 | 2275 | -0.002 [-0.155, 0.152] | 80.4% |  |
| A | pre-COVID | 3 | 806 | 0.052 [-0.159, 0.259] | 38.8% | 0.39 (1, 6), p=0.554, p_Holm=1.000 |
|  | COVID | 5 | 2275 | 0.160 [-0.115, 0.412] | 97.0% |  |
| N | pre-COVID | 4 | 827 | -0.050 [-0.173, 0.074] | 19.6% | 1.38 (1, 7), p=0.279, p_Holm=0.692 |
|  | COVID | 5 | 2275 | 0.058 [-0.124, 0.237] | 85.7% |  |

### Moderator: outcome_type

| Trait | Level | k | N | r [95% CI] | I² | F (df1, df2), p, p_Holm |
|-------|-------|---|---|-----------|-----|-------------------------|
| O | objective | 6 | 2367 | 0.106 [-0.109, 0.312] | 93.6% | 0.07 (1, 6), p=0.797, p_Holm=1.000 |
|  | self-report | 2 | 714 | 0.068 [-0.636, 0.710] | 65.2% |  |
| C | objective | 7 | 2388 | 0.172 [0.072, 0.268] | 57.3% | 0.72 (1, 7), p=0.425, p_Holm=0.850 |
|  | self-report | 2 | 714 | 0.224 [-0.187, 0.568] | 0.0% |  |
| E | objective | 6 | 2367 | -0.031 [-0.108, 0.047] | 40.5% | 7.96 (1, 6), p=0.030, p_Holm=0.061 |
|  | self-report | 2 | 714 | 0.116 [-0.143, 0.361] | 0.0% |  |
| A | objective | 6 | 2367 | 0.151 [-0.064, 0.352] | 96.6% | 0.28 (1, 6), p=0.616, p_Holm=1.000 |
|  | self-report | 2 | 714 | 0.068 [-0.742, 0.797] | 77.6% |  |
| N | objective | 7 | 2388 | -0.014 [-0.124, 0.097] | 71.1% | 1.72 (1, 7), p=0.231, p_Holm=0.692 |
|  | self-report | 2 | 714 | 0.117 [-0.922, 0.951] | 92.1% |  |

## Moderator tests with p < .10

- **E × region**: F(1, 6) = 12.96, p = 0.011 (Holm 0.034) — non-Asia (k=6): r = 0.058; Asia (k=2): r = -0.095
- **A × region**: F(1, 6) = 6.98, p = 0.038 (Holm 0.115) — non-Asia (k=6): r = 0.047; Asia (k=2): r = 0.319
- **E × outcome_type**: F(1, 6) = 7.96, p = 0.030 (Holm 0.061) — objective (k=6): r = -0.031; self-report (k=2): r = 0.116

Every moderator contrast rests on k ≤ 7 per level (k = 2 for Asia and for self-report outcomes); treat all of them as exploratory.

## Contributing studies per trait

### Trait O (k = 8)
- A-01 Abe 2020 (r)
- A-02 Alkis 2018 (r)
- A-22 Quigley 2022 (r)
- A-23 Rodrigues 2024 (r)
- A-28 Yu 2021 (beta_converted)
- A-29 Bahcekapili 2020 (r)
- A-30 Kaspar 2023 (beta_converted)
- A-31 Rivers 2021 (r)

### Trait C (k = 9)
- A-01 Abe 2020 (r)
- A-02 Alkis 2018 (r)
- A-15 Elvers 2003 (r)
- A-22 Quigley 2022 (r)
- A-23 Rodrigues 2024 (r)
- A-28 Yu 2021 (beta_converted)
- A-29 Bahcekapili 2020 (r)
- A-30 Kaspar 2023 (beta_converted)
- A-31 Rivers 2021 (r)

### Trait E (k = 8)
- A-01 Abe 2020 (r)
- A-02 Alkis 2018 (r)
- A-22 Quigley 2022 (r)
- A-23 Rodrigues 2024 (r)
- A-28 Yu 2021 (beta_converted)
- A-29 Bahcekapili 2020 (r)
- A-30 Kaspar 2023 (beta_converted)
- A-31 Rivers 2021 (r)

### Trait A (k = 8)
- A-01 Abe 2020 (r)
- A-02 Alkis 2018 (r)
- A-22 Quigley 2022 (r)
- A-23 Rodrigues 2024 (r)
- A-28 Yu 2021 (beta_converted)
- A-29 Bahcekapili 2020 (r)
- A-30 Kaspar 2023 (beta_converted)
- A-31 Rivers 2021 (r)

### Trait N (k = 9)
- A-01 Abe 2020 (r)
- A-02 Alkis 2018 (r)
- A-15 Elvers 2003 (r)
- A-22 Quigley 2022 (r)
- A-23 Rodrigues 2024 (r)
- A-28 Yu 2021 (beta_converted)
- A-29 Bahcekapili 2020 (r)
- A-30 Kaspar 2023 (beta_converted)
- A-31 Rivers 2021 (r)

---

## Notes
- **Power caveat**: With fewer than 10 studies per trait pool and the Asian subgroup limited to k = 2, moderator findings are **underpowered** and should be interpreted as exploratory.
- **β-to-r conversion**: Peterson & Brown (2005) applied where only β was reported. The exclude-converted sensitivity analysis is in `sensitivity_results.md`.
- **Sign conventions**: A-23 Rodrigues GPA sign-flipped so positive r = better performance; A-31 Rivers Emotional Stability sign-reversed to Neuroticism.
- **Remaining 6 pre-registered moderators** (instrument, publication year, sample size, RoB score, modality, education level) not quantitatively analyzed due to insufficient k per level; reported narratively in Methods Deviations subsection.
## Publication Bias Assessment

### Egger's Regression Test

| Trait | k | Intercept | SE | t | p |
|-------|---|-----------|-----|---|---|
| O | 8 | -6.006 | 2.794 | -2.15 | 0.075 |
| C | 9 | 2.460 | 0.872 | 2.82 | 0.026 🔴 |
| E | 8 | 1.307 | 1.759 | 0.74 | 0.485 |
| A | 8 | -7.926 | 4.101 | -1.93 | 0.101 |
| N | 9 | -1.741 | 1.738 | -1.00 | 0.350 |

### Duval & Tweedie (2000) Trim-and-Fill

| Trait | k_orig | k_imputed | Side | r_original | r_adjusted [95% CI] |
|-------|--------|-----------|------|-----------|---------------------|
| O | 8 | 0 | right | 0.096 | 0.096 [-0.052, 0.239] |
| C | 9 | 3 | left | 0.185 | 0.151 [0.069, 0.230] |
| E | 8 | 0 | right | 0.012 | 0.012 [-0.070, 0.093] |
| A | 8 | 3 | right | 0.130 | 0.216 [0.076, 0.347] |
| N | 9 | 1 | right | 0.017 | 0.030 [-0.087, 0.147] |

**Interpretation**:
- Egger's test: non-zero intercept suggests small-study effects or publication bias. With fewer than 10 studies per trait, the test has low power; p ≥ .05 does not rule out bias.
- Trim-and-fill (L0 estimator, side from the REML regression of y on its standard error, as in metafor::trimfill): imputes hypothetical missing studies on the under-represented side of the funnel. Reported as a *sensitivity check* only; adjusted estimates should NOT replace primary pooled estimates.
- C: Egger p = 0.026; trim-and-fill imputes 3 on the left.
