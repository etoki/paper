# Meta-Analysis Pooling Results

**Pre-registered protocol**: Random-effects meta-analysis, REML estimator, HKSJ confidence intervals, Fisher's z transformation.

**Input**: `data_extraction_populated.csv`
**Output**: `pooling_results.csv`, `moderator_results.csv`

## Per-trait pooled effects (primary analysis)

| Trait | k | N_total | r (95% CI) | 95% PI | I² | τ² | Q(df), p |
|-------|---|---------|-----------|--------|-----|-----|----------|
| **O** | 9 | 3363 | 0.085 [-0.044, 0.211] | [-0.293, 0.440] | 91.7% | 0.0237 | 96.80(8), p=0.000 |
| **C** | 10 | 3384 | 0.168 [0.090, 0.245] | [-0.039, 0.362] | 65.0% | 0.0070 | 25.74(9), p=0.002 |
| **E** | 9 | 3363 | 0.010 [-0.059, 0.080] | [-0.165, 0.185] | 63.5% | 0.0047 | 21.90(8), p=0.005 |
| **A** | 9 | 3363 | 0.116 [-0.020, 0.248] | [-0.309, 0.502] | 95.8% | 0.0301 | 189.85(8), p=0.000 |
| **N** | 10 | 3384 | 0.016 [-0.080, 0.112] | [-0.246, 0.276] | 78.5% | 0.0117 | 41.93(9), p=0.000 |

## Moderator analyses (exploratory; k < 10 per level caveats apply)

Three pre-registered moderators are reported; the remaining six (instrument, publication year, sample size, RoB score, modality, education level) are reported narratively in the manuscript due to insufficient k per level.

Moderator test: Knapp–Hartung mixed-effects meta-regression on the levels with k ≥ 2 (REML residual τ²), F(df1, df2). p_Holm: Holm correction within each trait across the three moderators.

### Moderator: region

| Trait | Level | k | N | r [95% CI] | I² | F (df1, df2), p, p_Holm |
|-------|-------|---|---|-----------|-----|-------------------------|
| O | non-Asia | 7 | 2062 | 0.054 [-0.063, 0.170] | 65.7% | 0.69 (1, 7), p=0.435, p_Holm=1.000 |
|  | Asia | 2 | 1301 | 0.155 [-0.988, 0.994] | 95.8% |  |
| C | non-Asia | 8 | 2083 | 0.187 [0.083, 0.287] | 69.9% | 0.58 (1, 8), p=0.470, p_Holm=1.000 |
|  | Asia | 2 | 1301 | 0.110 [-0.045, 0.260] | 0.0% |  |
| E | non-Asia | 7 | 2062 | 0.050 [-0.004, 0.104] | 0.0% | 12.94 (1, 7), p=0.009, p_Holm=0.026 |
|  | Asia | 2 | 1301 | -0.095 [-0.541, 0.393] | 23.0% |  |
| A | non-Asia | 7 | 2062 | 0.038 [-0.029, 0.105] | 31.5% | 9.13 (1, 7), p=0.019, p_Holm=0.058 |
|  | Asia | 2 | 1301 | 0.319 [-0.979, 0.994] | 95.4% |  |
| N | non-Asia | 8 | 2083 | -0.009 [-0.132, 0.115] | 81.0% | 0.99 (1, 8), p=0.348, p_Holm=0.582 |
|  | Asia | 2 | 1301 | 0.089 [0.005, 0.171] | 0.0% |  |

### Moderator: era

| Trait | Level | k | N | r [95% CI] | I² | F (df1, df2), p, p_Holm |
|-------|-------|---|---|-----------|-----|-------------------------|
| O | pre-COVID | 3 | 806 | 0.104 [-0.427, 0.581] | 84.2% | 0.00 (1, 6), p=0.954, p_Holm=1.000 |
|  | COVID | 5 | 2275 | 0.094 [-0.122, 0.300] | 93.9% |  |
|  | mixed | 1 | — | (k<2) | — |  |
| C | pre-COVID | 4 | 827 | 0.221 [-0.033, 0.448] | 71.7% | 0.02 (1, 7), p=0.889, p_Holm=1.000 |
|  | COVID | 5 | 2275 | 0.180 [0.097, 0.261] | 53.9% |  |
|  | mixed | 1 | — | (k<2) | — |  |
| E | pre-COVID | 3 | 806 | 0.033 [0.002, 0.063] | 0.0% | 0.22 (1, 6), p=0.659, p_Holm=0.659 |
|  | COVID | 5 | 2275 | -0.002 [-0.155, 0.152] | 80.4% |  |
|  | mixed | 1 | — | (k<2) | — |  |
| A | pre-COVID | 3 | 806 | 0.052 [-0.159, 0.259] | 38.8% | 0.39 (1, 6), p=0.554, p_Holm=1.000 |
|  | COVID | 5 | 2275 | 0.160 [-0.115, 0.412] | 97.0% |  |
|  | mixed | 1 | — | (k<2) | — |  |
| N | pre-COVID | 4 | 827 | -0.050 [-0.173, 0.074] | 19.6% | 1.38 (1, 7), p=0.279, p_Holm=0.582 |
|  | COVID | 5 | 2275 | 0.058 [-0.124, 0.237] | 85.7% |  |
|  | mixed | 1 | — | (k<2) | — |  |

### Moderator: outcome_type

| Trait | Level | k | N | r [95% CI] | I² | F (df1, df2), p, p_Holm |
|-------|-------|---|---|-----------|-----|-------------------------|
| O | objective | 7 | 2649 | 0.091 [-0.086, 0.262] | 93.2% | 0.03 (1, 7), p=0.862, p_Holm=1.000 |
|  | self-report | 2 | 714 | 0.068 [-0.636, 0.710] | 65.2% |  |
| C | objective | 8 | 2670 | 0.151 [0.050, 0.250] | 62.4% | 1.00 (1, 8), p=0.347, p_Holm=1.000 |
|  | self-report | 2 | 714 | 0.224 [-0.187, 0.568] | 0.0% |  |
| E | objective | 7 | 2649 | -0.028 [-0.090, 0.035] | 31.7% | 9.01 (1, 7), p=0.020, p_Holm=0.040 |
|  | self-report | 2 | 714 | 0.116 [-0.143, 0.361] | 0.0% |  |
| A | objective | 7 | 2649 | 0.129 [-0.052, 0.302] | 96.4% | 0.17 (1, 7), p=0.695, p_Holm=1.000 |
|  | self-report | 2 | 714 | 0.068 [-0.742, 0.797] | 77.6% |  |
| N | objective | 8 | 2670 | -0.010 [-0.100, 0.079] | 66.4% | 2.01 (1, 8), p=0.194, p_Holm=0.582 |
|  | self-report | 2 | 714 | 0.117 [-0.922, 0.951] | 92.1% |  |

## Moderator tests with p < .10

- **E × region**: F(1, 7) = 12.94, p = 0.009 (Holm 0.026) — non-Asia (k=7): r = 0.050; Asia (k=2): r = -0.095
- **A × region**: F(1, 7) = 9.13, p = 0.019 (Holm 0.058) — non-Asia (k=7): r = 0.038; Asia (k=2): r = 0.319
- **E × outcome_type**: F(1, 7) = 9.01, p = 0.020 (Holm 0.040) — objective (k=7): r = -0.028; self-report (k=2): r = 0.116

Every moderator contrast rests on k ≤ 7 per level (k = 2 for Asia and for self-report outcomes); treat all of them as exploratory.

## Contributing studies per trait

### Trait O (k = 9)
- A-01 Abe 2020 (r)
- A-02 Alkis 2018 (r)
- A-22 Quigley 2022 (r)
- A-23 Rodrigues 2024 (r)
- A-28 Yu 2021 (beta_converted)
- A-29 Bahcekapili 2020 (r)
- A-30 Kaspar 2023 (beta_converted)
- A-31 Rivers 2021 (r)
- A-37 Zheng 2023 (r)

### Trait C (k = 10)
- A-01 Abe 2020 (r)
- A-02 Alkis 2018 (r)
- A-15 Elvers 2003 (r)
- A-22 Quigley 2022 (r)
- A-23 Rodrigues 2024 (r)
- A-28 Yu 2021 (beta_converted)
- A-29 Bahcekapili 2020 (r)
- A-30 Kaspar 2023 (beta_converted)
- A-31 Rivers 2021 (r)
- A-37 Zheng 2023 (r)

### Trait E (k = 9)
- A-01 Abe 2020 (r)
- A-02 Alkis 2018 (r)
- A-22 Quigley 2022 (r)
- A-23 Rodrigues 2024 (r)
- A-28 Yu 2021 (beta_converted)
- A-29 Bahcekapili 2020 (r)
- A-30 Kaspar 2023 (beta_converted)
- A-31 Rivers 2021 (r)
- A-37 Zheng 2023 (r)

### Trait A (k = 9)
- A-01 Abe 2020 (r)
- A-02 Alkis 2018 (r)
- A-22 Quigley 2022 (r)
- A-23 Rodrigues 2024 (r)
- A-28 Yu 2021 (beta_converted)
- A-29 Bahcekapili 2020 (r)
- A-30 Kaspar 2023 (beta_converted)
- A-31 Rivers 2021 (r)
- A-37 Zheng 2023 (r)

### Trait N (k = 10)
- A-01 Abe 2020 (r)
- A-02 Alkis 2018 (r)
- A-15 Elvers 2003 (r)
- A-22 Quigley 2022 (r)
- A-23 Rodrigues 2024 (r)
- A-28 Yu 2021 (beta_converted)
- A-29 Bahcekapili 2020 (r)
- A-30 Kaspar 2023 (beta_converted)
- A-31 Rivers 2021 (r)
- A-37 Zheng 2023 (r)

---

## Notes
- **Power caveat**: With k = 9-10 per trait pool and Asian subgroup limited to k = 2, moderator findings are **underpowered** and should be interpreted as exploratory.
- **β-to-r conversion**: Peterson & Brown (2005) applied where only β was reported. The exclude-converted sensitivity analysis is in `sensitivity_results.md`.
- **Sign conventions**: A-23 Rodrigues GPA sign-flipped so positive r = better performance; A-31 Rivers Emotional Stability sign-reversed to Neuroticism.
- **Remaining 6 pre-registered moderators** (instrument, publication year, sample size, RoB score, modality, education level) not quantitatively analyzed due to insufficient k per level; reported narratively in Methods Deviations subsection.
## Publication Bias Assessment

### Egger's Regression Test

| Trait | k | Intercept | SE | t | p |
|-------|---|-----------|-----|---|---|
| O | 9 | -6.270 | 2.596 | -2.42 | 0.046 🔴 |
| C | 10 | 2.159 | 1.123 | 1.92 | 0.091 |
| E | 9 | 1.270 | 1.602 | 0.79 | 0.454 |
| A | 9 | -8.285 | 3.799 | -2.18 | 0.066 |
| N | 10 | -1.780 | 1.614 | -1.10 | 0.302 |

### Duval & Tweedie (2000) Trim-and-Fill

| Trait | k_orig | k_imputed | Side | r_original | r_adjusted [95% CI] |
|-------|--------|-----------|------|-----------|---------------------|
| O | 9 | 1 | right | 0.085 | 0.106 [-0.017, 0.226] |
| C | 10 | 2 | left | 0.168 | 0.149 [0.064, 0.231] |
| E | 9 | 0 | right | 0.010 | 0.010 [-0.059, 0.080] |
| A | 9 | 3 | right | 0.116 | 0.180 [0.061, 0.294] |
| N | 10 | 1 | right | 0.016 | 0.027 [-0.075, 0.128] |

**Interpretation**:
- Egger's test: non-zero intercept suggests small-study effects or publication bias. With k = 9–10, the test has low power; p ≥ .05 does not rule out bias.
- Trim-and-fill (L0 estimator, side from the REML regression of y on its standard error, as in metafor::trimfill): imputes hypothetical missing studies on the under-represented side of the funnel. Reported as a *sensitivity check* only; adjusted estimates should NOT replace primary pooled estimates.
- O: Egger p = 0.046; trim-and-fill imputes 1 on the right.
