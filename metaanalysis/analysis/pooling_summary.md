# Meta-Analysis Pooling Results

**Pre-registered protocol**: Random-effects meta-analysis, REML estimator, HKSJ confidence intervals, Fisher's z transformation.

**Input**: `data_extraction_populated.csv`
**Output**: `pooling_results.csv`, `moderator_results.csv`

## Per-trait pooled effects (primary analysis)

| Trait | k | N_total | r (95% CI) | 95% PI | I² | τ² | Q(df), p |
|-------|---|---------|-----------|--------|-----|-----|----------|
| **O** | 9 | 3363 | 0.086 [-0.044, 0.214] | [-0.297, 0.445] | 92.0% | 0.0244 | 100.17(8), p=0.000 |
| **C** | 10 | 3384 | 0.169 [0.090, 0.246] | [-0.038, 0.362] | 65.1% | 0.0070 | 25.79(9), p=0.002 |
| **E** | 9 | 3363 | 0.002 [-0.075, 0.080] | [-0.211, 0.215] | 75.5% | 0.0072 | 32.66(8), p=0.000 |
| **A** | 9 | 3363 | 0.111 [-0.031, 0.249] | [-0.332, 0.514] | 96.2% | 0.0332 | 208.49(8), p=0.000 |
| **N** | 10 | 3384 | 0.017 [-0.081, 0.114] | [-0.248, 0.279] | 79.0% | 0.0120 | 42.76(9), p=0.000 |

## Moderator analyses (exploratory; k < 10 per level caveats apply)

Three pre-registered moderators are reported; the remaining six (instrument, publication year, sample size, RoB score, modality, education level) are reported narratively in the manuscript due to insufficient k per level.

Moderator test: Knapp–Hartung mixed-effects meta-regression on the levels with k ≥ 2 (REML residual τ²), F(df1, df2).

### Moderator: region

| Trait | Level | k | N | r [95% CI] | I² | F (df1, df2), p |
|-------|-------|---|---|-----------|-----|-----------------|
| O | non-Asia | 7 | 2062 | 0.055 [-0.063, 0.170] | 65.9% | 0.71 (1, 7), p=0.427 |
|  | Asia | 2 | 1301 | 0.158 [-0.989, 0.994] | 96.0% |  |
| C | non-Asia | 8 | 2083 | 0.188 [0.083, 0.288] | 70.1% | 0.57 (1, 8), p=0.471 |
|  | Asia | 2 | 1301 | 0.111 [-0.039, 0.257] | 0.0% |  |
| E | non-Asia | 7 | 2062 | 0.050 [-0.004, 0.104] | 0.0% | 29.52 (1, 7), p=0.001 |
|  | Asia | 2 | 1301 | -0.131 [-0.314, 0.061] | 0.0% |  |
| A | non-Asia | 7 | 2062 | 0.032 [-0.045, 0.109] | 47.0% | 8.69 (1, 7), p=0.021 |
|  | Asia | 2 | 1301 | 0.324 [-0.981, 0.995] | 95.6% |  |
| N | non-Asia | 8 | 2083 | -0.009 [-0.133, 0.116] | 81.4% | 0.97 (1, 8), p=0.353 |
|  | Asia | 2 | 1301 | 0.089 [0.008, 0.169] | 0.0% |  |

### Moderator: era

| Trait | Level | k | N | r [95% CI] | I² | F (df1, df2), p |
|-------|-------|---|---|-----------|-----|-----------------|
| O | pre-COVID | 3 | 806 | 0.104 [-0.427, 0.581] | 84.2% | 0.00 (1, 6), p=0.962 |
|  | COVID | 5 | 2275 | 0.095 [-0.124, 0.305] | 94.1% |  |
|  | mixed | 1 | — | (k<2) | — |  |
| C | pre-COVID | 4 | 827 | 0.221 [-0.033, 0.448] | 71.7% | 0.02 (1, 7), p=0.901 |
|  | COVID | 5 | 2275 | 0.181 [0.098, 0.261] | 53.7% |  |
|  | mixed | 1 | — | (k<2) | — |  |
| E | pre-COVID | 3 | 806 | 0.033 [0.002, 0.063] | 0.0% | 0.34 (1, 6), p=0.582 |
|  | COVID | 5 | 2275 | -0.014 [-0.181, 0.153] | 86.3% |  |
|  | mixed | 1 | — | (k<2) | — |  |
| A | pre-COVID | 3 | 806 | 0.052 [-0.159, 0.259] | 38.8% | 0.28 (1, 6), p=0.616 |
|  | COVID | 5 | 2275 | 0.152 [-0.141, 0.420] | 97.4% |  |
|  | mixed | 1 | — | (k<2) | — |  |
| N | pre-COVID | 4 | 827 | -0.050 [-0.173, 0.074] | 19.6% | 1.39 (1, 7), p=0.277 |
|  | COVID | 5 | 2275 | 0.059 [-0.125, 0.240] | 86.0% |  |
|  | mixed | 1 | — | (k<2) | — |  |

### Moderator: outcome_type

| Trait | Level | k | N | r [95% CI] | I² | F (df1, df2), p |
|-------|-------|---|---|-----------|-----|-----------------|
| O | objective | 7 | 2649 | 0.092 [-0.087, 0.265] | 93.4% | 0.03 (1, 7), p=0.862 |
|  | self-report | 2 | 714 | 0.069 [-0.641, 0.715] | 66.1% |  |
| C | objective | 8 | 2670 | 0.152 [0.050, 0.250] | 62.3% | 1.04 (1, 8), p=0.337 |
|  | self-report | 2 | 714 | 0.225 [-0.167, 0.556] | 0.0% |  |
| E | objective | 7 | 2649 | -0.037 [-0.114, 0.040] | 60.8% | 6.92 (1, 7), p=0.034 |
|  | self-report | 2 | 714 | 0.117 [-0.136, 0.356] | 0.0% |  |
| A | objective | 7 | 2649 | 0.131 [-0.054, 0.307] | 96.5% | 0.32 (1, 7), p=0.587 |
|  | self-report | 2 | 714 | 0.043 [-0.861, 0.882] | 87.0% |  |
| N | objective | 8 | 2670 | -0.010 [-0.100, 0.080] | 66.6% | 2.04 (1, 8), p=0.191 |
|  | self-report | 2 | 714 | 0.119 [-0.926, 0.954] | 92.4% |  |

## Moderator tests with p < .10

- **E × region**: F(1, 7) = 29.52, p = 0.001 — non-Asia (k=7): r = 0.050; Asia (k=2): r = -0.131
- **A × region**: F(1, 7) = 8.69, p = 0.021 — non-Asia (k=7): r = 0.032; Asia (k=2): r = 0.324
- **E × outcome_type**: F(1, 7) = 6.92, p = 0.034 — objective (k=7): r = -0.037; self-report (k=2): r = 0.117

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
| O | 9 | -6.407 | 2.631 | -2.43 | 0.045 🔴 |
| C | 10 | 2.143 | 1.128 | 1.90 | 0.094 |
| E | 9 | 2.291 | 1.849 | 1.24 | 0.255 |
| A | 9 | -8.527 | 4.029 | -2.12 | 0.072 |
| N | 10 | -1.790 | 1.631 | -1.10 | 0.304 |

### Duval & Tweedie (2000) Trim-and-Fill

| Trait | k_orig | k_imputed | Side | r_original | r_adjusted [95% CI] |
|-------|--------|-----------|------|-----------|---------------------|
| O | 9 | 1 | right | 0.086 | 0.107 [-0.017, 0.229] |
| C | 10 | 0 | left | 0.169 | 0.169 [0.090, 0.246] |
| E | 9 | 2 | left | 0.002 | -0.035 [-0.116, 0.047] |
| A | 9 | 0 | right | 0.111 | 0.111 [-0.031, 0.249] |
| N | 10 | 0 | right | 0.017 | 0.017 [-0.081, 0.114] |

**Interpretation**:
- Egger's test: non-zero intercept suggests small-study effects or publication bias. With k = 9–10, the test has low power; p ≥ .05 does not rule out bias.
- Trim-and-fill: imputes hypothetical missing studies on the under-represented side of the funnel. Reported as a *sensitivity check* only; adjusted estimates should NOT replace primary pooled estimates.
- Openness (Egger p = .045) shows evidence of asymmetry, with trim-and-fill suggesting possible missing small-negative studies.
- Grey literature (dissertations, conference proceedings) was included in the search to partially mitigate file-drawer bias; unpublished/preprint studies were excluded for quality reasons.
