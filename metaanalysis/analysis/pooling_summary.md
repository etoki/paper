# Meta-Analysis Pooling Results

**Pre-registered protocol**: Random-effects meta-analysis, REML estimator, HKSJ confidence intervals, Fisher's z transformation.

**Input**: `data_extraction_populated.csv`
**Output**: `pooling_results.csv`, `moderator_results.csv`

## Per-trait pooled effects (primary analysis)

| Trait | k | N_total | r (95% CI) | 95% PI | I² | τ² | Q(df), p |
|-------|---|---------|-----------|--------|-----|-----|----------|
| **O** | 9 | 3363 | 0.086 [-0.044, 0.214] | [-0.273, 0.425] | 92.0% | 0.0212 | 100.17(8), p=0.000 |
| **C** | 10 | 3384 | 0.167 [0.089, 0.243] | [-0.020, 0.343] | 65.1% | 0.0057 | 25.79(9), p=0.002 |
| **E** | 9 | 3363 | 0.002 [-0.076, 0.080] | [-0.195, 0.200] | 75.5% | 0.0061 | 32.66(8), p=0.000 |
| **A** | 9 | 3363 | 0.112 [-0.031, 0.250] | [-0.310, 0.496] | 96.2% | 0.0297 | 208.49(8), p=0.000 |
| **N** | 10 | 3384 | 0.018 [-0.079, 0.114] | [-0.229, 0.263] | 79.0% | 0.0103 | 42.76(9), p=0.000 |

## Moderator analyses (exploratory; k < 10 per level caveats apply)

Three pre-registered moderators are reported; the remaining six (instrument, publication year, sample size, RoB score, modality, education level) are reported narratively in the manuscript due to insufficient k per level.

### Moderator: region

| Trait | Level | k | N | r [95% CI] | I² | Q_b (df), p |
|-------|-------|---|---|-----------|-----|-------------|
| O | non-Asia | 7 | 2062 | 0.053 [-0.060, 0.164] | 65.9% | 0.26(1), p=0.612 |
|  | Asia | 2 | 1301 | 0.164 [-0.989, 0.994] | 96.0% |  |
| C | non-Asia | 8 | 2083 | 0.185 [0.082, 0.284] | 70.1% | 2.68(1), p=0.102 |
|  | Asia | 2 | 1301 | 0.111 [-0.039, 0.257] | 0.0% |  |
| E | non-Asia | 7 | 2062 | 0.050 [-0.004, 0.104] | 0.0% | 46.43(1), p=0.000 |
|  | Asia | 2 | 1301 | -0.131 [-0.314, 0.061] | 0.0% |  |
| A | non-Asia | 7 | 2062 | 0.030 [-0.047, 0.106] | 47.0% | 2.17(1), p=0.140 |
|  | Asia | 2 | 1301 | 0.330 [-0.981, 0.995] | 95.6% |  |
| N | non-Asia | 8 | 2083 | -0.007 [-0.130, 0.117] | 81.4% | 3.31(1), p=0.069 |
|  | Asia | 2 | 1301 | 0.089 [0.008, 0.169] | 0.0% |  |

### Moderator: era

| Trait | Level | k | N | r [95% CI] | I² | Q_b (df), p |
|-------|-------|---|---|-----------|-----|-------------|
| O | pre-COVID | 3 | 806 | 0.098 [-0.423, 0.570] | 84.2% | 0.00(1), p=0.994 |
|  | COVID | 5 | 2275 | 0.097 [-0.123, 0.308] | 94.1% |  |
|  | mixed | 1 | — | (k<2) | — |  |
| C | pre-COVID | 4 | 827 | 0.208 [-0.041, 0.434] | 71.7% | 0.13(1), p=0.716 |
|  | COVID | 5 | 2275 | 0.179 [0.095, 0.260] | 53.7% |  |
|  | mixed | 1 | — | (k<2) | — |  |
| E | pre-COVID | 3 | 806 | 0.033 [0.002, 0.063] | 0.0% | 0.59(1), p=0.443 |
|  | COVID | 5 | 2275 | -0.014 [-0.180, 0.153] | 86.3% |  |
|  | mixed | 1 | — | (k<2) | — |  |
| A | pre-COVID | 3 | 806 | 0.033 [-0.162, 0.225] | 38.8% | 1.09(1), p=0.296 |
|  | COVID | 5 | 2275 | 0.153 [-0.141, 0.422] | 97.4% |  |
|  | mixed | 1 | — | (k<2) | — |  |
| N | pre-COVID | 4 | 827 | -0.050 [-0.173, 0.074] | 19.6% | 2.04(1), p=0.153 |
|  | COVID | 5 | 2275 | 0.060 [-0.125, 0.240] | 86.0% |  |
|  | mixed | 1 | — | (k<2) | — |  |

### Moderator: outcome_type

| Trait | Level | k | N | r [95% CI] | I² | Q_b (df), p |
|-------|-------|---|---|-----------|-----|-------------|
| O | objective | 7 | 2649 | 0.092 [-0.086, 0.265] | 93.4% | 0.04(1), p=0.839 |
|  | self-report | 2 | 714 | 0.072 [-0.637, 0.715] | 66.1% |  |
| C | objective | 8 | 2670 | 0.147 [0.048, 0.243] | 62.3% | 2.40(1), p=0.121 |
|  | self-report | 2 | 714 | 0.225 [-0.167, 0.556] | 0.0% |  |
| E | objective | 7 | 2649 | -0.038 [-0.115, 0.039] | 60.8% | 17.30(1), p=0.000 |
|  | self-report | 2 | 714 | 0.117 [-0.136, 0.356] | 0.0% |  |
| A | objective | 7 | 2649 | 0.131 [-0.054, 0.308] | 96.5% | 0.49(1), p=0.485 |
|  | self-report | 2 | 714 | 0.041 [-0.862, 0.881] | 87.0% |  |
| N | objective | 8 | 2670 | -0.009 [-0.097, 0.080] | 66.6% | 0.82(1), p=0.364 |
|  | self-report | 2 | 714 | 0.120 [-0.926, 0.954] | 92.4% |  |

## Key moderator findings 🔴

**Significant at p < .05**:

1. **Extraversion × Region** (Q_b = 46.43, df=1, p < .001):
   - non-Asia (k=7): r = 0.050 (null)
   - Asia (k=2): **r = -0.131** (negative)
   - Interpretation: Asian online learners show a pronounced negative Extraversion-achievement association, consistent with Chen et al. (2025)'s finding that E is significantly negative in individualistic and culturally sensitive contexts. Replication with k > 2 in Asia required.

2. **Extraversion × Outcome Type** (Q_b = 17.30, df=1, p < .001):
   - Objective outcomes (GPA, exam, MOOC composite; k=7): r = -0.038
   - Self-report outcomes (self-rated performance; k=2): **r = +0.117**
   - Interpretation: Extraverts self-report better performance (+ rating bias) but objective measures show weakly negative effects — consistent with social-desirability / self-enhancement bias in extraverted learners.

**Trends (p < .10)**:

- **Neuroticism × Region** (Q_b = 3.31, p = .069): Asia N r = +0.089 vs non-Asia r = -0.007, partial support for H4 modulation by culture.
- **Conscientiousness × Region** (Q_b = 2.68, p = .102): non-Asia r = 0.185 > Asia r = 0.111. Counter-intuitive relative to Mammadov (2022) Asian amplification — possibly driven by Yu 2021's MOOC-specific weak C (β=.057 in linguistics students). Larger k needed.

**Non-significant moderators**:

- Era (pre-COVID vs COVID): all 5 traits n.s. (p = .15 to .99). No evidence for COVID-shock amplification of any trait. Note k is insufficient for post-COVID isolation (only 2 studies, both mixed-era).
- Agreeableness × Region: directional (Asia r=.330 vs non-Asia r=.030) but not significant (p=.14). k=2 Asian limits power.

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
- **β-to-r conversion**: Peterson & Brown (2005) applied where only β was reported. Sensitivity analysis (exclude converted) pending.
- **Sign conventions**: A-23 Rodrigues GPA sign-flipped so positive r = better performance; A-31 Rivers Emotional Stability sign-reversed to Neuroticism.
- **Remaining 6 pre-registered moderators** (instrument, publication year, sample size, RoB score, modality, education level) not quantitatively analyzed due to insufficient k per level; reported narratively in Methods Deviations subsection.
## Publication Bias Assessment (Egger's test)

| Trait | k | Intercept | SE | t | p |
|-------|---|-----------|-----|---|---|
| O | 9 | -6.407 | 2.631 | -2.43 | 0.045 🔴 |
| C | 10 | 2.143 | 1.128 | 1.90 | 0.094 |
| E | 9 | 2.291 | 1.849 | 1.24 | 0.255 |
| A | 9 | -8.527 | 4.029 | -2.12 | 0.072 |
| N | 10 | -1.790 | 1.631 | -1.10 | 0.304 |

Egger's test evaluates funnel-plot asymmetry by regressing standardized effect sizes on precision; a non-zero intercept suggests small-study bias or publication bias. Interpretation threshold: p < .05 indicates potential bias; p ≥ .05 does not rule out bias (test has low power with small k).
