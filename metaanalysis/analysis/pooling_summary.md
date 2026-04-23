# Meta-Analysis Pooling Results

**Pre-registered protocol**: Random-effects meta-analysis, REML estimator, HKSJ confidence intervals, Fisher's z transformation.

**Input**: `data_extraction_populated.csv`
**Output**: `pooling_results.csv`

## Per-trait pooled effects

| Trait | k | N_total | r (95% CI) | 95% PI | I² | τ² | Q(df), p |
|-------|---|---------|-----------|--------|-----|-----|----------|
| **O** | 9 | 3363 | 0.086 [-0.044, 0.214] | [-0.273, 0.425] | 92.0% | 0.0212 | 100.17(8), p=0.000 |
| **C** | 10 | 3384 | 0.167 [0.089, 0.243] | [-0.020, 0.343] | 65.1% | 0.0057 | 25.79(9), p=0.002 |
| **E** | 9 | 3363 | 0.002 [-0.076, 0.080] | [-0.195, 0.200] | 75.5% | 0.0061 | 32.66(8), p=0.000 |
| **A** | 9 | 3363 | 0.112 [-0.031, 0.250] | [-0.310, 0.496] | 96.2% | 0.0297 | 208.49(8), p=0.000 |
| **N** | 10 | 3384 | 0.018 [-0.079, 0.114] | [-0.229, 0.263] | 79.0% | 0.0103 | 42.76(9), p=0.000 |

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

**Interpretation notes**:
- r_pooled is the random-effects pooled Pearson correlation (back-transformed from Fisher's z).
- 95% CI uses HKSJ adjustment (t-distribution, df = k-1).
- 95% PI is the range within which a future study's true effect is expected to fall (df = k-2).
- I² quantifies the proportion of variance due to heterogeneity (vs. sampling error).
- τ² is the REML-estimated between-study variance.
- β-only studies converted via Peterson & Brown (2005): r ≈ β + 0.05 if β ≥ 0 else β − 0.05.