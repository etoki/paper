# Sensitivity Analyses

Pre-registered sensitivity analyses per OSF Registration §15e.

## Primary pooled estimates (reference)

| Trait | k | r [95% CI] | I² |
|-------|---|-----------|-----|
| O | 8 | 0.096 [-0.052, 0.239] | 92.2% |
| C | 9 | 0.185 [0.112, 0.255] | 59.0% |
| E | 8 | 0.012 [-0.070, 0.093] | 68.0% |
| A | 8 | 0.130 [-0.022, 0.277] | 96.0% |
| N | 9 | 0.017 [-0.095, 0.128] | 80.7% |

## Exclude Author's Own Study (A-25 Tokiwa 2025, COI)

The author's own prior primary study was excluded to address the pre-declared conflict of interest.

| Trait | k | r [95% CI] | Δr vs primary |
|-------|---|-----------|---------------|
| O | 8 | 0.096 [-0.052, 0.239] | +0.000 |
| C | 9 | 0.185 [0.112, 0.255] | +0.000 |
| E | 8 | 0.012 [-0.070, 0.093] | +0.000 |
| A | 8 | 0.130 [-0.022, 0.277] | +0.000 |
| N | 9 | 0.017 [-0.095, 0.128] | +0.000 |

## Exclude Peterson-Brown β-converted Studies

Studies contributing only β (not zero-order r), requiring Peterson & Brown (2005) conversion, were excluded. Remaining pool uses direct r or Spearman ρ only.

| Trait | k | r [95% CI] | Δr vs primary |
|-------|---|-----------|---------------|
| O | 6 | 0.034 [-0.121, 0.188] | -0.062 🔴 |
| C | 7 | 0.206 [0.107, 0.301] | +0.022 |
| E | 6 | 0.016 [-0.090, 0.122] | +0.004 |
| A | 6 | 0.069 [-0.012, 0.150] | -0.061 🔴 |
| N | 7 | -0.040 [-0.129, 0.049] | -0.057 🔴 |

## Exclude Low-Quality Studies (RoB < 5)

Studies with JBI aggregate risk-of-bias score below 5 were excluded. Higher-quality subset.

| Trait | k | r [95% CI] | Δr vs primary |
|-------|---|-----------|---------------|
| O | 8 | 0.096 [-0.052, 0.239] | +0.000 |
| C | 9 | 0.185 [0.112, 0.255] | +0.000 |
| E | 8 | 0.012 [-0.070, 0.093] | +0.000 |
| A | 8 | 0.130 [-0.022, 0.277] | +0.000 |
| N | 9 | 0.017 [-0.095, 0.128] | +0.000 |

## Exclude Studies with Analytic N < 50

Studies with fewer than 50 participants in the analysed sample were excluded.

| Trait | k | r [95% CI] | Δr vs primary |
|-------|---|-----------|---------------|
| O | 8 | 0.096 [-0.052, 0.239] | +0.000 |
| C | 8 | 0.180 [0.105, 0.253] | -0.005 |
| E | 8 | 0.012 [-0.070, 0.093] | +0.000 |
| A | 8 | 0.130 [-0.022, 0.277] | +0.000 |
| N | 8 | 0.030 [-0.074, 0.134] | +0.014 |

## DerSimonian-Laird Estimator

All studies; tau² estimated with the DerSimonian-Laird moment estimator instead of REML.

| Trait | k | r [95% CI] | Δr vs primary |
|-------|---|-----------|---------------|
| O | 8 | 0.096 [-0.052, 0.239] | -0.000 |
| C | 9 | 0.184 [0.112, 0.255] | -0.000 |
| E | 8 | 0.012 [-0.070, 0.093] | +0.000 |
| A | 8 | 0.129 [-0.021, 0.273] | -0.001 |
| N | 9 | 0.017 [-0.095, 0.128] | +0.000 |

## Include A-37 Zheng (|r| < .10 entered as r = 0)

A-37 pooled face-to-face and online cohorts and reported only that all trait-grade correlations were below .10 in absolute value; the primary analysis leaves it out of the pool. Here it is added with r = 0. Not pre-registered.

| Trait | k | r [95% CI] | Δr vs primary |
|-------|---|-----------|---------------|
| O | 9 | 0.085 [-0.044, 0.211] | -0.011 |
| C | 10 | 0.168 [0.090, 0.245] | -0.016 |
| E | 9 | 0.010 [-0.059, 0.080] | -0.001 |
| A | 9 | 0.116 [-0.020, 0.248] | -0.014 |
| N | 10 | 0.016 [-0.080, 0.112] | -0.000 |

## Leave-One-Out Analysis

Each study is removed in turn to assess influence on the pooled estimate. Studies where removal causes |Δr| > 0.05 are flagged.

### Trait O  (primary r = 0.096)

| Dropped | r_pooled | Δr | Flag |
|---------|----------|-----|------|
| A-01 Abe(r) | 0.067 | -0.029 |  |
| A-02 Alkis(r) | 0.121 | +0.025 |  |
| A-22 Quigley(r) | 0.110 | +0.014 |  |
| A-23 Rodrigues(r) | 0.109 | +0.014 |  |
| A-28 Yu(beta_converted) | 0.049 | -0.047 |  |
| A-29 Bahcekapili(r) | 0.100 | +0.004 |  |
| A-30 Kaspar(beta_converted) | 0.091 | -0.005 |  |
| A-31 Rivers(r) | 0.117 | +0.021 |  |

### Trait C  (primary r = 0.185)

| Dropped | r_pooled | Δr | Flag |
|---------|----------|-----|------|
| A-01 Abe(r) | 0.168 | -0.016 |  |
| A-02 Alkis(r) | 0.185 | -0.000 |  |
| A-15 Elvers(r) | 0.180 | -0.005 |  |
| A-22 Quigley(r) | 0.172 | -0.013 |  |
| A-23 Rodrigues(r) | 0.180 | -0.004 |  |
| A-28 Yu(beta_converted) | 0.202 | +0.017 |  |
| A-29 Bahcekapili(r) | 0.202 | +0.017 |  |
| A-30 Kaspar(beta_converted) | 0.186 | +0.001 |  |
| A-31 Rivers(r) | 0.191 | +0.007 |  |

### Trait E  (primary r = 0.012)

| Dropped | r_pooled | Δr | Flag |
|---------|----------|-----|------|
| A-01 Abe(r) | 0.010 | -0.001 |  |
| A-02 Alkis(r) | 0.007 | -0.005 |  |
| A-22 Quigley(r) | -0.008 | -0.019 |  |
| A-23 Rodrigues(r) | 0.017 | +0.005 |  |
| A-28 Yu(beta_converted) | 0.032 | +0.020 |  |
| A-29 Bahcekapili(r) | 0.009 | -0.003 |  |
| A-30 Kaspar(beta_converted) | -0.003 | -0.015 |  |
| A-31 Rivers(r) | 0.029 | +0.018 |  |

### Trait A  (primary r = 0.130)

| Dropped | r_pooled | Δr | Flag |
|---------|----------|-----|------|
| A-01 Abe(r) | 0.126 | -0.004 |  |
| A-02 Alkis(r) | 0.135 | +0.005 |  |
| A-22 Quigley(r) | 0.127 | -0.003 |  |
| A-23 Rodrigues(r) | 0.149 | +0.018 |  |
| A-28 Yu(beta_converted) | 0.054 | -0.077 | ⚠ |
| A-29 Bahcekapili(r) | 0.152 | +0.022 |  |
| A-30 Kaspar(beta_converted) | 0.151 | +0.021 |  |
| A-31 Rivers(r) | 0.132 | +0.001 |  |

### Trait N  (primary r = 0.017)

| Dropped | r_pooled | Δr | Flag |
|---------|----------|-----|------|
| A-01 Abe(r) | 0.019 | +0.002 |  |
| A-02 Alkis(r) | 0.013 | -0.004 |  |
| A-15 Elvers(r) | 0.030 | +0.014 |  |
| A-22 Quigley(r) | 0.020 | +0.003 |  |
| A-23 Rodrigues(r) | 0.041 | +0.024 |  |
| A-28 Yu(beta_converted) | 0.003 | -0.014 |  |
| A-29 Bahcekapili(r) | 0.030 | +0.013 |  |
| A-30 Kaspar(beta_converted) | -0.013 | -0.030 |  |
| A-31 Rivers(r) | 0.005 | -0.012 |  |

---

## Interpretation notes
- **|Δr| > 0.05** is flagged as a potentially influential change per Cochrane Handbook guidance for correlation meta-analyses.
- COI sensitivity (exclude Tokiwa A-25) is particularly relevant given the author's prior study is included in the primary pool.
- β-conversion sensitivity tests robustness of the Peterson-Brown approximation for studies reporting only standardized regression coefficients.
- Low-quality exclusion tests whether findings are driven by studies below the pre-specified RoB threshold of 5.
- Not run: the pre-registered HEXACO crosswalk variants, because no HEXACO study is in the quantitative pool.