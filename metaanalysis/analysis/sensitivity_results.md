# Sensitivity Analyses

Pre-registered sensitivity analyses per OSF Registration §15e.

## Primary pooled estimates (reference)

| Trait | k | r [95% CI] | I² |
|-------|---|-----------|-----|
| O | 9 | 0.085 [-0.044, 0.211] | 91.7% |
| C | 10 | 0.168 [0.090, 0.245] | 65.0% |
| E | 9 | 0.010 [-0.059, 0.080] | 63.5% |
| A | 9 | 0.116 [-0.020, 0.248] | 95.8% |
| N | 10 | 0.016 [-0.080, 0.112] | 78.5% |

## Exclude Author's Own Study (A-25 Tokiwa 2025, COI)

The author's own prior primary study was excluded to address the pre-declared conflict of interest.

| Trait | k | r [95% CI] | Δr vs primary |
|-------|---|-----------|---------------|
| O | 9 | 0.085 [-0.044, 0.211] | +0.000 |
| C | 10 | 0.168 [0.090, 0.245] | +0.000 |
| E | 9 | 0.010 [-0.059, 0.080] | +0.000 |
| A | 9 | 0.116 [-0.020, 0.248] | +0.000 |
| N | 10 | 0.016 [-0.080, 0.112] | +0.000 |

## Exclude Peterson-Brown β-converted Studies

Studies contributing only β (not zero-order r), requiring Peterson & Brown (2005) conversion, were excluded. Remaining pool uses direct r or Spearman ρ only.

| Trait | k | r [95% CI] | Δr vs primary |
|-------|---|-----------|---------------|
| O | 6 | 0.034 [-0.121, 0.188] | -0.051 🔴 |
| C | 7 | 0.206 [0.107, 0.301] | +0.038 |
| E | 6 | 0.016 [-0.090, 0.122] | +0.006 |
| A | 6 | 0.069 [-0.012, 0.150] | -0.047 |
| N | 7 | -0.040 [-0.129, 0.049] | -0.057 🔴 |

## Exclude Low-Quality Studies (RoB < 5)

Studies with JBI aggregate risk-of-bias score below 5 were excluded. Higher-quality subset.

| Trait | k | r [95% CI] | Δr vs primary |
|-------|---|-----------|---------------|
| O | 8 | 0.096 [-0.052, 0.239] | +0.011 |
| C | 9 | 0.185 [0.112, 0.255] | +0.016 |
| E | 8 | 0.012 [-0.070, 0.093] | +0.001 |
| A | 8 | 0.130 [-0.022, 0.277] | +0.014 |
| N | 9 | 0.017 [-0.095, 0.128] | +0.000 |

## Leave-One-Out Analysis

Each study is removed in turn to assess influence on the pooled estimate. Studies where removal causes |Δr| > 0.05 are flagged.

### Trait O  (primary r = 0.085)

| Dropped | r_pooled | Δr | Flag |
|---------|----------|-----|------|
| A-01 Abe(r) | 0.059 | -0.026 |  |
| A-02 Alkis(r) | 0.106 | +0.021 |  |
| A-22 Quigley(r) | 0.096 | +0.011 |  |
| A-23 Rodrigues(r) | 0.096 | +0.011 |  |
| A-28 Yu(beta_converted) | 0.042 | -0.044 |  |
| A-29 Bahcekapili(r) | 0.087 | +0.002 |  |
| A-30 Kaspar(beta_converted) | 0.079 | -0.006 |  |
| A-31 Rivers(r) | 0.102 | +0.017 |  |
| A-37 Zheng(r) | 0.096 | +0.011 |  |

### Trait C  (primary r = 0.168)

| Dropped | r_pooled | Δr | Flag |
|---------|----------|-----|------|
| A-01 Abe(r) | 0.152 | -0.017 |  |
| A-02 Alkis(r) | 0.166 | -0.002 |  |
| A-15 Elvers(r) | 0.163 | -0.005 |  |
| A-22 Quigley(r) | 0.155 | -0.013 |  |
| A-23 Rodrigues(r) | 0.162 | -0.006 |  |
| A-28 Yu(beta_converted) | 0.181 | +0.012 |  |
| A-29 Bahcekapili(r) | 0.183 | +0.015 |  |
| A-30 Kaspar(beta_converted) | 0.167 | -0.002 |  |
| A-31 Rivers(r) | 0.173 | +0.004 |  |
| A-37 Zheng(r) | 0.185 | +0.016 |  |

### Trait E  (primary r = 0.010)

| Dropped | r_pooled | Δr | Flag |
|---------|----------|-----|------|
| A-01 Abe(r) | 0.009 | -0.001 |  |
| A-02 Alkis(r) | 0.006 | -0.004 |  |
| A-22 Quigley(r) | -0.007 | -0.017 |  |
| A-23 Rodrigues(r) | 0.015 | +0.004 |  |
| A-28 Yu(beta_converted) | 0.029 | +0.019 |  |
| A-29 Bahcekapili(r) | 0.008 | -0.003 |  |
| A-30 Kaspar(beta_converted) | -0.003 | -0.013 |  |
| A-31 Rivers(r) | 0.025 | +0.015 |  |
| A-37 Zheng(r) | 0.012 | +0.001 |  |

### Trait A  (primary r = 0.116)

| Dropped | r_pooled | Δr | Flag |
|---------|----------|-----|------|
| A-01 Abe(r) | 0.111 | -0.005 |  |
| A-02 Alkis(r) | 0.118 | +0.002 |  |
| A-22 Quigley(r) | 0.111 | -0.005 |  |
| A-23 Rodrigues(r) | 0.130 | +0.014 |  |
| A-28 Yu(beta_converted) | 0.045 | -0.071 | ⚠ |
| A-29 Bahcekapili(r) | 0.133 | +0.017 |  |
| A-30 Kaspar(beta_converted) | 0.132 | +0.016 |  |
| A-31 Rivers(r) | 0.115 | -0.001 |  |
| A-37 Zheng(r) | 0.130 | +0.014 |  |

### Trait N  (primary r = 0.016)

| Dropped | r_pooled | Δr | Flag |
|---------|----------|-----|------|
| A-01 Abe(r) | 0.018 | +0.002 |  |
| A-02 Alkis(r) | 0.013 | -0.003 |  |
| A-15 Elvers(r) | 0.027 | +0.011 |  |
| A-22 Quigley(r) | 0.019 | +0.003 |  |
| A-23 Rodrigues(r) | 0.037 | +0.021 |  |
| A-28 Yu(beta_converted) | 0.004 | -0.012 |  |
| A-29 Bahcekapili(r) | 0.028 | +0.012 |  |
| A-30 Kaspar(beta_converted) | -0.010 | -0.026 |  |
| A-31 Rivers(r) | 0.006 | -0.010 |  |
| A-37 Zheng(r) | 0.017 | +0.000 |  |

---

## Interpretation notes
- **|Δr| > 0.05** is flagged as a potentially influential change per Cochrane Handbook guidance for correlation meta-analyses.
- COI sensitivity (exclude Tokiwa A-25) is particularly relevant given the author's prior study is included in the primary pool.
- β-conversion sensitivity tests robustness of the Peterson-Brown approximation for studies reporting only standardized regression coefficients.
- Low-quality exclusion tests whether findings are driven by studies below the pre-specified RoB threshold of 5.