# Sensitivity Analyses

Pre-registered sensitivity analyses per OSF Registration §15e.

## Primary pooled estimates (reference)

| Trait | k | r [95% CI] | I² |
|-------|---|-----------|-----|
| O | 9 | 0.086 [-0.044, 0.214] | 92.0% |
| C | 10 | 0.169 [0.090, 0.246] | 65.1% |
| E | 9 | 0.002 [-0.075, 0.080] | 75.5% |
| A | 9 | 0.111 [-0.031, 0.249] | 96.2% |
| N | 10 | 0.017 [-0.081, 0.114] | 79.0% |

## Exclude Author's Own Study (A-25 Tokiwa 2025, COI)

The author's own prior primary study was excluded to address the pre-declared conflict of interest.

| Trait | k | r [95% CI] | Δr vs primary |
|-------|---|-----------|---------------|
| O | 9 | 0.086 [-0.044, 0.214] | +0.000 |
| C | 10 | 0.169 [0.090, 0.246] | +0.000 |
| E | 9 | 0.002 [-0.075, 0.080] | +0.000 |
| A | 9 | 0.111 [-0.031, 0.249] | +0.000 |
| N | 10 | 0.017 [-0.081, 0.114] | +0.000 |

## Exclude Peterson-Brown β-converted Studies

Studies contributing only β (not zero-order r), requiring Peterson & Brown (2005) conversion, were excluded. Remaining pool uses direct r or Spearman ρ only.

| Trait | k | r [95% CI] | Δr vs primary |
|-------|---|-----------|---------------|
| O | 6 | 0.034 [-0.121, 0.188] | -0.052 🔴 |
| C | 7 | 0.206 [0.107, 0.301] | +0.037 |
| E | 6 | 0.016 [-0.090, 0.122] | +0.013 |
| A | 6 | 0.069 [-0.012, 0.150] | -0.042 |
| N | 7 | -0.040 [-0.129, 0.049] | -0.057 🔴 |

## Exclude Low-Quality Studies (RoB < 5)

Studies with JBI aggregate risk-of-bias score below 5 were excluded. Higher-quality subset.

| Trait | k | r [95% CI] | Δr vs primary |
|-------|---|-----------|---------------|
| O | 8 | 0.097 [-0.052, 0.242] | +0.011 |
| C | 9 | 0.185 [0.113, 0.256] | +0.016 |
| E | 8 | 0.003 [-0.087, 0.093] | +0.001 |
| A | 8 | 0.125 [-0.036, 0.280] | +0.014 |
| N | 9 | 0.017 [-0.095, 0.129] | +0.001 |

## Leave-One-Out Analysis

Each study is removed in turn to assess influence on the pooled estimate. Studies where removal causes |Δr| > 0.05 are flagged.

### Trait O  (primary r = 0.086)

| Dropped | r_pooled | Δr | Flag |
|---------|----------|-----|------|
| A-01 Abe(r) | 0.060 | -0.026 |  |
| A-02 Alkis(r) | 0.107 | +0.021 |  |
| A-22 Quigley(r) | 0.097 | +0.011 |  |
| A-23 Rodrigues(r) | 0.097 | +0.011 |  |
| A-28 Yu(beta_converted) | 0.042 | -0.044 |  |
| A-29 Bahcekapili(r) | 0.088 | +0.002 |  |
| A-30 Kaspar(beta_converted) | 0.080 | -0.006 |  |
| A-31 Rivers(r) | 0.103 | +0.017 |  |
| A-37 Zheng(r) | 0.097 | +0.011 |  |

### Trait C  (primary r = 0.169)

| Dropped | r_pooled | Δr | Flag |
|---------|----------|-----|------|
| A-01 Abe(r) | 0.153 | -0.016 |  |
| A-02 Alkis(r) | 0.167 | -0.002 |  |
| A-15 Elvers(r) | 0.164 | -0.005 |  |
| A-22 Quigley(r) | 0.156 | -0.013 |  |
| A-23 Rodrigues(r) | 0.163 | -0.006 |  |
| A-28 Yu(beta_converted) | 0.181 | +0.012 |  |
| A-29 Bahcekapili(r) | 0.184 | +0.015 |  |
| A-30 Kaspar(beta_converted) | 0.167 | -0.002 |  |
| A-31 Rivers(r) | 0.173 | +0.004 |  |
| A-37 Zheng(r) | 0.185 | +0.016 |  |

### Trait E  (primary r = 0.002)

| Dropped | r_pooled | Δr | Flag |
|---------|----------|-----|------|
| A-01 Abe(r) | 0.001 | -0.002 |  |
| A-02 Alkis(r) | -0.003 | -0.005 |  |
| A-22 Quigley(r) | -0.016 | -0.018 |  |
| A-23 Rodrigues(r) | 0.006 | +0.004 |  |
| A-28 Yu(beta_converted) | 0.029 | +0.026 |  |
| A-29 Bahcekapili(r) | -0.001 | -0.003 |  |
| A-30 Kaspar(beta_converted) | -0.012 | -0.014 |  |
| A-31 Rivers(r) | 0.018 | +0.016 |  |
| A-37 Zheng(r) | 0.003 | +0.001 |  |

### Trait A  (primary r = 0.111)

| Dropped | r_pooled | Δr | Flag |
|---------|----------|-----|------|
| A-01 Abe(r) | 0.106 | -0.005 |  |
| A-02 Alkis(r) | 0.113 | +0.002 |  |
| A-22 Quigley(r) | 0.106 | -0.005 |  |
| A-23 Rodrigues(r) | 0.125 | +0.014 |  |
| A-28 Yu(beta_converted) | 0.040 | -0.072 | ⚠ |
| A-29 Bahcekapili(r) | 0.128 | +0.016 |  |
| A-30 Kaspar(beta_converted) | 0.134 | +0.022 |  |
| A-31 Rivers(r) | 0.110 | -0.001 |  |
| A-37 Zheng(r) | 0.125 | +0.014 |  |

### Trait N  (primary r = 0.017)

| Dropped | r_pooled | Δr | Flag |
|---------|----------|-----|------|
| A-01 Abe(r) | 0.019 | +0.002 |  |
| A-02 Alkis(r) | 0.014 | -0.003 |  |
| A-15 Elvers(r) | 0.028 | +0.011 |  |
| A-22 Quigley(r) | 0.020 | +0.003 |  |
| A-23 Rodrigues(r) | 0.038 | +0.021 |  |
| A-28 Yu(beta_converted) | 0.005 | -0.012 |  |
| A-29 Bahcekapili(r) | 0.029 | +0.012 |  |
| A-30 Kaspar(beta_converted) | -0.010 | -0.027 |  |
| A-31 Rivers(r) | 0.006 | -0.010 |  |
| A-37 Zheng(r) | 0.017 | +0.001 |  |

---

## Interpretation notes
- **|Δr| > 0.05** is flagged as a potentially influential change per Cochrane Handbook guidance for correlation meta-analyses.
- COI sensitivity (exclude Tokiwa A-25) is particularly relevant given the author's prior study is included in the primary pool.
- β-conversion sensitivity tests robustness of the Peterson-Brown approximation for studies reporting only standardized regression coefficients.
- Low-quality exclusion tests whether findings are driven by studies below the pre-specified RoB threshold of 5.