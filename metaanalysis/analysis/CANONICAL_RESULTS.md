# Canonical Results Snapshot — Big Five × Online Learning Meta-Analysis

**Generated**: 2026-05-01 from `data_extraction_populated.csv` (re-run of `pool.py`, `sensitivity.py`)
**Purpose**: Single source of truth for hallucination checking. Values below are AUTHORITATIVE.
**Convention**: When manuscript text disagrees with values here, the manuscript is wrong.

---

## 1. Study counts (from `data_extraction_populated.csv`)

| Category | n | Study IDs |
|----------|---|-----------|
| Total rows in extraction CSV | 31 | A-01 to A-37 (gaps: A-14, A-32–A-36) |
| **Retained (any include*)** | **13** | A-01, A-02, A-15, A-17, A-22, A-23, A-25, A-26, A-28, A-29, A-30, A-31, A-37 |
| Excluded — face-to-face / non-online | 3 | A-09, A-10, A-16 (+ A-14 in PRISMA only) |
| Excluded — no extractable r | 1 | A-24 |
| Excluded — sample overlap | 1 | A-05 |
| PDF unavailable | 1 | A-27 |
| Excluded from primary (secondary outcomes only) | 12 | A-03, A-04, A-06, A-07, A-08, A-11, A-12, A-13, A-18, A-19, A-20, A-21 |

**⚠ Manuscript currently states '31 primary studies' — actual retained = 13.**
**31 = total CSV row count, NOT studies passing eligibility.**

## 2. Education-level breakdown of RETAINED studies

| Level | k | Study IDs |
|-------|---|-----------|
| Graduate | 1 | A-37 |
| K-12 | 2 | A-25, A-26 |
| Mixed_UG_Grad | 2 | A-28, A-30 |
| Undergraduate | 8 | A-01, A-02, A-15, A-17, A-22, A-23, A-29, A-31 |
| **Total** | **13** | |

## 3. Region breakdown of RETAINED studies

| Region | k | Study IDs |
|--------|---|-----------|
| Asia | 4 | A-25, A-26, A-28, A-31 |
| Europe | 6 | A-02, A-17, A-22, A-23, A-29, A-30 |
| North_America | 3 | A-01, A-15, A-37 |

## 4. Era breakdown of RETAINED studies

| Era | k | Study IDs |
|-----|---|-----------|
| COVID | 5 | A-22, A-23, A-28, A-30, A-31 |
| Mixed_3era | 1 | A-37 |
| post-COVID | 3 | A-17, A-25, A-26 |
| pre-COVID | 4 | A-01, A-02, A-15, A-29 |

## 5. Risk of Bias (RoB) summary — RETAINED studies only

- Mean: **5.85**  (range 4–7; n = 13)
- ≥ 5 (low-bias): 12
- < 5 (high-bias): 1

## 6. Primary pooled random-effects meta-analysis (REML + HKSJ)

| Trait | k | N_total | r_pooled | 95% CI | 95% PI | I² | τ² | Q(df), p |
|-------|---|---------|----------|--------|--------|----|----|---------|
| O | 9 | 3363 | 0.086 | [-0.044, 0.214] | [-0.273, 0.425] | 92.0% | 0.0212 | 100.17(8), p=0.0000 |
| C | 10 | 3384 | 0.167 | [0.089, 0.243] | [-0.020, 0.343] | 65.1% | 0.0057 | 25.79(9), p=0.0022 |
| E | 9 | 3363 | 0.002 | [-0.076, 0.080] | [-0.196, 0.200] | 75.5% | 0.0061 | 32.66(8), p=0.0001 |
| A | 9 | 3363 | 0.112 | [-0.032, 0.250] | [-0.310, 0.496] | 96.2% | 0.0297 | 208.49(8), p=0.0000 |
| N | 10 | 3384 | 0.018 | [-0.079, 0.114] | [-0.229, 0.263] | 79.0% | 0.0103 | 42.76(9), p=0.0000 |

## 7. Contributing studies per trait pool

### Trait O (k = 9)
- A-01 Abe(r)
- A-02 Alkis(r)
- A-22 Quigley(r)
- A-23 Rodrigues(r)
- A-28 Yu(beta_converted)
- A-29 Bahcekapili(r)
- A-30 Kaspar(beta_converted)
- A-31 Rivers(r)
- A-37 Zheng(r)

### Trait C (k = 10)
- A-01 Abe(r)
- A-02 Alkis(r)
- A-15 Elvers(r)
- A-22 Quigley(r)
- A-23 Rodrigues(r)
- A-28 Yu(beta_converted)
- A-29 Bahcekapili(r)
- A-30 Kaspar(beta_converted)
- A-31 Rivers(r)
- A-37 Zheng(r)

### Trait E (k = 9)
- A-01 Abe(r)
- A-02 Alkis(r)
- A-22 Quigley(r)
- A-23 Rodrigues(r)
- A-28 Yu(beta_converted)
- A-29 Bahcekapili(r)
- A-30 Kaspar(beta_converted)
- A-31 Rivers(r)
- A-37 Zheng(r)

### Trait A (k = 9)
- A-01 Abe(r)
- A-02 Alkis(r)
- A-22 Quigley(r)
- A-23 Rodrigues(r)
- A-28 Yu(beta_converted)
- A-29 Bahcekapili(r)
- A-30 Kaspar(beta_converted)
- A-31 Rivers(r)
- A-37 Zheng(r)

### Trait N (k = 10)
- A-01 Abe(r)
- A-02 Alkis(r)
- A-15 Elvers(r)
- A-22 Quigley(r)
- A-23 Rodrigues(r)
- A-28 Yu(beta_converted)
- A-29 Bahcekapili(r)
- A-30 Kaspar(beta_converted)
- A-31 Rivers(r)
- A-37 Zheng(r)

**β-converted contributors (Peterson & Brown, 2005)**: A-28 Yu (2021), A-30 Kaspar et al. (2023). **THESE ARE THE ONLY TWO.**
**β-converted contributors NOT in primary pool (do NOT cite as contributors)**: A-04 Audet, A-06 Sahinidis, A-08 Bhagat, A-18 Keller, A-20 Mustafa.

## 8. Moderator analyses

### Moderator: region

| Trait | Level | k | N | r [95% CI] | I² | Q_b(df), p |
|-------|-------|---|---|------------|----|-----------|
| O | non-Asia | 7 | 2062 | 0.053 [-0.060, 0.164] | 65.9% | 0.26(1), p=0.6117 |
| O | Asia | 2 | 1301 | 0.164 [-0.989, 0.994] | 96.0% | 0.26(1), p=0.6117 |
| C | non-Asia | 8 | 2083 | 0.185 [0.082, 0.284] | 70.1% | 2.68(1), p=0.1016 |
| C | Asia | 2 | 1301 | 0.111 [-0.039, 0.257] | 0.0% | 2.68(1), p=0.1016 |
| E | non-Asia | 7 | 2062 | 0.050 [-0.004, 0.104] | 0.0% | 46.43(1), p=0.0000 |
| E | Asia | 2 | 1301 | -0.131 [-0.314, 0.061] | 0.0% | 46.43(1), p=0.0000 |
| A | non-Asia | 7 | 2062 | 0.030 [-0.047, 0.106] | 47.0% | 2.17(1), p=0.1403 |
| A | Asia | 2 | 1301 | 0.330 [-0.981, 0.995] | 95.6% | 2.17(1), p=0.1403 |
| N | non-Asia | 8 | 2083 | -0.007 [-0.130, 0.117] | 81.4% | 3.31(1), p=0.0687 |
| N | Asia | 2 | 1301 | 0.089 [0.008, 0.169] | 0.0% | 3.31(1), p=0.0687 |

### Moderator: era

| Trait | Level | k | N | r [95% CI] | I² | Q_b(df), p |
|-------|-------|---|---|------------|----|-----------|
| O | pre-COVID | 3 | 806 | 0.098 [-0.423, 0.570] | 84.2% | 0.00(1), p=0.9936 |
| O | COVID | 5 | 2275 | 0.097 [-0.123, 0.307] | 94.1% | 0.00(1), p=0.9936 |
| O | mixed | 1 | — | (k<2) | — | — |
| C | pre-COVID | 4 | 827 | 0.208 [-0.041, 0.434] | 71.7% | 0.13(1), p=0.7157 |
| C | COVID | 5 | 2275 | 0.179 [0.095, 0.260] | 53.7% | 0.13(1), p=0.7157 |
| C | mixed | 1 | — | (k<2) | — | — |
| E | pre-COVID | 3 | 806 | 0.033 [0.002, 0.063] | 0.0% | 0.59(1), p=0.4435 |
| E | COVID | 5 | 2275 | -0.014 [-0.180, 0.153] | 86.3% | 0.59(1), p=0.4435 |
| E | mixed | 1 | — | (k<2) | — | — |
| A | pre-COVID | 3 | 806 | 0.033 [-0.162, 0.225] | 38.8% | 1.09(1), p=0.2963 |
| A | COVID | 5 | 2275 | 0.153 [-0.141, 0.422] | 97.4% | 1.09(1), p=0.2963 |
| A | mixed | 1 | — | (k<2) | — | — |
| N | pre-COVID | 4 | 827 | -0.050 [-0.173, 0.074] | 19.6% | 2.04(1), p=0.1533 |
| N | COVID | 5 | 2275 | 0.060 [-0.124, 0.240] | 86.0% | 2.04(1), p=0.1533 |
| N | mixed | 1 | — | (k<2) | — | — |

### Moderator: outcome_type

| Trait | Level | k | N | r [95% CI] | I² | Q_b(df), p |
|-------|-------|---|---|------------|----|-----------|
| O | objective | 7 | 2649 | 0.092 [-0.086, 0.265] | 93.4% | 0.04(1), p=0.8395 |
| O | self-report | 2 | 714 | 0.072 [-0.637, 0.715] | 66.1% | 0.04(1), p=0.8395 |
| C | objective | 8 | 2670 | 0.147 [0.048, 0.243] | 62.3% | 2.40(1), p=0.1212 |
| C | self-report | 2 | 714 | 0.226 [-0.167, 0.556] | 0.0% | 2.40(1), p=0.1212 |
| E | objective | 7 | 2649 | -0.038 [-0.115, 0.039] | 60.8% | 17.30(1), p=0.0000 |
| E | self-report | 2 | 714 | 0.117 [-0.136, 0.356] | 0.0% | 17.30(1), p=0.0000 |
| A | objective | 7 | 2649 | 0.131 [-0.054, 0.308] | 96.5% | 0.49(1), p=0.4853 |
| A | self-report | 2 | 714 | 0.041 [-0.862, 0.881] | 87.0% | 0.49(1), p=0.4853 |
| N | objective | 8 | 2670 | -0.009 [-0.097, 0.080] | 66.6% | 0.82(1), p=0.3638 |
| N | self-report | 2 | 714 | 0.120 [-0.926, 0.954] | 92.4% | 0.82(1), p=0.3638 |

## 9. Significant moderator effects (p < .05)

1. **Extraversion × Region** — Q_between(1) = 46.43, p < .001
   - non-Asia (k=7): r = 0.050 [-0.004, 0.104]
   - Asia (k=2): r = -0.131 [-0.314, 0.061]

2. **Extraversion × Outcome Type** — Q_between(1) = 17.30, p < .001
   - Objective (k=7): r = -0.038 [-0.115, 0.039]
   - Self-report (k=2): r = +0.117 [-0.136, 0.356]

**No other moderator reached p < .05.**

## 10. Sensitivity analyses

### Exclude Author's own study (A-25 Tokiwa, COI)

| Trait | k | r [95% CI] | Δr |
|-------|---|------------|-----|
| O | 9 | 0.086 [-0.044, 0.214] | ±0.000 |
| C | 10 | 0.167 [0.089, 0.243] | ±0.000 |
| E | 9 | 0.002 [-0.076, 0.080] | ±0.000 |
| A | 9 | 0.112 [-0.031, 0.250] | ±0.000 |
| N | 10 | 0.018 [-0.079, 0.114] | ±0.000 |
**Reason for null Δ**: Tokiwa A-25 did not contribute correlations to primary pool.

### Exclude β-converted (Peterson-Brown approximation)

| Trait | k | r [95% CI] | Δr vs primary |
|-------|---|------------|---------------|
| O | 6 | 0.031 [-0.117, 0.178] | -0.055 🔴 |
| C | 7 | 0.203 [0.105, 0.298] | +0.036 |
| E | 6 | 0.018 [-0.087, 0.122] | +0.016 |
| A | 6 | 0.067 [-0.015, 0.148] | -0.045 |
| N | 7 | -0.043 [-0.126, 0.040] | -0.061 🔴 |

### Exclude RoB < 5 (low-quality)

| Trait | k | r [95% CI] | Δr |
|-------|---|------------|----|
| O | 8 | 0.097 [-0.052, 0.242] | +0.011 |
| C | 9 | 0.182 [0.111, 0.252] | +0.015 |
| E | 8 | 0.003 [-0.088, 0.093] | +0.001 |
| A | 8 | 0.126 [-0.036, 0.281] | +0.014 |
| N | 9 | 0.019 [-0.093, 0.130] | +0.001 |

## 11. Publication bias

### Egger's regression test

| Trait | k | Intercept | SE | t | p |
|-------|---|-----------|-----|---|---|
| O | 9 | -6.407 | 2.631 | -2.43 | **0.045** 🔴 |
| C | 10 | 2.143 | 1.128 | 1.90 | 0.094 |
| E | 9 | 2.291 | 1.849 | 1.24 | 0.255 |
| A | 9 | -8.527 | 4.029 | -2.12 | 0.072 |
| N | 10 | -1.790 | 1.631 | -1.10 | 0.304 |

### Duval & Tweedie trim-and-fill

| Trait | k_orig | k_imputed | Side | r_orig | r_adj [95% CI] |
|-------|--------|-----------|------|--------|-----------------|
| O | 9 | 1 | right | 0.086 | 0.107 [-0.017, 0.228] |
| C | 10 | 0 | left | 0.167 | 0.167 [0.089, 0.243] |
| E | 9 | 2 | left | 0.002 | -0.036 [-0.117, 0.046] |
| A | 9 | 0 | right | 0.112 | 0.112 [-0.031, 0.250] |
| N | 10 | 0 | right | 0.018 | 0.018 [-0.079, 0.114] |

## 12. Authoritative author / citation index

Each study's authoritative citation (from `deep_reading_notes.md`):

| ID | Authors | Year | Country | N_analytic | Region | Era |
|----|---------|------|---------|------------|--------|-----|
| A-01 | Abe | 2020 | US | 92 | North_America | pre-COVID |
| A-02 | Alkis | 2018 | Turkey | 189 | Europe | pre-COVID |
| A-15 | Elvers | 2003 | US | 47 | North_America | pre-COVID |
| A-17 | Kara | 2024 | Turkey | 437 | Europe | post-COVID |
| A-22 | Quigley | 2022 | UK | 301 | Europe | COVID |
| A-23 | Rodrigues | 2024 | Germany | 260 | Europe | COVID |
| A-25 | Tokiwa | 2025 | Japan | 103 | Asia | post-COVID |
| A-26 | Wang | 2023 | China | 1625 | Asia | post-COVID |
| A-28 | Yu | 2021 | China | 1152 | Asia | COVID |
| A-29 | Bahcekapili | 2020 | Turkey | 525 | Europe | pre-COVID |
| A-30 | Kaspar | 2023 | Germany | 413 | Europe | COVID |
| A-31 | Rivers | 2021 | Japan | 149 | Asia | COVID |
| A-37 | Zheng | 2023 | US | 282 | North_America | Mixed_3era |

---

**Use this file as the authoritative reference. If anything in the manuscript contradicts the values above, the manuscript should be corrected, not this file.**
