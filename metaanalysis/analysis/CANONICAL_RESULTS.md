# Canonical Results Snapshot — Big Five × Online Learning Meta-Analysis

**Generated**: 2026-05-01; sections 6–11 regenerated 2026-10-27 by `update_canonical_results.py`
**Purpose**: Single source of truth for hallucination checking. Values below are AUTHORITATIVE.
**Convention**: When manuscript text disagrees with values here, the manuscript is wrong.

---

## 1. Study counts (from `data_extraction_populated.csv`)

| Category | n | Study IDs |
|----------|---|-----------|
| **Total documented in Table 1 (PRISMA flow tracked)** | **31** | A-01 to A-37 (gaps: A-14 in PRISMA only, A-32–A-36 unused) |
| Retained in qualitative synthesis (`include*` + `exclude_from_primary`) | 25 | All except A-05, A-09, A-10, A-16, A-24, A-27 |
| **Primary quantitative pool (achievement r)** | **9** | A-01, A-02, A-15, A-22, A-23, A-28, A-29, A-30, A-31 (A-37 left out on 2026-10-27; see pool.POOL_EXCLUSIONS) |
| ↳ `include` (clean primary) | 6 | A-01 Abe, A-02 Alkis, A-23 Rodrigues, A-28 Yu, A-29 Bahcekapili, A-31 Rivers |
| ↳ `include_with_caveat` | 5 | A-15 Elvers, A-22 Quigley, A-26 Wang, A-30 Kaspar, A-37 Zheng |
| ↳ `include_COI` | 1 | A-25 Tokiwa (author's own; sensitivity excludes) |
| ↳ `include_secondary` | 1 | A-17 Kara (engagement only) |
| Secondary-outcome only (`exclude_from_primary`) | 12 | A-03, A-04, A-06, A-07, A-08, A-11, A-12, A-13, A-18, A-19, A-20, A-21 |
| `exclude` — face-to-face | 3 | A-09, A-10, A-16 (+ A-14 documented in PRISMA flow only, not in CSV) |
| `exclude` — no extractable r | 1 | A-24 Tlili |
| `exclude_overlap` | 1 | A-05 Audet 2023 (overlaps with A-04 Audet 2021; A-04 retained) |
| `PDF_unavailable` | 1 | A-27 Wu_Yu (PDF could not be obtained) |

**Numerical reconciliation**:
- **31** = total cataloged in Table 1, including studies excluded post-eligibility (with reasons disclosed per PRISMA 2020). This is the figure the manuscript reports.
- **25** = retained in systematic review (qualitative synthesis): 13 in `include*` + 12 in `exclude_from_primary` (secondary-outcome-only studies).
- **9** = primary quantitative pool. **k = 9 for Conscientiousness and Neuroticism** (A-15 Elvers reports only C and N); **k = 8 for Openness, Extraversion, Agreeableness** (A-15 dropped from those traits). A-37 Zheng (face-to-face and online cohorts pooled; |r| < .10 entered as 0) stays in the scoping map but is out of the pool; the include-A-37 sensitivity analysis restores it.

**The meta-analytic findings depend ONLY on the 10-study primary pool, which is unambiguously defined and exactly reproduces previous analyses on re-run. The meta-analysis is valid and reproducible.**

**⚠ Narrative-clarity caveat for the manuscript**: The wording "31 primary studies retained for the systematic review" is imprecise — 31 is the cataloged total (including exclusion records), while 25 is the count truly retained for qualitative synthesis and 10 contributed to the quantitative pool. Reviewers focused on PRISMA reporting may flag this. Recommend rewording to "31 primary studies cataloged at full-text assessment, of which 25 were retained for qualitative synthesis and 10 contributed direct or β-converted Pearson correlations to the primary achievement pool."

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

τ² is the exact REML estimate (matches PyMARE); β is converted with Peterson & Brown's r = .98β + .05λ; studies in pool.POOL_EXCLUSIONS (A-37) are left out of the pool.

| Trait | k | N_total | r_pooled | 95% CI | 95% PI | I² | τ² | Q(df), p |
|-------|---|---------|----------|--------|--------|----|----|---------|
| O | 8 | 3081 | 0.096 | [-0.052, 0.239] | [-0.319, 0.480] | 92.2% | 0.0265 | 89.58(7), p=0.0000 |
| C | 9 | 3102 | 0.185 | [0.112, 0.255] | [0.007, 0.351] | 59.0% | 0.0047 | 19.49(8), p=0.0124 |
| E | 8 | 3081 | 0.012 | [-0.070, 0.093] | [-0.191, 0.213] | 68.0% | 0.0058 | 21.90(7), p=0.0026 |
| A | 8 | 3081 | 0.130 | [-0.022, 0.277] | [-0.327, 0.538] | 96.0% | 0.0324 | 176.86(7), p=0.0000 |
| N | 9 | 3102 | 0.017 | [-0.095, 0.128] | [-0.275, 0.305] | 80.7% | 0.0138 | 41.45(8), p=0.0000 |

## 7. Contributing studies per trait pool

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

## 8. Moderator analyses

Test: Knapp–Hartung mixed-effects meta-regression over the levels with k ≥ 2 (REML residual τ²), F(df1, df2); p_Holm is Holm-corrected within each trait across the three moderators.

### Moderator: region

| Trait | Level | k | N | r [95% CI] | I² | F(df1, df2), p, p_Holm |
|-------|-------|---|---|------------|----|------------------------|
| O | non-Asia | 6 | 1780 | 0.066 [-0.082, 0.211] | 69.9% | 0.43(1, 6), p=0.5360, p_Holm=1.0000 |
| O | Asia | 2 | 1301 | 0.155 [-0.988, 0.994] | 95.8% | 0.43(1, 6), p=0.5360, p_Holm=1.0000 |
| C | non-Asia | 7 | 1801 | 0.211 [0.118, 0.300] | 59.0% | 1.83(1, 7), p=0.2187, p_Holm=0.6560 |
| C | Asia | 2 | 1301 | 0.110 [-0.045, 0.260] | 0.0% | 1.83(1, 7), p=0.2187, p_Holm=0.6560 |
| E | non-Asia | 6 | 1780 | 0.058 [-0.004, 0.120] | 2.0% | 12.96(1, 6), p=0.0114, p_Holm=0.0341 |
| E | Asia | 2 | 1301 | -0.095 [-0.541, 0.393] | 23.0% | 12.96(1, 6), p=0.0114, p_Holm=0.0341 |
| A | non-Asia | 6 | 1780 | 0.047 [-0.036, 0.128] | 40.6% | 6.98(1, 6), p=0.0384, p_Holm=0.1152 |
| A | Asia | 2 | 1301 | 0.319 [-0.979, 0.994] | 95.4% | 6.98(1, 6), p=0.0384, p_Holm=0.1152 |
| N | non-Asia | 7 | 1801 | -0.013 [-0.163, 0.138] | 83.7% | 0.87(1, 7), p=0.3812, p_Holm=0.6925 |
| N | Asia | 2 | 1301 | 0.089 [0.005, 0.171] | 0.0% | 0.87(1, 7), p=0.3812, p_Holm=0.6925 |

### Moderator: era

| Trait | Level | k | N | r [95% CI] | I² | F(df1, df2), p, p_Holm |
|-------|-------|---|---|------------|----|------------------------|
| O | pre-COVID | 3 | 806 | 0.104 [-0.427, 0.581] | 84.2% | 0.00(1, 6), p=0.9542, p_Holm=1.0000 |
| O | COVID | 5 | 2275 | 0.094 [-0.122, 0.300] | 93.9% | 0.00(1, 6), p=0.9542, p_Holm=1.0000 |
| C | pre-COVID | 4 | 827 | 0.221 [-0.033, 0.448] | 71.7% | 0.02(1, 7), p=0.8895, p_Holm=0.8895 |
| C | COVID | 5 | 2275 | 0.180 [0.097, 0.261] | 53.9% | 0.02(1, 7), p=0.8895, p_Holm=0.8895 |
| E | pre-COVID | 3 | 806 | 0.033 [0.002, 0.063] | 0.0% | 0.22(1, 6), p=0.6586, p_Holm=0.6586 |
| E | COVID | 5 | 2275 | -0.002 [-0.155, 0.152] | 80.4% | 0.22(1, 6), p=0.6586, p_Holm=0.6586 |
| A | pre-COVID | 3 | 806 | 0.052 [-0.159, 0.259] | 38.8% | 0.39(1, 6), p=0.5542, p_Holm=1.0000 |
| A | COVID | 5 | 2275 | 0.160 [-0.115, 0.412] | 97.0% | 0.39(1, 6), p=0.5542, p_Holm=1.0000 |
| N | pre-COVID | 4 | 827 | -0.050 [-0.173, 0.074] | 19.6% | 1.38(1, 7), p=0.2786, p_Holm=0.6925 |
| N | COVID | 5 | 2275 | 0.058 [-0.124, 0.237] | 85.7% | 1.38(1, 7), p=0.2786, p_Holm=0.6925 |

### Moderator: outcome_type

| Trait | Level | k | N | r [95% CI] | I² | F(df1, df2), p, p_Holm |
|-------|-------|---|---|------------|----|------------------------|
| O | objective | 6 | 2367 | 0.106 [-0.109, 0.312] | 93.6% | 0.07(1, 6), p=0.7968, p_Holm=1.0000 |
| O | self-report | 2 | 714 | 0.068 [-0.636, 0.710] | 65.2% | 0.07(1, 6), p=0.7968, p_Holm=1.0000 |
| C | objective | 7 | 2388 | 0.172 [0.072, 0.268] | 57.3% | 0.72(1, 7), p=0.4252, p_Holm=0.8503 |
| C | self-report | 2 | 714 | 0.224 [-0.187, 0.568] | 0.0% | 0.72(1, 7), p=0.4252, p_Holm=0.8503 |
| E | objective | 6 | 2367 | -0.031 [-0.108, 0.047] | 40.5% | 7.96(1, 6), p=0.0303, p_Holm=0.0605 |
| E | self-report | 2 | 714 | 0.116 [-0.143, 0.361] | 0.0% | 7.96(1, 6), p=0.0303, p_Holm=0.0605 |
| A | objective | 6 | 2367 | 0.151 [-0.064, 0.352] | 96.6% | 0.28(1, 6), p=0.6155, p_Holm=1.0000 |
| A | self-report | 2 | 714 | 0.068 [-0.742, 0.797] | 77.6% | 0.28(1, 6), p=0.6155, p_Holm=1.0000 |
| N | objective | 7 | 2388 | -0.014 [-0.124, 0.097] | 71.1% | 1.72(1, 7), p=0.2308, p_Holm=0.6925 |
| N | self-report | 2 | 714 | 0.117 [-0.922, 0.951] | 92.1% | 1.72(1, 7), p=0.2308, p_Holm=0.6925 |

## 9. Significant moderator effects (p < .05, uncorrected)

1. **Extraversion × region** — F(1, 6) = 12.96, p = 0.011, Holm p = 0.034
   - non-Asia (k=6): r = 0.058 [-0.004, 0.120]
   - Asia (k=2): r = -0.095 [-0.541, 0.393]

2. **Agreeableness × region** — F(1, 6) = 6.98, p = 0.038, Holm p = 0.115
   - non-Asia (k=6): r = 0.047 [-0.036, 0.128]
   - Asia (k=2): r = 0.319 [-0.979, 0.994]

3. **Extraversion × outcome_type** — F(1, 6) = 7.96, p = 0.030, Holm p = 0.061
   - objective (k=6): r = -0.031 [-0.108, 0.047]
   - self-report (k=2): r = 0.116 [-0.143, 0.361]

Every contrast has k = 2 in its smaller subgroup; treat them as exploratory.

## 10. Sensitivity analyses

### Exclude Author's own study (A-25 Tokiwa, COI)

| Trait | k | r [95% CI] | Δr vs primary |
|-------|---|-----------|---------------|
| O | 8 | 0.096 [-0.052, 0.239] | +0.000 |
| C | 9 | 0.185 [0.112, 0.255] | +0.000 |
| E | 8 | 0.012 [-0.070, 0.093] | +0.000 |
| A | 8 | 0.130 [-0.022, 0.277] | +0.000 |
| N | 9 | 0.017 [-0.095, 0.128] | +0.000 |

### Exclude β-converted (A-28 Yu, A-30 Kaspar)

| Trait | k | r [95% CI] | Δr vs primary |
|-------|---|-----------|---------------|
| O | 6 | 0.034 [-0.121, 0.188] | -0.062 🔴 |
| C | 7 | 0.206 [0.107, 0.301] | +0.022 |
| E | 6 | 0.016 [-0.090, 0.122] | +0.004 |
| A | 6 | 0.069 [-0.012, 0.150] | -0.061 🔴 |
| N | 7 | -0.040 [-0.129, 0.049] | -0.057 🔴 |

### Exclude RoB < 5

| Trait | k | r [95% CI] | Δr vs primary |
|-------|---|-----------|---------------|
| O | 8 | 0.096 [-0.052, 0.239] | +0.000 |
| C | 9 | 0.185 [0.112, 0.255] | +0.000 |
| E | 8 | 0.012 [-0.070, 0.093] | +0.000 |
| A | 8 | 0.130 [-0.022, 0.277] | +0.000 |
| N | 9 | 0.017 [-0.095, 0.128] | +0.000 |

### Exclude analytic N < 50 (A-15 Elvers)

| Trait | k | r [95% CI] | Δr vs primary |
|-------|---|-----------|---------------|
| O | 8 | 0.096 [-0.052, 0.239] | +0.000 |
| C | 8 | 0.180 [0.105, 0.253] | -0.005 |
| E | 8 | 0.012 [-0.070, 0.093] | +0.000 |
| A | 8 | 0.130 [-0.022, 0.277] | +0.000 |
| N | 8 | 0.030 [-0.074, 0.134] | +0.014 |

### DerSimonian–Laird τ² instead of REML

| Trait | k | r [95% CI] | Δr vs primary |
|-------|---|-----------|---------------|
| O | 8 | 0.096 [-0.052, 0.239] | -0.000 |
| C | 9 | 0.184 [0.112, 0.255] | -0.000 |
| E | 8 | 0.012 [-0.070, 0.093] | +0.000 |
| A | 8 | 0.129 [-0.021, 0.273] | -0.001 |
| N | 9 | 0.017 [-0.095, 0.128] | +0.000 |

### Include A-37 Zheng (|r| < .10 entered as r = 0)

| Trait | k | r [95% CI] | Δr vs primary |
|-------|---|-----------|---------------|
| O | 9 | 0.085 [-0.044, 0.211] | -0.011 |
| C | 10 | 0.168 [0.090, 0.245] | -0.016 |
| E | 9 | 0.010 [-0.059, 0.080] | -0.001 |
| A | 9 | 0.116 [-0.020, 0.248] | -0.014 |
| N | 10 | 0.016 [-0.080, 0.112] | -0.000 |

## 11. Publication bias

### Egger's regression test

| Trait | k | Intercept | SE | t | p |
|-------|---|-----------|-----|---|---|
| O | 8 | -6.006 | 2.794 | -2.15 | 0.075 |
| C | 9 | 2.460 | 0.872 | 2.82 | 0.026 🔴 |
| E | 8 | 1.307 | 1.759 | 0.74 | 0.485 |
| A | 8 | -7.926 | 4.101 | -1.93 | 0.101 |
| N | 9 | -1.741 | 1.738 | -1.00 | 0.350 |

### Duval & Tweedie trim-and-fill (L0; metafor::trimfill procedure)

| Trait | k_orig | k_imputed | Side | r_original | r_adjusted [95% CI] |
|-------|--------|-----------|------|-----------|---------------------|
| O | 8 | 0 | right | 0.096 | 0.096 [-0.052, 0.239] |
| C | 9 | 3 | left | 0.185 | 0.151 [0.069, 0.230] |
| E | 8 | 0 | right | 0.012 | 0.012 [-0.070, 0.093] |
| A | 8 | 3 | right | 0.130 | 0.216 [0.076, 0.347] |
| N | 9 | 1 | right | 0.017 | 0.030 [-0.087, 0.147] |

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
