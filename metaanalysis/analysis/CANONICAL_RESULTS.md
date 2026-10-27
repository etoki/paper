# Canonical Results Snapshot — Big Five × Online Learning Meta-Analysis

**Generated**: 2026-05-01; sections 6 and 8–11 regenerated 2026-10-27 after the switch to exact REML τ², Knapp–Hartung moderator tests and the corrected Peterson–Brown conversion (re-run of `pool.py`, `sensitivity.py`)
**Purpose**: Single source of truth for hallucination checking. Values below are AUTHORITATIVE.
**Convention**: When manuscript text disagrees with values here, the manuscript is wrong.

---

## 1. Study counts (from `data_extraction_populated.csv`)

| Category | n | Study IDs |
|----------|---|-----------|
| **Total documented in Table 1 (PRISMA flow tracked)** | **31** | A-01 to A-37 (gaps: A-14 in PRISMA only, A-32–A-36 unused) |
| Retained in qualitative synthesis (`include*` + `exclude_from_primary`) | 25 | All except A-05, A-09, A-10, A-16, A-24, A-27 |
| **Primary quantitative pool (achievement r)** | **10** | A-01, A-02, A-15, A-22, A-23, A-28, A-29, A-30, A-31, A-37 |
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
- **10** = primary quantitative pool. **k = 10 for Conscientiousness and Neuroticism** (A-15 Elvers reports only C and N); **k = 9 for Openness, Extraversion, Agreeableness** (A-15 dropped from those traits).

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

τ² is the exact REML estimate (restricted likelihood maximised directly; matches PyMARE), and β is converted with Peterson & Brown's r = .98β + .05λ. Values before 2026-10-27 used a fixed-point τ² update that under-estimated τ² and a β ± .05 conversion that pushed negative β a further .05 from zero.

| Trait | k | N_total | r_pooled | 95% CI | 95% PI | I² | τ² | Q(df), p |
|-------|---|---------|----------|--------|--------|----|----|---------|
| O | 9 | 3363 | 0.085 | [-0.044, 0.211] | [-0.293, 0.440] | 91.7% | 0.0237 | 96.80(8), p=0.0000 |
| C | 10 | 3384 | 0.168 | [0.090, 0.245] | [-0.039, 0.362] | 65.0% | 0.0070 | 25.74(9), p=0.0023 |
| E | 9 | 3363 | 0.010 | [-0.059, 0.080] | [-0.165, 0.185] | 63.5% | 0.0047 | 21.90(8), p=0.0051 |
| A | 9 | 3363 | 0.116 | [-0.020, 0.248] | [-0.309, 0.502] | 95.8% | 0.0301 | 189.85(8), p=0.0000 |
| N | 10 | 3384 | 0.016 | [-0.080, 0.112] | [-0.246, 0.276] | 78.5% | 0.0117 | 41.93(9), p=0.0000 |

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

Test: Knapp–Hartung mixed-effects meta-regression over the levels with k ≥ 2 (REML residual τ²), F(df1, df2); p_Holm is Holm-corrected within each trait across the three moderators. Before 2026-10-27 this section reported Q_between from HKSJ subgroup SEs in a χ² test, which overstated the evidence (E × Region Q = 46.43; E × Outcome type Q = 17.30).

### Moderator: region

| Trait | Level | k | N | r [95% CI] | I² | F(df1, df2), p, p_Holm |
|-------|-------|---|---|------------|----|------------------------|
| O | non-Asia | 7 | 2062 | 0.054 [-0.063, 0.170] | 65.7% | 0.69(1, 7), p=0.4350, p_Holm=1.0000 |
| O | Asia | 2 | 1301 | 0.155 [-0.988, 0.994] | 95.8% | 0.69(1, 7), p=0.4350, p_Holm=1.0000 |
| C | non-Asia | 8 | 2083 | 0.187 [0.083, 0.287] | 69.9% | 0.58(1, 8), p=0.4700, p_Holm=1.0000 |
| C | Asia | 2 | 1301 | 0.110 [-0.045, 0.260] | 0.0% | 0.58(1, 8), p=0.4700, p_Holm=1.0000 |
| E | non-Asia | 7 | 2062 | 0.050 [-0.004, 0.104] | 0.0% | 12.94(1, 7), p=0.0088, p_Holm=0.0263 |
| E | Asia | 2 | 1301 | -0.095 [-0.541, 0.393] | 23.0% | 12.94(1, 7), p=0.0088, p_Holm=0.0263 |
| A | non-Asia | 7 | 2062 | 0.038 [-0.029, 0.105] | 31.5% | 9.13(1, 7), p=0.0193, p_Holm=0.0580 |
| A | Asia | 2 | 1301 | 0.319 [-0.979, 0.994] | 95.4% | 9.13(1, 7), p=0.0193, p_Holm=0.0580 |
| N | non-Asia | 8 | 2083 | -0.009 [-0.132, 0.115] | 81.0% | 0.99(1, 8), p=0.3480, p_Holm=0.5819 |
| N | Asia | 2 | 1301 | 0.089 [0.005, 0.171] | 0.0% | 0.99(1, 8), p=0.3480, p_Holm=0.5819 |

### Moderator: era

| Trait | Level | k | N | r [95% CI] | I² | F(df1, df2), p, p_Holm |
|-------|-------|---|---|------------|----|------------------------|
| O | pre-COVID | 3 | 806 | 0.104 [-0.427, 0.581] | 84.2% | 0.00(1, 6), p=0.9542, p_Holm=1.0000 |
| O | COVID | 5 | 2275 | 0.094 [-0.122, 0.300] | 93.9% | 0.00(1, 6), p=0.9542, p_Holm=1.0000 |
| O | mixed | 1 | — | (k<2) | — | — |
| C | pre-COVID | 4 | 827 | 0.221 [-0.033, 0.448] | 71.7% | 0.02(1, 7), p=0.8895, p_Holm=1.0000 |
| C | COVID | 5 | 2275 | 0.180 [0.097, 0.261] | 53.9% | 0.02(1, 7), p=0.8895, p_Holm=1.0000 |
| C | mixed | 1 | — | (k<2) | — | — |
| E | pre-COVID | 3 | 806 | 0.033 [0.002, 0.063] | 0.0% | 0.22(1, 6), p=0.6586, p_Holm=0.6586 |
| E | COVID | 5 | 2275 | -0.002 [-0.155, 0.152] | 80.4% | 0.22(1, 6), p=0.6586, p_Holm=0.6586 |
| E | mixed | 1 | — | (k<2) | — | — |
| A | pre-COVID | 3 | 806 | 0.052 [-0.159, 0.259] | 38.8% | 0.39(1, 6), p=0.5542, p_Holm=1.0000 |
| A | COVID | 5 | 2275 | 0.160 [-0.115, 0.412] | 97.0% | 0.39(1, 6), p=0.5542, p_Holm=1.0000 |
| A | mixed | 1 | — | (k<2) | — | — |
| N | pre-COVID | 4 | 827 | -0.050 [-0.173, 0.074] | 19.6% | 1.38(1, 7), p=0.2786, p_Holm=0.5819 |
| N | COVID | 5 | 2275 | 0.058 [-0.124, 0.237] | 85.7% | 1.38(1, 7), p=0.2786, p_Holm=0.5819 |
| N | mixed | 1 | — | (k<2) | — | — |

### Moderator: outcome_type

| Trait | Level | k | N | r [95% CI] | I² | F(df1, df2), p, p_Holm |
|-------|-------|---|---|------------|----|------------------------|
| O | objective | 7 | 2649 | 0.091 [-0.086, 0.262] | 93.2% | 0.03(1, 7), p=0.8620, p_Holm=1.0000 |
| O | self-report | 2 | 714 | 0.068 [-0.636, 0.710] | 65.2% | 0.03(1, 7), p=0.8620, p_Holm=1.0000 |
| C | objective | 8 | 2670 | 0.151 [0.050, 0.250] | 62.4% | 1.00(1, 8), p=0.3468, p_Holm=1.0000 |
| C | self-report | 2 | 714 | 0.224 [-0.187, 0.568] | 0.0% | 1.00(1, 8), p=0.3468, p_Holm=1.0000 |
| E | objective | 7 | 2649 | -0.028 [-0.090, 0.035] | 31.7% | 9.01(1, 7), p=0.0199, p_Holm=0.0398 |
| E | self-report | 2 | 714 | 0.116 [-0.143, 0.361] | 0.0% | 9.01(1, 7), p=0.0199, p_Holm=0.0398 |
| A | objective | 7 | 2649 | 0.129 [-0.052, 0.302] | 96.4% | 0.17(1, 7), p=0.6948, p_Holm=1.0000 |
| A | self-report | 2 | 714 | 0.068 [-0.742, 0.797] | 77.6% | 0.17(1, 7), p=0.6948, p_Holm=1.0000 |
| N | objective | 8 | 2670 | -0.010 [-0.100, 0.079] | 66.4% | 2.01(1, 8), p=0.1940, p_Holm=0.5819 |
| N | self-report | 2 | 714 | 0.117 [-0.922, 0.951] | 92.1% | 2.01(1, 8), p=0.1940, p_Holm=0.5819 |

## 9. Significant moderator effects (p < .05, uncorrected)

1. **Extraversion × region** — F(1, 7) = 12.94, p = .009, Holm p = .026
   - non-Asia (k=7): r = 0.050 [-0.004, 0.104]
   - Asia (k=2): r = -0.095 [-0.541, 0.393]

2. **Agreeableness × region** — F(1, 7) = 9.13, p = .019, Holm p = .058
   - non-Asia (k=7): r = 0.038 [-0.029, 0.105]
   - Asia (k=2): r = 0.319 [-0.979, 0.994]

3. **Extraversion × outcome_type** — F(1, 7) = 9.01, p = .020, Holm p = .040
   - objective (k=7): r = -0.028 [-0.090, 0.035]
   - self-report (k=2): r = 0.116 [-0.143, 0.361]

All of these rest on k = 2 in the smaller subgroup (A-28 Yu and A-31 Rivers for Asia; A-22 Quigley and A-30 Kaspar for self-report), so they are exploratory.

## 10. Sensitivity analyses

### Exclude Author's own study (A-25 Tokiwa, COI)

| Trait | k | r [95% CI] | Δr vs primary |
|-------|---|-----------|---------------|
| O | 9 | 0.085 [-0.044, 0.211] | +0.000 |
| C | 10 | 0.168 [0.090, 0.245] | +0.000 |
| E | 9 | 0.010 [-0.059, 0.080] | +0.000 |
| A | 9 | 0.116 [-0.020, 0.248] | +0.000 |
| N | 10 | 0.016 [-0.080, 0.112] | +0.000 |

### Exclude β-converted (A-28 Yu, A-30 Kaspar)

| Trait | k | r [95% CI] | Δr vs primary |
|-------|---|-----------|---------------|
| O | 7 | 0.026 [-0.093, 0.145] | -0.059 🔴 |
| C | 8 | 0.181 [0.073, 0.284] | +0.012 |
| E | 7 | 0.015 [-0.069, 0.098] | +0.005 |
| A | 7 | 0.057 [-0.013, 0.126] | -0.059 🔴 |
| N | 8 | -0.035 [-0.107, 0.037] | -0.051 🔴 |

### Exclude RoB < 5 (low-quality; drops A-37)

| Trait | k | r [95% CI] | Δr vs primary |
|-------|---|-----------|---------------|
| O | 8 | 0.096 [-0.052, 0.239] | +0.011 |
| C | 9 | 0.185 [0.112, 0.255] | +0.016 |
| E | 8 | 0.012 [-0.070, 0.093] | +0.001 |
| A | 8 | 0.130 [-0.022, 0.277] | +0.014 |
| N | 9 | 0.017 [-0.095, 0.128] | +0.000 |

### Exclude analytic N < 50 (drops A-15 Elvers)

| Trait | k | r [95% CI] | Δr vs primary |
|-------|---|-----------|---------------|
| O | 9 | 0.085 [-0.044, 0.211] | +0.000 |
| C | 9 | 0.163 [0.082, 0.242] | -0.005 |
| E | 9 | 0.010 [-0.059, 0.080] | +0.000 |
| A | 9 | 0.116 [-0.020, 0.248] | +0.000 |
| N | 9 | 0.027 [-0.063, 0.117] | +0.011 |

### DerSimonian–Laird τ² instead of REML

| Trait | k | r [95% CI] | Δr vs primary |
|-------|---|-----------|---------------|
| O | 9 | 0.085 [-0.045, 0.212] | -0.000 |
| C | 10 | 0.167 [0.089, 0.243] | -0.001 |
| E | 9 | 0.010 [-0.059, 0.080] | +0.000 |
| A | 9 | 0.115 [-0.018, 0.244] | -0.001 |
| N | 10 | 0.016 [-0.081, 0.112] | -0.000 |

### Exclude A-37 Zheng (|r| < .10 entered as r = 0; not pre-registered)

| Trait | k | r [95% CI] | Δr vs primary |
|-------|---|-----------|---------------|
| O | 8 | 0.096 [-0.052, 0.239] | +0.011 |
| C | 9 | 0.185 [0.112, 0.255] | +0.016 |
| E | 8 | 0.012 [-0.070, 0.093] | +0.001 |
| A | 8 | 0.130 [-0.022, 0.277] | +0.014 |
| N | 9 | 0.017 [-0.095, 0.128] | +0.000 |

## 11. Publication bias

### Egger's regression test

| Trait | k | Intercept | SE | t | p |
|-------|---|-----------|-----|---|---|
| O | 9 | -6.270 | 2.596 | -2.42 | 0.046 🔴 |
| C | 10 | 2.159 | 1.123 | 1.92 | 0.091 |
| E | 9 | 1.270 | 1.602 | 0.79 | 0.454 |
| A | 9 | -8.285 | 3.799 | -2.18 | 0.066 |
| N | 10 | -1.780 | 1.614 | -1.10 | 0.302 |

### Duval & Tweedie trim-and-fill (L0; metafor::trimfill procedure)

| Trait | k_orig | k_imputed | Side | r_original | r_adjusted [95% CI] |
|-------|--------|-----------|------|-----------|---------------------|
| O | 9 | 1 | right | 0.085 | 0.106 [-0.017, 0.226] |
| C | 10 | 2 | left | 0.168 | 0.149 [0.064, 0.231] |
| E | 9 | 0 | right | 0.010 | 0.010 [-0.059, 0.080] |
| A | 9 | 3 | right | 0.116 | 0.180 [0.061, 0.294] |
| N | 10 | 1 | right | 0.016 | 0.027 [-0.075, 0.128] |

Before 2026-10-27 the trim-and-fill routine used argsort positions as ranks and re-ranked only the trimmed set; it reported 0 imputed for C and A.

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
