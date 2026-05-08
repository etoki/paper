# Canonical Numbers — HEXACO Harassment Microsimulation v2.0

**Generated**: from `output/supplementary/*.h5` after `make reproduce` (seed 20260429)  
**Pre-registration**: OSF DOI 10.17605/OSF.IO/3Y54U  
**Purpose**: Single source of truth for every numerical claim in `simulation/paper/`. 
Use this file as the reference for hallucination/integrity checks instead of re-running the pipeline.

**How to update**: 
```
cd simulation
make reproduce                                    # regenerate stage0..8 H5
uv run python -m code.build_canonical_numbers     # rewrite this file
```

---

## Stage 0 — 14-cell propensity (Results §3.1, Tables 1/6)

- N total: 354
- Cells: 14 (7 HEXACO clusters × 2 genders, gender 0=female, 1=male)
- Cell N range: 10–70 (median 18)
- Power-harassment propensity range: 0.0000–0.4000 (mean across cells = 0.1807)
- Degenerate cells (N=0): 0; Degenerate cells (X∈{0,N}): 1 (idx [11])
- CI cascade resolution: BCa = 13, Clopper-Pearson = 1, BC = 0, Percentile = 0

| cell_idx | cluster | gender | N | X (power) | p̂ (power) | CI method |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 0 | female | 10 | 4 | 0.4000 | bca |
| 1 | 0 | male | 13 | 5 | 0.3846 | bca |
| 2 | 1 | female | 10 | 2 | 0.2000 | bca |
| 3 | 1 | male | 32 | 2 | 0.0625 | bca |
| 4 | 2 | female | 15 | 3 | 0.2000 | bca |
| 5 | 2 | male | 34 | 5 | 0.1471 | bca |
| 6 | 3 | female | 28 | 2 | 0.0714 | bca |
| 7 | 3 | male | 21 | 2 | 0.0952 | bca |
| 8 | 4 | female | 13 | 3 | 0.2308 | bca |
| 9 | 4 | male | 38 | 9 | 0.2368 | bca |
| 10 | 5 | female | 14 | 1 | 0.0714 | bca |
| 11 | 5 | male | 12 | 0 | 0.0000 | clopper_pearson |
| 12 | 6 | female | 44 | 12 | 0.2727 | bca |
| 13 | 6 | male | 70 | 11 | 0.1571 | bca |

## Stage 1 — National prevalence (Results §3.2)

- P̂ baseline (Stage 2 bootstrap mean): **0.1745** (used internally; differs from
  the headline value below by 0.0001 due to bootstrap-replicate noise — Stage 2
  uses B = 10,000 per Methods Clarification m3, while Stage 7 uses B = 2,000
  for counterfactual contrasts; the paper consistently reports the Stage 7
  value 0.1744 as the headline P̂ baseline)
- MIC Labor Force Survey 2022 marginals: F = 0.4498, M = 0.5502 (total 6,723万人)
- Cluster proportions M3-fixed at Tokiwa (2026, IEEE Access) values

## Stage 2 — H1 MAPE (Results §3.2 Table 1, Table 6 H1 row)

| Period | Observed | MAPE | 95% CI lo | 95% CI hi | Tier |
|---|---:|---:|---:|---:|---|
| FY2016 | 0.325 | 46.34% | 32.36% | 59.30% | PARTIAL SUCCESS |
| FY2020 | 0.314 | 44.46% | 30.00% | 57.88% | PARTIAL SUCCESS |
| FY2023 | 0.193 | 9.65% | 0.52% | 31.52% | Standard SUCCESS |

- B = 10,000 (headline national MAPE bootstrap, Methods Clarification m3)
- B = 2,000 (per-cell bootstrap)

## Stage 4 — B0–B4 baselines (Results §3.4 Table 3)

| Baseline | MAPE | 95% CI |
|---|---:|---|
| B0 (uniform = MHLW grand mean) | 0.00% | [0.00, 0.00] |
| B1 (gender-only logistic) | 46.02% | [46.02, 46.02] |
| B2 (HEXACO 6-domain logistic) | 43.98% | [40.33, 47.35] |
| B3 (14-cell conditional, main pipeline) | 46.34% | [46.34, 46.34] |
| B4 (extended: age + age × cluster) | 47.04% | [45.02, 49.05] |

- Bonferroni-Holm decisions (B0-B1, B1-B2, B2-B3, B3-B4): [0, 0, 0, 0] → **0/4 confirmed**
- BH p-values: [1.0, 0.1145, 0.916, 0.7795]
- Page's L statistic: 36.00, p = 0.9757
- H2 decision: ambiguous_or_reversed → **REJECTED**

## Stage 5 — CMV diagnostic (Results §3.5)

- N total: 354; N used after listwise deletion: 353
- 11 standardized variables (6 HEXACO + 3 Dark Triad + 2 harassment)
- Harman first-factor variance: **24.08%** (threshold 50%; from stage5 print log)
- CMV concern flag: False

Marker-variable correction (Lindell & Whitney 2001) using HEXACO O as theoretical marker:
- r(O, power harassment) = +0.068 (CMV estimate)
- r(O, gender harassment) = −0.181

| Variable | r_raw (power) | r_adjusted (power) |
|---|---:|---:|
| H | −0.265 | −0.356 |
| E | −0.037 | −0.112 |
| X | +0.030 | −0.041 |
| A | −0.230 | −0.319 |
| C | −0.100 | −0.180 |
| Machiavellianism | +0.072 | +0.005 |
| Narcissism | +0.154 | +0.093 |
| Psychopathy | +0.391 | +0.347 |

## Stage 7 — Counterfactual ΔP_x and H7 IUT (Results §3.6 Table 4)

- P̂ baseline: **0.1744** (17.44%) — headline value used in the paper Abstract,
  Results §3.2, and all downstream tables. Differs from Stage 2 internal
  bootstrap mean (0.1745) by 0.0001 due to bootstrap-replicate noise (B
  differs across stages per Methods Clarification m3); the headline 0.1744 is
  the value the paper consistently reports and the value embedded in
  Tables 1, 4, 5 and Figure 3.

| Counterfactual | Operationalization | ΔP point | 95% CI |
|---|---|---:|---|
| A (universal) | do(HH+0.3σ; A+0.3σ; E+0.3σ) for all individuals | -0.0061 | [-0.0207, +0.0062] |
| B (targeted) | do(HH+0.40σ) for individuals with baseline cluster ∈ {0, 4, 6} | -0.0059 | [-0.0207, +0.0066] |
| C (structural) | p_c × 0.80 for all 14 cells | +0.0349 | [+0.0264, +0.0435] |

- H7 IUT lower bounds: L_BA = -0.0011, L_BC = -0.0544
- H7 classification: **REVERSAL** (m7 priority-1: point ΔP_B < point ΔP_C; consistent with L_BC < 0)
- |ΔP_C| / |ΔP_A| = 5.70

Positivity (Methods Clarification m5):
- ρ_B per cell: [1.0, 1.0, 0.091, 0.03, 0.0, 0.029, 0.034, 0.125, 1.0, 1.0, 0.067, 0.143, 1.0, 1.0]
- B flagged_weight (population-weighted share of cells with ρ_B < 0.10): 44.5% → m5 downgrade triggered (≥ 20%)
- A: ρ ≡ 1 (universal intervention; no extrapolation)
- C: ρ ≡ 1 (cell-level rate adjustment; no individual reassignment)

## Stage 8 — Transportability (Results §3.7 Table 5)

| F | Anchor | ΔP_A | ΔP_B | ΔP_C | L_BA | L_BC | H7 |
|---:|---|---:|---:|---:|---:|---:|---|
| 0.3 | Conservative cross-cultural worst case | -0.0018 | -0.0018 | +0.0105 | -0.0003 | -0.0163 | REVERSAL |
| 0.5 | Nielsen et al. 2017 Asia/Oceania | -0.0031 | -0.0029 | +0.0174 | -0.0006 | -0.0272 | REVERSAL |
| 0.7 | Mild attenuation | -0.0043 | -0.0041 | +0.0244 | -0.0008 | -0.0380 | REVERSAL |
| 1.0 | Reference (no attenuation) | -0.0061 | -0.0059 | +0.0349 | -0.0011 | -0.0544 | REVERSAL |

## H5 / H6 cluster-rank correlations (Results §3.8 Table 6)

**H5 — Gender invariance** (cluster-level rank concordance across genders, 7-cluster):
- Spearman ρ(female, male) = **0.8729** (p = 0.0103) → **CONFIRMED**

**H6 — Cross-domain triangulation** (cluster-level rank concordance between power and gender harassment):
- Spearman ρ(power, gender) at 7-cluster: **0.0357** (p = 0.9394)
- Spearman ρ(power, gender) at 14-cell: **0.1444** (p = 0.6223)
- → **REJECTED**

## External validation targets (verified against primary sources)

MHLW Workplace Harassment Survey past-3-year power-harassment victimization rates:
- FY2016 (H28): **32.5%** (https://www.mhlw.go.jp/file/06-Seisakujouhou-11200000-Roudoukijunkyoku/0000165751.pdf)
- FY2020 (R2):  **31.4%**
- FY2023 (R5):  **19.3%** (https://www.mhlw.go.jp/content/11909000/001259093.pdf)
- FY2016 → FY2023: **−13.2 pp** (−40.6% relative)
- FY2020 → FY2023: −12.1 pp

Statistics Bureau (MIC) Labor Force Survey 2022 Annual Average:
- Total employed: **6,723 万人**
- Female: **3,024 万人** (44.98%)
- Male: **3,699 万人** (55.02%)

Power Harassment Prevention Law (改正労働施策総合推進法):
- Large enterprises: **2020-06-01** enforcement
- SMEs: **2022-04-01** enforcement

## Hypothesis outcomes summary (Table 6)

| Hypothesis | Outcome | Tier |
|---|---|---|
| H1 (latent prevalence) | MAPE_FY2016 = 46.34% | PARTIAL SUCCESS |
| H2 (B0–B4 ordinal trend) | 0/4 BH pairs confirmed; Page's L p = 0.9757 | REJECTED |
| H3 (centroid concordance) | Externally validated in Tokiwa (2026, IEEE Access) on the N=13,668 source sample; not separately re-tested in the present N=354 pipeline | EXTERNALLY CONFIRMED |
| H4 (CI cascade stability) | 13/14 BCa + 1 Clopper-Pearson per M4 | CONFIRMED |
| H5 (gender invariance) | Spearman ρ = 0.87 (p = 0.010) | CONFIRMED |
| H6 (cross-domain triangulation) | Spearman ρ = 0.04 (cluster), 0.14 (cell) | REJECTED |
| H7 (counterfactual ordering) | REVERSAL via m7 priority-1; robust F=0.3..1.0 | REVERSAL |

## Headline values used in abstract / cover letter

- ΔP_A = -0.0061 (-0.61 pp), 95% CI [-0.0207, +0.0062]
- ΔP_B = -0.0059 (-0.59 pp), 95% CI [-0.0207, +0.0066]
- ΔP_C = +0.0349 (+3.49 pp), 95% CI [+0.0264, +0.0435]
- |ΔP_C| / |ΔP_A| ≈ **5.7×** (rounded to 1 decimal)
- H7 = REVERSAL (4/4 transportability factors)
- P̂ baseline = 17.44%

---

*End of canonical numbers file.*
