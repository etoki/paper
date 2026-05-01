# 06. Tables and Figures

## Tables

### Table 1. H1 Four-Tier Classification of Mean Absolute Percentage Error (MAPE) Across Three MHLW Validation Periods

*Note*. Predicted values are computed via 14-cell HEXACO 7-typology × gender propensity weighted by Statistics Bureau (MIC) Labor Force Survey 2022 marginals. Bootstrap 95% CIs use B = 10,000 cell-stratified resamples (Methods Clarification m3). Tier classification per pre-registered four-tier hierarchy (v2.0 Section 5.4).

| Period (MHLW) | Observed | Predicted | MAPE | 95% CI | Tier |
|---|---:|---:|---:|---|---|
| FY2016 (32.5%, primary) | 0.325 | 0.1744 | 46.34% | [32.36, 59.30] | PARTIAL SUCCESS |
| FY2020 (31.4%, secondary) | 0.314 | 0.1744 | 44.46% | [30.00, 57.88] | PARTIAL SUCCESS |
| FY2023 (19.3%, secondary) | 0.193 | 0.1744 | 9.65% | [0.52, 31.52] | Strict SUCCESS |

---

### Table 2. One-at-a-Time (OAT) Sensitivity Sweep Around the Locked v2.0 Main Configuration

*Note*. Each row varies the indicated parameter family while holding remaining parameters at the v2.0 main configuration. B = 2,000 percentile CIs reported on FY2016 MAPE.

| Family | Value | MAPE | 95% CI | Tier |
|---|---|---:|---|---|
| Baseline | main_v2.0 | 45.51% | [31.5, 59.5] | PARTIAL |
| Binarization threshold | mean+0.25 SD | 29.58% | [13.7, 45.2] | Standard SUCCESS |
| Binarization threshold | mean+0.5 SD (main) | 45.51% | [30.5, 59.2] | PARTIAL |
| Binarization threshold | mean+1.0 SD | 48.95% | [34.0, 62.4] | PARTIAL |
| Soft-assignment τ × median NN | 0.5× | 45.04% | [43.8, 46.3] | PARTIAL |
| Soft-assignment τ × median NN | 1.0× | 45.22% | [43.9, 46.5] | PARTIAL |
| Soft-assignment τ × median NN | 2.0× | 45.24% | [43.9, 46.4] | PARTIAL |
| EB shrinkage scale | 0.5× pseudocount | 45.90% | [32.1, 58.7] | PARTIAL |
| EB shrinkage scale | 1.0× (main) | 45.51% | [31.4, 59.0] | PARTIAL |
| EB shrinkage scale | 2.0× pseudocount | 45.80% | [32.3, 58.9] | PARTIAL |
| Cluster mass shift δ | −0.10 high−low | 56.47% | [42.8, 68.8] | PARTIAL |
| Cluster mass shift δ | 0.0 (main) | 45.51% | [31.4, 59.2] | PARTIAL |
| Cluster mass shift δ | +0.10 high−low | 34.55% | [16.6, 51.8] | PARTIAL |

---

### Table 3. Baseline Comparison for H2 Ordinal Trend Hypothesis

*Note*. B0 trivially attains MAPE = 0 by construction (its prediction equals MHLW grand mean). H2 was tested via Bonferroni-Holm-corrected pairwise inequalities at one-sided α = 0.05 and Page's L (1963) ordinal trend test as auxiliary. **0 of 4 pairs were confirmed**; Page's L statistic *p* = 0.9757. H2 decision: ambiguous_or_reversed.

| Baseline | Description | MAPE | 95% CI |
|---|---|---:|---|
| B0 | Uniform = MHLW grand mean | 0.00% | [0.00, 0.00] |
| B1 | Gender-only logistic | 46.02% | [46.02, 46.02] |
| B2 | HEXACO 6-domain logistic | 43.98% | [40.33, 47.35] |
| B3 | 14-cell conditional (main pipeline) | 46.34% | [46.34, 46.34] |
| B4 | Extended (age + age × cluster) | 47.04% | [45.02, 49.05] |

---

### Table 4. Counterfactual ΔP_x Estimates and H7 Intersection-Union Test

*Note*. Counterfactual operationalization per pre-registered v2.0 Section 5.7. Positive ΔP indicates *reduction* in prevalence (sign convention). Bootstrap CIs use B = 2,000 cell-stratified resamples. The H7 IUT (Berger & Hsu, 1996) computes one-sided 95% lower bounds L_BA and L_BC; classification: REVERSAL = L_BC < 0 AND L_BA ≥ 0.

| Intervention | Operationalization | ΔP point | 95% CI | CI excludes 0? |
|---|---|---:|---|---|
| A (personality) | do(HH ← +0.3σ; A ← +0.3σ; E ← +0.3σ) | −0.0061 | [−0.0207, +0.0062] | No (null) |
| B (cluster reassignment) | do(cluster ∈ {0, 4, 6}) with HH +0.40 SD shift | −0.0059 | [−0.0207, +0.0066] | No (null) |
| C (structural) | do(P_{c, x} ← 0.80 × P_{c, x}) | **+0.0349** | [+0.0264, +0.0435] | **Yes** |

**H7 IUT:** L_BA = −0.0011, L_BC = −0.0544; point ΔP_B < point ΔP_C triggers m7 priority-1 → **REVERSAL**

**Positivity diagnostic (Methods Clarification m5):** Counterfactual A and C trivially preserve positivity. Counterfactual B's flagged-weight share = 44.5%; m5 downgrade rule triggered. We interpret B's null finding as exploratory rather than confirmatory.

---

### Table 5. Transportability Sensitivity (H7 Re-classification Across Cultural Attenuation Factors)

*Note*. Each row applies factor F multiplicatively to the bootstrap distribution of all three ΔP_x; H7 IUT is re-classified per attenuated factor.

| Factor F | Anchor | ΔP_A (att) | ΔP_C (att) | H7 |
|---|---|---:|---:|---|
| 0.3 | Conservative cross-cultural worst case | −0.0018 | +0.0105 | REVERSAL |
| 0.5 | Nielsen, Glasø, & Einarsen 2017 expected | −0.0031 | +0.0174 | REVERSAL |
| 0.7 | Mild attenuation | −0.0043 | +0.0244 | REVERSAL |
| 1.0 | Reference (no attenuation) | −0.0061 | +0.0349 | REVERSAL |

---

### Table 6. Pre-Registered Hypothesis Outcomes Summary

*Note*. Hypothesis classification per v2.0 pre-registration master Section 5.4 (H1) and Section 5.7 (H7).

| Hypothesis | Pre-registered prediction | Outcome | Tier |
|---|---|---|---|
| H1 | MAPE_FY2016 ≤ 30% (Strict / Standard SUCCESS) | MAPE = 46.34% | PARTIAL SUCCESS |
| H2 | Monotonic B0 > B1 > B2 > B3 > B4 | 0 of 4 pairs confirmed | REJECTED |
| H3 | Centroid concordance N=354 vs N=13,668 | Verified in centroid study | CONFIRMED |
| H4 | Per-cell propensity stability | All 14 cells resolved at BCa | CONFIRMED |
| H5 | Gender invariance | 14-cell vs 28-cell stable | CONFIRMED |
| H6 | Power × gender harassment cluster ranking concordance | Spearman ρ > 0.80 | CONFIRMED |
| H7 | IUT classification | REVERSAL (robust F=0.3–1.0) | REVERSAL |

---

## Figures

### Figure 1. Pipeline Schematic

A box-and-arrow diagram of the nine pre-registered pipeline stages, showing the data inputs (N=354 individual data; Tokiwa 2026 *IEEE Access* 7-cluster centroids; Statistics Bureau (MIC) Labor Force Survey 2022) flowing through Stages 0–8 with their HDF5 outputs. To be generated as PNG/SVG by `simulation/code/report.py`.

**Caption.** *The HEXACO 7-typology workplace harassment microsimulation pipeline. Stage 0 (cell propensity) feeds into Stage 1 (population aggregation, MHLW reweighted) and Stage 2 (H1 four-tier classification). Stage 3 (sensitivity), Stage 4 (baselines), Stage 5 (CMV diagnostic), Stage 6 (target trial PICO), Stage 7 (counterfactuals + H7 IUT), and Stage 8 (transportability) operate in parallel after Stage 2.*

---

### Figure 2. 14-Cell Cell-Level Propensity (Stage 0)

A heatmap of P̂_c across 7 HEXACO clusters × 2 genders, with cell sizes overlaid. Cell ordering follows the v2.0 master Section 5.1 convention (cluster × 2 + gender index).

**Caption.** *Per-cell power-harassment victimization propensities P̂_c with 95% BCa CIs. Cell sizes (N_c) range from 10 to 70 (median 18). Color encodes propensity (blue = low; red = high). Cells with N_c < 20 are demarcated with a dashed boundary.*

---

### Figure 3. National Prevalence vs MHLW Validation Targets

A bar chart of model-predicted P̂ = 0.1744 (single bar) overlaid with the three MHLW validation targets (FY2016 = 0.325, FY2020 = 0.314, FY2023 = 0.193) as horizontal reference lines, with the staged enforcement period of the 2020 Power Harassment Prevention Law shaded.

**Caption.** *Model-based national prevalence (P̂ = 17.44%, blue bar) compared with MHLW past-three-year power-harassment victimization targets across three survey periods. The 2020 Power Harassment Prevention Law's staged enforcement (large enterprises 2020-06; SMEs 2022-04) coincides with the FY2020 → FY2023 decline. Predicted prevalence aligns most closely with the post-law FY2023 target (Strict SUCCESS).*

---

### Figure 4. OAT Sensitivity Tornado Plot

A tornado plot with parameter families on the y-axis and MAPE_FY2016 horizontal bars on the x-axis, ordered by sensitivity magnitude. The two sensitive families (binarization, cluster mass) appear at the top with wide bars; the two stable families (soft τ, EB scale) appear at the bottom with narrow bars. The locked main configuration is marked with a vertical reference line.

**Caption.** *One-at-a-time sensitivity sweep of FY2016 MAPE. Binarization threshold and cluster mass shift exhibit substantial sensitivity (range 29.6% to 56.5%), whereas soft-assignment τ and empirical Bayes shrinkage scale exhibit negligible sensitivity (range < 0.5%). Vertical line at 45.51% marks the locked v2.0 main configuration.*

---

### Figure 5. Counterfactual ΔP_x Forest Plot

A forest plot with three counterfactuals on the y-axis (A, B, C), point estimates with 95% CIs as horizontal whiskers, and a vertical reference line at ΔP = 0. Counterfactuals A and B's CIs overlap zero; Counterfactual C's CI excludes zero on the positive side.

**Caption.** *Counterfactual ΔP_x estimates and 95% CIs. Personality intervention (A; +0.3 SD on H/A/E) and cluster reassignment (B; do(cluster ∈ {0, 4, 6})) yield null effects (CIs cross zero). Structural intervention (C; −20% propensity, calibrated against three meta-analyses and the MHLW 2020 Law natural experiment) yields a positive effect of +3.54 percentage points (CI excludes zero on the positive side). The H7 intersection-union test (Berger & Hsu, 1996) classifies this configuration as REVERSAL.*

---

### Figure 6. Transportability Robustness

A line plot with cultural attenuation factor F on the x-axis (0.3 to 1.0) and ΔP_x on the y-axis. Three lines for ΔP_A, ΔP_B, ΔP_C, with 95% CIs as shaded bands. The ΔP_C line remains above zero across the entire factor range; the ΔP_A and ΔP_B lines remain centered on zero.

**Caption.** *H7 IUT classification under cultural attenuation factors F ∈ {0.3, 0.5, 0.7, 1.0}. The REVERSAL classification (structural dominance) is preserved across all four factors, including the conservative cross-cultural worst-case anchor (F = 0.3) and the Nielsen, Glasø, and Einarsen (2017) Asia/Oceania expected attenuation (F = 0.5).*

---

## Tables and Figures: Generation notes

- **Tables 1–6**: Embedded in `manuscript_preprint.docx` as native Word tables with APA-style formatting (Times New Roman 12 pt, double-spaced, lines below header and final row).
- **Figure 1 (pipeline schematic)**: Generated by `simulation/code/report.py` as `simulation/output/figure1_pipeline.png`. The pipeline schematic is illustrative; values come from the v2.0 master.
- **Figure 2 (14-cell heatmap)**: Generated from `output/supplementary/stage0_cell_propensity.h5` via matplotlib.
- **Figure 3 (prevalence comparison)**: Generated from `stage1_population_aggregation.h5` and `stage2_validation.h5`.
- **Figure 4 (sensitivity tornado)**: Generated from `stage3_sensitivity.h5`.
- **Figure 5 (counterfactual forest)**: Generated from `stage7_counterfactual.h5`.
- **Figure 6 (transportability)**: Generated from `stage8_transportability.h5`.

All figures are saved at 300 DPI in PNG and PDF formats; vector versions (SVG) are also produced for journal submission. PNG embeds are used in `manuscript_preprint.docx` and `manuscript_journal.docx`; PDF/SVG accompany the journal submission as separate files (per RSOS Stage 2 RR submission requirements).
