# Hallucination Check Protocol — harassment/

**Purpose**: split the manuscript hallucination audit into focused tasks
that can be re-run individually, mirroring `metaanalysis/HALLUCINATION_CHECK_PROTOCOL.md`.

**Targets**: `paper/Manuscript_only.docx`, `paper/Table.docx`,
`paper/Title page with Declarations.docx`.

**Source of truth**: `harassment/CANONICAL_RESULTS.md`, derived from
`res/*.csv` (regenerable via `python3 analysis.py`) and
`alfa/hexaco_alpha_results.csv`.

**Run all**:

```bash
python3 harassment/check_hallucinations.py            # all tasks
python3 harassment/check_hallucinations.py --task t3  # one task
python3 harassment/check_hallucinations.py --verbose  # show passes
```

---

## Task list (MECE)

| ID | Task | Auto | Source |
|----|------|------|--------|
| **T1** | Table 1 Descriptives (M, SD for 13 vars) | Y | `res/descriptive_statistics.csv` |
| **T2** | Table 2 Spearman key correlations | Y | `res/spearman_rho.csv` |
| **T3** | Tables 3 & 4 Regression (β, R², ΔR², F, p) | Y | `res/regression_coefficients_extended.csv` + `res/model_fit_incremental.csv` |
| **T4** | Table 5 Sex-stratified R² | Y | `res/sex_stratified_R2.csv` |
| **T5** | Tables 6 & 7 (DW, BP, Cook's D, VIF) | Y | `res/model_fit_incremental.csv` + `res/vif_modelB.csv` |
| **T6** | Cronbach's α (Methods) | Y | `alfa/hexaco_alpha_results.csv` (item-level data) |
| **T7** | Sample / counts (N, gender, age bins, missing) | Y | `raw.csv` |
| **T8** | Reference list audit (in-text ⇔ ref list) | Y | manuscript text only |

## Typical failure patterns

- **T1–T5**: hand-typed numeric mismatch (e.g. β = .14 vs canonical .143).
- **T6**: α reported in Methods does not match what `alfa/` actually
  computes; α stated as "in the present sample" while really computed
  on the combined N = 4168 psychometric pool.
- **T7**: 134 vs 133 — total N includes all 354, but listwise N for
  regression (DT has 1 missing) drops one male → 133.
- **T8**: in-text citations missing from reference list (the most
  common hallucination historically). Surname spelling drift between
  body and reference list (Saltuküğlu ↔ Saltukoğlu).

## When to run

Re-run after every edit to `paper/*.docx`. Re-run `analysis.py`
first if `raw.csv` changes. Re-run `alfa/alpha_hexaco.py` if any of
the three psychometric files changes.
