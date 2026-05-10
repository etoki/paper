# ICERI 2026 — Education-Level x Discipline Cross-Tabulation: Analysis Plan

**Author**: Eisuke Tokiwa (single author)
**Venue**: ICERI 2026, Seville, Spain
**Deadline**: Abstract 2026-07-09; Full Paper TBD (typically ~6 weeks after
abstract acceptance).
**Parent preprint**: doi:10.21203/rs.3.rs-9513298/v1
**Novel contribution (not in preprint)**:
A 3 (education level: K-12 / Undergraduate / Graduate) x 3 (discipline:
STEM / Humanities / Mixed) cross-tabulated meta-analytic interaction model
of the Big Five — achievement correlations.

---

## 1. Research questions

- **RQ1.** Does the pooled Big Five — achievement effect differ across
  education levels (K-12 vs UG vs Graduate)?
- **RQ2.** Does it differ across disciplines (STEM vs Humanities vs Mixed)?
- **RQ3.** Is there an education-level x discipline interaction effect on
  the trait — achievement correlations?

## 2. Data

- **Source**: `metaanalysis/conference_submissions/inputs/studies.csv`, restricted to
  the primary pool.
- **Education-level categories** (collapsed):
  - K-12 (= K-12 + HS_Year3 + HS_Grade12)
  - Undergraduate (= Undergraduate + Mixed_secondary_postsecondary)
  - Graduate (= Graduate)
  - Mixed_UG_Grad treated as a separate cell — too heterogeneous to merge.
- **Discipline categories** (re-derived in `inputs/derive_studies_csv.py`):
  STEM / Humanities / Psychology / Mixed.

## 3. Methods

- For each cell of the 3x3 (or 4x4 with Mixed_UG_Grad) cross-tab and each
  trait, run a per-cell random-effects pool when k >= 2; otherwise report
  point estimates without inference.
- Cross-tab interaction test on Fisher's z scale (long format,
  per study x trait observation):
  z_ij = beta_0 + level_i + discipline_j + (level x discipline)_ij + e_ij.
  Random intercept per study.
- Heterogeneity attribution: how much of the overall tau-squared is absorbed
  by the cross-tab structure.

## 4. Outputs

| File | Content |
|------|---------|
| `results/cross_tab_pools.csv` | Per (level, discipline, trait) pooled r when k>=2. |
| `results/interaction_terms.csv` | Long-format interaction model coefficients. |
| `results/cross_tab_summary.md` | Human-readable cross-tab table. |

## 5. Pre-analysis declarations

- Many cells will have k < 2 or even k = 0 in the primary pool
  (the corpus is undergraduate-dominated). The paper reports the empty
  cells transparently as a *finding about the field*: education-level x
  discipline coverage is uneven.
- Confirmatory claims are restricted to cells with k >= 4. All others are
  descriptive.

## 6. Limitations to acknowledge in the paper

- Per-cell k is small to non-existent for Graduate x STEM, K-12 x
  Humanities, etc.
- The corpus is convenience-sample dominated; cross-tab cells reflect
  publication patterns, not population stratification.
- Single coder, single author.
