# ICERI 2026 — Full Paper Outline

**Target length**: ~10 pages, IATED Word template

---

## 1. Introduction (~1 page)

- The pooled meta-analytic estimate is a coarse instrument. Its policy
  value depends on whether interaction structure (education x discipline)
  is present and detectable.
- Stated novel contribution: 3x3 (or 4x4) cross-tab interaction — not in
  the parent Research Square preprint.

## 2. Related Work (~1 page)

- Education-level moderators in personality x achievement (Poropat 2009 by
  level; Mammadov 2022 by stage).
- Discipline moderators (STEM vs Humanities; Vedel 2014).
- Online-learning context: most evidence is undergraduate / mixed-major.

## 3. Method (~2 pages)

- Cross-tab definitions; collapsing rules.
- Per-cell random-effects pool (REML + HKSJ).
- Long-format mixed-effects meta-regression specification.
- tau-squared decomposition.

## 4. Results (~3 pages)

### 4.1. Cross-tab cell map (Figure 1)

- Heatmap of k per (level, discipline) cell.

### 4.2. Per-cell pooled effects (Table 1)

- For each (level, discipline) cell with k >= 2: pooled r per trait, 95 %
  CI, I-squared.

### 4.3. Interaction model (Table 2)

- Coefficients for level main effect, discipline main effect, interaction
  terms.
- Wald chi-squared joint test.

### 4.4. Heterogeneity attribution (Table 3)

- tau-squared overall vs tau-squared after structuring; percent absorbed.

## 5. Discussion (~1.5 pages)

- The empirical message: where evidence is dense, interaction is
  detectable; where it is sparse, the model cannot speak.
- The structural message: K-12 STEM-online and Graduate-Humanities-online
  are evidence deserts.
- Implications for ICERI audience (educators / programme designers): the
  pooled effects in the parent preprint should not be applied uniformly
  across teaching contexts.

## 6. Limitations (~0.5 page)

- Most cells have k < 2.
- Discipline classification is heuristic (script `derive_studies_csv.py`).
- Single coder, single author.

## 7. Conclusion (~0.5 page)

- Where you teach (level x discipline) shapes which Big Five trait
  matters more in online instruction. The current corpus only allows that
  claim to be made for the dense undergraduate-mixed cell with confidence;
  elsewhere it is hypothesis-generating.

---

## Tables / figures backed by CSVs

- Figure 1: heatmap from `results/cross_tab_pools.csv`
- Table 1: `results/cross_tab_pools.csv`
- Table 2: `results/interaction_terms.csv`
- Table 3: tau-squared decomposition (auto-generated section in
  `results/cross_tab_summary.md`)
