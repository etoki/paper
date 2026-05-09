# ECEL 2026 — Modality-Stratified Meta-Analysis: Analysis Plan

**Author**: Eisuke Tokiwa (single author)
**Venue**: ECEL 2026, Lund, Sweden
**Deadline**: Abstract 2026-05-14, Full Paper 2026-05-21
**Parent preprint**: doi:10.21203/rs.3.rs-9513298/v1 (Research Square, 2026-04-27)
**Novel contribution (not in preprint)**: Modality-stratified random-effects
meta-regression of Big Five — academic-achievement correlations, plus a
modality x trait interaction model.

---

## 1. Research questions

- **RQ1.** Do the pooled Big Five — achievement correlations vary across
  online learning modalities (synchronous / asynchronous / blended /
  mixed-online / unspecified)?
- **RQ2.** Is there a modality x trait interaction — i.e., do specific
  Big Five traits show modality-dependent effects on achievement?
- **RQ3.** Within the asynchronous subset (the largest modality cell in
  the corpus), is Conscientiousness the only trait whose pooled effect is
  robustly positive after sensitivity perturbations?

## 2. Data

- **Source**: `papers/P3_meta_analysis/inputs/studies.csv`, derived from
  `metaanalysis/analysis/data_extraction_populated.csv` via
  `inputs/derive_studies_csv.py`.
- **Primary pool**: studies with `inclusion_status in {include,
  include_with_caveat, include_COI}` and `primary_achievement in {yes,
  partial}` AND at least one extractable `r_*`. Per-trait k matches the
  preprint's canonical pool (`metaanalysis/analysis/CANONICAL_RESULTS.md`):
  C / N: k = 10; O / E / A: k = 9.
- **Modality coding**: derived from `modality_subtype` in the master
  extraction. Where the source paper does not state synchrony explicitly,
  platform context is used (Moodle / MOOC platforms → asynchronous unless
  stated otherwise). Inference rules are documented in
  `inputs/derive_studies_csv.py::classify_modality`.
- **Modality categories used in the analysis**:
  - **A** asynchronous
  - **M** mixed-online (within an online-only setting, both sync and async)
  - **S** synchronous
  - **U** unspecified
  - **B** blended (mix of online and face-to-face) — currently k=0 in primary pool

## 3. Statistical model

### 3.1. Pooled per-trait estimates (replication of preprint)

- Random-effects with REML estimator for tau-squared.
- Hartung—Knapp—Sidik—Jonkman (HKSJ) adjustment for confidence intervals.
- Fisher's z transformation; back-transformed Pearson r reported.
- 95 % prediction interval reported alongside CI.
- Implemented in `metaanalysis/analysis/pool.py`; the ECEL script imports
  the existing pooling functions to guarantee numerical identity with the
  preprint.

### 3.2. Modality moderator (NEW)

For each trait t in {O, C, E, A, N}:

- Fit one mixed-effects model per modality level with k >= 2:
  `pool_random_effects(y_t_in_level, v_t_in_level)`.
- Q_between test contrasting modality levels.
- Single-study modality levels (e.g. S with k=1) are reported narratively,
  not pooled.

### 3.3. Modality x trait interaction (NEW)

- Stack all primary-pool effect sizes into a long-format matrix
  (one row per study x trait observation; rows with missing r dropped).
- Fit a mixed-effects meta-regression on Fisher's z with predictors:
  modality (3-level factor restricted to A / M / U; S contributes a
  narrative footnote), trait (5-level factor), and the modality x trait
  interaction term. Random intercept per study.
- Joint Wald test on interaction coefficients reported as the primary
  inferential statistic for RQ2.
- Assumptions documented; small-cell warnings issued where any
  modality x trait cell has k < 2.

### 3.4. Sensitivity layer (preserves preprint's robustness checks)

1. **Drop beta-converted studies** (A-28 Yu, A-30 Kaspar): re-run modality
   pools.
2. **Drop COI study** (A-25 Tokiwa, in modality A): re-run modality A.
3. **Drop low-RoB (< 5)** studies: re-run modality pools.
4. **Drop the unspecified-modality cell**: confirm that the
   modality-stratified picture survives without the U bucket.

### 3.5. Heterogeneity attribution

- Report I-squared and tau-squared per modality cell.
- Attribute heterogeneity reduction (delta tau-squared) to modality stratification:
  delta tau-squared = tau-squared_overall − weighted-average(tau-squared_per_modality).

## 4. Outputs

| File | Content |
|------|---------|
| `results/modality_pools.csv` | Per-modality x per-trait pooled r (95 % CI), k, N, I-squared, tau-squared. |
| `results/modality_qbetween.csv` | Per-trait Q_between, df, p across modality levels with k >= 2. |
| `results/interaction_terms.csv` | Modality x trait interaction coefficients (Fisher z scale + back-transformed delta r). |
| `results/sensitivity.csv` | Re-run estimates under sensitivity scenarios (1)–(4). |
| `results/forestplot_<trait>.png` | Per-trait forest plot grouped by modality (Matplotlib via `metaanalysis/analysis/plots.py`). |

## 5. Pre-analysis declarations

- **Primary hypothesis**: Conscientiousness x asynchronous shows the
  largest positive pooled effect; Extraversion x asynchronous is null or
  weakly negative. (Pre-registered direction; ECEL paper reports as
  *exploratory* because the modality moderator was not pre-registered with
  the preprint.)
- **Stopping rule**: All listed analyses are run unconditionally; none is
  selected after viewing results.
- **Multiplicity**: 5 trait pools x 4 modality levels x 4 sensitivity
  scenarios = 80 cell-level estimates. The interaction Wald test is
  treated as primary inference; cell-level estimates are descriptive.

## 6. Limitations to acknowledge in the paper

- Per-modality k is small (k = 1–4 for any single modality x trait cell).
- The unspecified-modality cell mixes potentially heterogeneous designs.
- The B (blended) modality is empty in the primary pool; the sub-paper
  cannot speak to genuine blended vs purely-online contrasts directly.
- The corpus is dominated by undergraduate samples; modality conclusions
  do not transfer to K-12 or graduate populations.
- **Single author**, no second coder for the modality re-classification;
  audit-trail is the script `derive_studies_csv.py`.
