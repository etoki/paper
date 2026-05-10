# IEEE TALE 2026 — Engineering / STEM Subset + ML Predictive Layer: Analysis Plan

**Author**: Eisuke Tokiwa (single author)
**Venue**: IEEE TALE 2026, Pattaya, Thailand
**Deadline**: Full Paper 2026-06-30
**Parent preprint**: doi:10.21203/rs.3.rs-9513298/v1
**Novel contribution (not in preprint)**:
1. Engineering / STEM subset analysis of the Big Five — achievement pool.
2. Study-level ML predictive layer (Logistic Regression / Random Forest /
   XGBoost) trained on extracted moderator features, with SHAP
   interpretability and a fairlearn-based fairness audit across region
   and era.

---

## 1. Research questions

- **RQ1.** Within engineering / STEM samples, do the Big Five —
  achievement correlations replicate the preprint's pooled estimates?
- **RQ2.** Can a study-level classifier trained on the extracted
  features (modality, era, sample size, RoB, region, instrument)
  predict whether a study reports a non-null Big Five effect on
  achievement, and which features carry the most predictive weight?
- **RQ3.** Are these classifier predictions fair across (a) Asia vs
  non-Asia, and (b) pre-COVID vs COVID-and-after? Fairness operationalised
  via demographic parity and equalised-odds gaps.

## 2. Data

- **Source CSV**: `papers/P3_meta_analysis/inputs/studies.csv`.
- **Engineering / STEM subset**: rows with `discipline == "STEM"`
  *or* `subject_domain` matching {IT, Engineering, Computer Science,
  Health-related disciplines with STEM emphasis}. The current corpus
  contains a small set of explicit STEM samples (e.g., A-02 Alkis 2018 IT
  course); to enlarge the subset for the ML pipeline the analysis falls
  back to *all primary-pool studies* and uses STEM as a sub-stratum
  reported alongside the full-corpus model.
- **Per-study label** for the classifier (`y_label`): binary indicator
  "any of |r_O|, |r_C|, |r_E|, |r_A|, |r_N| >= 0.10" — a Cohen-style
  small-effect threshold. Studies with no extractable r are excluded from
  the ML stage.
- **Per-study features** (`X`): modality (one-hot S/A/M/U), era (one-hot
  pre/COVID/post), sample size N (log-transformed), RoB score (numeric),
  region (one-hot Asia/Europe/NA), instrument family (BFI / NEO / IPIP /
  TIPI / Mini-IPIP — one-hot).

## 3. Methods

### 3.1. STEM-subset replication (Aim 1)

- Subset the primary pool to STEM studies; re-run REML + HKSJ pooled
  estimates per trait.
- Report side-by-side with the full-corpus estimates from
  `metaanalysis/analysis/pooling_results.csv`.
- k per trait will be tiny (order 1–3); inferences are exploratory.

### 3.2. ML pipeline (Aim 2)

- Three classifiers: Logistic Regression (L2, C tuned by 5-fold inner CV),
  Random Forest (200 trees, default depth), XGBoost (depth 3, eta 0.1,
  100 rounds, early stopping disabled given small N).
- Outer evaluation: leave-one-study-out cross-validation (LOSO-CV);
  metrics: AUROC, balanced accuracy, F1.
- Calibration: Brier score; reliability diagram saved to
  `results/calibration_<model>.png`.
- Class imbalance handled by `class_weight="balanced"`.
- **The N is small (~12 primary-pool studies with extractable r).
  All performance numbers are reported as proof-of-concept and explicitly
  caveated in the discussion. The ML stage's primary contribution is
  *interpretability*, not predictive performance.**

### 3.3. SHAP interpretation

- Compute TreeSHAP values for the XGBoost classifier (best-fit model, or
  whichever has the highest LOSO AUROC).
- Per-feature mean(|SHAP|) ranking saved to `results/shap_ranking.csv`.
- Global summary plot + per-prediction force plots for the most
  influential 3 studies, saved to `results/shap_*.png`.

### 3.4. Fairness audit (fairlearn)

- Sensitive attributes: `region` (Asia vs non-Asia) and `era` (pre vs
  post-pre). Engineered to be binary for fairlearn metrics.
- Metrics: demographic parity difference / ratio, equalised-odds difference.
- Mitigation pass (only if disparity > 0.10 absolute): rerun with
  `fairlearn.reductions.ExponentiatedGradient(constraints=DemographicParity())`,
  report disparity vs accuracy trade-off.
- Output: `results/fairness_metrics.csv`.

## 4. Outputs

| File | Content |
|------|---------|
| `results/stem_subset_pools.csv` | STEM-only pooled r per trait (k, N, CI). |
| `results/ml_loso_metrics.csv` | LOSO-CV AUROC / balanced acc / F1 / Brier per model. |
| `results/shap_ranking.csv` | Mean(\|SHAP\|) per feature per model. |
| `results/fairness_metrics.csv` | Disparity metrics per sensitive attribute. |
| `results/calibration_<model>.png` | Reliability diagrams. |
| `results/shap_summary.png` | Global SHAP beeswarm (best model). |
| `results/fairness_dashboard.html` | Optional fairlearn dashboard export. |

## 5. Pre-analysis declarations

- **No model selection on the test fold**: Hyperparameters are fixed
  before LOSO-CV; only inner CV is used for any tuning.
- **No multiple-testing correction**: With only 12 study-level
  observations and 3 models, all classifier comparisons are exploratory.
- **Reproducibility**: random seed 20260509 (today's date); all RNG
  consumers seeded explicitly.

## 6. Limitations to acknowledge in the paper

- N at the study level (~12) is far below conventional ML thresholds.
  The ML pipeline is reported as an *interpretability case study*, not as
  a deployable model.
- Engineering/STEM-only k is too small for confirmatory inference.
- Fairness metrics on N ~ 6 per stratum are unstable; report point
  estimates with bootstrap CI.
- Single-author submission; no second coder for feature definitions.
