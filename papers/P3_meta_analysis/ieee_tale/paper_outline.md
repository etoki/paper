# IEEE TALE 2026 — Full Paper Outline

**Target length**: 8 pages, IEEE 2-column conference template
**Status**: Outline only; numerical placeholders filled after running
`scripts/run_ml_pipeline.py`.

---

## 1. Introduction (~0.75 page)

- Personality x online learning: established trait-level correlations;
  unmet need for *study-level* prediction of which research designs will
  surface non-null effects.
- TALE's audience is engineering / STEM education; we deliver an
  interpretability case study, not a deployable model.
- Stated novel contribution: study-level ML predictive layer + SHAP +
  fairlearn audit, **NOT in the parent Research Square preprint**.

## 2. Related Work (~0.75 page)

- Big Five — academic-achievement meta-analyses (Poropat 2009, 2014;
  Mammadov 2022; Trapmann et al. 2007).
- Personality x online learning specifically (preprint references list).
- Use of ML on small meta-analytic corpora (cite the few precedents in
  educational data mining and clinical meta-analyses).
- Algorithmic-fairness frameworks adapted to research-synthesis settings.

## 3. Data and Features (~1.25 pages)

- Source: `papers/P3_meta_analysis/inputs/studies.csv` (N studies in
  primary pool; engineering / STEM substratum reported separately).
- Feature engineering:
  - One-hot modality (S/A/M/U), era (pre/COVID/post), region
    (Asia/Europe/NA), instrument family.
  - log(N), RoB score.
- Label: binary indicator "max(|r_O|, |r_C|, |r_E|, |r_A|, |r_N|) >= 0.10".
- Listing 1: Python code for feature construction (excerpt of
  `scripts/run_ml_pipeline.py`).

## 4. Methods (~1.25 pages)

- LOSO-CV protocol; rationale (study-level dependency, no leakage).
- Model specifications:
  - Logistic Regression (L2, C grid {0.1, 1, 10}).
  - Random Forest (200 trees, default depth, class-weight balanced).
  - XGBoost (max_depth=3, eta=0.1, 100 rounds, scale_pos_weight tuned).
- Calibration via Brier; reliability diagram.
- SHAP: TreeSHAP for tree models; KernelSHAP for LR.
- Fairness: demographic-parity gap, equalised-odds gap; mitigation only
  if gap > 0.10.

## 5. STEM-subset Replication (~0.75 page)

- Side-by-side per-trait pooled r for STEM-only vs full-corpus.
- Discuss interpretation given small k.

## 6. Predictive Layer Results (~1.25 pages)

- Table: LOSO AUROC, balanced accuracy, F1, Brier per model.
- Figure: per-model calibration curves.
- Figure: SHAP beeswarm (best model).
- Top-3 force plots for studies with most extreme predictions.

## 7. Fairness Audit (~0.75 page)

- Disparity table by region and era.
- Discussion: are the disparities driven by genuine heterogeneity in the
  underlying corpus, or by feature redundancy?
- Mitigation result, if triggered.

## 8. Discussion (~0.75 page)

- The ML layer's value is interpretive, not predictive.
- Convergence between SHAP top features and the preprint's moderator
  analyses supports both pipelines.
- Fairness gaps caution against extrapolation to under-represented regions.

## 9. Limitations (~0.25 page)

- N at the study level is too small for confirmatory ML claims.
- LOSO-CV is the most defensible split given the data structure but
  inflates variance.
- Single coder, single author.

## 10. Conclusion (~0.25 page)

- The ML interpretability layer complements rather than replaces classical
  meta-analytic moderator analysis.
- Future work: extend the corpus to k > 30 studies; deploy the same
  pipeline for confirmatory prediction.

---

## Tables / figures backed by CSVs

- Table 1: `results/stem_subset_pools.csv`
- Table 2: `results/ml_loso_metrics.csv`
- Table 3: `results/shap_ranking.csv`
- Table 4: `results/fairness_metrics.csv`
- Figure 1: `results/calibration_<model>.png`
- Figure 2: `results/shap_summary.png`
- Figure 3 (optional): `results/fairness_dashboard.html`
