# IEEE TALE 2026 — Abstract

**Title**:
*From correlations to predictions: A study-level machine-learning
interpretability layer over a Big Five — online-learning meta-analysis,
with a fairness audit across region and era.*

**Author**: Eisuke Tokiwa (SUNBLAZE Co., Ltd.; ORCID 0009-0009-7124-6669)
**Word target**: ~250 words (IEEE conference abstract).

> Numbers in `{{...}}` are filled from
> `papers/P3_meta_analysis/ieee_tale/results/*.csv` after running
> `scripts/run_ml_pipeline.py`.

---

**Background.** Meta-analysis of personality and academic achievement in
online learning has matured enough to ask why specific moderators —
modality, era, instrument family, sample region — predict whether a study
will report a non-null Big Five effect. The Research Square preprint by
the present author (DOI 10.21203/rs.3.rs-9513298/v1) reports trait-level
pooled correlations but does not exploit study-level features
predictively.

**Aim.** We train three classifiers (Logistic Regression, Random Forest,
XGBoost) on the extracted study-level features of the meta-analytic
corpus, evaluate them with leave-one-study-out cross-validation, and
interpret them with SHAP. We then audit predictions for fairness across
(a) Asia vs non-Asia and (b) pre-COVID vs COVID-and-after using fairlearn.

**Methods.** Features: modality, era, sample size (log), RoB, region,
instrument family. Label: "any |r| >= 0.10" across the five Big Five
traits. LOSO-CV; AUROC; calibration via Brier score; TreeSHAP for the best
model; demographic-parity / equalised-odds gaps.

**Results.** LOSO AUROC: LR = {{auroc_lr}}, RF = {{auroc_rf}}, XGB =
{{auroc_xgb}}. Top SHAP features: {{shap_top_feature_1}},
{{shap_top_feature_2}}, {{shap_top_feature_3}}. Demographic parity
difference (region): {{dp_region}}; equalised-odds difference (era):
{{eo_era}}.

**Discussion.** With study-level N ~ 12, predictive performance is
underpowered, but the *interpretability* layer is informative: the
features the model leans on align with the moderator analyses already in
the preprint, providing a complementary qualitative read on
heterogeneity sources. Fairness gaps {{exceed / stay below}} the 10%
threshold, motivating cautious extrapolation from the existing corpus to
new regions and eras.

**Keywords**: meta-analysis, Big Five, online learning, machine learning,
SHAP, algorithmic fairness, IEEE TALE.

---

## Submission notes

- Disclose preprint per `templates/preprint_disclosure_template.md` (TALE block).
- IEEE template: 8 pages max, 2-column.
- Single-author; ORCID required.
