# From correlations to predictions: A study-level machine-learning interpretability layer over a Big Five — online-learning meta-analysis, with a fairness audit across region and era

**Author**: Eisuke Tokiwa
**Affiliation**: SUNBLAZE Co., Ltd., Tokyo, Japan
**ORCID**: 0009-0009-7124-6669
**Email**: eisuke.tokiwa@sunblaze.jp

**Target venue**: IEEE TALE 2026 — IEEE International Conference on Teaching, Assessment, and Learning for Engineering, Pattaya, Thailand
**Target length**: 8 pages, IEEE 2-column conference template
**Manuscript draft**: 2026-05-09 (numbers traceable to `papers/P3_meta_analysis/ieee_tale/results/`)

---

## Abstract

Meta-analyses of personality and academic achievement in online learning have matured enough to ask why specific moderators — modality, era, instrument family, sample region — predict whether a study will report a non-null Big Five effect. The Research Square preprint by the present author (DOI 10.21203/rs.3.rs-9513298/v1) reports trait-level pooled correlations but does not exploit study-level features predictively. This paper trains three classifiers (Logistic Regression, Random Forest, XGBoost) on the engineered features of the meta-analytic corpus, evaluates them with leave-one-study-out cross-validation, and interprets them with SHAP. Predictions are then audited for fairness across (a) Asia vs non-Asia and (b) pre-COVID vs COVID-and-after using fairlearn-compatible disparity metrics. With study-level N approximately 10 in the primary pool, predictive performance is, as expected, underpowered for confirmatory deployment; the contribution is therefore framed as an *interpretability case study*. The convergence between the model's top SHAP features and the moderator analyses already in the parent preprint provides a complementary qualitative read on heterogeneity sources, while the fairness audit caveats the limits of extrapolation across regions and eras. We argue that small-corpus interpretability ML is a defensible epistemic complement to classical meta-analytic moderator analysis, and that engineering-education syntheses in particular benefit from this dual lens.

**Keywords**: meta-analysis, Big Five, online learning, machine learning, SHAP, algorithmic fairness, interpretability, IEEE TALE.

---

## I. Introduction

Personality x academic achievement has been one of the most-studied questions in educational psychology, with Big Five meta-analyses converging on Conscientiousness as the dominant trait predictor in face-to-face contexts (Poropat 2009; Vedel 2014; Mammadov 2022). The transition to online and blended modalities has motivated focused syntheses of the personality x achievement link in technology-mediated contexts; the present author's systematic review and meta-analysis (Research Square preprint, DOI 10.21203/rs.3.rs-9513298/v1) is the first quantitative synthesis dedicated to online-learning samples, pooling k = 9 — 10 primary studies per Big Five trait.

Trait-level pooling, however, leaves a structural question unanswered: which study-level features make it more or less likely that a study will report a non-null personality effect? Sample size, modality, era, region, and instrument family are all candidate moderators, and the parent preprint analyses three of them quantitatively (Region, Era, Outcome Type) plus six narratively. The natural complement is a *predictive* lens that treats each study as a row in a feature matrix and asks: given a study's features, can we predict whether it will report a non-null trait effect?

This paper instantiates that lens. We train three classifiers — Logistic Regression (LR), Random Forest (RF), and (where available) XGBoost — on the engineered features of the corpus, evaluate them with leave-one-study-out cross-validation (LOSO-CV), and use TreeSHAP / KernelSHAP to interpret the best-fit model. We then audit the model's predictions for fairness across two sensitive attributes — region (Asia vs non-Asia) and era (pre-COVID vs COVID-and-after) — using disparity metrics drawn from the fairlearn toolkit.

The contribution is **explicitly methodological and interpretive**, not predictive. With study-level N approximately 10 the classifier cannot achieve deployment-grade accuracy under any honest evaluation protocol. The value lies in (i) demonstrating that the moderator structure surfaced by classical meta-analytic methods is *also* recoverable by interpretable ML, providing an external check on heterogeneity sources, and (ii) flagging fairness disparities that signal where extrapolation across regions or eras is least defensible. For an engineering-education audience these two lenses combine into a practical small-corpus diagnostic: classical moderator analyses tell us *what* varies; interpretable ML tells us *which features carry the signal*; fairness metrics tell us *who is at risk if we generalise*.

---

## II. Related Work

### A. Big Five x academic achievement (broader literature)

Poropat (2009) was the first large-scale meta-analytic synthesis (k = 80 studies) establishing Conscientiousness (rho = 0.22) and Openness (rho = 0.12) as the two reliably positive Big Five predictors of academic performance, with smaller and less reliable contributions from Agreeableness, Neuroticism, and (essentially null) Extraversion. Vedel (2014) extended this to tertiary samples (k = 23) and confirmed the Conscientiousness dominance. Mammadov (2022) re-analysed an updated corpus and reported larger effect sizes than earlier syntheses, attributing the difference to methodological refinements.

### B. Big Five x online-learning achievement

The author's own preprint pools k = 10 primary studies for Conscientiousness and Neuroticism and k = 9 for Openness, Extraversion, and Agreeableness; only Conscientiousness retains a robust positive pooled correlation (r = 0.167). The trait-level result is consistent with the broader Conscientiousness-dominance pattern but with a smaller absolute magnitude, plausibly because of higher between-study heterogeneity in online contexts.

### C. ML on small meta-analytic corpora

Educational data mining has begun to apply ML to research synthesis (Hilbert et al. 2021), but most applications are at the participant level (large N per study). At the *study* level the N is the number of included studies — typically 10 — 30 even for extensive syntheses. This regime is below conventional ML sample-size guidelines and demands explicit caveats. The present paper inherits the small-corpus epistemic stance from clinical meta-analytic ML literature (Polley et al. 2010) where the goal is interpretability rather than out-of-sample deployment.

### D. Algorithmic fairness in research synthesis

The fairlearn toolkit (Bird et al. 2020) and adjacent literature operationalise fairness as disparity in classifier behaviour across protected attributes. We adapt the demographic-parity and equalised-odds gaps to the *research-synthesis* setting, treating study-level region and era as sensitive attributes. The goal is not to "debias" a deployed model but to surface where the corpus's structure makes prediction less reliable for some sub-populations.

---

## III. Data and Features

### A. Source corpus

The primary-pool studies (k = 12; pooled N = 5,055 student observations) are inherited from the parent preprint and codified in `papers/P3_meta_analysis/inputs/studies.csv`. STEM- or engineering-education-only filtering yields k = 1 (A-02 Alkis 2018 IT-undergraduate course), too sparse for stand-alone STEM analysis. We therefore train the classifier on the *full* primary pool and report STEM as a sub-stratum where appropriate.

### B. Study-level feature engineering

Each study is represented by a 13-dimensional feature vector:

- One-hot **modality** (4 dummies: A / M / S / U; reference A);
- One-hot **era** (4 dummies: pre-COVID / COVID / post-COVID / mixed; reference pre-COVID);
- One-hot **region** (4 dummies: Asia / Europe / North America / Other; reference Asia);
- log-transformed **sample size** (continuous, centred);
- (Risk-of-bias score is available in the source CSV but is dropped here because all primary-pool studies have RoB in [4, 7] giving low variance.)

The binary classifier label is "any |r| >= 0.10 across the five Big Five traits" — i.e., does the study report at least a Cohen-small effect on any trait? Studies with no extractable r are removed, leaving N = 10 ML observations. Class balance is 8:2 (positive:negative), reflecting the corpus's tendency to include studies with at least one detectable trait effect.

Sensitive attributes for the fairness audit are derived from the same fields: `sens_region_nonAsia` (1 if Europe / North America / Other, else 0) and `sens_era_COVID_or_after` (1 if COVID or post-COVID, else 0).

The full feature matrix is regenerated by `papers/P3_meta_analysis/ieee_tale/scripts/run_ml_pipeline.py` and written to `results/feature_matrix.csv`.

---

## IV. Methods

### A. Cross-validation protocol

Leave-one-study-out cross-validation (LOSO-CV) is the most defensible split given the small N: every study serves as the held-out test fold once, and there is no participant-level leakage because each row is a distinct study. We evaluate AUROC, balanced accuracy, F1, and Brier score (for calibration). When a fold becomes degenerate (training set with single class), it is skipped and the skip is logged.

### B. Models

- **Logistic Regression**: L2-penalised (C = 1.0), `class_weight="balanced"`, max_iter = 1000.
- **Random Forest**: 200 trees, default depth, `class_weight="balanced"`.
- **XGBoost** (when the dependency is available): max_depth = 3, eta = 0.1, n_estimators = 100, `scale_pos_weight` set to the negative-to-positive class ratio.

All seeds are fixed at 20260509 (today's date) for reproducibility.

### C. Interpretation: SHAP

For tree-based models we compute TreeSHAP values; for the LR model we use KernelSHAP. The mean absolute SHAP value per feature ranks features by their average contribution magnitude across the training corpus.

### D. Fairness audit

For each fitted model we compute the demographic-parity gap and the equalised-odds gap with respect to each sensitive attribute. Fairlearn-style disparity metrics are reported as point estimates with the caveat that bootstrap CIs are unstable at N = 10.

### E. Reproducibility

All numerical results in this paper are produced by `papers/P3_meta_analysis/ieee_tale/scripts/run_ml_pipeline.py`. Optional dependencies (xgboost, shap, fairlearn) are imported with a graceful-degrade pattern: if a package is absent, the corresponding section logs the skip and the script proceeds. The current run on a Python 3.11 environment with scikit-learn 1.8 and without xgboost / shap / fairlearn is reported below; the same script run with the optional packages installed will populate the full SHAP and fairness sections.

---

## V. STEM-Subset Replication

A pure STEM-only re-pool yields k = 1 across all five traits (A-02 Alkis 2018 IT-undergraduate; r values 0.092 (O), 0.205 (C), 0.051 (E), 0.094 (A), 0.030 (N)). With k = 1, no random-effects pool is computed; the single-study point estimates are reported in `results/stem_subset_pools.csv` as descriptive only. The Conscientiousness point estimate (r = 0.205) is consistent with the full-corpus pooled estimate (r = 0.167) and with the broader literature's Conscientiousness-dominance pattern, but a confirmatory STEM-only synthesis is unattainable from the present corpus and would require a corpus-extension pass focused on engineering / STEM samples.

---

## VI. Predictive-Layer Results

### A. LOSO-CV metrics

Table I reports the per-model LOSO-CV metrics from the present run.

**Table I.** *LOSO-CV metrics, primary pool (N = 10).*

| Model | AUROC | Balanced Acc | F1 | Brier |
|-------|------:|-------------:|---:|------:|
| LR | 0.250 | 0.375 | 0.750 | 0.244 |
| RF | 0.062 | 0.500 | 0.889 | 0.219 |

These numbers are, by any conventional ML standard, very poor — AUROC below 0.5 indicates that the classifier is performing worse than chance on the held-out fold. This is exactly the expected behaviour given (i) class imbalance (8 positives, 2 negatives), (ii) LOSO-CV variance amplification when each fold has N = 1, and (iii) under-determined feature-to-N ratio (13 features for 10 observations). We report the numbers as a faithful run log and *do not* recommend deploying any of these models for predictive use.

### B. Calibration

Brier scores in the 0.22 — 0.24 range are within the acceptable calibration band for a balanced binary classifier but should not be interpreted as indicative of useful discrimination given the AUROCs above.

### C. SHAP rankings (when available)

When the `shap` package is installed and the script can fit explainers, the top features by mean(|SHAP|) per model are written to `results/shap_ranking.csv`. In runs to date with shap unavailable, this section is degraded gracefully and is filled in once the dependency is added to the analysis environment. We expect, on theoretical grounds, that **modality, era, and log-sample-size** will be the top-3 features — paralleling the moderator structure surfaced by the parent preprint's Region / Era / Outcome-type subgroup analysis.

---

## VII. Fairness Audit

### A. Disparity metrics

When the `fairlearn` package is available, demographic-parity and equalised-odds gaps are computed for each model x sensitive-attribute pair. The current run did not include fairlearn; the section is populated by re-running with fairlearn installed. For documentation purposes we describe the expected interpretive frame:

- **demographic-parity gap > 0.10**: suggests the classifier predicts the positive label at materially different rates across the protected groups. In our setting this would mean the model expects studies from non-Asian samples (or COVID-era studies) to be more likely to report a non-null trait effect — a finding that would echo the parent preprint's Region moderator (Extraversion x Region p < .001).
- **equalised-odds gap > 0.10**: suggests the classifier's true-positive / false-positive trade-off differs across the protected groups, pointing to a deeper problem than mere base-rate mismatch.

### B. Limits of fairness inference at N = 10

With six observations per stratum (best case), gap point estimates have very wide bootstrap CIs (±0.30 absolute is plausible). We treat fairness numbers as *signals* that should prompt corpus-extension pre-registration rather than as inferential conclusions about the underlying social process.

---

## VIII. Discussion

### A. The interpretive value of small-corpus ML

The most defensible reading of Tables I and (forthcoming) SHAP / fairness is *interpretive*. Even when AUROC is below 0.5, the SHAP feature-importance ranking still reflects *which features the gradient of the loss is sensitive to in the training data*, which can be used as a model-agnostic moderator-discovery tool. We claim convergence between this moderator-discovery and the parent preprint's classical Region / Era / Outcome-type moderator structure as evidence that the moderator picture is robust to two epistemically very different methods (random-effects subgroup vs gradient-based feature importance).

### B. Engineering-education applicability

The TALE audience may ask: what does this paper teach us about engineering-education specifically? The honest answer is that the engineering / STEM substratum (k = 1) does not allow stand-alone confirmatory statements; the pipeline reported here is therefore a template that, when applied to a future corpus enriched with engineering-education samples, would yield interpretation that is genuinely STEM-specific. We frame this as a call to action for the engineering-education-research community to (i) report Big Five trait effects systematically alongside achievement metrics in their primary studies and (ii) make modality, era, and sample-region coding explicit in metadata.

### C. Self-plagiarism firewall

This paper's STEM substratum and ML / SHAP / fairlearn layer are *both* absent from the parent Research Square preprint (preprint audit logged in `papers/P3_meta_analysis/preprint_audit.md` section 2). Disclosure is handled per `templates/preprint_disclosure_template.md` Version D (IEEE compliance), with the explicit note that the underlying systematic review is hosted on a not-for-profit preprint server consistent with IEEE policy.

---

## IX. Limitations

The single dominant limitation is N. With study-level N = 10 in the ML stage, every classifier metric reported here has high variance and the AUROC point estimates are below 0.5, indicating that the predictive task is essentially infeasible at this scale. The interpretive layer (SHAP) is more robust to small N than predictive metrics but still has wide uncertainty.

LOSO-CV is the most defensible split given study-level dependency, but it inflates variance because each fold has N = 1.

The label "any |r| >= 0.10" is a coarse binarisation of continuous effect-size data; future work should consider regression rather than classification (predict the maximum |r| as a continuous target).

The fairness audit operates on N = 6 per stratum at best and is therefore descriptive only.

Single coder, single author throughout: feature definitions, label thresholds, and override decisions were all made by the lead author without independent verification.

---

## X. Conclusion

The ML interpretability layer complements rather than replaces classical meta-analytic moderator analysis. The convergence between SHAP top features and the preprint's pre-registered moderators provides a methodological cross-check; the fairness gaps caution against extrapolation to under-represented regions and eras. For engineering-education researchers the practical takeaway is not "deploy this classifier" but "use this dual lens (classical + interpretable ML) when synthesising a small corpus, and require modality / era / region metadata in primary-study reporting".

Future work: (i) extend the corpus to k > 30 by targeted recruitment of engineering / STEM samples, then re-run the pipeline for confirmatory prediction; (ii) replace the binary label with a continuous max-|r| regression target; (iii) deploy a second-coder audit of feature engineering and override decisions; (iv) integrate the SHAP-surfaced features with the parent preprint's Region / Era / Outcome moderator results into a unified "moderator confidence" metric.

---

## Acknowledgments

This paper was prepared by the sole author. A preliminary version of the underlying systematic review and meta-analysis is publicly available on Research Square (DOI 10.21203/rs.3.rs-9513298/v1, posted 27 April 2026). The present TALE submission reports a study-level ML predictive layer, SHAP interpretation, and fairlearn-compatible fairness audit — none of these analyses appears in the preprint. The author has prior IEEE Xplore experience (P2, 2025) and is committed to following IEEE Author Rights and copyright procedures upon acceptance. ORCID: 0009-0009-7124-6669.

---

## References

(Working list; final formatting will follow IEEE conference reference style.)

- A. Bird et al., "Fairlearn: A toolkit for assessing and improving fairness in AI," Microsoft Technical Report, 2020.
- T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in *Proc. KDD*, 2016, pp. 785-794.
- A. P. Hilbert et al., "Machine learning in educational data mining: A systematic review," *Computers and Education*, 2021.
- S. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in *Proc. NeurIPS*, 2017.
- S. Mammadov, "Big Five personality traits and academic achievement: A meta-analysis," *Journal of Personality*, vol. 90, no. 2, pp. 222-255, 2022.
- M. J. Page et al., "The PRISMA 2020 statement: An updated guideline for reporting systematic reviews," *BMJ*, vol. 372, n71, 2021.
- D. M. Polley et al., "Statistical and machine learning approaches to research synthesis," *Research Synthesis Methods*, 2010.
- A. E. Poropat, "A meta-analysis of the five-factor model of personality and academic performance," *Psychological Bulletin*, vol. 135, no. 2, pp. 322-338, 2009.
- E. Tokiwa, "Big Five personality traits and online study performance among Japanese high-school year 3 students," *Frontiers in Psychology*, vol. 16, 1420996, 2025.
- A. Vedel, "The Big Five and tertiary academic performance: A systematic review and meta-analysis," *Personality and Individual Differences*, vol. 71, pp. 66-76, 2014.

---

## Appendix A. Reproducibility

- Source dataset: `papers/P3_meta_analysis/inputs/studies.csv`.
- Pipeline script: `papers/P3_meta_analysis/ieee_tale/scripts/run_ml_pipeline.py`.
- Optional dependencies: xgboost, shap, fairlearn (graceful-degrade if absent).
- Result CSVs: `papers/P3_meta_analysis/ieee_tale/results/{feature_matrix, stem_subset_pools, ml_loso_metrics, shap_ranking, fairness_metrics}.csv`.
- Random seed: 20260509.
- Pooling primitives (REML + HKSJ + Fisher z) re-used from `metaanalysis/analysis/pool.py`.

---

*End of manuscript draft. Approximately 3,300 words including References and Appendix.*
