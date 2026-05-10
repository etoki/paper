# Cover Letter — IEEE TALE 2026 Paper Submission

**To**: Programme Committee, IEEE TALE 2026
**Submission system**: EDAS (https://edas.info/)
**Conference**: IEEE TALE 2026, Pattaya, Thailand, 2–4 December 2026
**From**: Eisuke Tokiwa, SUNBLAZE Co., Ltd., Tokyo, Japan (ORCID 0009-0009-7124-6669)
**Date**: 2026-06-30 (paper submission deadline)
**Subject**: IEEE TALE 2026 paper submission — *From correlations to predictions: A study-level machine-learning interpretability layer over a Big Five — online-learning meta-analysis*
**Track**: AI for Education (Learning Analytics) / Online & Blended Learning

---

Dear IEEE TALE 2026 Programme Committee,

I am pleased to submit the enclosed paper, *From correlations to predictions: A study-level machine-learning interpretability layer over a Big Five — online-learning meta-analysis, with a fairness audit across region and era*, to the IEEE Conference on Teaching, Assessment, and Learning for Engineers 2026.

## Manuscript summary

The manuscript layers a study-level machine-learning interpretability and fairness pipeline on top of a Big Five × online-learning meta-analysis corpus restricted to STEM / engineering primary studies (k ≈ 8–10 studies depending on availability filter). Three predictive models — logistic regression, random forest, and XGBoost — are trained leave-one-study-out (LOSO) on study-level features (trait, modality, region, era, instrument, sample size, education level) to predict whether each per-trait correlation is "moderate or stronger" (|r| ≥ 0.10). SHAP feature attribution decomposes which study-level descriptors most predict the trait–achievement correlation magnitude. A Fairlearn-based subgroup audit then evaluates demographic parity and equalised odds across (i) region (Asia vs non-Asia) and (ii) era (pre-COVID / COVID / post-COVID).

The paper's framing is deliberately **methodological for the engineering-education audience**: it shows how interpretable ML can serve as a meta-analytic auditing tool, surfacing which study-level features (rather than which traits) most explain heterogeneity across the synthesised evidence. The fairness audit is a transparent acknowledgement that meta-analytic conclusions are themselves subject to subgroup-coverage bias.

## Fit with TALE 2026 theme

The 2026 theme — *AI-Augmented Learning: Redefining the Future of Engineering & Technology Education* — and the AI for Education track (Learning Analytics, Ethical AI, Adaptive Learning) align directly with the paper's combination of (i) ML-based learning-analytics methodology applied to (ii) a meta-analytic corpus from engineering / STEM online-learning studies, with (iii) explicit fairness considerations.

## Preprint disclosure (IEEE policy compliance)

The systematic review and meta-analysis on which this paper builds was deposited on Research Square as a preprint (DOI 10.21203/rs.3.rs-9513298/v1, posted 27 April 2026). The present TALE submission reports a **study-level machine-learning pipeline (Logistic Regression / Random Forest / XGBoost) trained on the STEM / engineering subset, together with SHAP-based interpretation and a Fairlearn-based fairness audit**. None of these analyses appears in the preprint.

I acknowledge that IEEE's preprint policy lists permitted preprint locations (arXiv, TechRxiv, PSPB-approved non-profit servers, institutional repositories, personal websites) and that Research Square may not appear on the PSPB-approved list. The preprint is hosted on Research Square under CC BY 4.0; the author has full author rights to submit elsewhere and has not transferred copyright. Upon acceptance, the author will execute the IEEE Copyright Form per standard procedure.

I have submitted a separate brief inquiry to the TALE 2026 chairs regarding (a) the explicit PSPB status of Research Square and (b) the availability of virtual presentation. **If Research Square is not acceptable for IEEE TALE, please advise; I will withdraw this submission rather than migrate the preprint to another server, since the parent preprint is the canonical reference for the broader research portfolio.**

## Statements

- This is single-authored work. There are no co-authors.
- The manuscript has not been published in any peer-reviewed venue and is not under consideration elsewhere.
- The author declares one disclosed conflict of interest: the corpus includes one of the author's own previously published primary studies — Tokiwa, E. (2025), *Who excels in online learning in Japan?*, *Frontiers in Psychology*, 16, 1420996 (CC BY, open access; https://doi.org/10.3389/fpsyg.2025.1420996). Inclusion is flagged in the corpus, and the leave-one-study-out (LOSO) cross-validation design naturally handles per-study leakage in the ML pipeline.
- The author reports no competing financial interests and received no external funding for this work.

Sincerely,

**Eisuke Tokiwa**
SUNBLAZE Co., Ltd.
Tokyo, Japan
ORCID 0009-0009-7124-6669
sub.ashuman@gmail.com
