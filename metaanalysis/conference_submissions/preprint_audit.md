# Preprint Audit (2026-05-09)

**Goal:** confirm that each of the four conference sub-paper "novel
analyses" is genuinely absent from the Research Square preprint
(DOI 10.21203/rs.3.rs-9513298/v1, deposited 2026-04-27), so each
conference submission is a defensible non-self-plagiarising contribution
under Strategy 2 (preprint maintained, sub-papers extended).

**Audit method:** keyword search of the preprint full-text PDF using
`pdfminer.six`, plus structural read of the `Moderator Analyses` section
and the abstract.

---

## 1. What the preprint actually runs

From the preprint Methods/Results:

> "Three pre-registered moderators met the minimum k-per-level requirement
> of the present synthesis and were analyzed quantitatively via subgroup
> random-effects meta-analyses: **region (Asia vs. non-Asia)**, **era
> (pre-COVID vs. COVID-era)**, and **outcome type (objective vs. self-
> reported achievement)**. The remaining six pre-registered moderators
> (personality instrument, publication year, log-sample size,
> risk-of-bias score, **modality, education level**) did not meet the
> k ≥ 10 per predictor level requirement, and are reported narratively
> below and in the Methods Deviations subsection."

So the preprint:

- **executes** Region, Era, Outcome-type moderators
- **registers but explicitly does not execute** Modality, Education-level,
  Instrument, Pub-year, log-N, RoB
- the abstract reports Conscientiousness pooled r = .167 [.089, .243];
  Extraversion x Region p < .001 and Extraversion x Outcome-type p < .001
  as the two highly-significant moderator effects

## 2. Coverage table for the 4 venues

| Conference | Novel analysis claimed | preprint coverage | Verdict |
|------------|----------------------|-------------------|---------|
| **ECEL 2026** | Modality-stratified meta-regression (sync / async / blended / mixed / unspec) + modality x trait interaction | Modality is **pre-registered** but **not executed** (preprint section "Methods Deviations" says k<10 per level). Modality is discussed 21 times in narrative but no quantitative moderator table. | **Distinct contribution.** The ECEL paper completes the registered-but-unrun modality moderator and adds an interaction model. Disclosure must say so explicitly (do not pretend modality is unregistered). |
| **IEEE TALE 2026** | Engineering / STEM substratum + study-level ML predictive layer (LR / RF / XGB) + SHAP + fairlearn | "STEM" appears 1 time as a domain-specificity reference (citing Stadler 2024). No ML, no SHAP, no fairness analysis. No engineering-only subset. | **Novel.** Both the STEM substratum and the ML / interpretability / fairness layer are absent. |
| **ICEEL 2026** | Hofstede cultural-dimensions moderator on the East-Asian subset + Japan focus | Region (Asia vs non-Asia) is run, with significant Extraversion effect (Q_b = 46.43, p < .001). **Hofstede appears 0 times. East-Asia (within-Asia) appears 0 times. Japan-specific synthesis appears 0 times** beyond inclusion of A-25 Tokiwa and A-31 Rivers in the primary pool. | **Novel.** Within-Asia structure (Hofstede dimensions, country-level meta-regression) is genuinely new. The 2-level region moderator from the preprint is acknowledged as the *parent* contrast; ICEEL paper goes deeper. |
| **ICERI 2026** | 3 (education level) x 3 (discipline) cross-tabulated meta-analytic interaction | Education-level pre-registered but not executed (k constraint). Discipline / domain not pre-registered. **Cross-tab appears 0 times.** | **Novel.** Education-level partially overlaps with preprint's pre-registration (so disclosure must mention this), but discipline and the cross-tab interaction are entirely new. |

## 3. Disclosure language adjustments

These additions go into `templates/preprint_disclosure_template.md`
per-venue blocks (already drafted; tweak per the verdicts above):

- **ECEL**: add the phrase "completes the modality moderator that was
  pre-registered but not quantitatively executed in the preprint due to
  the k>=10 per-level rule". This is honest and sidesteps reviewer pushback
  about hidden pre-registration.
- **IEEE TALE**: state plainly that the STEM substratum and the ML layer
  are not in the preprint.
- **ICEEL**: acknowledge that the binary Asia/non-Asia contrast IS in the
  preprint, then position the Hofstede meta-regression as a *within-Asia*
  decomposition that the preprint did not run.
- **ICERI**: acknowledge that education-level was pre-registered but not
  run; position the cross-tab interaction as novel.

## 4. Modality re-classification of unspecified-bucket studies

The preprint records `modality_subtype = ""` for 4 of the 12 primary-pool
studies. PDF re-reading (2026-05-09) yields:

| Study | preprint code | Evidence in source PDF | New code |
|-------|---------------|------------------------|----------|
| **A-15 Elvers 2003** | (blank) | "Web-based class" with audio lectures, graphics, videos, activities; "students in the online class came to class only to take tests"; Web server logged date-stamped Web page access (Methods, p. 160). Pure self-paced LMS. | **A** (asynchronous) |
| **A-23 Rodrigues 2024** | (blank) | Discussion p. 380: "home study was also partly asynchronous, self-discipline and organization, partial characteristics of the conscientiousness factor, also played a major role". Methods describe "online-based formats that were used from home" with materials "accessed and worked through at any time" — but "home study" in COVID-era German universities included synchronous Zoom lectures (cited literature within the paper). | **M** (mixed-online) |
| **A-26 Wang 2023** | (blank) | China K-12 post-COVID 2023; measurement scale assesses "network platform usage, school management and services, **teacher teaching**, and learning task arrangement"; the "teacher teaching" sub-scale and the post-COVID school-day context strongly imply synchronous live classes alongside async tasks. | **M** (mixed-online) |
| **A-30 Kaspar 2023** | (blank) | German university COVID-2021; cites "synchronous online learning at the start of the Covid-19 pandemic (Besser et al.)" and "Zooming Their Way through University" — strong synchronous component plus asynchronous materials. | **M** (mixed-online) |

These overrides are now hard-coded in
`metaanalysis/conference_submissions/inputs/derive_studies_csv.py::MODALITY_OVERRIDES`
with the citations above as comments.

## 5. Resulting modality cell sizes

After overrides (per-trait, primary pool, k = 10 for C / N; k = 9 for O / E / A):

- **A** (asynchronous): A-01, A-25 (no r), A-28, A-31, **A-15** (was U) — k = 4 for C and N (A-15 contributes only those two), k = 3 for O / E / A
- **M** (mixed-online): A-02, A-22, A-37, **A-23, A-26 (no r), A-30** (was U), k = 5 for C and N, k = 5 for O / E / A
  - (A-26 has primary_achievement=partial but no extractable r, so doesn't enter the trait pool numerically; still affects narrative count)
- **S** (synchronous): A-29, k = 1 — narrative only
- **U** (unspecified) after overrides: empty
- **B** (blended): empty (no primary-pool study reports a primary blended condition)

This change collapses the "U" bucket entirely and shifts the analysis from
"4 cells (A/M/S/U)" to "3 cells (A/M/S) + 1 cell (S) is narrative", which
substantially tightens the modality-stratified estimates.
