# ECEL 2026 — Excellence Awards Extended Abstract

**Award category nomination**: Excellence in Research(open), or Excellence in Practitioner Research, depending on which category the awards committee considers most appropriate. The author is a single-author independent researcher (founder, SUNBLAZE Co., Ltd.) pursuing a thesis-by-publication doctorate at Keio University SFC.

**Submission deadline**: 2026-05-14
**Submission contact**: helen@academic-conferences.org / info@academic-conferences.org

---

## Title

**Modality matters for Extraversion: A modality-stratified meta-regression of the Big Five personality traits and academic achievement in online learning environments.**

## Author

Eisuke Tokiwa, MEng — Founder, SUNBLAZE Co., Ltd., Tokyo, Japan
ORCID: 0009-0009-7124-6669 — eisuke.tokiwa@sunblaze.jp

## Why the work is excellent (≈400 words)

The personality–achievement literature has matured for face-to-face education over fifty years, with multiple landmark meta-analyses (Poropat 2009; Mammadov 2022; Meyer et al. 2023) establishing Conscientiousness as the dominant Big Five predictor. Yet for **online and digital learning environments — the central concern of ECEL — no quantitative meta-analytic synthesis dedicated to the modality has previously been published**. The author's parent systematic review and meta-analysis (Research Square preprint, DOI 10.21203/rs.3.rs-9513298/v1, deposited 27 April 2026; manuscript v2 currently under review at *Frontiers in Education*) fills that gap by pooling 12 primary studies (combined N = 3,384) and reporting two pre-registered moderator findings (Extraversion × Region, Extraversion × Outcome Type) that diverge sharply from the face-to-face literature. The present ECEL paper extends the parent work along the dimension that **ECEL's scope foregrounds first-order**: synchronous / asynchronous / blended modality.

What makes the present analysis worthy of nomination is the combination of three ingredients rarely brought together in the e-learning evidence-synthesis literature. **First**, methodological rigour: a PRISMA 2020-compliant flow, REML τ² estimation with Hartung–Knapp–Sidik–Jonkman small-k CI adjustment, and seven pre-registered sensitivity analyses time-stamped on OSF (DOI 10.17605/OSF.IO/E5W47, registered 23 April 2026 prior to formal extraction). **Second**, modality re-coding as a measurable methodological act: where the parent extraction left modality as "unspecified" for four primary-pool studies, each was re-checked against the original PDF and reclassified, with the override rules and citations published as code in the open repository so any reader can audit and rerun. **Third**, a transparent conflict-of-interest treatment: one of the corpus's twelve primary studies (Tokiwa, 2025, *Frontiers in Psychology* 16:1420996, CC BY) is the author's own published research; it is included with a flag and a no-op pre-registered sensitivity drop, and contributes zero extractable correlations to the pooled estimate (|Δr| < .001).

The headline empirical result is a **modality × Extraversion interaction** that is detectable only at the modality-stratified resolution: the Extraversion–achievement correlation is *negative* in fully-asynchronous online settings (pooled r = −0.121 [−0.246, 0.007], k = 6) and reverses to weakly positive in mixed-online settings (r = +0.059, k = 5); the joint Wald test on the four trait × modality interaction terms returns χ²(4) = 13.64, p = .0085. The trait-only pool reported Extraversion as null (r ≈ 0.002); the modality-stratified pool reveals that the null is the average of two opposing signals.

The practical implication for ECEL's audience is asymmetric. Conscientiousness pays off in both async and mixed-online settings (modality-stratified r = 0.180–0.190 across cells, Q_between p = .90); course designers cannot escape it. Extraversion, however, is a **silent disadvantage in async-only programmes** for which course design — not student selection — must compensate. The paper provides three concrete design recommendations and a reproducible code base under the project's OSF Registration.

## Reproducibility statement

All analytic code, derived datasets, intermediate result CSVs, and PRISMA flow diagrams for the present paper live under `metaanalysis/conference_submissions/ecel/` in the project repository. A cross-paper numeric consistency checker (`scripts/check_numbers.py`, 0 failures at 2026-05-10) and DOI resolution checker (`scripts/check_dois.py`, 0 syntax failures across 19 entries) accompany every submission. The corresponding parent meta-analysis manuscript v2 is currently under review at *Frontiers in Education* with a fully audited bibliographic record.

---

## Submission notes

- Format check: ECEL Awards typically expects 400–500 words extended-abstract format. Body above is ≈ 480 words.
- All numbers traceable to `results/modality_pools.csv`, `results/modality_qbetween.csv`, `results/interaction_terms.csv`. **Do not hand-edit**; rerun `scripts/run_modality_meta.py` if anything changes.
- Author review required before sending. Do not auto-send.
