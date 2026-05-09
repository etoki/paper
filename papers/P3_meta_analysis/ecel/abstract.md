# ECEL 2026 — Abstract (numbers filled from results CSVs)

**Title**:
*Modality matters for Extraversion: A modality-stratified meta-regression
of the Big Five personality traits and academic achievement in online
learning environments.*

**Author**: Eisuke Tokiwa (SUNBLAZE Co., Ltd.; ORCID 0009-0009-7124-6669)

**Word count target**: 250–300 words.

**Numerical sources** (do NOT hand-edit; rerun
`scripts/run_modality_meta.py`):
- `results/modality_pools.csv`
- `results/modality_qbetween.csv`
- `results/interaction_terms.csv`
- `metaanalysis/analysis/CANONICAL_RESULTS.md` (preprint replication numbers)

---

## Draft (numbers filled 2026-05-09 after modality re-classification)

**Background.** A recent systematic review and meta-analysis by the
present author (Research Square preprint, DOI
10.21203/rs.3.rs-9513298/v1) pooled k = 9–10 primary studies per trait
across online-learning settings. Modality was pre-registered as a
moderator but was **not quantitatively executed in the preprint** because
no single modality cell met the k >= 10 per-level rule. The present paper
fills that gap.

**Aims.** Two analyses not reported in the preprint: (1) a
modality-stratified random-effects model contrasting asynchronous (A,
k = 3 — 4) and mixed-online (M, k = 5) instruction (synchronous evidence,
k = 1, reported narratively), with the unspecified-modality bucket from
the master extraction reclassified to A or M based on PDF re-reading; and
(2) a long-format weighted modality x trait interaction model on Fisher
z-transformed correlations.

**Methods.** Modality codes were derived from each study's reported
delivery format and re-checked against the original PDFs for the four
studies whose preprint extraction was blank (A-15 Elvers, A-23 Rodrigues,
A-26 Wang, A-30 Kaspar; reclassification rules in
`inputs/derive_studies_csv.py::MODALITY_OVERRIDES`). Random-effects models
used REML with HKSJ-adjusted CIs. The interaction model uses
weights = 1 / (v + tau-squared), with tau-squared set to the median of
per-trait REML estimates.

**Results.** Modality moderates the trait–achievement correlation in a
**trait-specific** way. Q_between is highly significant for Extraversion
(Q = 17.60, df = 1, p < .001), trend-level for Agreeableness (Q = 3.28,
p = .070), and not significant for Conscientiousness (p = .90),
Neuroticism (p = .52), or Openness (p = .16). In the asynchronous bucket
**Extraversion shows a negative pooled r = -0.121 [-0.246, 0.007]**,
reversing in mixed-online to r = +0.059 [-0.027, 0.145]; the
Extraversion x Mixed interaction term is +0.385 Fisher z, p = .009. The
joint Wald test on all four interaction terms is **chi-squared(4) = 13.64,
p = .0085**.

**Implications.** Modality does not change the bottom line for
Conscientiousness — its modality-stratified pooled r is 0.180–0.190 across
A and M cells — but materially changes the picture for Extraversion, a
trait that the trait-only pool reported as null (r = .002). For online
course designers the practical message is asymmetric: support
self-regulation regardless of synchrony, but expect Extraversion to be a
silent disadvantage in async-only programmes.

**Keywords**: meta-analysis, Big Five, online learning, modality,
synchronous, asynchronous, Extraversion, academic achievement.

---

## Submission notes

- Disclose preprint per `templates/preprint_disclosure_template.md` (ECEL
  block). Disclosure must say explicitly that this paper "completes the
  modality moderator that was pre-registered but not quantitatively
  executed in the preprint due to the k>=10 per-level rule" — see
  `papers/P3_meta_analysis/preprint_audit.md` section 3.
- Single-author submission; ORCID required.
- All numbers traceable to CSVs in `results/`. If a number changes after a
  rerun, update this abstract by re-running this section's substitution
  pass — do **not** hand-edit numbers.
- Author review required before submission. **Do not auto-send.**
