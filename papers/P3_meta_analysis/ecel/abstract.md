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

## Draft (numbers filled 2026-05-09)

**Background.** A recent systematic review and meta-analysis by the
present author (Research Square preprint, DOI
10.21203/rs.3.rs-9513298/v1) pooled k = 9–10 primary studies per trait
across online-learning settings and reported that only Conscientiousness
retained a robust positive association with academic achievement
(r = 0.167, 95 % CI [0.089, 0.243]); the other four Big Five traits were
null or directionally inconsistent. Whether these pooled estimates mask
modality-dependent variation has not been tested.

**Aims.** This paper extends the parent meta-analysis with two analyses
that are NOT reported in the preprint: (1) a modality-stratified
random-effects model contrasting asynchronous (A), mixed-online (M), and
unspecified (U) instruction (synchronous studies, k = 1, are reported
narratively); and (2) a long-format weighted modality x trait
interaction model.

**Methods.** Modality codes were derived from each study's reported
delivery format and platform; ambiguous cases were resolved by
transparent rules in `inputs/derive_studies_csv.py`. Random-effects
models used REML with HKSJ-adjusted CIs. The interaction model is fit
on Fisher z-transformed correlations with weights = 1 / (v + tau-squared).

**Results.** Modality moderates the trait — achievement correlation in a
trait-specific way. The Q_between contrast across modality levels is
significant for Extraversion (Q = 15.52, df = 2, p < .001),
Neuroticism (Q = 12.24, df = 2, p = .002), and Agreeableness
(Q = 9.11, df = 2, p = .011), but not for Conscientiousness (p = .65)
or Openness (p = .10). In the asynchronous bucket Extraversion shows a
negative pooled r = -0.121 [-0.246, 0.007], reversing in mixed-online
settings (r = +0.067). The modality x trait interaction Wald test is
chi-squared(8) = 14.27, p = .075.

**Implications.** Modality does not change the bottom line for
Conscientiousness, but it materially changes the picture for Extraversion
— a trait that the trait-only pool reported as null (r = 0.002). Future
syntheses should refuse to extract studies that fail to code synchrony.

**Keywords**: meta-analysis, Big Five, online learning, modality,
synchronous, asynchronous, Extraversion, academic achievement.

---

## Submission notes

- Disclose preprint per `templates/preprint_disclosure_template.md`
  (ECEL block).
- Single-author submission; ORCID required.
- All numbers traceable to CSVs in `results/`. If a number changes after a
  rerun, update this abstract by re-running this section's substitution
  pass — do **not** hand-edit numbers.
- Author review required before submission. **Do not auto-send.**
