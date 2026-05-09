# ECEL 2026 — Full Paper Outline

**Target length**: ~10 pages, ACI Word template
**Status**: Outline only; numerical placeholders are filled after running
`scripts/run_modality_meta.py`.

---

## 1. Introduction (~1 page)

- Why Big Five x online learning is unsettled despite four decades of work.
- The preprint's key finding: only Conscientiousness has a robust pooled
  positive r with achievement; the rest are null or weak.
- Why modality matters in principle:
  - Synchronous instruction reintroduces social cues, so Extraversion and
    Agreeableness should re-acquire predictive value.
  - Asynchronous instruction shifts the load to self-regulation, so
    Conscientiousness should dominate.
  - Mixed-online and blended instruction *should* sit in between but
    generates the most heterogeneous outcomes empirically.
- Stated novel contribution (not in preprint): modality-stratified
  meta-regression + modality x trait interaction.

## 2. Method (~2 pages)

- Inheritance from the parent preprint: PRISMA 2020 flow, eligibility
  criteria, search log, RoB scoring.
- New step: modality classification (S / A / M / B / U), with the script
  `inputs/derive_studies_csv.py` reproduced as Listing 1.
- Statistical model:
  - REML + HKSJ; Fisher's z; 95 % CI and 95 % PI.
  - Modality moderator via `pool_by_subgroup` extension.
  - Modality x trait interaction via long-format mixed-effects meta-regression.
- Sensitivity layer (4 scenarios).

## 3. Results (~3 pages)

### 3.1. Replication of preprint (Table 1)

- Per-trait pooled r (95 % CI, 95 % PI) reproduced from
  `pooling_results.csv` to demonstrate identity with parent work.

### 3.2. Modality-stratified pools (Table 2)

- Per-modality x per-trait pooled r with k, N, I-squared, tau-squared.
- Highlight cells with k = 1 (S bucket) reported narratively only.

### 3.3. Modality x trait interaction (Table 3 + Figure 1)

- Interaction coefficients from the long-format model.
- Forest plot per trait grouped by modality (Figure 1).

### 3.4. Sensitivity (Table 4)

- Re-runs under: (i) drop beta-converted; (ii) drop COI; (iii) drop low-RoB;
  (iv) drop unspecified-modality bucket.

## 4. Discussion (~2 pages)

The headline empirical result is that **modality matters for Extraversion
but not for Conscientiousness**. After running the script:

- Q_between is highly significant for Extraversion (Q = 15.52, p < .001),
  Neuroticism (Q = 12.24, p = .002), and Agreeableness (Q = 9.11,
  p = .011), but not for Conscientiousness (Q = 0.85, p = .65) or
  Openness (Q = 4.55, p = .10).
- The asynchronous Extraversion pool is *negative* (r = -0.121
  [-0.246, 0.007]), the mixed-online pool is *weakly positive*
  (r = +0.067), and the single synchronous study (A-29 Bahcekapili 2020)
  reports r = +0.027.
- The interaction Wald test on the long-format model is
  chi-squared(8) = 14.27, p = .075 — trend-level evidence consistent
  with the per-trait Q_between picture.

Discussion subsections:

- **Conscientiousness is robust across modality.** All four modality
  cells produce r in the 0.07–0.22 band; no Q_between detection.
  The preprint's headline holds at finer resolution.
- **Extraversion is modality-dependent.** The negative async cell and
  the positive mixed/unspecified cells are theoretically consistent with
  the social-presence-of-extraverts hypothesis: extraverts thrive when
  there *is* a social channel and lose ground when there is not.
- **Synchronous evidence (k = 1)**: A-29 Bahcekapili 2020 provides the
  only synchronous data point. This sets a hypothesis for replication
  rather than a confirmatory result.
- **Blended (k = 0 primary)**: the field needs *more reporting* of
  modality. This is itself a finding.
- Theoretical bridge to self-regulation models (Zimmerman, Pintrich) —
  why C dominates async — is now bolstered by a *negative* result for E
  in async, exactly the signal those models predict.

## 5. Limitations (~0.5 page)

- Modality coding is post-hoc; another coder is not available
  (single-author study). Audit trail is the script.
- k per modality cell is 1–4 for the larger modalities; null findings can
  not be interpreted as evidence of absence.
- The COVID confound (modality and era are partly entangled): explored but
  not fully disentangled in this corpus.

## 6. Conclusion (~0.5 page)

- A modality lens **partially** changes the picture: the
  Conscientiousness story is unchanged, but Extraversion becomes a
  modality-dependent predictor (negative in async, positive in
  mixed-online).
- The dominant message remains: **report modality**. Future syntheses
  should refuse to extract studies that fail to code synchrony.
- Practical recommendation for course designers: in async-dominant
  programmes, Conscientiousness is the trait with the cleanest empirical
  link to achievement, and Extraversion is *not* a useful predictor —
  in fact it is a mild negative; design interventions for self-regulation
  rather than for personality "fit".

## 7. Acknowledgments

- Single-author, self-funded.
- Disclosure of preprint (DOI 10.21203/rs.3.rs-9513298/v1) per ECEL block in
  `templates/preprint_disclosure_template.md`.
- ORCID 0009-0009-7124-6669.

## 8. References

- Inherited from `metaanalysis/paper_v2/references_data.py` (regenerate
  with the build pipeline).
- Plus modality-related additions:
  - Hrastinski (2008) *Asynchronous and synchronous e-learning*.
  - Roblyer & Marshall (2003) on online vs face-to-face equivalence.
  - Means et al. (2014) blended-learning evaluation.
  - Pintrich (2004) self-regulation framework.

---

## Tables / figures backed by CSVs

- Table 1: `metaanalysis/analysis/pooling_results.csv` (preprint replication)
- Table 2: `papers/P3_meta_analysis/ecel/results/modality_pools.csv`
- Table 3: `papers/P3_meta_analysis/ecel/results/interaction_terms.csv`
- Table 4: `papers/P3_meta_analysis/ecel/results/sensitivity.csv`
- Figure 1: `papers/P3_meta_analysis/ecel/results/forestplot_<trait>.png`

If a table or figure cannot be backed by a generated CSV, it does not go
into the manuscript.
