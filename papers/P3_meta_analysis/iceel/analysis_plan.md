# ICEEL 2026 — East-Asia + Japan Synthesis: Analysis Plan

**Author**: Eisuke Tokiwa (single author)
**Venue**: ICEEL 2026, Tokyo, Japan
**Deadline**: Full Paper 2026-06-30
**Parent preprint**: doi:10.21203/rs.3.rs-9513298/v1
**Novel contribution (not in preprint)**:
1. Hofstede cultural-dimensions moderator analysis on the East-Asian subset.
2. A focused narrative synthesis of the Japanese online-learning context,
   drawing on A-25 Tokiwa 2025 and A-31 Rivers 2021 (the only Japan-based
   primary-pool studies in the corpus) plus regional context.

---

## 1. Research questions

- **RQ1.** Do the East-Asian primary-pool studies show a Big Five
  trait — achievement pattern that differs from the non-Asian subset?
  (Replicates and extends the region moderator already in the preprint
  by analysing the *Asian* subset internally rather than as a single bin.)
- **RQ2.** Are Hofstede's cultural dimensions (Power Distance,
  Individualism, Uncertainty Avoidance, Long-Term Orientation,
  Indulgence) systematically associated with the trait — achievement
  correlations across countries within the Asian subset?
- **RQ3.** What does the Japan-specific evidence (k = 2: A-25, A-31)
  add to a synthesis primarily drawn from China-dominant samples?

## 2. Data

- **Primary pool, Asian subset**: filter `region == "Asia"`. Currently
  contains A-25 Tokiwa (Japan), A-26 Wang (China; partial), A-28 Yu
  (China), A-31 Rivers (Japan), and from the secondary corpus A-07 Cohen,
  A-11 Cheng, A-12 Baruth, A-13 Dang, A-18 Bhagat, A-20 Mustafa, A-21
  Nakayama (most are exclude_from_primary, used here for narrative
  context and for cultural-dimension annotation).
- **Hofstede scores**: appended at the country level using the canonical
  Hofstede Insights / Minkov-revised scores. Sources documented in
  `inputs/hofstede_country_scores.csv` (to be added by the script).
- **Country-aggregated effect sizes**: where multiple studies share a
  country, aggregate via random-effects pooling first; the aggregated
  effect then feeds the Hofstede regression.

## 3. Methods

### 3.1. Asian-subset replication (Aim 1)

- Re-run the per-trait random-effects pool on the Asian subset.
- Report alongside the preprint's region-moderator table.

### 3.2. Hofstede meta-regression (Aim 2)

- Per trait, fit a meta-regression: Fisher's z = beta_0 + beta_1 *
  PowerDistance + beta_2 * Individualism + beta_3 * UncertaintyAvoidance
  + beta_4 * LongTermOrientation + beta_5 * Indulgence.
- Centre Hofstede dimensions at the corpus mean.
- With k <= 4 Asian studies per trait, restrict the regression to a
  *single* Hofstede dimension at a time; joint multi-dimension models are
  run only as illustrative diagnostics (clearly labelled as such).
- Report coefficient, SE, t / z, and p per dimension per trait.
- Acknowledge that the k is too small for stable meta-regression and
  treat results as hypothesis-generating.

### 3.3. Japan synthesis (Aim 3)

- Narrative comparison of A-25 Tokiwa (K-12, asynchronous, BFI-2-J) and
  A-31 Rivers (undergraduate, asynchronous, TIPI-J).
- Discuss instrument differences (60-item BFI-2-J vs 10-item TIPI-J)
  as a potential confound.
- Place Japan against the East-Asian backdrop (China-dominant studies)
  using the country-aggregated effects.

## 4. Outputs

| File | Content |
|------|---------|
| `results/asia_subset_pools.csv` | Per-trait pooled r within the Asian subset. |
| `results/hofstede_meta_regression.csv` | Coefficient table per trait per dimension. |
| `results/japan_synthesis.md` | Narrative table of Japan studies + comparison. |
| `inputs/hofstede_country_scores.csv` | Country -> dimension scores actually used. |

## 5. Pre-analysis declarations

- Hofstede meta-regression is exploratory; cell sizes are too small
  (k <= 4) for confirmatory inference.
- Hofstede scores are *country-level* — they do not represent
  individual-level cultural variation. This is a known limitation of the
  framework and is acknowledged.

## 6. Limitations

- k = 2 for Japan; meta-analytic pooling within Japan is not justifiable.
- Hofstede framework is criticised in modern cross-cultural psychology
  (see Minkov critiques); the paper engages with this and reports
  Minkov-revised dimensions where available.
- Single author, no second coder for cultural-dimension assignment
  (mitigated by sourcing scores from the Hofstede Insights canonical file
  rather than hand-coding).
