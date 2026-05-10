# ICEEL 2026 — Full Paper Outline

**Target length**: ~6 pages, ICEEL conference template

---

## 1. Introduction (~0.5 page)

- Within-Asia heterogeneity of personality — achievement evidence is
  under-explored. East-Asian samples dominate the regional bin in most
  meta-analyses but China-vs-Japan-vs-Korea differences are rarely
  modelled.
- Stated novel contribution: Hofstede meta-regression + Japan focus —
  not in the parent preprint.

## 2. Related Work (~0.5 page)

- Cross-cultural personality — academic achievement (Mammadov 2022; Chen
  et al. 2025).
- Hofstede's framework and its critiques (Minkov-Hofstede 2014).
- Japanese online learning context (Rivers 2021; Tokiwa 2025).

## 3. Method (~1 page)

- Asian-subset filter (region == "Asia").
- Country-level aggregation when k > 1 within country.
- Hofstede dimension table appended from `inputs/hofstede_country_scores.csv`.
- Single-dimension meta-regression per trait per dimension.

## 4. Results (~2 pages)

### 4.1. Asian-subset pooled effects (Table 1)

- Per-trait k, N, r, 95 % CI, I-squared within Asia.

### 4.2. Hofstede meta-regression (Table 2)

- Coefficient, SE, p per (trait, dimension) pair.
- Highlight cells with the lowest p — flagged as *exploratory*.

### 4.3. Japan synthesis (Table 3 + narrative)

- A-25 Tokiwa 2025 vs A-31 Rivers 2021: side-by-side on N, modality,
  instrument, outcome, r per trait.
- Narrative on the role of LMS choice (StudySapuri vs Moodle) in the
  Japanese ecosystem.

## 5. Discussion (~1 page)

- Region-as-Asia is too coarse; within-Asia variation matters.
- Hofstede dimensions are explanatory hooks only — k is too small to
  carry confirmatory weight.
- Practical implication for Japanese ed-tech designers: do not personalise
  on Big Five at scale on the current evidence.

## 6. Limitations (~0.5 page)

- k = 4 Asian primary-pool studies; k = 2 in Japan.
- Hofstede is country-level; ecological-fallacy risk.
- Single coder, single author.

## 7. Conclusion (~0.25 page)

- Cultural moderators warrant explicit modelling, but only with
  considerably larger evidence bases than the current corpus offers.

---

## Tables / figures backed by CSVs

- Table 1: `results/asia_subset_pools.csv`
- Table 2: `results/hofstede_meta_regression.csv`
- Table 3: `results/japan_synthesis.md` (narrative, with linked CSV cells)
