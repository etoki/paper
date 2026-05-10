# ICEEL 2026 — Hofstede x East Asia synthesis

## Asian-subset pooled correlations

| Trait | k | N | r [95% CI] | I^2 | tau^2 |
|-------|---|---|-----------|------|-------|
| O | 2 | 1301 | 0.164 [-0.989, 0.994] | 96.0% | 0.0449 |
| C | 2 | 1301 | 0.111 [-0.039, 0.257] | 0.0% | 0.0000 |
| E | 2 | 1301 | -0.131 [-0.314, 0.061] | 0.0% | 0.0000 |
| A | 2 | 1301 | 0.330 [-0.981, 0.995] | 95.6% | 0.0410 |
| N | 2 | 1301 | 0.089 [0.008, 0.169] | 0.0% | 0.0000 |

## Hofstede single-dimension meta-regressions (per trait)

Each cell is a weighted-OLS on Fisher z, slope coefficient + p-value.
All numbers are exploratory; k <= 4 within Asia means the regression is severely underpowered.

| Trait | Dimension | k | slope | SE | t | p | note |
|-------|-----------|---|------:|---:|---:|---:|------|
| O | PDI | 2 | +0.0168 | — | — | — | descriptive only (df_resid<1) |
| O | IDV | 2 | -0.0168 | — | — | — | descriptive only (df_resid<1) |
| O | MAS | 2 | -0.0151 | — | — | — | descriptive only (df_resid<1) |
| O | UAI | 2 | -0.0071 | — | — | — | descriptive only (df_resid<1) |
| O | LTO | 2 | -0.4372 | — | — | — | descriptive only (df_resid<1) |
| O | IND | 2 | -0.0243 | — | — | — | descriptive only (df_resid<1) |
| C | PDI | 2 | -0.0014 | — | — | — | descriptive only (df_resid<1) |
| C | IDV | 2 | +0.0014 | — | — | — | descriptive only (df_resid<1) |
| C | MAS | 2 | +0.0013 | — | — | — | descriptive only (df_resid<1) |
| C | UAI | 2 | +0.0006 | — | — | — | descriptive only (df_resid<1) |
| C | LTO | 2 | +0.0376 | — | — | — | descriptive only (df_resid<1) |
| C | IND | 2 | +0.0021 | — | — | — | descriptive only (df_resid<1) |
| E | PDI | 2 | +0.0018 | — | — | — | descriptive only (df_resid<1) |
| E | IDV | 2 | -0.0018 | — | — | — | descriptive only (df_resid<1) |
| E | MAS | 2 | -0.0017 | — | — | — | descriptive only (df_resid<1) |
| E | UAI | 2 | -0.0008 | — | — | — | descriptive only (df_resid<1) |
| E | LTO | 2 | -0.0481 | — | — | — | descriptive only (df_resid<1) |
| E | IND | 2 | -0.0027 | — | — | — | descriptive only (df_resid<1) |
| A | PDI | 2 | +0.0162 | — | — | — | descriptive only (df_resid<1) |
| A | IDV | 2 | -0.0162 | — | — | — | descriptive only (df_resid<1) |
| A | MAS | 2 | -0.0145 | — | — | — | descriptive only (df_resid<1) |
| A | UAI | 2 | -0.0068 | — | — | — | descriptive only (df_resid<1) |
| A | LTO | 2 | -0.4201 | — | — | — | descriptive only (df_resid<1) |
| A | IND | 2 | -0.0233 | — | — | — | descriptive only (df_resid<1) |
| N | PDI | 2 | -0.0008 | — | — | — | descriptive only (df_resid<1) |
| N | IDV | 2 | +0.0008 | — | — | — | descriptive only (df_resid<1) |
| N | MAS | 2 | +0.0007 | — | — | — | descriptive only (df_resid<1) |
| N | UAI | 2 | +0.0003 | — | — | — | descriptive only (df_resid<1) |
| N | LTO | 2 | +0.0202 | — | — | — | descriptive only (df_resid<1) |
| N | IND | 2 | +0.0011 | — | — | — | descriptive only (df_resid<1) |

## Japan focus (k = 2)

- **A-25 Tokiwa 2025**: N=103, modality=A, education=K-12
- **A-31 Rivers 2021**: N=149, modality=A, education=Undergraduate

See `japan_synthesis.md` for the side-by-side comparison.

## Caveats
- Asian primary-pool k is **2 per trait** (A-28 Yu China; A-31 Rivers Japan). Two further Asian studies in the qualitative synthesis (A-25 Tokiwa, A-26 Wang) do not contribute extractable Pearson r values.
- A 2-parameter regression with k=2 has df_resid = 0; slopes are reported as **descriptive only** (no SE / t / p).
- Country-level Hofstede scores are an *ecological* proxy; no individual-level cultural data are available in the corpus.
- The Hofstede meta-regression in this paper is therefore a *coverage demonstration* — it shows what the within-Asia structure looks like with the present evidence base, not a confirmatory test of cultural-dimension effects.