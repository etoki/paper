# ICEEL 2026 — Hofstede x East Asia synthesis

## Asia vs non-Asia contrast (Knapp-Hartung meta-regression)

Replaces the parent preprint's Q_between (E: Q = 46.43), which fed HKSJ subgroup SEs into a chi-square test.

| Trait | k Asia | r Asia | k non-Asia | r non-Asia | F (df) | p |
|-------|--------|--------|------------|------------|--------|---|
| O | 2 | +0.155 | 7 | +0.054 | 0.69 (1, 7) | 0.4350 |
| C | 2 | +0.110 | 8 | +0.187 | 0.58 (1, 8) | 0.4700 |
| E | 2 | -0.095 | 7 | +0.050 | 12.94 (1, 7) | 0.0088 |
| A | 2 | +0.319 | 7 | +0.038 | 9.13 (1, 7) | 0.0193 |
| N | 2 | +0.089 | 8 | -0.009 | 0.99 (1, 8) | 0.3480 |

## Asian-subset pooled correlations

| Trait | k | N | r [95% CI] | I^2 | tau^2 |
|-------|---|---|-----------|------|-------|
| O | 2 | 1301 | 0.155 [-0.988, 0.994] | 95.8% | 0.0887 |
| C | 2 | 1301 | 0.110 [-0.045, 0.260] | 0.0% | 0.0000 |
| E | 2 | 1301 | -0.095 [-0.541, 0.393] | 23.0% | 0.0012 |
| A | 2 | 1301 | 0.319 [-0.979, 0.994] | 95.4% | 0.0796 |
| N | 2 | 1301 | 0.089 [0.005, 0.171] | 0.0% | 0.0000 |

## Hofstede single-dimension meta-regressions (per trait)

Each cell is a weighted-OLS on Fisher z, slope coefficient + p-value.
All numbers are exploratory; k <= 4 within Asia means the regression is severely underpowered.

| Trait | Dimension | k | slope | SE | t | p | note |
|-------|-----------|---|------:|---:|---:|---:|------|
| O | PDI | 2 | +0.0165 | — | — | — | descriptive only (df_resid<1) |
| O | IDV | 2 | -0.0226 | — | — | — | descriptive only (df_resid<1) |
| O | MAS | 2 | -0.0148 | — | — | — | descriptive only (df_resid<1) |
| O | UAI | 2 | -0.0069 | — | — | — | descriptive only (df_resid<1) |
| O | LTO | 2 | -0.0187 | — | — | — | descriptive only (df_resid<1) |
| O | IND | 2 | -0.0239 | — | — | — | descriptive only (df_resid<1) |
| C | PDI | 2 | -0.0015 | — | — | — | descriptive only (df_resid<1) |
| C | IDV | 2 | +0.0020 | — | — | — | descriptive only (df_resid<1) |
| C | MAS | 2 | +0.0013 | — | — | — | descriptive only (df_resid<1) |
| C | UAI | 2 | +0.0006 | — | — | — | descriptive only (df_resid<1) |
| C | LTO | 2 | +0.0017 | — | — | — | descriptive only (df_resid<1) |
| C | IND | 2 | +0.0022 | — | — | — | descriptive only (df_resid<1) |
| E | PDI | 2 | +0.0039 | — | — | — | descriptive only (df_resid<1) |
| E | IDV | 2 | -0.0053 | — | — | — | descriptive only (df_resid<1) |
| E | MAS | 2 | -0.0035 | — | — | — | descriptive only (df_resid<1) |
| E | UAI | 2 | -0.0016 | — | — | — | descriptive only (df_resid<1) |
| E | LTO | 2 | -0.0044 | — | — | — | descriptive only (df_resid<1) |
| E | IND | 2 | -0.0056 | — | — | — | descriptive only (df_resid<1) |
| A | PDI | 2 | +0.0157 | — | — | — | descriptive only (df_resid<1) |
| A | IDV | 2 | -0.0215 | — | — | — | descriptive only (df_resid<1) |
| A | MAS | 2 | -0.0141 | — | — | — | descriptive only (df_resid<1) |
| A | UAI | 2 | -0.0066 | — | — | — | descriptive only (df_resid<1) |
| A | LTO | 2 | -0.0178 | — | — | — | descriptive only (df_resid<1) |
| A | IND | 2 | -0.0227 | — | — | — | descriptive only (df_resid<1) |
| N | PDI | 2 | -0.0008 | — | — | — | descriptive only (df_resid<1) |
| N | IDV | 2 | +0.0011 | — | — | — | descriptive only (df_resid<1) |
| N | MAS | 2 | +0.0007 | — | — | — | descriptive only (df_resid<1) |
| N | UAI | 2 | +0.0003 | — | — | — | descriptive only (df_resid<1) |
| N | LTO | 2 | +0.0009 | — | — | — | descriptive only (df_resid<1) |
| N | IND | 2 | +0.0012 | — | — | — | descriptive only (df_resid<1) |

## Japan focus (k = 2)

- **A-25 Tokiwa 2025**: N=103, modality=A, education=K-12
- **A-31 Rivers 2021**: N=149, modality=A, education=Undergraduate

See `japan_synthesis.md` for the side-by-side comparison.

## Caveats
- Asian primary-pool k is **2 per trait** (A-28 Yu China; A-31 Rivers Japan). Two further Asian studies in the qualitative synthesis (A-25 Tokiwa, A-26 Wang) do not contribute extractable Pearson r values.
- A 2-parameter regression with k=2 has df_resid = 0; slopes are reported as **descriptive only** (no SE / t / p).
- Country-level Hofstede scores are an *ecological* proxy; no individual-level cultural data are available in the corpus.
- The Hofstede meta-regression in this paper is therefore a *coverage demonstration* — it shows what the within-Asia structure looks like with the present evidence base, not a confirmatory test of cultural-dimension effects.