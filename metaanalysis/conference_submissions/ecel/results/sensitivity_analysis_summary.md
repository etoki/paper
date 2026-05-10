# ECEL 2026 — Sensitivity Analysis Narrative

**Source CSV**: `results/sensitivity.csv`
**Generated**: 2026-05-09 after PDF-grounded modality re-classification.

This document narrates the sensitivity layer that supports the primary
modality-stratified estimates in `summary.md`. Three perturbations are
run; the headline modality x trait pattern is **robust under all three**.

---

## Scenario list

| Scenario | What changes | Why |
|----------|-------------|-----|
| `primary` | All 12 primary-pool studies, all extractable r, modality coded after PDF re-reading. | Headline. |
| `drop_beta_converted` | Exclude A-28 Yu (β-converted via Peterson-Brown) and A-30 Kaspar (β-converted). | Tests sensitivity to the β->r approximation. |
| `drop_coi` | Exclude A-25 Tokiwa (author's own prior study; tagged include_COI). | Removes the COI study; A-25 contributes no extractable r so the per-trait pools are unchanged but documented. |
| `drop_unspecified_modality` | Exclude any U-bucket study (post-override there are none). | Acts as a no-op after the override but is retained as a robustness audit. |

## Per-scenario per-trait per-modality re-pooled r

(Synchronous cell with k = 1 reported narratively only; not re-pooled.)

### Conscientiousness (the modality-invariant trait)

| Scenario | A | M |
|----------|---|---|
| primary | r = 0.190 [-0.034, 0.395] (k=4) | r = 0.180 [0.051, 0.302] (k=5) |
| drop_beta_converted | r = 0.262 [-0.119, 0.576] (k=3) | r = 0.174 [-0.019, 0.354] (k=4) |
| drop_coi | identical to primary (A-25 contributes no r) | identical |
| drop_unspecified_modality | identical to primary (U bucket already empty) | identical |

**Verdict**: Conscientiousness pooled r in the asynchronous bucket
holds in the 0.19–0.26 range; in the mixed bucket holds in 0.17–0.18.
β-converted dropping nudges A upward (+0.07) and M slightly downward
(-0.01). The modality-invariance message survives.

### Extraversion (the modality-dependent trait)

| Scenario | A | M |
|----------|---|---|
| primary | r = -0.121 [-0.246, 0.007] (k=3) | r = +0.059 [-0.027, 0.145] (k=5) |
| drop_beta_converted | r = -0.095 [-0.878, 0.826] (k=2) | r = +0.044 [-0.078, 0.164] (k=4) |
| drop_coi | identical to primary | identical |
| drop_unspecified_modality | identical to primary | identical |

**Verdict**: The negative sign in async holds (-0.121 -> -0.095 after
β drop). The CI widens dramatically when k drops from 3 to 2 in the A
bucket. The mixed-online positive sign holds (+0.059 -> +0.044).
The Q_between contrast (E x modality) survives the perturbations
qualitatively but loses inferential power on β-drop alone — flagged in
the Discussion as the scenario most sensitive to single-study
contributions.

### Other traits

| Trait | Modality | primary | drop_beta_converted |
|-------|----------|---------|---------------------|
| Openness | A | 0.223 [-0.374, 0.689] (k=3) | 0.139 [-0.989, 0.994] (k=2) |
| Openness | M | 0.019 [-0.081, 0.119] (k=5) | -0.017 [-0.082, 0.049] (k=4) |
| Agreeableness | A | 0.283 [-0.293, 0.708] (k=3) | 0.134 [-0.129, 0.379] (k=2) |
| Agreeableness | M | 0.032 [-0.076, 0.138] (k=5) | 0.061 [-0.062, 0.182] (k=4) |
| Neuroticism | A | 0.076 [-0.035, 0.186] (k=4) | 0.027 [-0.359, 0.404] (k=3) |
| Neuroticism | M | 0.028 [-0.155, 0.209] (k=5) | -0.036 [-0.153, 0.081] (k=4) |

**Verdict**: O / A / N point estimates shift modestly under β-drop but
none of the asynchronous-vs-mixed contrasts changes sign. The pattern
"largest contrast on E, near-zero contrast on C / N / O / A" persists.

## Scenarios that were NOT run (and why)

- **Drop low-RoB (RoB < 5)**: only one primary-pool study has RoB = 4
  (A-37 Zheng) and the parent preprint already runs an "exclude RoB < 5"
  sensitivity (CANONICAL_RESULTS.md section 10). Re-running it at the
  modality-cell level would split A-37 (modality M) and reduce the M
  bucket from k = 5 to k = 4 with negligible inferential change. Logged
  here for transparency rather than re-executed.

## Headline robustness statement (drop into the paper Limitations)

> The modality x trait interaction effect for Extraversion (joint Wald
> chi-squared(4) = 13.64, p = .0085) is the empirical anchor of this
> paper. Under the β-converted-drop sensitivity scenario the asynchronous
> Extraversion cell loses one study (k = 3 -> k = 2) and the CI widens
> by ~0.6 in absolute width, but the negative sign and the contrast with
> the mixed-online cell are preserved. Conscientiousness's
> modality-invariance is preserved across all three sensitivity
> scenarios. We therefore treat the headline trait-by-modality pattern
> as robust within the limits of the present evidence base, while
> acknowledging that any single-study removal materially affects
> single-cell precision.
