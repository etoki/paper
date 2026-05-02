# Canonical Results — harassment/

**Purpose**: Single source of truth for all numbers reported in the
manuscript. Generated from `analysis.py` against `raw.csv`, plus
`alfa/hexaco_alpha_results.csv` (psychometric alphas).

All values below were directly read from `res/*.csv` after
re-running `python3 analysis.py` and confirming reproducibility
(only float-rounding diffs from prior commit).

---

## 1. Sample

| Field | Value |
|-------|-------|
| Recruited (responses) | 380 (manuscript text only) |
| Final analytic N | **354** |
| Men (gender = 0) | **134** |
| Women (gender = 1) | **220** |
| Age 10s decade (manuscript "18–19") | 32 |
| Age 20s | 100 |
| Age 30s | 124 |
| Age 40s | 70 |
| Age ≥ 50 | 28 (50s = 22, 60s = 6) |
| Mach/Narc/Psy missing rate | 1/354 = **0.28 %** |
| Listwise N for regression | **353** (134 → **133** men, 220 women) |

**NOTE**: `raw.csv` codes `age` in decade bins (10, 20, 30, 40, 50, 60),
not exact age. The minimum age value is 10. The manuscript's claim
"32 participants aged 18–19" is technically the 10s bin; this should
probably be relabelled "in their teens (under 20)" but is not a
fabrication.

## 2. Descriptive Statistics (Table 1)

From `res/descriptive_statistics.csv`:

| Variable | N | M | SD | min | max | sk | ku |
|----------|---|---|----|-----|-----|----|----|
| Honesty-Humility | 354 | 3.25 | 0.59 | 1.5 | 5.0 | -0.18 | 0.42 |
| Emotionality | 354 | 3.30 | 0.78 | 1.1 | 5.0 | -0.39 | 0.01 |
| Extraversion | 354 | 2.70 | 0.84 | 1.0 | 4.8 | 0.25 | -0.40 |
| Agreeableness | 354 | 2.88 | 0.73 | 1.0 | 4.8 | -0.11 | -0.07 |
| Conscientiousness | 354 | 3.21 | 0.68 | 1.3 | 4.9 | 0.15 | -0.10 |
| Openness | 354 | 3.46 | 0.80 | 1.4 | 5.0 | -0.11 | -0.62 |
| Machiavellianism | 353 | 3.58 | 0.66 | 1.1 | 4.9 | -0.46 | 0.24 |
| Narcissism | 353 | 2.40 | 0.72 | 1.0 | 4.9 | 0.34 | -0.06 |
| Psychopathy | 353 | 2.50 | 0.66 | 1.1 | 4.4 | 0.13 | -0.40 |
| Power Harassment | 354 | 1.26 | 0.32 | 1.0 | 2.5 | 1.81 | 3.04 |
| Gender Harassment | 354 | 1.70 | 0.70 | 1.0 | 4.2 | 1.14 | 0.65 |
| age | 354 | 29.10 | 11.13 | 10.0 | 60.0 | 0.35 | -0.09 |
| gender | 354 | 0.62 | 0.49 | 0.0 | 1.0 | -0.50 | -1.76 |

## 3. Cronbach's α (manuscript Methods → Measurement Instruments)

The harassment study `raw.csv` contains only domain-level scores, so
α cannot be reproduced from it. The reported values match
`alfa/hexaco_alpha_results.csv` (combined N = 4168 from three
psychometric files), **except Honesty–Humility** (see § Findings):

| Domain | Manuscript | alfa script (combined N=4168) |
|--------|-----------|-------------------------------|
| Honesty–Humility | .671 | **.571** ❌ DISCREPANCY |
| Emotionality | .830 | .830 ✓ |
| Extraversion | .621 | .621 ✓ |
| Agreeableness | .783 | .783 ✓ |
| Conscientiousness | .815 | .815 ✓ |
| Openness | .804 | .804 ✓ |

The α values are *not* computed on the present harassment N=354
sample (the script combines three sources totaling 4168), but the
manuscript states "In the present sample, internal consistency
coefficients were…". This wording is misleading.

For SD3-J (Methods), the manuscript reports α = .767 / .778 / .708
and total .842. These are not reproducible from `raw.csv` alone
(item-level data is in `alfa/raw_sd3j.csv`).

## 4. Spearman correlations (Table 2)

Authoritative: `res/spearman_corr_table.csv`. Selected key values:

| Pair | ρ | sig |
|------|---|-----|
| H-H × Mach | -0.32 | *** |
| H-H × Narc | -0.36 | *** |
| H-H × Psy | -0.38 | *** |
| H-H × Power | -0.23 | *** |
| H-H × Gender | -0.18 | *** |
| Agr × Mach | -0.27 | *** |
| Agr × Psy | -0.36 | *** |
| Agr × Power | -0.26 | *** |
| Ext × Narc | 0.56 | *** |
| Ext × Psy | 0.16 | ** |
| Ext × Gender | 0.11 | * |
| Power × Gender | 0.35 | *** |

## 5. Regression Model Fit (Table 4)

From `res/model_fit_incremental.csv`:

| Outcome | n | R²(A) | R²(B) | ΔR² | F-change | p-change | R²(B_sens) | R²(C) | DW(B) | Max Cook D | n>4/n |
|---------|---|-------|-------|-----|----------|----------|-----------|-------|-------|------------|-------|
| Power harassment | 353 | **0.166** | **0.198** | **0.032** | **2.280** | **0.036** | **0.221** | **0.218** | 1.95 | 0.058 | 23 |
| Gender harassment | 353 | **0.117** | **0.213** | **0.096** | **6.910** | <.001 (6.13e-07) | **0.203** | **0.219** | 1.84 | 0.043 | 15 |

Power for incremental ΔR² (HEXACO):
- Power harassment: f² = **0.040**, achieved power = **0.803**
- Gender harassment: f² = **0.122**, achieved power = **0.9997**

## 6. Standardized β — Model B (Table 3)

From `res/regression_coefficients_extended.csv`,
`Model == "B_controls+DT+HEXACO"`.

### Power harassment (Model B)
| Predictor | β | SE | t | p | sig |
|-----------|---|----|---|---|-----|
| age | -0.063 | 0.054 | -1.17 | 0.242 | |
| gender | -0.144 | 0.108 | -1.34 | 0.181 | |
| Machiavellianism | -0.109 | 0.072 | -1.52 | 0.130 | |
| Narcissism | 0.004 | 0.082 | 0.05 | 0.964 | |
| Psychopathy | **0.317** | 0.065 | 4.89 | 9.88e-07 | *** |
| Honesty-Humility | **-0.143** | 0.073 | -1.97 | 0.0494 | * |
| Emotionality | -0.030 | 0.055 | -0.54 | 0.588 | |
| Extraversion | -0.056 | 0.069 | -0.81 | 0.417 | |
| Agreeableness | **-0.108** | 0.058 | -1.86 | 0.063 | (marginal) |
| Conscientiousness | -0.054 | 0.059 | -0.91 | 0.363 | |
| Openness | 0.033 | 0.054 | 0.60 | 0.546 | |

### Gender harassment (Model B)
| Predictor | β | SE | t | p | sig |
|-----------|---|----|---|---|-----|
| age | 0.078 | 0.058 | 1.36 | 0.174 | |
| gender | **-0.312** | 0.113 | -2.76 | 0.006 | ** |
| Machiavellianism | **-0.164** | 0.055 | -2.96 | 0.003 | ** |
| Narcissism | **0.190** | 0.066 | 2.86 | 0.004 | ** |
| Psychopathy | **0.154** | 0.059 | 2.62 | 0.009 | ** |
| Honesty-Humility | **-0.230** | 0.065 | -3.52 | 0.00043 | *** |
| Emotionality | 0.031 | 0.049 | 0.63 | 0.530 | |
| Extraversion | -0.031 | 0.063 | -0.49 | 0.625 | |
| Agreeableness | 0.102 | 0.054 | 1.89 | 0.0586 | (marginal) |
| Conscientiousness | 0.031 | 0.049 | 0.65 | 0.518 | |
| Openness | **-0.236** | 0.056 | -4.20 | 2.61e-05 | *** |

## 7. Model A (DT only) — Selected (Power)

From `res/regression_coefficients_extended.csv`:

| Predictor | Outcome | β | p | sig |
|-----------|---------|----|---|-----|
| Machiavellianism | Power | -0.062 | 0.342 | |
| Narcissism | Power | -0.005 | 0.933 | |
| Psychopathy | Power | **0.396** | 2.87e-11 | *** |
| Machiavellianism | Gender | **-0.138** | 0.006 | ** |
| Narcissism | Gender | **0.180** | 0.00064 | *** |
| Psychopathy | Gender | **0.165** | 0.0044 | ** |

## 8. Sex-stratified Model B (Table 5)

From `res/sex_stratified_R2.csv`:

| Outcome | gender code | n | R² | Adj. R² |
|---------|-------------|---|-----|--------|
| Power | 0 (male) | 133 | 0.287 | 0.215 |
| Power | 1 (female) | 220 | 0.203 | 0.157 |
| Gender | 0 (male) | 133 | 0.283 | 0.211 |
| Gender | 1 (female) | 220 | 0.166 | 0.118 |

## 9. VIF (Table 7)

From `res/vif_modelB.csv` (identical for both DVs because predictor set is the same):

| Predictor | VIF |
|-----------|-----|
| age | 1.24 |
| gender | 1.11 |
| Kinki | 1.17 |
| Other | 1.16 |
| Machiavellianism | 1.31 |
| Narcissism | **2.00** (max) |
| Psychopathy | 1.56 |
| Honesty-Humility | 1.57 |
| Emotionality | 1.33 |
| Extraversion | 1.70 |
| Agreeableness | 1.53 |
| Conscientiousness | 1.16 |
| Openness | 1.31 |

Mean VIF (excluding const, all 13 predictors): **1.40**.

## 10. Diagnostic statistics (Table 6)

| Outcome | DW | Shapiro-Wilk p | BP-LM p | BP-F p | Max Cook D | n > 4/n |
|---------|----|----|----|----|----|----|
| Power | 1.95 | 2.14e-14 | 1.22e-04 | 7.10e-05 | 0.058 | 23 |
| Gender | 1.84 | 6.30e-09 | 3.53e-06 | 1.29e-06 | 0.043 | 15 |

## 11. Reference List Author (T5 source)

The 22 lines of the title page contain author identifiers. The
manuscript main file `Manuscript_only.docx` lists all references
in `Manuscript_only.txt:80–198` (with hyperlinks to DOIs).
