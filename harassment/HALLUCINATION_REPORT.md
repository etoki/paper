# Hallucination Check Report — harassment/

**Date**: 2026-05-02
**Scope**: `paper/Manuscript_only.docx`, `paper/Table.docx`, `paper/Title page with Declarations.docx`
**Tool**: `python3 harassment/check_hallucinations.py`

## Summary

| Task | Status | Findings |
|------|--------|----------|
| T1 Descriptives | ✅ pass | All 13 means / SDs match `res/descriptive_statistics.csv`. |
| T2 Spearman | ✅ pass | All 12 spot-checked correlations match `res/spearman_rho.csv`. |
| T3 Regression (β, R², ΔR², F) | ✅ pass | All Model A / B / sensitivity / interaction numbers match. |
| T4 Sex-stratified R² | ✅ pass | n = 133 / 220, R² values match. |
| T5 Diagnostics & VIF | ✅ pass | DW, Cook's D, exceedance counts, all VIFs match. |
| T6 Cronbach's α | ❌ **FAIL** | One discrepancy: Honesty–Humility α. |
| T7 Sample / counts | ✅ pass | N = 354, 134 ♂ / 220 ♀, age bins match, listwise N = 353. |
| T8 References | ❌ **FAIL** | 14 in-text citations missing from reference list, 1 surname spelling typo, 1 unused reference. |

**Overall**: **163 passes, 4 failures** across 8 tasks. The numeric
results in Tables 1–7 and the Results section all reproduce exactly
from `analysis.py` against `raw.csv`. The defects are concentrated
in (a) one Cronbach's α value and (b) the reference list.

---

## ❌ Finding 1 — Honesty–Humility α discrepancy (T6)

**Manuscript text** (Methods → Measurement Instruments,
`Manuscript_only.docx`):

> In the present sample, internal consistency coefficients were
> α = **.671** for Honesty–Humility, α = .830 for Emotionality,
> α = .621 for Extraversion, α = .783 for Agreeableness,
> α = .815 for Conscientiousness, and α = .804 for Openness.

**`alfa/hexaco_alpha_results.csv`** (combined N = 4168 over three
psychometric files, computed by `alfa/alpha_hexaco.py`):

| Domain | Manuscript | Script |
|--------|-----------|--------|
| Honesty–Humility | **.671** | **.571** ❌ |
| Emotionality | .830 | .830 ✅ |
| Extraversion | .621 | .621 ✅ |
| Agreeableness | .783 | .783 ✅ |
| Conscientiousness | .815 | .815 ✅ |
| Openness | .804 | .804 ✅ |

5 of 6 values match exactly. The single H-H value differs by exactly
0.10, suggesting a digit transposition (.571 → .671) when copying
from the script output to the manuscript.

**Secondary issue — wording**: the alphas are computed on the combined
4168-row pool, *not* on the present harassment sample (N = 354).
`raw.csv` does not contain item-level HEXACO data, so the alphas in
the Methods section cannot have been computed on the harassment
sample. The phrase **"In the present sample"** is therefore
misleading. (See `alfa/alpha_hexaco.py` and `hexaco_alpha_results.csv`.)

**Recommendation**:
1. Fix the typo: `α = .671` → `α = .571` for Honesty–Humility.
2. Either (a) re-compute α's on the actual harassment item-level
   data (currently not in the repo) or (b) reword to "Cronbach's α
   coefficients for the HEXACO-60 (estimated from the combined
   validation pool, N = 4168) were …".

---

## ❌ Finding 2 — In-text citations missing from the reference list (T8)

The body cites the following 14 papers (all real publications), but
the reference list at lines 80–198 of `Manuscript_only.docx` contains
no entry for any of them:

| In-text citation | Probable target |
|------------------|----------------|
| `Babiak & Hare, 2006` | Babiak, P., & Hare, R. D. (2006). *Snakes in suits.* HarperCollins. |
| `Bass, 1990` | Bass, B. M. (1990). *Bass & Stogdill's handbook of leadership* (3rd ed.). |
| `Breevaart & de Vries, 2019` | (only 2017 and 2021 entries present) |
| `Cherniss & Goleman, 2001` | Cherniss, C., & Goleman, D. (Eds.) (2001). *The emotionally intelligent workplace.* |
| `Christie & Geis, 1970` | Christie, R., & Geis, F. L. (1970). *Studies in Machiavellianism.* |
| `Cleckley, 1941` | Cleckley, H. (1941). *The mask of sanity.* |
| `Fehr et al., 1992` | Fehr, B., Samsom, D., & Paulhus, D. L. (1992). *The construct of Machiavellianism.* |
| `Gandolfi et al., 2017` | Gandolfi, F., Stone, S., & Deno, F. (2017). |
| `Hare, 2003` | Hare, R. D. (2003). *Hare PCL-R* (2nd ed.). |
| `Jones & Paulhus, 2009` | Jones, D. N., & Paulhus, D. L. (2009). *Machiavellianism.* In M. R. Leary & R. H. Hoyle (Eds.). |
| `Jones & Paulhus, 2010` | Jones, D. N., & Paulhus, D. L. (2010). *Different provocations trigger aggression…* |
| `Lynam & Derefinko, 2006` | Lynam, D. R., & Derefinko, K. J. (2006). *Psychopathy and personality.* |
| `Paulhus, 2001` | Paulhus, D. L. (2001). *Normal narcissism: Two minimalist accounts.* |
| `Zuckerman, 1994` | Zuckerman, M. (1994). *Behavioral expressions and biosocial bases of sensation seeking.* |

A 15th item, **`Saltuküğlu et al., 2019`**, is also flagged by the
checker: see Finding 3.

**Recommendation**: add full reference entries for all 14 above.
Verification command:

```bash
python3 harassment/check_hallucinations.py --task t8
```

---

## ❌ Finding 3 — Surname spelling typo: Saltuküğlu vs Saltukoğlu (T8)

- **Body** (`Manuscript_only.docx`): `Saltuküğlu et al., 2019` (uses
  *ü* in the third position).
- **Reference list**: `Saltukoğlu, G., Tatar, A., & Özdemir, H. (2019).
  The HEXACO personality measure as a predictor of job performance
  and job satisfaction…` (uses *o* in the third position).

The reference DOI (`10.36315/2019inpact040`) and journal title
(*Psychological Applications and Trends 2019*) confirm the correct
surname is **Saltukoğlu** (with *o*). The body should be corrected.

**Recommendation**: replace `Saltuküğlu` → `Saltukoğlu` in the body.
Once fixed, the "unused references" warning for `Saltukoğlu (2019)`
also resolves.

---

## ❌ Finding 4 — Unused reference: Jones & Paulhus (2011) (T8)

The reference list contains:

> Jones, D. N., & Paulhus, D. L. (2011). Differentiating the Dark
> Triad within the interpersonal circumplex. In L. M. Horowitz &
> S. Strack (Eds.), *Handbook of interpersonal psychology…*

…but the body never cites Jones & Paulhus 2011 — only **2009** and
**2010** (both of which are missing from the reference list, see
Finding 2).

This pattern is consistent with a year transcription error: the
intended reference for `Jones & Paulhus, 2010` (or 2009) may have
been replaced in the reference list by a 2011 chapter that was never
cited in body text.

**Recommendation**: verify which year(s) the body intends to cite,
add the missing 2009/2010 entries, and either delete or add a citation
for the 2011 chapter.

---

## ✅ Findings that PASSED (no action needed)

- **All 13 descriptive M / SD values** in Table 1 reproduce exactly.
- **All Spearman correlations** referenced in the body (12 spot
  checks) match `res/spearman_rho.csv`.
- **All Model A / B / C / sensitivity coefficients** mentioned in
  Results match `res/regression_coefficients_extended.csv` to three
  decimals.
- **Model fit numbers**: R²(A) = .166, R²(B) = .198, ΔR² = .032,
  F-change = 2.28, p = .036 (power); R²(A) = .117, R²(B) = .213,
  ΔR² = .096, F-change = 6.91, p < .001 (gender); R²(B sens) =
  .221 / .203; R²(C) = .218 / .219 — all match.
- **Post-hoc power**: f² = 0.040 / 0.122, achieved power
  0.803 / 0.9997 — all match.
- **Sex-stratified n and R²**: 133 / 220, .287 / .203 / .283 / .166 — match.
- **Diagnostics**: DW 1.95 / 1.84, max Cook's D 0.058 / 0.043,
  exceedances 23 / 15 — match.
- **VIF**: max = 2.00 (Narcissism), N = 2.00 / X = 1.70 / H-H = 1.57 — match.
- **Sample**: N = 354, 134 ♂ / 220 ♀, age bins 32/100/124/70/28,
  Mach/Narc/Psy missing rate 0.28% (< 0.3%), listwise N = 353 — all match.
- **Stratified n vs total n** (134 vs 133): the manuscript correctly
  reports both because Mach/Narc/Psy each have 1 missing case (in a
  male respondent), so listwise drops 1 from the male stratum.

## Minor caveat — age coding

The manuscript states "32 participants aged 18–19". `raw.csv` codes
`age` in decade bins (10, 20, 30, 40, 50, 60), so the youngest cohort
is the **10-decade bin** (32 participants). The data itself does not
distinguish 18 from 10, so "18–19" is an *interpretation* (consistent
with the inclusion criterion of "currently employed adult") rather
than a fabrication, but the wording is technically imprecise. Consider
"32 participants aged under 20".

---

## Reproducing this report

```bash
# 1. Regenerate canonical results from raw data
cd /home/user/paper/harassment && python3 analysis.py

# 2. Re-run the hallucination checker
python3 harassment/check_hallucinations.py
```
