# Cover Letters — harassment/paper/

## Files

| File | Status | Notes |
|------|--------|-------|
| `Cover Letter (need edit).docx` | original template | kept as-is per request |
| `Cover Letter_Behavioral Sciences.docx` | already submitted | kept as-is |
| `Cover Letter_Heliyon.docx` | **new** | top recommendation |
| `Cover Letter_Acta Psychologica.docx` | **new** | Elsevier hybrid OA |
| `Cover Letter_PeerJ.docx` | **new** | low-APC, indexed |
| `Cover Letter_Royal Society Open Science.docx` | **new** | learned-society OA |
| `Cover Letter_Discover Psychology.docx` | **new** | newer Springer OA |
| `Cover Letter_Healthcare.docx` | **new** | occupational-health frame |
| `build_cover_letters.py` | builder | regenerates all six new letters |

## Improvements over the uploaded template

The new letters were built from the same factual core as the template
but with the following corrections — these are **not** applied to the
two original files (the template and the Behavioral Sciences letter
are kept intact per request, but the same fixes can be ported by hand
if those are reused).

### 1. Sole-author voice

The Title page (`Title page with Declarations.docx`) declares Eisuke
Tokiwa as the only author. The original template uses both "I" and
"our / We" — inconsistent and grammatically incorrect for a sole-author
submission. Most journals' submission systems flag this. The new
letters use **I / my** throughout.

### 2. Preprint disclosure

The manuscript is publicly preprinted on Research Square
(https://doi.org/10.21203/rs.3.rs-7756124/v1) — declared in the
Title page but **omitted from the original cover letter**. Several
target journals (Heliyon, PeerJ, Royal Society Open Science, BMC,
Discover) require explicit preprint disclosure in the cover letter.
The new letters include a dedicated *Open science and preprint*
paragraph.

### 3. Softened applied-implications wording

The original template ends the Methods paragraph with
"underscoring the applied value of integrating HEXACO into risk
assessment and prevention". For a cross-sectional, single-source,
self-report design, this is overclaiming for selection or screening
applications. The new letters use:

> "informing tentative implications for harassment-prevention
> frameworks rather than deterministic personnel-selection rules"

This matches the Limitations / Practical Implications wording in the
manuscript itself ("personality-informed approaches should be viewed
as complementary to systemic interventions rather than replacements
for organizational reforms").

### 4. Journal-tailored fit paragraphs

Each new letter has a *Fit with [Journal]* paragraph that names the
journal's scope, audience, or editorial culture. The original template
had a generic "advances personality and organizational behavior
research" paragraph that did not differentiate between targets.

### 5. Data availability statement

The new letters explicitly state where the data and code are deposited
(`https://github.com/etoki/paper`) — required by Heliyon, Royal Society
Open Science, PeerJ, Discover and recommended by all others.

### 6. Numeric values quote-quoted to manuscript

The original template wrote `ΔR² ≈ .03 / .10`. The new letters use
the precise canonical values from `res/model_fit_incremental.csv`:
**ΔR² = .032, p = .036** (power) and **ΔR² = .096, p < .001**
(gender). These match the manuscript's Results section exactly so the
cover letter cannot drift from the paper during revisions.

## Regenerating

```bash
python3 harassment/paper/build_cover_letters.py
```

If a journal-specific edit is required (e.g., suggested reviewers,
specific section editor), edit `JOURNALS["<name>"]` in the builder
and re-run.

## Excluded venues (already submitted)

- BMC Psychology
- Frontiers in Psychology
- Current Psychology
- Behavioral Sciences (MDPI) — letter present in this directory
- SAGE Open
