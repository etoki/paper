# IEEE Access Manuscript (HEXACO Workplace Harassment Microsim)

This folder produces an **IEEE Access**-format manuscript variant of the
HEXACO 7-Typology Workplace Harassment Microsimulation paper, modeled on
the layout used in the previously accepted companion paper
`clustering/paper_IEEE/Manuscript_IEEE_rivision.docx`.

## Why an IEEE Access variant?

IEEE Access is a methodologically broad, computation-positive venue where
the heavy statistical machinery of this study (BCa bootstrap cascade,
Beta-Binomial empirical-Bayes shrinkage, Berger–Hsu IUT, target-trial
emulation, microsimulation, transportability sweep) lands more naturally
than in a behaviorally-typed psychology journal. The companion HEXACO
clustering paper was accepted at IEEE Access on first submission largely
because the rigour of its computational pipeline was understood as a
contribution. This variant prepares the same paper in the same venue's
formatting expectations.

## Output

```
manuscript_ieee.docx
```

## Format specifications (matched to the clustering IEEE Access submission)

| Field | Value |
|---|---|
| Page size | 8.00" × 10.875" |
| Margins | top 0.89", bottom 0.72", left/right 0.51" |
| Title block (section 0) | single-column, full page width |
| Body (section 1) | 2-column, 3.35" each, 0.28" gap |
| Body font | Times New Roman 10pt, justified |
| Section heading | 9pt bold, ALL CAPS |
| Subsection heading | 10pt bold, mixed case |
| Sub-subsection heading | 10pt bold italic, mixed case |
| Title | 22pt bold |
| Author | 10pt bold + 7pt superscript |
| Affiliation | 7pt |
| Corresponding author / Funding | 8pt |
| Abstract / Index Terms | inline 10pt bold label + 10pt body |
| References | numbered [N], 8pt body, 9pt bold key |
| Citation style | numbered IEEE [N] (replaces APA `(Author, Year)`) |

## Source

The IEEE-format docx is generated **directly from the existing APA markdown**
in `simulation/paper/0[1-4]_*.md` and `simulation/paper/05_refs.md` by
`build_ieee_docx.py`. There is no separate markdown source file: the
single source of truth remains the APA paper, and the IEEE variant is a
pure rendering transform.

The build script does three transformations on the fly:

1. **Layout**: page setup + section break for title block (1-column) →
   body (2-column).
2. **Heading hierarchy**: `## Section` → 9pt ALL CAPS;
   `### Subsection` → 10pt mixed case; `#### Sub-sub` → 10pt italic.
3. **Citations**: walks the body, builds an order-of-first-appearance
   index of every APA citation it finds (e.g., `(Hudson & Fraley, 2015)`),
   replaces them with IEEE numbered form (e.g., `[14]`), and emits the
   References section in IEEE author-initials style.

## Build

```bash
cd simulation
uv run python paper_IEEE/build_ieee_docx.py
```

Outputs `paper_IEEE/manuscript_ieee.docx` (~64 KB; ~7,900 words; 6 tables).

## Verifying the IEEE rendering

```python
from docx import Document
from docx.oxml.ns import qn
doc = Document('paper_IEEE/manuscript_ieee.docx')
# Page setup should match clustering/paper_IEEE
sec0 = doc.sections[0]
assert abs(sec0.page_width.inches - 8.00) < 0.01
assert abs(sec0.page_height.inches - 10.875) < 0.01
# Section 0: 1-column (title); section 1: 2-column (body)
import re
text = '\n'.join(p.text for p in doc.paragraphs)
n_brackets = len(re.findall(r'\[\d+\]', text))
assert n_brackets > 50, "Expected ≥50 numbered citations"
# No APA-style citations should remain in body
n_apa = len(re.findall(r'\([A-Z][a-z]+\s*&\s*[A-Z][a-z]+,\s*\d{4}\)', text))
assert n_apa == 0, "All citations should be IEEE numbered"
```

## Citation index

The build script's `IEEE_REFS` dict is the canonical mapping from
APA-style citation key → IEEE-formatted reference text. It currently
contains 37 reference entries spanning the full bibliography
(`05_refs.md`); only entries that are actually cited in the body appear
in the rendered References list (with order-of-first-appearance
numbering).

If a new citation is introduced in the APA markdown that is not in
`IEEE_REFS`, the build script leaves the original `(Author, Year)` form
in place and emits no `[N]` for it — i.e., a missing-key fallback. To
add a new citation, edit `IEEE_REFS` in `build_ieee_docx.py`.

## Comparison with the APA build (`paper/manuscript_preprint.docx`)

| Aspect | APA preprint | IEEE Access |
|---|---|---|
| Page size | 8.5" × 11" (US Letter) | 8.00" × 10.875" (IEEE trim) |
| Columns | 1 (full page) | 1 (title) + 2 (body) |
| Body font size | 12pt | 10pt |
| Line spacing | Double | Single |
| Section heading | Bold centered | 9pt ALL CAPS bold |
| Citation style | `(Author, Year)` | Numbered `[N]` |
| References | APA 7th edition | IEEE author-initials style |
| Running head | `STRUCTURE DOMINATES PERSONALITY` | (none in this build) |

Both builds reuse the same source markdown — no content drift between
them is possible by construction.
