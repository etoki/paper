# paper_v3_scoping — Scoping Review reframing

Scoping-review reframing of the paper_v2 manuscript, prepared in
response to Heliyon Reviewer 1 (decision: implicit reject / editor
recommendation: major revision) and to the parallel "search
deviation" concern raised by Frontiers in Education Reviewer 1.

## Background — why reframe rather than re-search

The pre-registered protocol (OSF Registries, 2026-04-23) specified
six bibliographic databases (PubMed/MEDLINE, PsycINFO, ERIC, Web of
Science, Scopus, ProQuest Dissertations) as the information sources.
At execution time, only the open subset (PubMed via WebSearch
equivalent, ERIC web search, and open-access repositories) was
actually accessible: the subscription-gated databases (PsycINFO,
Web of Science, Scopus) require institutional access that is not
available to the author, and individual subscriptions are not
offered by those providers. Direct PubMed E-utilities, OpenAlex,
and Semantic Scholar API access was also blocked by the execution
environment's network allowlist.

Reviewer 1 of Heliyon characterised this gap as a *decisive issue*
that is "not fixable by editing the manuscript" and offered two
honorable paths: (i) conduct the intended six-database search and
resubmit, or (ii) reframe the manuscript as a scoping review with
correspondingly modest claims. Path (i) is precluded by the
combination of (a) absent institutional access, (b) the absence of
individual subscriptions for the three subscription-gated databases,
and (c) the revision deadlines on the three journals where the
manuscript is currently under review (Frontiers in Education,
Humanities and Social Sciences Communications, Heliyon).

This directory implements Path (ii): a full reframing of the
manuscript as a scoping review with exploratory quantitative
synthesis, which (a) honestly characterises the actual evidence-
search activity performed, (b) replaces the PRISMA 2020 framework
with PRISMA-ScR (Tricco et al., 2018), (c) re-positions the
quantitative meta-analytic component as an exploratory secondary
analysis rather than the primary claim, and (d) recalibrates all
claims throughout.

## Differences from paper_v2

| Element | paper_v2 | paper_v3_scoping |
|---|---|---|
| Research type | Systematic review and meta-analysis | Scoping review with exploratory quantitative synthesis |
| Reporting framework | PRISMA 2020 | PRISMA-ScR (Tricco et al., 2018) |
| Information sources | "six pre-registered databases" wording | Honest listing of actually-searched sources only |
| Primary claim | "First quantitative synthesis of online ..." | "Scoping mapping of the online-modality Big Five literature ..." |
| Quantitative pool | "Primary analytical engine" (k = 10) | "Exploratory meta-analytic estimate, secondary to the scoping map" |
| Practical implications | "Practical Implications" subsection | "Tentative research-agenda implications" subsection |
| GRADE | Adapted GRADE for confidence | Not applicable to scoping reviews; replaced by completeness-of-evidence mapping |
| Hypotheses | H1-H5 (confirmatory framing) | Mapping questions MQ1-MQ5 (descriptive framing) |
| Risk of bias | JBI per-study scores | Retained, but described as descriptive completeness signal |

paper_v2 itself is retained as the canonical history of the
systematic-review attempt and is not deleted, but it should not be
submitted to any journal in its current form going forward.

## Targets

A single source build produces journal-tailored outputs for all
three active submissions:

* Frontiers in Education (Manuscript ID 1866537) — Reviewer 1
  major-revision response.
* Humanities and Social Sciences Communications, Springer Nature
  (Submission ID 72427190-7674-4b5f-ac51-9518b8c16eaf) — anonymised
  pre-peer-review revision.
* Heliyon, Elsevier (Manuscript ID HELIYON-D-26-02879) — Reviewer 1
  major-revision response, currently pending the editor's response
  to the parallel deadline-extension request.

## Files in this directory

(populated incrementally as the reframing proceeds)

* `references_data.py` — copied verbatim from paper_v2 (no change).
* `build_docx_scoping.py` — to be derived from paper_v2/build_docx.py
  with the reframing edits described above.
* `build_prisma_scr_checklist.py` — to be created; PRISMA-ScR (rather
  than PRISMA 2020) checklist supplement.
* Cover letters / point-by-point responses — to be created per-journal
  once the reframed manuscript is stable.
