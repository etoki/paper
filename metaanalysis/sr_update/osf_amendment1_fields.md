# OSF に入れる文面（Amendment 1）

`protocol_amendment.md` の英語部分を、OSF の欄に合わせて分けたもの。各ブロックを該当する欄の**先頭**に足し、元の文面はそのまま残す。

---

## Justification（修正理由の欄）

```
Amendment 1 (27 October 2026): replication search in the registered information sources.

The search run in April 2026 did not follow "Data collection procedures": the computing environment could not reach the registered databases, so eight web search engine queries, previously known studies and the reference lists of benchmark meta-analyses were used instead, and record counts before the full-text stage were not logged. This amendment specifies a replication search in the registered sources, run before any updated synthesis.

The amendment is registered after the first synthesis was completed and its results were known (for example, Conscientiousness r = .185, k = 9). It does not change the eligibility criteria, the outcomes, the effect-size metric, or the analysis model; deviations of the first synthesis from the analysis plan are reported in the manuscript. Three sections are amended, each by a block added at its top; the original text of 23 April 2026 is kept below each block:
- Data collection procedures: information sources, search terms, selection, full-text retrieval, data management, studies with mixed delivery, framing of the report.
- Starting and stopping rules: start and stopping of the replication search.
- Missing data: authors are not contacted.

Full text of the amendment and the search scripts: https://github.com/etoki/paper (metaanalysis/sr_update/).
```

---

## Sampling → Data collection procedures（先頭に足す）

```
=== AMENDMENT 1 (27 October 2026) ===
This block supersedes the parts of this section on information sources, search terms, screening tools and full-text retrieval. The eligibility criteria are unchanged. The original text of 23 April 2026 follows below the line, unchanged.

A1. INFORMATION SOURCES
All sources are searched on the same date through their public APIs, with a script that saves every response unmodified.
- PubMed/MEDLINE (NCBI E-utilities): registered; to be run.
- ERIC (IES ERIC API): registered; to be run.
- OpenAlex: added, as an open substitute for Web of Science and Scopus, to which the author has no access.
- PsycINFO, Web of Science, Scopus, ProQuest Dissertations and Theses: registered; not run (no institutional access); reported as not searched.
- Google Scholar: registered; run once by the author with Publish or Perish (Harzing), on the first 200 records of one simplified query, because Google Scholar has no API and limits query length. The exported records are merged and de-duplicated with the other sources.
- Citation searching (registered), run through OpenAlex. Backward: the full reference list of each seed. Forward: every work citing each seed. Seeds: the 25 studies included in manuscript v4; the four benchmark meta-analyses named above (Poropat, 2009; Vedel, 2014; Stajkovic et al., 2018; Mammadov, 2022); two later reviews (Meyer et al., 2023; Hunter et al., 2025). Seeds without a DOI are searched by hand. Forward citations of the six reviews are restricted to works whose title or abstract matches the online-learning block, because those reviews are cited thousands of times, mostly by studies of face-to-face instruction. As registered, citation searching is repeated from newly included studies until two consecutive rounds yield no new eligible study.
- Author contact is no longer a source (see Missing data).

A2. SEARCH TERMS
The three registered blocks below are used unchanged. Terms are added to catch spelling variants, plural forms and close synonyms; no registered term is removed.
- Personality: "five factor", "extroversion", "personality".
- Online learning: "elearning", "online education", "distance education", "online instruction", "MOOCs", "massive open online courses", "online courses", "emergency remote teaching", "emergency remote learning", "online class", "online classes".
- Outcome: "test scores", "course grades", "learning outcomes", "grades", "exam performance", "examination performance", "final grade", "course performance", "course completion", "academic outcomes".
For each source, hits are reported twice: for the registered terms alone and for the amended terms. Limits: English language; no limit on publication year. The exact strings run in each source are archived with the results.

A3. SELECTION
As registered: ET screens every record; at the title/abstract stage a 10% random subsample is re-screened after at least seven days, and at the full-text stage 20%; Cohen's kappa is reported for each. The 37 reports assessed at full text for manuscript v4 keep their recorded decisions; new records go through the full process.
Full-text retrieval: open-access copies are retrieved by script (OpenAlex open-access locations, PubMed Central, ERIC, OSTI); the author obtains the other reports through the author's own access. Reports that cannot be obtained are reported as not retrieved.

A4. DATA MANAGEMENT (replaces Zotero and Rayyan)
Records are de-duplicated by script, first by DOI and then by normalised title plus publication year, and screened in a spreadsheet with one row per unique record and the decision and reason at each stage. All files and scripts are published at https://github.com/etoki/paper (metaanalysis/sr_update/).

A5. STUDIES WITH MIXED DELIVERY
Studies that pool students taught face to face with students taught online (for example cohorts before, during and after the COVID-19 pandemic) without statistics for the online students alone are included in the evidence map but left out of the quantitative pool. This rule was applied post hoc to one study in manuscript v4 (Zheng and Zheng, 2023) and is applied prospectively to every study found by this search.

A6. FRAMING OF THE REPORT
The report is framed by search completeness, not by the results. If the searches in A1 are run and selection follows A3, the report is a systematic review with meta-analysis (PRISMA 2020); otherwise it stays a scoping review (PRISMA-ScR). Either way, the sources not searched and all deviations are reported.

=== ORIGINAL TEXT (23 April 2026) ===
```

---

## Sampling → Starting and stopping rules（先頭に足す）

```
=== AMENDMENT 1 (27 October 2026) ===
This block supersedes "Search start" and items (a) and (b) of "Search stopping" below. The original text of 23 April 2026 follows below the line, unchanged.

SEARCH START: the replication search specified in "Data collection procedures" (Amendment 1) is run within one month of this amendment, all sources on the same date.

SEARCH STOPPING:
(a) PubMed, ERIC and OpenAlex have been searched with the full strategy; PsycINFO, Web of Science, Scopus and ProQuest are not searched (no institutional access);
(b) Google Scholar has returned its first 200 records for one simplified query, exported with Publish or Perish;
(c) unchanged: citation searching has reached saturation (two consecutive rounds yielding no new eligible study).

The search update rule is unchanged: if more than six months pass before the final analysis, the search is re-run.

=== ORIGINAL TEXT (23 April 2026) ===
```

---

## Analysis Plan → Missing data（先頭に足す）

```
=== AMENDMENT 1 (27 October 2026) ===
This block supersedes procedure (A), "Author contact", below. The original text of 23 April 2026 follows below the line, unchanged.

Authors are not contacted for missing statistics, for studies found by the replication search or any other. When a statistic is missing, it is converted from a reported alternative under the registered conversion rules (procedure B); if that is not possible, the study is left out of that trait's pool and the reason is logged (procedure C).

Reason: none of the 37 reports assessed for manuscript v4 involved author contact. Using the same procedure for every study keeps their treatment consistent. As a consequence, the counts of studies excluded after failed author contact and the sensitivity analysis by author-response status do not apply.

=== ORIGINAL TEXT (23 April 2026) ===
```
