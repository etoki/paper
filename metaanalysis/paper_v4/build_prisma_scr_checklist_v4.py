"""
build_prisma_scr_checklist_v4.py — PRISMA-ScR (Tricco et al., 2018)
22-item checklist for manuscript v4 (paper_v4/).

The "Location reported" column names the section and subsection headings
of manuscript_journal_v4.docx exactly as they appear in build_docx_v4.py,
so a reviewer can find each item by heading rather than by page number.
Study counts and the Conscientiousness estimate are taken from values.py,
the same source build_docx_v4.py uses, so the checklist cannot drift from
the manuscript.

Outputs (next to this script):
    prisma_scr_checklist_v4.docx        — identified version
    prisma_scr_checklist_v4_anon.docx   — repository and registration
                                          links withheld (double-anonymous
                                          review)

Usage:
    python metaanalysis/paper_v4/build_prisma_scr_checklist_v4.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from values import V  # noqa: E402

OUT_NON_ANON = HERE / "prisma_scr_checklist_v4.docx"
OUT_ANON = HERE / "prisma_scr_checklist_v4_anon.docx"

TITLE = ("Big Five Personality Traits and Academic Achievement in Online "
         "Learning Environments: A Scoping Review with Exploratory "
         "Quantitative Synthesis")

REPO = "https://github.com/etoki/paper"
OSF = "https://osf.io/e5w47/ (DOI 10.17605/OSF.IO/E5W47)"
WITHHELD = "[withheld for double-anonymous review]"


# --------------------------------------------------------------------------
# docx helpers
# --------------------------------------------------------------------------

def _set_run_font(run, size_pt=10, bold=None, font_name="Calibri"):
    run.font.name = font_name
    run.font.size = Pt(size_pt)
    if bold is not None:
        run.bold = bold


def _set_cell_shading(cell, hex_color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tcPr.append(shd)


def _set_cell_borders(cell):
    tcPr = cell._tc.get_or_add_tcPr()
    borders = OxmlElement("w:tcBorders")
    for edge in ("top", "left", "bottom", "right"):
        b = OxmlElement(f"w:{edge}")
        b.set(qn("w:val"), "single")
        b.set(qn("w:sz"), "4")
        b.set(qn("w:color"), "808080")
        borders.append(b)
    tcPr.append(borders)


# --------------------------------------------------------------------------
# The 22 items
# --------------------------------------------------------------------------

def _build_items(anonymous: bool):
    repo = WITHHELD if anonymous else REPO
    osf = WITHHELD if anonymous else OSF
    reviewer = "the author" if anonymous else "ET"

    n_cat = V["n_catalogued"]
    n_ret = V["n_retained"]
    n_pool = V["n_pool"]
    n_direct = V["n_direct"]
    n_beta = V["n_beta"]
    k_c = V["C.k"]
    k_o = V["O.k"]
    c_r = V["C.r"]
    c_ci = V["C.ci"]
    c_pi = V["C.pi"]
    n_total = V["C.N"]

    # (section, item number, item text, location reported)
    return [
        ("TITLE", "", "", ""),
        ("Title", "1",
         "Identify the report as a scoping review.",
         "Title page. The title ends \"A Scoping Review with Exploratory "
         "Quantitative Synthesis\"."),

        ("ABSTRACT", "", "", ""),
        ("Structured summary", "2",
         "Provide a structured summary that includes (as applicable): "
         "background, objectives, eligibility criteria, sources of "
         "evidence, charting methods, results, and conclusions that relate "
         "to the review questions and objectives.",
         "Abstract (single paragraph, in the order background, objective, "
         "sources of evidence, synthesis method, results, conclusions). "
         "It names PRISMA-ScR as the reporting guideline, states which "
         "pre-registered databases could not be accessed and what was "
         "searched instead, and gives the counts "
         f"({n_cat} catalogued, {n_ret} retained, {n_pool} pooled)."),

        ("INTRODUCTION", "", "", ""),
        ("Rationale", "3",
         "Describe the rationale for the review in the context of what is "
         "already known. Explain why the review questions/objectives lend "
         "themselves to a scoping review approach.",
         "Introduction, subsections \"Personality and Academic Achievement "
         "in Face-to-Face Contexts\", \"The Shift to Online Learning "
         "Environments\", \"Big Five Personality Traits in Online "
         "Learning: Fragmented Evidence\", and \"Potential Moderators of "
         "the Personality-Achievement Association in Online Contexts\". "
         "The reason for the scoping-review approach (the pre-registered "
         "databases could not be searched) is given in Methods, "
         "\"Pre-registration and Reporting Standards\" and \"Information "
         "Sources and Search Strategy\"."),
        ("Objectives", "4",
         "Provide an explicit statement of the questions and objectives "
         "being addressed with reference to their key elements (e.g., "
         "population or participants, concepts, and context) or other "
         "relevant key elements used to conceptualize the review questions "
         "and/or objectives.",
         "Introduction, subsection \"The Present Study\": two mapping "
         "questions (MQ1, MQ2) and five mapping priors (MP1 to MP5). "
         "Population (learners in online environments), concept (Big Five "
         "traits and academic achievement), and context (online, blended, "
         "and MOOC settings) are stated there and defined in Methods, "
         "\"Eligibility Criteria\"."),

        ("METHODS", "", "", ""),
        ("Protocol and registration", "5",
         "Indicate whether a review protocol exists; state if and where it "
         "can be accessed (e.g., a Web address); and if available, provide "
         "registration information, including the registration number.",
         "Methods, \"Pre-registration and Reporting Standards\": protocol "
         f"registered on OSF Registries on April 23, 2026, {osf}. Also in "
         "the Declarations (\"Pre-registration\"). Departures from the "
         "protocol are listed in Methods, \"Deviations from the "
         "Pre-registered Protocol\" (ten numbered deviations)."),
        ("Eligibility criteria", "6",
         "Specify characteristics of the sources of evidence used as "
         "eligibility criteria (e.g., years considered, language, and "
         "publication status), and provide a rationale.",
         "Methods, \"Eligibility Criteria\": PICOS criteria (population, "
         "exposure, comparator, outcome, study design); English language; "
         "peer-reviewed journals, peer-reviewed proceedings, or graduate "
         "theses; preprints excluded; no restriction on publication year; "
         "samples of at least 10 learners; fully face-to-face samples "
         "excluded."),
        ("Information sources", "7",
         "Describe all information sources (e.g., databases with dates of "
         "coverage and contact with authors to identify additional "
         "sources), as well as the date the most recent search was "
         "executed.",
         "Methods, \"Information Sources and Search Strategy\": the six "
         "pre-registered databases are named; PsycINFO, Scopus, Web of "
         "Science, and ProQuest were not searched (no institutional "
         "access) and programmatic access to the open databases was "
         "blocked. Executed sources: (a) a web search engine, run on "
         "April 23, 2026; (b) the ERIC web interface; (c) open-access "
         "repositories (PubMed Central, Frontiers, MDPI, Open Praxis); "
         "(d) forward and backward citation tracking from eligible studies "
         "and eight benchmark meta-analyses. Authors were not contacted "
         "(Methods, \"Deviations\", item 2)."),
        ("Search", "8",
         "Present the full electronic search strategy for at least 1 "
         "database, including any limits used, such that it could be "
         "repeated.",
         "Methods, \"Information Sources and Search Strategy\", second "
         "paragraph: the three concept blocks are reproduced in full with "
         "every term, and the split into eight web queries is explained. "
         "The executed queries, dates, and hit counts are in the search "
         f"log in the project repository ({repo}, file "
         "metaanalysis/search_log.md)."),
        ("Selection of sources of evidence", "9",
         "State the process for selecting sources of evidence (i.e., "
         "screening and eligibility) included in the scoping review.",
         f"Methods, \"Study Selection\": one reviewer ({reviewer}) screened "
         "titles and abstracts and assessed full texts, with a large "
         "language model assisting in locating and summarising candidate "
         "reports; every inclusion decision was made by the reviewer from "
         "the full text. The pre-registered intra-rater re-screening was "
         "not carried out, and records excluded before the full-text stage "
         "were not logged (Methods, \"Deviations\", items 2 and 3; "
         "Discussion, \"Limitations\")."),
        ("Data charting process", "10",
         "Describe the methods of charting data from the included sources "
         "of evidence (e.g., calibrated forms or forms that have been "
         "tested by the team before their use, and whether data charting "
         "was done independently or in duplicate) and any processes for "
         "obtaining and confirming data from investigators.",
         "Methods, \"Data Extraction\": the same reviewer extracted data "
         "into a pre-specified form (data_extraction_populated.csv, in the "
         f"project repository, {repo}). Extraction was not done in "
         "duplicate, the pre-registered 10% re-extraction was not carried "
         "out, and authors were not contacted for missing statistics "
         "(Methods, \"Deviations\", item 2)."),
        ("Data items", "11",
         "List and define all variables for which data were sought and any "
         "assumptions and simplifications made.",
         "Methods, \"Data Extraction\" (study identification, sample "
         "characteristics, learning context, personality and outcome "
         "measurement, effect-size statistics) and Table 1 (the charted "
         "fields for each study). Assumptions: Methods, \"Effect Size "
         "Metric and Transformations\" (Fisher z; Peterson and Brown "
         f"β-to-r conversion for {n_beta} studies; sign reversal for "
         "Emotional Stability; the |r| < .10 study kept out of the pool; "
         "HEXACO crosswalk not needed)."),
        ("Critical appraisal of individual sources of evidence", "12",
         "If done, provide a rationale for conducting a critical appraisal "
         "of included sources of evidence; describe the methods used and "
         "how this information was used in any data synthesis (if "
         "appropriate).",
         "Methods, \"Risk of Bias Assessment\": JBI Critical Appraisal "
         "Checklist for Analytical Cross-Sectional Studies (eight items, "
         "score 0 to 8), rated by the single reviewer and not re-rated. "
         "The score defines the sensitivity analysis that excludes studies "
         "scoring below 5 (Methods, \"Moderator and Sensitivity "
         "Analyses\") and enters the completeness-of-evidence rating "
         "(Methods, \"Completeness-of-Evidence Mapping (adapted from "
         "GRADE)\")."),
        ("Synthesis of results", "13",
         "Describe the methods of handling and summarizing the data that "
         "were charted.",
         "Methods, \"Statistical Analysis and Synthesis Strategy\" "
         "(random-effects model per trait, REML τ², Hartung-Knapp-Sidik-"
         "Jonkman intervals, prediction intervals), \"Moderator and "
         "Sensitivity Analyses\" (Knapp–Hartung meta-regression with Holm "
         "correction; seven sensitivity analyses), \"Publication Bias\" "
         "(funnel plots, Egger's test, trim-and-fill, reported "
         "descriptively), and \"Completeness-of-Evidence Mapping (adapted "
         "from GRADE)\". Studies with only secondary outcomes are "
         "summarised narratively (Results, \"Characteristics of Included "
         "Studies\")."),

        ("RESULTS", "", "", ""),
        ("Selection of sources of evidence", "14",
         "Give numbers of sources of evidence screened, assessed for "
         "eligibility, and included in the review, with reasons for "
         "exclusions at each stage, ideally using a flow diagram.",
         "Results, \"Study Selection\" and Figure 1 (PRISMA-ScR flow "
         "diagram). Counts before the full-text stage are approximate "
         "because they were not logged; from the full-text stage on they "
         "are exact (37 reports sought, 1 not retrieved, 36 assessed, 11 "
         f"excluded with reasons, {n_ret} retained, {n_pool} pooled). "
         "Exclusion reasons per study are in Table 1."),
        ("Characteristics of sources of evidence", "15",
         "For each included source of evidence, present characteristics "
         "for which data were charted and provide the citations.",
         f"Table 1 (all {n_cat} catalogued studies: author and year, N, "
         "country, mode, education level, era, region, instrument, "
         "outcome, inclusion status, risk-of-bias score) and Results, "
         "\"Characteristics of Included Studies\". Full citations are in "
         "the References."),
        ("Critical appraisal within sources of evidence", "16",
         "If done, present data on critical appraisal of included sources "
         "of evidence (see item 12).",
         "Results, \"Risk of Bias Across Included Studies\" (mean, SD, "
         "range, and the studies scoring below 5) and the RoB column of "
         "Table 1. The sensitivity analysis excluding low-scoring studies "
         "is in Table 4."),
        ("Results of individual sources of evidence", "17",
         "For each included source of evidence, present the relevant data "
         "that were charted that relate to the review questions and "
         "objectives.",
         f"Figures 2 to 6 (forest plots): the correlation of each of the "
         f"{n_pool} pooled studies with every trait it reports "
         f"({n_direct} direct, {n_beta} converted from β). Table 1 gives "
         "each study's outcome and inclusion status, and Results, "
         "\"Characteristics of Included Studies\" describes the outcomes "
         "of the studies that were not pooled."),
        ("Synthesis of results", "18",
         "Summarize and/or present the charting results as they relate to "
         "the review questions and objectives.",
         "Results, \"Exploratory Pooled Effect Sizes\" and Table 2 (MQ1; "
         f"e.g., Conscientiousness k = {k_c}, N = {n_total}, r = {c_r}, "
         f"95% CI {c_ci}, 95% PI {c_pi}; k = {k_o} for the "
         "other traits except Neuroticism), \"Between-Study "
         "Heterogeneity\", \"Moderator Analyses\" and Table 3 (MQ2), "
         "\"Sensitivity Analyses\" and Table 4, \"Publication-Bias "
         "Diagnostics (descriptive only)\" and Figures 7 to 11, and "
         "\"Completeness-of-Evidence Summary\" and Table 5."),

        ("DISCUSSION", "", "", ""),
        ("Summary of evidence", "19",
         "Summarize the main results (including an overview of concepts, "
         "themes, and types of evidence available), link to the review "
         "questions and objectives, and consider the relevance to key "
         "groups.",
         "Discussion, \"Summary of Findings\", \"Evaluation of the Mapping "
         "Priors\" (MP1 to MP5 against the face-to-face benchmarks), "
         "\"Theoretical Implications\", \"Hypotheses for Future Practical "
         "Testing\" (HPT-1 to HPT-3, framed as hypotheses rather than "
         "recommendations), and \"Distinguishing Robust from Fragile "
         "Findings\"."),
        ("Limitations", "20",
         "Discuss the limitations of the scoping review process.",
         "Discussion, \"Limitations\": eight limitations, including the "
         f"small pool (k = {n_pool}), the search that did not cover the "
         "pre-registered databases, unlogged screening counts, "
         "single-reviewer screening, extraction, and appraisal with "
         "language-model assistance and without the pre-registered "
         "reliability checks, analysis errors found and corrected during "
         "revision, the post hoc removal of one study from the pool, the "
         "English-language restriction, outcome heterogeneity, the "
         "author's own study among the retained studies, and the absence "
         "of facet-level data."),
        ("Conclusions", "21",
         "Provide a general interpretation of the results with respect to "
         "the review questions and objectives, as well as potential "
         "implications and/or next steps.",
         "Conclusion, and Discussion, \"Future Research Directions\": the "
         "review is presented as a map of the evidence and its gaps rather "
         "than a settled effect size; next steps are a replication search "
         "of the pre-registered databases with record-level logging and a "
         "second reviewer, and primary studies listed (a) to (e)."),

        ("FUNDING", "", "", ""),
        ("Funding", "22",
         "Describe sources of funding for the included sources of "
         "evidence, as well as sources of funding for the scoping review. "
         "Describe the role of the funders of the scoping review.",
         "Declarations, \"Funding\": the review received no specific grant. "
         "Funding of the included primary studies was not extracted."),
    ]


# --------------------------------------------------------------------------
# Document
# --------------------------------------------------------------------------

def build_checklist(anonymous: bool, out_path: Path):
    doc = Document()
    s = doc.sections[0]
    s.top_margin = Cm(1.5)
    s.bottom_margin = Cm(1.5)
    s.left_margin = Cm(1.8)
    s.right_margin = Cm(1.8)
    s.page_width, s.page_height = s.page_height, s.page_width  # landscape

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(4)
    _set_run_font(p.add_run("PRISMA-ScR Checklist (Tricco et al., 2018)"),
                  size_pt=14, bold=True)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(8)
    _set_run_font(p.add_run(TITLE), size_pt=11)

    if anonymous:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(10)
        _set_run_font(p.add_run(
            "[Author identifying information removed for double-anonymous "
            "peer review]"), size_pt=10)

    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(10)
    _set_run_font(p.add_run(
        "This checklist follows the PRISMA Extension for Scoping Reviews "
        "(PRISMA-ScR): Tricco, A. C., Lillie, E., Zarin, W., et al. (2018). "
        "PRISMA Extension for Scoping Reviews (PRISMA-ScR): Checklist and "
        "explanation. Annals of Internal Medicine, 169(7), 467–473. "
        "https://doi.org/10.7326/M18-0850. Items are numbered 1–22. The "
        "\"Location reported\" column names the section and subsection "
        "headings of the manuscript where each item is reported, rather "
        "than page numbers, which change between the submitted and the "
        "typeset versions. Where an item was done only in part, the "
        "column says so and points to the Deviations and Limitations "
        "subsections."), size_pt=10)

    items = _build_items(anonymous)
    headers = ["Section", "#", "Checklist item", "Location reported"]
    widths = [Cm(3.2), Cm(1.2), Cm(8.5), Cm(11.5)]
    table = doc.add_table(rows=len(items) + 1, cols=len(headers))
    table.autofit = False

    for ci_, h in enumerate(headers):
        cell = table.rows[0].cells[ci_]
        cell.width = widths[ci_]
        cell.text = ""
        _set_run_font(cell.paragraphs[0].add_run(h), size_pt=10, bold=True)
        _set_cell_shading(cell, "D9D9D9")
        _set_cell_borders(cell)

    for ri, (section, num, item, loc) in enumerate(items, start=1):
        is_header = (num == "" and item == "" and loc == "")
        for ci_, value in enumerate([section, num, item, loc]):
            cell = table.rows[ri].cells[ci_]
            cell.width = widths[ci_]
            cell.text = ""
            _set_run_font(cell.paragraphs[0].add_run(value), size_pt=9,
                          bold=is_header)
            if is_header:
                _set_cell_shading(cell, "F2F2F2")
            _set_cell_borders(cell)

    doc.save(str(out_path))
    print(f"Wrote {out_path}")


def main():
    build_checklist(anonymous=False, out_path=OUT_NON_ANON)
    build_checklist(anonymous=True, out_path=OUT_ANON)


if __name__ == "__main__":
    main()
