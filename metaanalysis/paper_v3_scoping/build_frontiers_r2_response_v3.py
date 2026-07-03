"""
build_frontiers_r2_response_v3.py — Updated point-by-point response
to Reviewer 2 of Frontiers in Education (Manuscript ID 1866537) for
the scoping-review-reframed v3 manuscript.

Context:
    * Reviewer 2 previously recommended "Accept in current form" on
      04 Jun 2026 with seven minor recommendations for strengthening
      the manuscript before publication.
    * The Frontiers editorial office reactivated both reviewers on
      02 Jul 2026 to evaluate the scoping-reframed v3 manuscript.
    * Reviewer 2's response is therefore needed to acknowledge the
      structural reframing and to confirm that the seven earlier
      recommendations remain addressed.

Output: response_frontiers_reviewer2_v3.docx
"""

from datetime import date
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt


HERE = Path(__file__).resolve().parent
OUT = HERE / "response_frontiers_reviewer2_v3.docx"


def _add_p(doc, text, *, bold=False, italic=False, size=11, align=None,
           space_after=Pt(6), space_before=Pt(0)):
    p = doc.add_paragraph()
    if align is not None:
        p.alignment = align
    p.paragraph_format.space_before = space_before
    p.paragraph_format.space_after = space_after
    p.paragraph_format.line_spacing = 1.3
    run = p.add_run(text)
    run.font.size = Pt(size)
    if bold:
        run.bold = True
    if italic:
        run.italic = True
    return p


def _bullet(doc, text, size=10):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(3)
    run = p.add_run(text)
    run.font.size = Pt(size)


def main():
    doc = Document()
    s = doc.sections[0]
    s.top_margin = Cm(2.2)
    s.bottom_margin = Cm(2.2)
    s.left_margin = Cm(2.5)
    s.right_margin = Cm(2.5)

    _add_p(
        doc, "Updated Response to Reviewer 2 (scoping-review reframing)",
        bold=True, size=14, align=WD_ALIGN_PARAGRAPH.CENTER,
        space_after=Pt(4),
    )
    _add_p(
        doc,
        "Big Five Personality Traits and Academic Achievement in Online "
        "Learning Environments: A Scoping Review with Exploratory "
        "Quantitative Synthesis",
        size=11, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=Pt(8),
    )
    _add_p(
        doc,
        f"Frontiers in Education  |  Manuscript ID 1866537  |  "
        f"Response date: {date.today().strftime('%d %B %Y')}",
        size=10, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=Pt(18),
    )

    _add_p(
        doc,
        "I am very grateful to Reviewer 2 for the supportive evaluation "
        "of the previous version (\"Accept in current form,\" 04 Jun "
        "2026) and for the seven constructive recommendations for "
        "strengthening the manuscript before publication. Following "
        "the reactivation of the review by the Frontiers in Education "
        "editorial office on 02 Jul 2026, I would like to briefly "
        "explain the structural change made to the manuscript since "
        "the previous review round and to confirm that each of "
        "Reviewer 2's earlier recommendations has been addressed in "
        "the revised version.",
        size=11,
    )

    _add_p(
        doc,
        "Since the last review round, the manuscript has undergone a "
        "reframing from a systematic review and meta-analysis to a "
        "scoping review with an exploratory quantitative synthesis. "
        "The reason is a structural concern regarding the pre-"
        "registered six-database search: PsycINFO, Scopus, Web of "
        "Science, and ProQuest Dissertations require institutional "
        "subscription access that I do not have, and the providers "
        "(Clarivate, Elsevier, the American Psychological Association) "
        "do not offer individual subscriptions. On closer reflection I "
        "did not consider it possible to publish the manuscript "
        "honestly as a systematic review when the systematic-search "
        "criterion had not in fact been met. The reframing therefore "
        "revises the Title, Abstract, Methods → Information Sources "
        "subsection, and Conclusion to describe honestly what was and "
        "was not searched; adopts PRISMA-ScR (Tricco et al., 2018) as "
        "the reporting framework in place of PRISMA 2020; reframes the "
        "Hypotheses H1–H5 as descriptive Mapping Priors MP1–MP5; and "
        "recalibrates the language throughout from confirmatory to "
        "exploratory. The Results, the Methods sections other than "
        "Information Sources, the analytical pipeline, and the "
        "underlying data are unchanged from the previously reviewed "
        "version.",
        size=11,
    )

    _add_p(
        doc,
        "Reviewer 2's seven earlier recommendations have been "
        "addressed as follows in the revised (scoping-reframed) "
        "manuscript:",
        size=11,
    )

    items = [
        ("R2-1 — Small quantitative synthesis pool implications",
         "The Limitations section first paragraph now states "
         "explicitly that the exploratory quantitative pool of k = 10 "
         "(6 direct + 4 β-converted correlations) is at the lower "
         "bound of robust random-effects estimation, that pooled "
         "estimates are unstable, that heterogeneity is poorly "
         "estimated, and that prediction intervals are wide. The "
         "Abstract and Conclusion now lead with this small-k caveat, "
         "and the Discussion adds a Distinguishing Robust from "
         "Fragile Findings subsection classifying Conscientiousness "
         "as the relatively robust finding and marking the other "
         "traits and the subgroup contrasts as fragile."),

        ("R2-2 — Single-reviewer screening/extraction rationale",
         "The Limitations third paragraph now states that intra-rater "
         "reliability cannot replicate inter-rater independence and "
         "that single-reviewer workflows are known to be susceptible "
         "to confirmation effects in study inclusion and extraction-"
         "convention drift. Actual observed intra-rater Cohen's kappa "
         "values are reported in the Methods → Study Selection "
         "subsection rather than only the target threshold."),

        ("R2-3 — Blended vs fully online modality clarification",
         "The Methods → Eligibility Criteria subsection now explicitly "
         "distinguishes fully asynchronous, fully synchronous online, "
         "blended, and MOOC modalities, and reports the per-study "
         "modality coding in Table 1. The Discussion notes explicitly "
         "that pooling these modalities in the exploratory pool is a "
         "methodological simplification imposed by the small k and "
         "that the descriptive modality subgroup contrast in Table 3 "
         "should be treated as exploratory."),

        ("R2-4 — Converted effect-size limitations",
         "The Methods → Effect-Size Conversion subsection expands the "
         "discussion of the Peterson and Brown (2005) β-to-r "
         "conversion assumptions, the two-predictor rule applied, and "
         "the limits of comparability of β-converted estimates with "
         "zero-order correlations. The β-converted-excluded "
         "sensitivity analysis is now reported in the Results section "
         "with explicit attention to the |Δr| shifts for Neuroticism "
         "and Openness."),

        ("R2-5 — Publication-bias power",
         "The Methods → Publication-Bias Assessment subsection and "
         "the Limitations section now state explicitly that Egger's "
         "regression and trim-and-fill analyses are underpowered at "
         "k = 10 and that the publication-bias results are presented "
         "descriptively rather than as inferential tests of bias."),

        ("R2-6 — Cautious causal language",
         "An audit pass has been applied to the Abstract, Results, "
         "and Discussion to ensure that the language consistently "
         "emphasises association rather than causal prediction; "
         "\"predict,\" \"predictor,\" and \"effect\" have been "
         "replaced with \"correlate with,\" \"is associated with,\" "
         "or \"pooled estimate\" wherever the analytic claim is "
         "correlational rather than causal."),

        ("R2-7 — Tightened practical-implications subsection",
         "The previous Practical Implications subsection has been "
         "renamed Tentative Practical Implications and now opens with "
         "an explicit caveat paragraph stating that the implications "
         "are tentative working hypotheses rather than actionable "
         "design or pedagogical recommendations. The implications "
         "themselves have been consolidated and reframed as "
         "exploratory leads for primary-research replication."),

        ("Additional — Subject discipline as a future moderator",
         "Reviewer 2's suggestion of subject discipline (or "
         "discipline group) as a potential moderator has been added "
         "to the Future Research Directions subsection as a priority "
         "for a subsequent meta-analytic update, alongside facet-"
         "level analyses and within-learner modality contrasts."),
    ]

    for heading, body in items:
        _add_p(doc, heading, bold=True, size=12, space_before=Pt(10))
        _add_p(doc, "Response.", bold=True, size=11)
        _add_p(doc, body, size=11)

    _add_p(doc, "Closing", bold=True, size=12, space_before=Pt(14))
    _add_p(
        doc,
        "All seven Reviewer 2 recommendations have been incorporated "
        "into the revised (scoping-reframed) manuscript, and the "
        "additional research-direction suggestion (subject discipline "
        "as a moderator) has been added to the Future Research "
        "Directions subsection. I am grateful to Reviewer 2 for the "
        "constructive evaluation and hope that the reframing "
        "clarifies rather than complicates the manuscript's "
        "presentation.",
        size=11,
    )

    _add_p(doc, "Sincerely,", size=11, space_before=Pt(6))
    _add_p(doc, "", size=11)
    for line in [
        "Eisuke Tokiwa",
        "Founder, SUNBLAZE Co., Ltd., Tokyo, Japan",
        "ORCID: 0009-0009-7124-6669",
        "Email: eisuke.tokiwa@sunblaze.jp",
    ]:
        _add_p(doc, line, size=10, space_after=Pt(0))

    doc.save(str(OUT))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
