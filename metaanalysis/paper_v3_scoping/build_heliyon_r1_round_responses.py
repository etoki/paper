"""
build_heliyon_r1_round_responses.py — Response letters for the
Heliyon minor-revision round (2026-07-14, R1 outcome).

Reviewer 1 essentially accepted the previous revision ("Good luck!"),
so their response is a short thank-you.

Reviewer 2 raised four minor items and one verification item; those
are addressed point-by-point.

Outputs:
    response_heliyon_reviewer1_r1round.docx  (thank-you)
    response_heliyon_reviewer2_r1round.docx  (point-by-point)
"""

from datetime import date
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt


HERE = Path(__file__).resolve().parent


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


def build_r1_thank_you(out_path):
    doc = Document()
    s = doc.sections[0]
    s.top_margin = Cm(2.2)
    s.bottom_margin = Cm(2.2)
    s.left_margin = Cm(2.5)
    s.right_margin = Cm(2.5)

    _add_p(
        doc, "Response to Reviewer 1 — Minor-Revision Round",
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
        f"Heliyon  |  Manuscript ID HELIYON-D-26-02879R1  |  "
        f"Response date: {date.today().strftime('%d %B %Y')}",
        size=10, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=Pt(18),
    )

    _add_p(
        doc,
        "I am very grateful to Reviewer 1 for the generous evaluation of "
        "the revised manuscript and for confirming that the substantive "
        "changes made in the previous revision — the structural reframing "
        "to a scoping review under PRISMA-ScR, the addition of Trait "
        "Activation Theory alongside the Personality-Achievement "
        "Saturation Hypothesis, the reporting of actual intra-rater "
        "reliability values and the detailed β-to-r conversion "
        "procedures, and the careful contextualisation of the fragile "
        "pooled estimates — have resulted in a rigorous and well-"
        "calibrated contribution.",
        size=11,
    )
    _add_p(
        doc,
        "Reviewer 1 did not raise additional revision requests in this "
        "round, so no substantive new changes were made in response to "
        "Reviewer 1 specifically. The minor-revision items raised by "
        "Reviewer 2 have been addressed in a separate point-by-point "
        "response, and the corresponding changes to the manuscript "
        "(language audit for residual causal / predictive terminology; "
        "explicit descriptive-only framing of the publication-bias "
        "diagnostics; explicit inclusion of subject discipline as a "
        "candidate moderator in Future Research Directions; further "
        "softening of Practical Implications into a \"Hypotheses for "
        "Future Practical Testing\" subsection) are indicated by yellow "
        "highlighting in the revised manuscript.",
        size=11,
    )
    _add_p(
        doc,
        "Verification note on Trait Activation Theory integration "
        "(Comment #3 / #18 of the previous round). The previous "
        "response letter reported that Trait Activation Theory (Tett "
        "& Burnett, 2003) had been added as a complementary theoretical "
        "anchor alongside PASH; on preparing this round of revisions, I "
        "audited the manuscript and confirmed — following the same "
        "check that Reviewer 2 correctly applied to the subject-"
        "discipline moderator addition — that the TAT paragraph was not "
        "in fact present in the previously submitted manuscript body, "
        "even though it was described in the previous response letter. "
        "This has now been remedied. An explicit TAT paragraph has "
        "been added to the Introduction alongside the PASH review, "
        "articulating three specific TAT-derived expectations for the "
        "Big Five in online contexts (Extraversion attenuation in "
        "asynchronous formats, Conscientiousness preservation across "
        "modalities, and Agreeableness amplification in cooperative "
        "cultural contexts). A second TAT reference has been added to "
        "the Discussion's Extraversion subsection, and TAT is used "
        "alongside PASH in the Discussion's Conscientiousness "
        "subsection to distinguish two competing accounts of the "
        "preserved-but-attenuated pattern. I apologise for the "
        "oversight in the previous round and am grateful for the "
        "opportunity to correct it before publication.",
        size=11,
    )

    _add_p(doc, "", size=11, space_before=Pt(4))
    _add_p(
        doc,
        "Thank you again for the careful review across both rounds. "
        "I look forward to the editor's final decision.",
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

    doc.save(str(out_path))
    print(f"Wrote {out_path}")


def build_r2_pbp(out_path):
    doc = Document()
    s = doc.sections[0]
    s.top_margin = Cm(2.2)
    s.bottom_margin = Cm(2.2)
    s.left_margin = Cm(2.5)
    s.right_margin = Cm(2.5)

    _add_p(
        doc, "Point-by-Point Response to Reviewer 2 — Minor-Revision Round",
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
        f"Heliyon  |  Manuscript ID HELIYON-D-26-02879R1  |  "
        f"Response date: {date.today().strftime('%d %B %Y')}",
        size=10, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=Pt(18),
    )

    _add_p(
        doc,
        "I am grateful to Reviewer 2 for a second thoughtful review and "
        "for the four specific minor-revision items raised in this "
        "round. Each item is addressed in turn below. Corresponding "
        "changes to the manuscript are indicated by yellow highlighting "
        "in the revised file.",
        size=11,
    )

    items = [
        ("R2-1 — Residual causal / directional wording",
         "Reviewer 2 flagged specific terms — \"predictor,\" \"predictive "
         "validity,\" \"effects,\" and \"causal pathway\" — as still "
         "implying causality that the predominantly correlational, "
         "cross-sectional evidence base does not support.",
         "A second-pass language audit has been applied to the entire "
         "manuscript to replace these terms with association / "
         "correlation language wherever the claim is correlational "
         "rather than causal or regression-model-technical. Specifically: "
         "\"predictor\" (used as a substantive claim about traits) has "
         "been replaced with \"correlate\" throughout the Introduction, "
         "Results, and Discussion; \"predictive validity\" has been "
         "replaced with \"correlational validity\"; \"causal pathway\" "
         "has been replaced with \"hypothesised association pathway\"; "
         "and residual instances of \"predicts,\" \"predicting,\" and "
         "\"effect(s)\" have been replaced with \"is associated with,\" "
         "\"correlated with,\" and \"estimate\" / \"association\" as "
         "appropriate to each context. The five remaining occurrences of "
         "the string \"predictor\" in the revised manuscript are all "
         "statistical-model or bibliographic in nature and do not carry "
         "causal implications: (i) \"at most two predictors\" and "
         "(ii) \"single-predictor models\" in the β-to-r conversion "
         "procedure describe the source regression models; (iii) and "
         "(iv) \"k ≥ 10 per predictor level\" describe the meta-"
         "regression power threshold from Borenstein et al. (2021); "
         "(v) is inside the title of the Quigley et al. (2022) reference "
         "list entry and cannot be altered."),

        ("R2-2 — Practical implications framed as tentative "
         "recommendations",
         "Reviewer 2 asked that the practical implications be softened "
         "further and framed more explicitly as \"hypotheses for future "
         "testing\" rather than as recommendations.",
         "The subsection previously titled \"Tentative Practical "
         "Implications\" has been renamed \"Hypotheses for Future "
         "Practical Testing.\" The subsection now opens with an "
         "explicit statement that \"every item below is framed as a "
         "hypothesis that requires empirical validation in independent "
         "primary research before being operationalised\" and that "
         "\"the subsection is deliberately not framed as 'practical "
         "implications' or 'recommendations'.\" Each of the three "
         "substantive items has been rewritten as a labelled "
         "\"Hypothesis for future practical testing\" (HPT-1, HPT-2, "
         "HPT-3), and the wording within each hypothesis is now "
         "explicit that the hypothesis \"requires empirical validation\" "
         "and is \"offered ... for future empirical testing rather "
         "than as actionable guidance.\" Design and instructional "
         "language has been removed."),

        ("R2-3 — Subject discipline as a future moderator (verification)",
         "Reviewer 2 noted that the previous response letter claimed "
         "the subject-discipline moderator was incorporated, but that "
         "this addition is not clearly identifiable in the revised "
         "manuscript.",
         "Reviewer 2 is entirely correct: in the previous round the "
         "subject-discipline addition was flagged in the response "
         "letter but did not, on inspection, appear as an "
         "identifiable, self-contained paragraph in the manuscript "
         "itself. This has now been remedied. A new stand-alone "
         "paragraph (item \"Sixth\") has been added to the Future "
         "Research Directions subsection. The paragraph explicitly "
         "(a) attributes the suggestion to the peer-review process, "
         "(b) describes the current heterogeneity of disciplines in "
         "the scoping-retained studies (psychology, education, IT / "
         "computing, health sciences, business, and mixed general-"
         "education contexts), (c) explains why a formal discipline-"
         "moderator analysis is not possible in the present k, "
         "(d) recommends that discipline be recorded as a first-order "
         "extraction field in future updates, (e) proposes two "
         "specific testable sub-hypotheses (Conscientiousness × STEM-"
         "vs-humanities, Extraversion × discipline-typical peer-"
         "interaction level), and (f) recommends categorical rather "
         "than free-text coding for reliable pooling."),

        ("R2-4 — Publication-bias analyses framed as descriptive only",
         "Reviewer 2 asked that the publication-bias analyses be framed "
         "as descriptive only, with no implication that bias has been "
         "ruled out.",
         "Both the Methods → Publication Bias subsection and the "
         "Results → Publication Bias Assessment subsection have been "
         "revised to state this framing explicitly. The Methods "
         "subsection now opens with the sentence: \"Publication-bias "
         "diagnostics were computed per trait using four complementary "
         "procedures, and are reported for descriptive purposes only "
         "rather than as inferential tests capable of ruling out "
         "bias.\" It further states that all four diagnostics and the "
         "p-curve analysis are underpowered at the present k = 9–10 "
         "and that \"non-significant results from any of the "
         "diagnostics should not be interpreted as evidence that "
         "publication bias is absent.\" The Results subsection has "
         "been retitled \"Publication-Bias Diagnostics (descriptive "
         "only)\" and now opens with an explicit descriptive-only "
         "caveat that names each of the four diagnostics (funnel plot, "
         "Egger's, Peters', trim-and-fill), states that none of them "
         "can rule out publication bias at k = 9–10, and warns that "
         "the trim-and-fill imputed studies should not be interpreted "
         "as bias-corrected estimates. The narrative discussion of the "
         "diagnostic results is reworded to emphasise that "
         "\"non-significant results should not be interpreted as "
         "evidence that publication bias is absent\" and that \"small-"
         "study positive bias for Conscientiousness in particular "
         "cannot be ruled out on the basis of the reported "
         "diagnostics.\""),
    ]

    for heading, reviewer, response in items:
        _add_p(doc, heading, bold=True, size=12, space_before=Pt(10))
        _add_p(doc, "Reviewer 2's comment.", italic=True, size=11)
        _add_p(doc, reviewer, size=11)
        _add_p(doc, "Response.", bold=True, size=11)
        _add_p(doc, response, size=11)

    _add_p(doc, "Closing", bold=True, size=12, space_before=Pt(14))
    _add_p(
        doc,
        "All four minor-revision items raised by Reviewer 2 have been "
        "addressed, and the missing subject-discipline paragraph "
        "correctly noted by Reviewer 2 has been added as a stand-alone "
        "paragraph in the Future Research Directions subsection. I am "
        "grateful to Reviewer 2 for the careful checking across both "
        "rounds and for catching the missing paragraph.",
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

    doc.save(str(out_path))
    print(f"Wrote {out_path}")


def main():
    build_r1_thank_you(HERE / "response_heliyon_reviewer1_r1round.docx")
    build_r2_pbp(HERE / "response_heliyon_reviewer2_r1round.docx")


if __name__ == "__main__":
    main()
