"""
Build APA-formatted Introduction docx for the Big Five × Online Learning meta-analysis.
Matches the style of Manuscript_sample.docx (Times New Roman 12pt, 1-inch margins).

Built incrementally: Title page + Declarations first, then Abstract, then Intro sections.
"""
from docx import Document
from docx.shared import Pt, Inches, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml.ns import qn
from docx.oxml import OxmlElement


OUTPUT = "/home/user/paper/metaanalysis/paper/introduction.docx"


def set_cell_font(paragraph, font_name="Times New Roman", size_pt=12, bold=None):
    for run in paragraph.runs:
        run.font.name = font_name
        run.font.size = Pt(size_pt)
        if bold is not None:
            run.bold = bold
        # East Asia font fallback
        rPr = run._element.get_or_add_rPr()
        rFonts = rPr.find(qn("w:rFonts"))
        if rFonts is None:
            rFonts = OxmlElement("w:rFonts")
            rPr.append(rFonts)
        rFonts.set(qn("w:eastAsia"), font_name)


def set_double_space(paragraph):
    pf = paragraph.paragraph_format
    pf.line_spacing = 2.0
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)


def add_para(doc, text, style=None, bold=False, align=None, indent_first=False):
    p = doc.add_paragraph(text, style=style)
    if align is not None:
        p.alignment = align
    if indent_first:
        p.paragraph_format.first_line_indent = Inches(0.5)
    set_double_space(p)
    set_cell_font(p, bold=bold if bold else None)
    return p


def configure_styles(doc):
    """Set default styles to APA (Times New Roman 12pt, double-spaced)."""
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(12)
    # East Asia
    rPr = normal.element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.append(rFonts)
    rFonts.set(qn("w:eastAsia"), "Times New Roman")

    # Heading 1
    h1 = doc.styles["Heading 1"]
    h1.font.name = "Times New Roman"
    h1.font.size = Pt(12)
    h1.font.bold = True
    h1.font.color.rgb = RGBColor(0, 0, 0)
    h1.paragraph_format.space_before = Pt(0)
    h1.paragraph_format.space_after = Pt(0)

    # Heading 2
    h2 = doc.styles["Heading 2"]
    h2.font.name = "Times New Roman"
    h2.font.size = Pt(12)
    h2.font.bold = True
    h2.font.color.rgb = RGBColor(0, 0, 0)
    h2.paragraph_format.space_before = Pt(0)
    h2.paragraph_format.space_after = Pt(0)


def configure_page(doc):
    s = doc.sections[0]
    s.top_margin = Inches(1.0)
    s.bottom_margin = Inches(1.0)
    s.left_margin = Inches(1.0)
    s.right_margin = Inches(1.0)


def build_title_page(doc):
    """APA-style title page (centered title, author block)."""
    # A few blank lines before title (APA top spacing)
    for _ in range(4):
        add_para(doc, "", align=WD_ALIGN_PARAGRAPH.CENTER)

    # Title (bold, centered)
    add_para(
        doc,
        "Big Five Personality Traits and Academic Achievement in Online Learning "
        "Environments: A Systematic Review and Meta-Analysis",
        bold=True,
        align=WD_ALIGN_PARAGRAPH.CENTER,
    )
    add_para(doc, "", align=WD_ALIGN_PARAGRAPH.CENTER)
    add_para(doc, "", align=WD_ALIGN_PARAGRAPH.CENTER)

    # Author block
    for line in [
        "Eisuke Tokiwa",
        "Founder of SUNBLAZE Co., Ltd.",
        "Tokyo, Japan",
        "eisuke.tokiwa@sunblaze.jp",
        "ORCID: 0009-0009-7124-6669",
    ]:
        add_para(doc, line, align=WD_ALIGN_PARAGRAPH.CENTER)


def build_declarations(doc):
    add_para(doc, "")  # blank line
    add_para(doc, "Declarations", bold=True)

    sections = [
        ("Conflict of Interest",
         "The author declares that there are no financial or personal relationships with "
         "other people or organizations that could inappropriately influence this work, "
         "except that one of the author's prior primary studies (Tokiwa, 2025) is "
         "potentially eligible for inclusion in the present meta-analysis. This conflict "
         "is addressed through a pre-specified sensitivity analysis excluding the "
         "author's own study."),
        ("Funding",
         "This research did not receive any specific grant from funding agencies in the "
         "public, commercial, or not-for-profit sectors."),
        ("Ethics Approval",
         "Not applicable. This meta-analysis synthesizes data from previously published "
         "studies and does not involve direct collection of data from human participants. "
         "Ethical approval for the original studies was the responsibility of the authors "
         "of those studies."),
        ("Informed Consent",
         "Not applicable. All included studies obtained informed consent from their "
         "participants in accordance with institutional and journal requirements at the "
         "time of original data collection."),
        ("Pre-registration",
         "This systematic review and meta-analysis was pre-registered on OSF Registries "
         "(https://osf.io/e5w47/; DOI: 10.17605/OSF.IO/E5W47) on April 23, 2026, prior to "
         "formal data extraction and quantitative synthesis. The full protocol (PRISMA-P "
         "2015 compliant) and supplementary materials are publicly available in the "
         "associated OSF project (https://osf.io/79m5j/)."),
        ("Availability of data and material",
         "All extracted data, analysis code, search logs, and supplementary materials are "
         "publicly available on the OSF project page (https://osf.io/79m5j/) and the "
         "accompanying GitHub repository (https://github.com/etoki/paper, directory "
         "metaanalysis/)."),
        ("Authors' contributions",
         "The author (ET) conceived and designed the review, developed the search strategy "
         "and eligibility criteria, performed the database searches, conducted screening, "
         "extracted data, performed the statistical analyses, and drafted and revised the "
         "manuscript. The author approves the final version."),
    ]

    for heading, body in sections:
        add_para(doc, heading, bold=True)
        add_para(doc, body)
        add_para(doc, "")  # blank line between sections


def build_abstract(doc):
    """APA Abstract section (Heading 1 style, block paragraph, keywords)."""
    p = doc.add_paragraph("Abstract", style="Heading 1")
    set_double_space(p)

    abstract_body = (
        "Academic achievement in online learning environments has become a central "
        "concern for post-secondary education following the global shift toward digital "
        "instruction, yet existing meta-analyses of Big Five personality traits and "
        "academic performance have pooled samples across face-to-face, blended, and "
        "online modalities without testing delivery mode as a substantive moderator. "
        "The present systematic review and meta-analysis is the first quantitative "
        "synthesis dedicated to online learning environments. The review was pre-"
        "registered on OSF Registries prior to data extraction. Following PRISMA 2020 "
        "reporting standards, correlational primary studies reporting associations "
        "between Big Five (or HEXACO) personality traits and academic achievement in "
        "fully online, blended, MOOC, synchronous online, or asynchronous online "
        "environments were identified through a structured search of multiple "
        "databases and supplementary sources. Effect sizes were pooled using random-"
        "effects meta-analysis with Restricted Maximum Likelihood estimation and "
        "Hartung-Knapp-Sidik-Jonkman confidence-interval adjustment; Fisher's z "
        "transformation was applied prior to pooling, and robust variance estimation "
        "was used to handle dependent effect sizes. Pre-specified moderator analyses "
        "examined learning modality, education level, region, era, outcome type, and "
        "personality instrument. [Results and conclusions to be finalized after the "
        "quantitative synthesis is completed. This placeholder will be replaced with "
        "concrete pooled effect sizes, heterogeneity statistics, moderator findings, "
        "and GRADE-based confidence ratings.]"
    )
    add_para(doc, abstract_body)
    add_para(doc, "")

    p_kw = add_para(
        doc,
        "Keywords: Big Five; Five-Factor Model; HEXACO; online learning; e-learning; "
        "MOOC; academic achievement; meta-analysis; systematic review; PRISMA",
    )
    # APA keywords use italicized "Keywords:" label; for simplicity we keep plain.


def add_h2(doc, text):
    p = doc.add_paragraph(text, style="Heading 2")
    set_double_space(p)
    return p


def build_intro_part1(doc):
    """Introduction opening + Personality benchmarks subsection."""
    p = doc.add_paragraph("Introduction", style="Heading 1")
    set_double_space(p)

    opening = (
        "Academic performance in higher education has shifted dramatically in the past "
        "decade, first through the gradual adoption of learning management systems and "
        "massive open online courses, and then, abruptly, through the COVID-19 pandemic, "
        "which forced millions of students worldwide into fully online or blended "
        "learning environments. As of 2023, a substantial proportion of post-secondary "
        "instruction continues to be delivered online or in hybrid modalities, even in "
        "contexts where face-to-face teaching is again available. This structural shift "
        "raises a fundamental question: do the non-cognitive predictors of academic "
        "achievement that were established in face-to-face classrooms retain their "
        "predictive validity when instruction is mediated by technology? Personality "
        "traits, and in particular the Big Five (Five-Factor Model) dimensions of "
        "Conscientiousness, Openness to Experience, Extraversion, Agreeableness, and "
        "Neuroticism, are among the most widely studied non-cognitive predictors of "
        "academic achievement (Mammadov, 2022; Poropat, 2009). Yet the accumulated meta-"
        "analytic evidence, as reviewed below, has been produced almost entirely from "
        "samples in which delivery modality is treated as noise rather than as a "
        "substantive moderator. The present review is the first to address this gap "
        "quantitatively."
    )
    add_para(doc, opening, indent_first=True)

    add_h2(doc, "Personality and Academic Achievement in Face-to-Face Contexts")

    p1 = (
        "Eight large-scale meta-analyses have established a robust pattern of "
        "associations between the Big Five personality traits and academic achievement. "
        "Poropat (2009), synthesizing 138 samples and over 70,000 participants, "
        "reported a corrected correlation of ρ = .22 between Conscientiousness and "
        "academic performance, with smaller but significant effects for Openness "
        "(ρ = .12) and Agreeableness (ρ = .07), and essentially null effects for "
        "Extraversion (ρ = −.01) and Neuroticism (ρ = .02). These findings were "
        "replicated and extended by McAbee and Oswald (2013; k = 57, N = 26,382; C "
        "ρ = .26) and by Vedel (2014; k = 21, N = 17,717; C ρ = .26), both restricted "
        "to tertiary education. Stajkovic, Bandura, Locke, Lee, and Sergent (2018) "
        "provided additional evidence that Conscientiousness predicts academic "
        "achievement both directly and indirectly through self-efficacy, with the "
        "self-efficacy path carrying a substantial proportion of the Conscientiousness "
        "effect (β = .24 to .33)."
    )
    add_para(doc, p1, indent_first=True)

    p2 = (
        "More recent syntheses have substantially expanded the evidence base. Mammadov "
        "(2022), in the largest meta-analysis to date (267 independent samples, N = "
        "413,074), reported a corrected pooled correlation of ρ = .27 for "
        "Conscientiousness and ρ = .16 for Openness, and documented a striking "
        "cultural moderation effect in which Asian samples showed markedly stronger "
        "associations (C ρ = .35; A ρ = .23; N ρ = −.19). Meyer, Jansen, Hübner, and "
        "Lüdtke (2023), focusing exclusively on K-12 samples (110 samples, N = "
        "500,218), obtained even larger estimates (C ρ = .24; O ρ = .21) and "
        "identified domain specificity as a critical moderator: Openness was "
        "substantially stronger for language than for STEM domains, while "
        "Conscientiousness effects were larger for grades than for standardized tests. "
        "This latter finding was theorized under the Personality-Achievement Saturation "
        "Hypothesis (PASH), which posits that the behavioral signals captured by "
        "Conscientiousness (e.g., on-task engagement, homework completion) are more "
        "visible to teachers assigning grades than to standardized test scorers. Zell "
        "and Lesick (2021), in a second-order synthesis of 54 meta-analyses, confirmed "
        "the stability of these effects, with academic-specific estimates of C ρ = "
        ".28 and E ρ = −.01. Most recently, Chen, Cheung, and Zeng (2025), examining "
        "84 articles and 370 independent correlations restricted to university students "
        "with samples of at least 200 participants, reported somewhat smaller estimates "
        "(C r = .206; O r = .081; A r = .082; E r = −.009; N r = −.029), consistent "
        "with the idea that stricter methodological inclusion criteria attenuate "
        "apparent trait-performance relationships."
    )
    add_para(doc, p2, indent_first=True)

    p3 = (
        "Across these eight meta-analyses, a convergent picture emerges: "
        "Conscientiousness is the dominant predictor of academic achievement "
        "(ρ = .19 to .28), with Openness serving as a reliable secondary predictor "
        "(ρ = .07 to .21). Agreeableness exhibits small positive effects "
        "(ρ = .04 to .10), and Extraversion and Neuroticism are generally near zero "
        "in academic contexts, although both may show meaningful directional effects "
        "under specific moderators. Critically, however, none of these eight meta-"
        "analyses has tested learning modality—online, blended, face-to-face, or "
        "MOOC—as a moderator. The samples pooled in these syntheses were drawn almost "
        "entirely from traditional classroom contexts, or from contexts in which "
        "delivery modality was not reported."
    )
    add_para(doc, p3, indent_first=True)


def main():
    doc = Document()
    configure_page(doc)
    configure_styles(doc)
    build_title_page(doc)
    doc.add_page_break()
    build_declarations(doc)
    doc.add_page_break()
    build_abstract(doc)
    build_intro_part1(doc)
    doc.save(OUTPUT)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
