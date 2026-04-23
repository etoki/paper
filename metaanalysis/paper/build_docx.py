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


def main():
    doc = Document()
    configure_page(doc)
    configure_styles(doc)
    build_title_page(doc)
    doc.add_page_break()
    build_declarations(doc)
    doc.add_page_break()
    build_abstract(doc)
    doc.save(OUTPUT)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
