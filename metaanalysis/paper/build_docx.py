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


def build_intro_part2(doc):
    """Online Shift + Fragmented Evidence subsections."""
    add_h2(doc, "The Shift to Online Learning Environments")

    p1 = (
        "Online learning environments differ from face-to-face instruction along at "
        "least four dimensions that have theoretical implications for the personality-"
        "achievement relationship. First, online environments impose heightened self-"
        "regulation demands on learners, who must manage their own time, attention, "
        "and task sequencing without the physical and social scaffolding provided by "
        "a traditional classroom (Broadbent & Poon, 2015). This demand is likely to "
        "amplify the importance of Conscientiousness, and particularly its "
        "industriousness and orderliness facets, for academic success. Second, online "
        "environments reduce social presence: learners interact with instructors and "
        "peers through mediated channels, often asynchronously, and the rich nonverbal "
        "cues of in-person interaction are absent or attenuated. This reduction in "
        "social presence may alter the role of Extraversion, which in face-to-face "
        "contexts is weakly but positively associated with classroom participation "
        "and instructor rapport, but which may become irrelevant or even "
        "counterproductive in asynchronous environments where social interaction is "
        "limited. Third, online environments afford greater temporal flexibility, "
        "which can either support self-directed learning or, conversely, facilitate "
        "procrastination (Cheng, Chang, Quilantan-Garza, & Gutierrez, 2023). Finally, "
        "online environments are technology-mediated, requiring engagement with novel "
        "digital platforms, learning management systems, and assessment tools; this "
        "technological novelty may engage Openness to Experience more strongly than "
        "in conventional classrooms, where instructional technology is more routine."
    )
    add_para(doc, p1, indent_first=True)

    p2 = (
        "These four dimensions—self-regulation demands, reduced social presence, "
        "temporal flexibility, and technology mediation—motivate specific predictions "
        "about how the personality-achievement relationship should differ in online "
        "as compared with face-to-face contexts. In particular, if the meta-analytic "
        "Conscientiousness effect of approximately ρ = .22–.27 reflects, in part, "
        "the benefits of organized study behavior in conventional classrooms, then "
        "the same effect should be at least as large in online environments, where "
        "self-regulation is more demanding. Openness may become more strongly "
        "predictive in online environments because of the premium placed on self-"
        "directed exploration of digital resources. Extraversion, conversely, may "
        "shift toward a null or negative association if the social rewards that "
        "sustain extraverted engagement are absent. Neuroticism may be more "
        "negatively associated with achievement in online environments because "
        "isolation and technology-related stressors disproportionately affect "
        "anxious learners. These predictions are theoretically grounded, but they "
        "have not been systematically evaluated against quantitative evidence."
    )
    add_para(doc, p2, indent_first=True)

    add_h2(doc, "Big Five Personality Traits in Online Learning: Fragmented Evidence")

    p3 = (
        "Individual primary studies of the Big Five and academic achievement in "
        "online learning have accumulated in substantial numbers since approximately "
        "2018, with a marked acceleration during the COVID-19 pandemic (2020–2022). "
        "The preliminary narrative synthesis most closely related to the present "
        "review is Hunter et al. (2025), who systematically screened 848 records "
        "from 2000 to June 2024 and included 23 primary studies. Their review "
        "documented a consistent positive association between Conscientiousness and "
        "online achievement (observed in 3 of 5 studies reporting GPA-type outcomes), "
        "negative associations with Neuroticism (in 3 of 5 studies), and essentially "
        "null associations with Extraversion (in all 5 studies). However, Hunter et "
        "al. (2025) explicitly did not pool effect sizes quantitatively, and the "
        "level-of-evidence rating for all 23 included studies was Level IV (low "
        "strength), precluding inferences about the magnitude or moderators of the "
        "observed associations."
    )
    add_para(doc, p3, indent_first=True)

    p4 = (
        "Examining the primary literature directly reveals considerable heterogeneity "
        "in reported effect sizes. In fully online asynchronous contexts, Abe (2020) "
        "reported strong correlations between Conscientiousness and both quiz "
        "performance (r = .48) and paper grades (r = .37), with Openness also "
        "predicting paper grades (r = .35). Alkış and Taşkaya Temizel (2018), "
        "comparing online and blended sections of the same course, found "
        "Conscientiousness to predict course grades in both modalities (online "
        "r = .205; blended r = .244), with other traits exhibiting only null "
        "effects. In the COVID-era German higher education context, Rodrigues, "
        "Rose, and Hewig (2024) reported a moderate negative correlation between "
        "Conscientiousness and self-reported GPA (r = −.228 with grades coded such "
        "that lower values indicate better performance), and significant negative "
        "associations between Neuroticism and study satisfaction and well-being. "
        "Rivers (2021), studying Japanese undergraduates in an asynchronous Moodle "
        "course (N = 149), reported a direct negative effect of Extraversion on "
        "course achievement (β = −.168, p < .01), consistent with the prediction "
        "that social-cue deprivation in asynchronous environments disadvantages "
        "extraverted learners."
    )
    add_para(doc, p4, indent_first=True)

    p5 = (
        "Other primary studies reveal patterns that diverge from conventional face-"
        "to-face findings. In a large Chinese K-12 sample studied by Wang, Wang, "
        "and Li (2023; N = 1,625), Big Five traits were associated with academic "
        "achievement only indirectly through online learning engagement, with "
        "Conscientiousness (β = .322) and Openness (β = .253) serving as the "
        "strongest predictors of engagement. Yu (2021), in a Chinese MOOC platform "
        "sample (N = 1,152), reported that Agreeableness (β = .442) and Openness "
        "(β = .305) were stronger predictors of objective MOOC composite scores "
        "than Conscientiousness (β = .057), suggesting that cooperative behaviors "
        "and novelty orientation may be especially salient in Chinese collectivistic "
        "contexts (cf. Mammadov, 2022; Chen et al., 2025). Bahçekapılı and Karaman "
        "(2020), in a Turkish distance-education context (N = 525), found that all "
        "five Big Five traits exhibited non-significant direct correlations with "
        "GPA, with their effects fully mediated through self-efficacy and external "
        "locus of control."
    )
    add_para(doc, p5, indent_first=True)

    p6 = (
        "In aggregate, this primary literature suggests that the Conscientiousness-"
        "achievement link observed in face-to-face contexts is generally preserved "
        "in online settings, but that (a) the magnitude of the effect varies widely "
        "across samples, (b) the role of Extraversion may shift in theoretically "
        "predictable directions when social cues are attenuated, and (c) Openness "
        "and Agreeableness effects may be amplified in specific cultural or "
        "disciplinary contexts. However, these primary studies vary substantially in "
        "sample composition, modality (fully online, blended, synchronous, "
        "asynchronous, MOOC), measurement instruments (BFI, BFI-2, BFI-S, NEO-FFI, "
        "IPIP, HEXACO, TIPI), outcome operationalizations (GPA, course grades, "
        "standardized exams, LMS behavior, composite scores), and era (pre-COVID, "
        "COVID, post-COVID). Without a quantitative synthesis, it is not possible "
        "to determine whether these apparent heterogeneities reflect true moderator "
        "effects, sampling noise, or methodological artifacts."
    )
    add_para(doc, p6, indent_first=True)


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
    build_intro_part2(doc)
    doc.save(OUTPUT)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
