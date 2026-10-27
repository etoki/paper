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

import math

from values import MINUS, V, TRAITS, TRAIT_NAMES, apa, ci


OUTPUT_PREPRINT = "/home/user/paper/metaanalysis/paper_v4/manuscript_preprint_v4.docx"
OUTPUT_JOURNAL = "/home/user/paper/metaanalysis/paper_v4/manuscript_journal_v4.docx"
# Legacy path kept for backwards compat during transition; will be removed
OUTPUT = OUTPUT_PREPRINT


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

    # Heading 1 — APA 7 Level 1: centered, bold
    h1 = doc.styles["Heading 1"]
    h1.font.name = "Times New Roman"
    h1.font.size = Pt(12)
    h1.font.bold = True
    h1.font.color.rgb = RGBColor(0, 0, 0)
    h1.paragraph_format.space_before = Pt(0)
    h1.paragraph_format.space_after = Pt(0)
    h1.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER

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

    # APA 7 running head: left-aligned uppercase short title + right-aligned page number
    # Max 50 chars for running head
    RUNNING_HEAD = "BIG FIVE AND ONLINE LEARNING"  # 30 chars, uppercase per APA 7
    header = s.header
    # Use default paragraph and add tab-separated running head + page number
    p = header.paragraphs[0]
    # Clear existing runs
    for r in list(p.runs):
        r._element.getparent().remove(r._element)
    run = p.add_run(RUNNING_HEAD + "\t\t")
    run.font.name = "Times New Roman"
    run.font.size = Pt(12)
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.append(rFonts)
    rFonts.set(qn("w:eastAsia"), "Times New Roman")

    # Add page number field via XML (right-aligned)
    fldChar_begin = OxmlElement("w:fldChar")
    fldChar_begin.set(qn("w:fldCharType"), "begin")
    instrText = OxmlElement("w:instrText")
    instrText.set(qn("xml:space"), "preserve")
    instrText.text = "PAGE"
    fldChar_end = OxmlElement("w:fldChar")
    fldChar_end.set(qn("w:fldCharType"), "end")

    pn_run = p.add_run()
    pn_run.font.name = "Times New Roman"
    pn_run.font.size = Pt(12)
    pn_run._element.append(fldChar_begin)
    pn_run._element.append(instrText)
    pn_run._element.append(fldChar_end)

    # Set tab stops: center tab + right tab at right margin (~6.5" usable page width)
    from docx.enum.text import WD_TAB_ALIGNMENT
    tab_stops = p.paragraph_format.tab_stops
    tab_stops.add_tab_stop(Inches(6.5), WD_TAB_ALIGNMENT.RIGHT)


def build_title_page(doc):
    """APA 7 title page: centered title, author, affiliation + Author Note."""
    # A few blank lines before title (APA top spacing)
    for _ in range(4):
        add_para(doc, "", align=WD_ALIGN_PARAGRAPH.CENTER)

    # Title (bold, centered)
    add_para(
        doc,
        "Big Five Personality Traits and Academic Achievement in Online Learning "
        "Environments: A Scoping Review with Exploratory Quantitative Synthesis",
        bold=True,
        align=WD_ALIGN_PARAGRAPH.CENTER,
    )
    add_para(doc, "", align=WD_ALIGN_PARAGRAPH.CENTER)

    # Author name (bold is not standard in APA 7; plain centered)
    add_para(doc, "Eisuke Tokiwa", align=WD_ALIGN_PARAGRAPH.CENTER)
    # Affiliation line (APA 7: affiliation below author, no department prefix needed for
    # industry-affiliated authors; city/country is conventional)
    add_para(
        doc,
        "SUNBLAZE Co., Ltd., Tokyo, Japan",
        align=WD_ALIGN_PARAGRAPH.CENTER,
    )

    # --- Author Note (APA 7 formal heading on title page) ---
    add_para(doc, "", align=WD_ALIGN_PARAGRAPH.CENTER)
    add_para(doc, "", align=WD_ALIGN_PARAGRAPH.CENTER)
    add_para(doc, "Author Note", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)

    # Paragraph 1: ORCID
    add_para(
        doc,
        "Eisuke Tokiwa https://orcid.org/0009-0009-7124-6669",
        indent_first=True,
    )

    # Paragraph 2: disclosures (conflicts already in Declarations; here we
    # provide brief APA-standard author-note content pointing to the full
    # Declarations section, which is APA 7 best practice).
    add_para(
        doc,
        "The author has no known conflict of interest to disclose beyond that "
        "stated in the Declarations section, where a pre-specified sensitivity "
        "analysis addresses the author's own potentially eligible primary study. "
        "This research received no external funding. The review was "
        "pre-registered on OSF Registries "
        "(https://doi.org/10.17605/OSF.IO/E5W47), and the extraction data, "
        "analysis code, and outputs are openly available at "
        "https://github.com/etoki/paper.",
        indent_first=True,
    )

    # Paragraph 3: correspondence
    add_para(
        doc,
        "Correspondence concerning this article should be addressed to Eisuke "
        "Tokiwa, SUNBLAZE Co., Ltd., Tokyo, Japan. Email: "
        "eisuke.tokiwa@sunblaze.jp",
        indent_first=True,
    )


def build_declarations(doc, for_journal=False):
    add_para(doc, "")  # blank line
    add_para(doc, "Declarations", bold=True)

    sections = [
        ("Conflict of Interest",
         "The author declares no financial or personal relationships that "
         "could inappropriately influence this work. One of the author's "
         "own primary studies (Tokiwa, 2025) met the eligibility criteria "
         "and is among the retained studies; it contributed no correlation "
         "to the quantitative pool, and a pre-specified sensitivity "
         "analysis excluding it is reported."),
        ("Funding",
         "This research did not receive any specific grant from funding agencies in the "
         "public, commercial, or not-for-profit sectors."),
        ("Ethics Approval",
         "Not applicable. This scoping review synthesises previously "
         "published research and involved no data collection from human "
         "participants by the author; ethical approval for the primary "
         "studies was the responsibility of their investigators."),
        ("Consent to Participate",
         "Not applicable."),
        ("Consent for Publication",
         "Not applicable. This manuscript does not contain any individual person's data "
         "in any form (including individual details, images, or videos)."),
        ("Pre-registration",
         "The protocol was pre-registered on OSF Registries "
         "(https://osf.io/e5w47/; DOI: 10.17605/OSF.IO/E5W47) on April 23, "
         "2026. Deviations from it are listed in the Methods."),
        ("Data Availability Statement",
         "The extraction file (data_extraction_populated.csv), the search "
         "log, the reading notes, the analysis code, and the outputs behind "
         "every table and figure are openly available at "
         "https://github.com/etoki/paper (folder metaanalysis). The "
         "completed PRISMA-ScR checklist is provided as supplementary "
         "material."),
        ("Declaration of Generative AI Use",
         "The author used a large language model (Claude, Anthropic) "
         "throughout this project: to support the literature search and "
         "the summarising of candidate reports, to write and run the "
         "Python analysis code, and to draft and edit the manuscript text. "
         "All inclusion decisions and extracted values were checked by the "
         "author against the full texts, and the author takes full "
         "responsibility for the content. Two analysis errors in an earlier "
         "version (an incorrect β-to-r conversion and an inappropriate "
         "subgroup test) were introduced in AI-generated code and were "
         "corrected for this version; the corrected estimators were "
         "checked against an independent implementation (PyMARE)."),
        ("Authors' contributions",
         "The author (ET) conceived and designed the review, developed the "
         "search strategy and eligibility criteria, ran the searches, "
         "screened and extracted the studies, performed the statistical "
         "analyses, and drafted and revised the manuscript. The author "
         "approves the final version."),
        ("Acknowledgments",
         "The author thanks the investigators of the included primary "
         "studies for reporting their correlation matrices, and the "
         "developers of NumPy, SciPy, Matplotlib, PyMARE, and python-docx."),
    ]

    if for_journal:
        sections.insert(len(sections) - 1, (
            "Preprint Statement",
            "An earlier version of this work was posted on Research Square "
            "(https://doi.org/10.21203/rs.3.rs-9513298) as a systematic "
            "review and meta-analysis. That version has not been peer "
            "reviewed, and its numerical results differ from the present "
            "ones because of the corrections described in the Declaration "
            "of Generative AI Use and in the Methods."
        ))

    for heading, body in sections:
        add_para(doc, heading, bold=True)
        add_para(doc, body)
        add_para(doc, "")  # blank line between sections


def build_abstract(doc):
    """APA Abstract section (Heading 1 style, block paragraph, keywords)."""
    p = doc.add_paragraph("Abstract", style="Heading 1")
    set_double_space(p)

    abstract_body = (
        "Meta-analyses of the Big Five personality traits and academic "
        "achievement have pooled face-to-face, blended, and online samples "
        "without examining online learning separately. This pre-registered "
        "scoping review maps the evidence for online, blended, and MOOC "
        "settings and adds an exploratory quantitative synthesis. Because "
        "the pre-registered subscription databases (PsycINFO, Scopus, Web "
        "of Science, ProQuest Dissertations) could not be accessed, the "
        "search used a web search engine, ERIC, open-access repositories, "
        "and citation snowballing, and the review follows PRISMA-ScR. "
        f"Of {V['n_catalogued']} catalogued full-text reports, "
        f"{V['n_retained']} studies were retained, and {V['n_pool']} "
        "contributed trait–achievement correlations to random-effects "
        f"meta-analyses (REML, Hartung-Knapp-Sidik-Jonkman; N = {V['C.N']}). "
        f"Conscientiousness showed a small positive association (r = "
        f"{V['C.r']}, 95% CI {V['C.ci']}) that held in all sensitivity "
        "analyses but was somewhat below face-to-face benchmarks. "
        f"Agreeableness (r = {V['A.r']}, {V['A.ci']}), Openness (r = "
        f"{V['O.r']}, {V['O.ci']}), Neuroticism (r = {V['N.r']}, "
        f"{V['N.ci']}), and Extraversion (r = {V['E.r']}, {V['E.ci']}) were "
        "small and imprecise, and the Agreeableness and Openness estimates "
        "depended on a single study. Exploratory moderator tests "
        "(Knapp–Hartung meta-regression) suggested a lower Extraversion "
        "estimate in the two Asian studies and a higher one for self-rated "
        "outcomes, but each contrast rests on two studies per subgroup, and "
        "the contrasts overlap with each other and with asynchronous "
        f"delivery. {pi_summary()} The review's "
        "contribution is a map of the evidence and its gaps — no "
        "comprehensive database search, few K-12 or Asian samples, and "
        "almost no facet-level or modality-contrast data — rather than an "
        "established effect size. The estimates should not be used as a "
        "basis for instructional design."
    )
    add_para(doc, abstract_body)
    add_para(doc, "")

    add_para(
        doc,
        "Keywords: Big Five, Five-Factor Model, online learning, e-learning, "
        "MOOC, academic achievement, scoping review, PRISMA-ScR",
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
        "raises a fundamental question: do the non-cognitive correlates of academic "
        "achievement that were established in face-to-face classrooms retain their "
        "correlational validity when instruction is mediated by technology? Personality "
        "traits, and in particular the Big Five (Five-Factor Model) dimensions of "
        "Conscientiousness, Openness to Experience, Extraversion, Agreeableness, and "
        "Neuroticism, are among the most widely studied non-cognitive correlates of "
        "academic achievement (Mammadov, 2022; Poropat, 2009). Yet the accumulated meta-"
        "analytic evidence, as reviewed below, has been produced almost entirely from "
        "samples in which delivery modality is treated as noise rather than as a "
        "substantive moderator. The present review is , to the author\'s knowledge, the first to address this gap "
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
        "to tertiary education. Stajkovic et al. (2018) "
        "provided additional evidence that Conscientiousness is associated with academic "
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
        "associations (C ρ = .35; A ρ = .23; N ρ = −.19). Meyer et al. (2023), "
        "focusing exclusively on K-12 samples (110 samples, N = "
        "500,218), obtained even larger estimates (C ρ = .24; O ρ = .21) and "
        "identified domain specificity as a critical moderator: Openness was "
        "substantially stronger for language than for STEM domains, while "
        "Conscientiousness effects were larger for grades than for standardized tests. "
        "This latter finding was theorized under the Personality-Achievement Saturation "
        "Hypothesis (PASH), which posits that the behavioral signals captured by "
        "Conscientiousness (e.g., on-task engagement, homework completion) are more "
        "visible to teachers assigning grades than to standardized test scorers. Zell "
        "and Lesick (2022), in a second-order synthesis of 54 meta-analyses, confirmed "
        "the stability of these effects, with academic-specific estimates of C ρ = "
        ".28 and E ρ = −.01. Most recently, Chen et al. (2025), examining "
        "84 articles and 370 independent correlations restricted to university students "
        "with samples of at least 200 participants, reported somewhat smaller estimates "
        "(C r = .206; O r = .081; A r = .082; E r = −.009; N r = −.029), consistent "
        "with the idea that stricter methodological inclusion criteria attenuate "
        "apparent trait-performance relationships."
    )
    add_para(doc, p2, indent_first=True)

    p2b = (
        "A complementary theoretical lens is provided by Trait Activation "
        "Theory (Tett & Burnett, 2003), which predicts that stable "
        "dispositional traits are expressed behaviourally only when the "
        "situation contains cues relevant to that trait. TAT complements "
        "PASH by shifting the theoretical question from \"why is a trait "
        "captured by an outcome measure?\" (PASH) to \"why is a trait "
        "expressed at all in a given learning context?\" (TAT). For the "
        "present scoping review, TAT motivates three concrete "
        "expectations that would otherwise remain descriptive. First, "
        "Extraversion should be less strongly associated with academic "
        "outcomes in asynchronous online contexts than in face-to-face "
        "contexts, because the social and participatory cues that "
        "activate extraverted study behaviour (classroom discussion, "
        "study groups, in-person feedback) are attenuated in "
        "asynchronous formats. Second, Conscientiousness should remain "
        "associated with achievement across modalities, because the "
        "structured-task cues that activate conscientious study "
        "behaviour (deadlines, assessments, syllabi) are preserved in "
        "online formats. Third, Agreeableness should be more strongly "
        "associated with achievement in online contexts whose formats "
        "reward cooperative behaviour (e.g., peer-graded assignments, "
        "collaborative MOOCs), particularly in cultural contexts where "
        "cooperation is normative. TAT is therefore used, alongside "
        "PASH, as the second theoretical anchor for the mapping priors "
        "in the present review."
    )
    add_para(doc, p2b, indent_first=True)

    p3 = (
        "Across these eight meta-analyses, a convergent picture emerges: "
        "Conscientiousness is the dominant correlate of academic achievement "
        "(ρ = .19 to .28), with Openness serving as a reliable secondary correlate "
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
        "procrastination (Cheng et al., 2023). Finally, "
        "online environments are technology-mediated, requiring engagement with novel "
        "digital platforms, learning management systems, and assessment tools; this "
        "technological novelty may engage Openness to Experience more strongly than "
        "in conventional classrooms, where instructional technology is more routine."
    )
    add_para(doc, p1, indent_first=True)

    p2 = (
        "These four dimensions—self-regulation demands, reduced social presence, "
        "temporal flexibility, and technology mediation—motivate specific expectations "
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
        "anxious learners. These expectations are theoretically motivated, but they "
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
        "correlated with paper grades (r = .35). Alkış and Taşkaya Temizel (2018), "
        "comparing online and blended sections of the same course, found "
        "Conscientiousness to correlate with course grades in both modalities (online "
        "r = .205; blended r = .244), with other traits exhibiting only null "
        "effects. In the COVID-era German higher education context, Rodrigues et "
        "al. (2024) reported a moderate negative correlation between "
        "Conscientiousness and self-reported GPA (r = −.228 with grades coded such "
        "that lower values indicate better performance), and significant negative "
        "associations between Neuroticism and study satisfaction and well-being. "
        "Rivers (2021), studying Japanese undergraduates in an asynchronous Moodle "
        "course (N = 149), reported a direct negative effect of Extraversion on "
        "course achievement (β = −.168, p < .01), consistent with the mapping prior "
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
        "strongest correlates of engagement. Yu (2021), in a Chinese MOOC platform "
        "sample (N = 1,152), reported that Agreeableness (β = .442) and Openness "
        "(β = .305) were stronger correlates of objective MOOC composite scores "
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


def build_intro_part3(doc):
    """Moderators + Present Study subsections."""
    add_h2(doc, "Potential Moderators of the Personality-Achievement Association in Online Contexts")

    p1 = (
        "Several moderators are theoretically and empirically motivated for the "
        "present review. First, learning modality itself—fully online, blended, MOOC, "
        "synchronous online, asynchronous online—may moderate the personality-"
        "achievement association because the four dimensions of online learning "
        "identified above vary in intensity across modalities. Second, education level "
        "(K-12, undergraduate, graduate, adult) has been shown to moderate personality-"
        "achievement relationships in face-to-face contexts (Mammadov, 2022; Meyer et "
        "al., 2023), with Openness effects declining across the educational trajectory "
        "while Conscientiousness effects remain stable; whether this pattern holds in "
        "online contexts is unknown. Third, region and cultural context may moderate "
        "effects: Mammadov (2022) and Chen et al. (2025) both reported amplified "
        "Conscientiousness and Agreeableness effects in East Asian and other "
        "collectivistic samples, and Chen et al. (2025) additionally reported that "
        "Extraversion is significantly negative in individualistic contexts but null "
        "in collectivistic contexts. Fourth, the COVID-19 era may constitute a "
        "meaningful moderator: pre-COVID online learning was typically self-selected "
        "and involved learners with strong motivation for online study, whereas COVID-"
        "era online learning was imposed on all learners regardless of preference, "
        "and post-COVID online learning reflects a mixture of voluntary and mandatory "
        "adoption. Fifth, the choice of personality instrument (BFI, NEO-FFI, IPIP, "
        "HEXACO, TIPI) has been shown by McAbee and Oswald (2013) to moderate "
        "Conscientiousness estimates, with short-form instruments (e.g., Mini-Markers, "
        "TIPI) yielding smaller effects; this moderator is particularly relevant for "
        "the present review because online studies more frequently use short-form "
        "instruments than traditional paper-and-pencil contexts."
    )
    add_para(doc, p1, indent_first=True)

    add_h2(doc, "The Present Study")

    p2 = (
        "The present scoping review with exploratory quantitative synthesis is, to the author's knowledge, the "
        "first mapping of the association between Big Five personality "
        "traits and academic achievement that focuses specifically on online learning "
        "environments. Consistent with PRISMA-ScR convention (Tricco et al., 2018), "
        "the review is organised around mapping questions and directional priors "
        "rather than around hypothesis tests. Two mapping questions are addressed. "
        "First (MQ1), how large is the available pooled estimate of the association "
        "between each Big Five trait and academic achievement in online learning "
        "environments within the mappable evidence base, and how does this estimate "
        "compare with published face-to-face benchmarks? Second (MQ2), does the "
        "association vary descriptively across learning modality, education level, "
        "region, era, outcome type, and personality instrument within the mappable "
        "evidence base? Because the evidence base is small and the executed search "
        "is incomplete relative to the pre-registered information-source plan, both "
        "mapping questions are addressed descriptively rather than as "
        "confirmatory tests. A third question, whether mediators "
        "such as self-efficacy account for the associations, could not be "
        "addressed because too few studies reported mediation models."
    )
    add_para(doc, p2, indent_first=True)

    p3 = (
        "Five directional priors, pre-specified in the OSF registration "
        "(https://osf.io/e5w47/) as hypotheses, are reframed here as mapping priors "
        "(MP1 through MP5) that describe expectations against which the mapped "
        "evidence can be compared descriptively. Mapping Prior 1 (MP1) states that "
        "Conscientiousness is expected to show the strongest positive pooled "
        "correlation with academic achievement in online learning environments, "
        "with an expected pooled ρ of .20 to .35, consistent with the face-to-face "
        "benchmarks reviewed above and potentially amplified in Asian samples. "
        "Mapping Prior 2 (MP2) states that Openness is expected to show the second-"
        "strongest positive correlation, potentially stronger than in face-to-face "
        "contexts, reflecting the premium placed on self-directed exploration in "
        "online environments. Mapping Prior 3 (MP3) states that Agreeableness is "
        "expected to show a small positive correlation, potentially weaker than in "
        "face-to-face contexts because cooperative group behaviors are less central "
        "in asynchronous online environments. Mapping Prior 4 (MP4) states that "
        "Neuroticism is expected to show a negative correlation, more pronounced in "
        "fully online than in blended modalities, reflecting the isolating and "
        "technology-mediated nature of fully online environments. Mapping Prior 5 "
        "(MP5) states that Extraversion is expected to show a null or weak negative "
        "correlation, with possible facet-level cancellation between sociability "
        "(expected negative) and assertiveness (expected positive) components, "
        "reflecting the reduced social presence of online environments. Mapping "
        "Priors MP2, MP4, and MP5 constitute the modality-specific contribution of "
        "this review, as they describe expected modality-related divergences from "
        "established patterns observed in face-to-face meta-analyses; the review "
        "evaluates the available evidence descriptively against these priors rather "
        "than performing formal hypothesis tests."
    )
    add_para(doc, p3, indent_first=True)

    p4 = (
        "Because the mapped evidence is limited, agreement or disagreement with "
        "a prior is descriptive: a small pooled estimate shows that the "
        "evidence is compatible with no difference from face-to-face samples, "
        "not that the two settings are equivalent. Within those limits, the "
        "review is, to the author's knowledge, the first attempt to examine "
        "whether personality relates to achievement differently in online "
        "learning, and it identifies the primary studies that are still "
        "missing."
    )
    add_para(doc, p4, indent_first=True)


def build_methods_part1(doc):
    """Methods heading + Pre-registration statement + Eligibility criteria."""
    p = doc.add_paragraph("Methods", style="Heading 1")
    set_double_space(p)

    add_h2(doc, "Pre-registration and Reporting Standards")

    p1 = (
        "The protocol was pre-registered on OSF Registries "
        "(https://osf.io/e5w47/; DOI: 10.17605/OSF.IO/E5W47) on April 23, "
        "2026, before the formal search and the quantitative synthesis, "
        "using the OSF Preregistration template with content structured "
        "according to PRISMA-P 2015 (Moher et al., 2015). The protocol was "
        "written for a systematic review and meta-analysis; because the "
        "pre-registered bibliographic databases could not be searched "
        "(see Information Sources), the work is reported as a scoping "
        "review with an exploratory quantitative synthesis, following the "
        "PRISMA extension for scoping reviews (PRISMA-ScR; Tricco et al., "
        "2018). The completed PRISMA-ScR checklist is provided as "
        "supplementary material. All departures from the protocol are "
        "listed in Deviations from the Pre-registered Protocol."
    )
    add_para(doc, p1, indent_first=True)

    add_h2(doc, "Eligibility Criteria")

    p2 = (
        "Eligibility was defined using the PICOS (Population, Intervention/Exposure, "
        "Comparator, Outcome, Study design) framework. Studies were included if they "
        "satisfied all of the following criteria. Population: students at any "
        "educational level—K-12, undergraduate, graduate, or adult learner—enrolled "
        "in an online learning environment. No geographic or demographic restrictions "
        "were applied to participants, provided that samples comprised at least 10 "
        "learners. Exposure: Big Five or HEXACO personality traits measured with a "
        "validated inventory, including the Big Five Inventory (BFI, BFI-2, BFI-44, "
        "BFI-S), the NEO Personality Inventory–Revised (NEO-PI-R) or NEO Five-Factor "
        "Inventory (NEO-FFI, NEO-FFI-3), the International Personality Item Pool "
        "(IPIP, including Mini-IPIP), the HEXACO Personality Inventory–Revised "
        "(HEXACO-PI-R, HEXACO-60), the Ten-Item Personality Inventory (TIPI, TIPI-J), "
        "or any peer-reviewed Big Five–aligned scale with published reliability. "
        "Measures based on typological frameworks (e.g., Myers-Briggs Type Indicator) "
        "or on single-trait constructs (e.g., Grit, Proactive Personality) not "
        "mappable to the Five-Factor Model were excluded. Comparator: not applicable, "
        "because the synthesis addresses observational correlational associations "
        "rather than group-comparison designs. Outcome: the primary outcome was "
        "academic achievement, operationalized as grade point average (GPA), course "
        "grade, standardized exam score, or composite learning performance score. "
        "Studies reporting only secondary outcomes (academic satisfaction, "
        "engagement, learning-related behaviors, persistence) were retained for "
        "the narrative map but not pooled. Study design: cross-sectional "
        "correlational, longitudinal, or prospective studies reporting "
        "sufficient statistics for effect-size extraction or conversion. "
        "Qualitative-only studies, commentaries, editorials, narrative "
        "reviews, and single-case studies were excluded, although the "
        "reference lists of narrative reviews were hand-searched for primary "
        "studies."
    )
    add_para(doc, p2, indent_first=True)

    p3 = (
        "Studies were restricted to those published in English, in peer-reviewed "
        "journals, peer-reviewed conference proceedings, or graduate theses. "
        "Unpublished manuscripts and preprints that had not undergone peer review "
        "were excluded. No restriction was placed on publication year. Studies "
        "of fully face-to-face samples with no online learning component were "
        "excluded. When a study's modality was ambiguous, it was classified "
        "from the full text; corresponding authors were not contacted (see "
        "Deviations)."
    )
    add_para(doc, p3, indent_first=True)


def build_methods_part2(doc):
    """Information sources + Search strategy + Screening + Data extraction."""
    add_h2(doc, "Information Sources and Search Strategy")

    p1 = (
        "Scoping reviews map the available evidence on a topic — its "
        "breadth, conceptual boundaries, methodological characteristics, "
        "and gaps — rather than aiming for an exhaustive synthesis of every "
        "relevant primary study (Munn et al., 2018; Tricco et al., 2018). "
        "The protocol pre-registered a six-database search (PubMed/MEDLINE, "
        "PsycINFO, ERIC, Web of Science, Scopus, and ProQuest Dissertations "
        "& Theses Global), supplemented by Google Scholar and citation "
        "snowballing. PsycINFO, Scopus, Web of Science, and ProQuest "
        "Dissertations require institutional subscriptions that were not "
        "available to the author, and programmatic access to the open "
        "databases was blocked in the computing environment used. The "
        "executed search therefore consisted of: (a) a web search engine "
        "search run on April 23, 2026, applying the pre-specified "
        "three-concept strategy split into eight queries; (b) the ERIC web "
        "interface; (c) targeted retrieval from open-access repositories "
        "(PubMed Central, Frontiers, MDPI, Open Praxis); and (d) forward and "
        "backward citation snowballing from included primary studies and "
        "from the eight prior face-to-face Big Five–achievement "
        "meta-analyses that form the comparison corpus. This search is "
        "sufficient to map the visible literature but not to claim "
        "exhaustive coverage, which is why the review is reported as a "
        "scoping review. The search log with the executed queries, dates, "
        "and hit counts is available in the project repository "
        "(https://github.com/etoki/paper, file metaanalysis/search_log.md)."
    )
    add_para(doc, p1, indent_first=True)

    p2 = (
        "The search strategy combined three concept blocks with the Boolean operator "
        "AND. Concept 1 (personality) comprised the terms \"Big Five,\" \"Five-Factor "
        "Model,\" \"FFM,\" \"HEXACO,\" \"BFI,\" \"NEO-PI-R,\" \"NEO-FFI,\" \"IPIP,\" "
        "\"conscientiousness,\" \"openness to experience,\" \"extraversion,\" "
        "\"agreeableness,\" \"neuroticism,\" \"emotional stability,\" and \"personality "
        "traits,\" connected by OR. Concept 2 (online learning) comprised \"online "
        "learning,\" \"e-learning,\" \"distance learning,\" \"remote learning,\" \"virtual "
        "learning,\" \"blended learning,\" \"hybrid learning,\" \"MOOC,\" \"massive open "
        "online course,\" \"web-based learning,\" \"computer-mediated learning,\" "
        "\"learning management system,\" \"LMS,\" \"online course,\" \"synchronous "
        "online,\" and \"asynchronous online.\" Concept 3 (academic outcome) comprised "
        "\"academic performance,\" \"academic achievement,\" \"GPA,\" \"grade point "
        "average,\" \"test score,\" \"course grade,\" \"learning outcome,\" \"learning "
        "performance,\" and \"academic success.\" Because a web search engine "
        "does not support the full Boolean syntax, the strategy was split "
        "into eight shorter queries. Language and publication-type filters "
        "were applied during screening."
    )
    add_para(doc, p2, indent_first=True)

    p3 = (
        "Forward citation tracking (cited-by) and backward citation tracking "
        "(reference list scanning) were performed for the studies identified "
        "as eligible and for the eight benchmark meta-analyses that "
        "constitute the face-to-face comparison corpus (Poropat, 2009; "
        "McAbee & Oswald, 2013; Vedel, 2014; Stajkovic et al., 2018; "
        "Mammadov, 2022; Zell & Lesick, 2022; Meyer et al., 2023; Chen et "
        "al., 2025). The reference list of the most directly relevant "
        "narrative review (Hunter et al., 2025) was also hand-searched."
    )
    add_para(doc, p3, indent_first=True)

    add_h2(doc, "Study Selection")

    p4 = (
        "One reviewer (ET) screened titles and abstracts and then assessed "
        "full texts against the eligibility criteria, with assistance from "
        "a large language model in locating and summarising candidate "
        "reports (see the Declaration of Generative AI Use); every "
        "inclusion decision was made by the author from the full text. "
        "Records were not managed in a reference manager, and the number of "
        "records excluded at each screening stage was not logged. Every "
        "full-text report that was catalogued received a recorded decision "
        "and, where excluded, a reason (Table 1). The pre-registered "
        "intra-rater reliability checks (re-screening a random subsample "
        "after a wash-out interval) were not carried out (see Deviations)."
    )
    add_para(doc, p4, indent_first=True)

    add_h2(doc, "Data Extraction")

    p5 = (
        "Data were extracted by the same reviewer into a pre-specified "
        "extraction form covering study identification (author, year, DOI, "
        "journal, country), sample characteristics (analytic N, gender "
        "composition, education level), learning context (modality, era), "
        "personality measurement (instrument and item count), outcome "
        "measurement, and effect-size statistics (Pearson r or standardised "
        "β with the associated sample size). The completed extraction file "
        "(data_extraction_populated.csv) is available in the project "
        "repository. The pre-registered re-extraction of a random 10% of "
        "studies was not carried out, and authors were not contacted for "
        "missing statistics (see Deviations). When a statistic was missing, "
        "the effect size was converted from an alternative statistic where "
        "possible (see below) or the study was left out of the trait pool "
        "with the reason recorded."
    )
    add_para(doc, p5, indent_first=True)


def build_methods_part3(doc):
    """Risk of bias + Effect size metric + Transformations."""
    add_h2(doc, "Risk of Bias Assessment")

    p1 = (
        "Risk of bias for each included study was assessed using the Joanna Briggs "
        "Institute (JBI) Critical Appraisal Checklist for Analytical Cross-Sectional "
        "Studies (Moola et al., 2017), which comprises eight items: (1) clearly "
        "defined inclusion criteria for the sample, (2) detailed description of "
        "study subjects and setting, (3) valid and reliable measurement of exposure "
        "(the personality inventory), (4) objective and standardized measurement of "
        "the outcome, (5) identification of confounding factors, (6) strategies to "
        "address confounding, (7) valid and reliable measurement of the outcome, "
        "and (8) appropriate statistical analysis. Each item was rated as Yes, No, "
        "or Unclear, and an aggregate score from 0 to 8 was computed, with "
        "higher scores indicating lower risk of bias. The aggregate score was "
        "used to define the low-quality exclusion sensitivity analysis "
        "(score < 5). The ratings were made by the single reviewer and were "
        "not re-assessed (see Deviations)."
    )
    add_para(doc, p1, indent_first=True)

    add_h2(doc, "Effect Size Metric and Transformations")

    p2 = (
        "The effect-size metric was the Pearson correlation (r) between "
        "each Big Five trait and the academic achievement outcome. Each r "
        "was transformed to Fisher's z (Borenstein et al., 2009), "
        "z = 0.5 × ln[(1 + r) / (1 − r)], with sampling variance "
        "Var(z) = 1 / (N − 3), where N is the analytic sample size. All "
        "pooling, moderator regression, and confidence-interval "
        "construction were performed on the Fisher z scale, and pooled "
        "values were back-transformed to r for reporting."
    )
    add_para(doc, p2, indent_first=True)

    beta_studies = ", ".join(V["beta_ids"])
    p3 = (
        "Two kinds of non-standard effect size entered the pool. First, "
        f"{V['n_beta']} studies ({beta_studies}) reported standardised "
        "regression coefficients (β) rather than correlations. These were "
        "converted with the rule of Peterson and Brown (2005), "
        "r = .98β + .05λ, where λ = 1 if β ≥ 0 and λ = 0 if β < 0, and "
        "were flagged for a sensitivity analysis that excludes them. "
        "Second, one study (A-37 Zheng & Zheng, 2023) reported only that "
        "every trait–grade correlation was below .10 in absolute value, "
        "and its graduate cohorts were taught face to face before and after "
        "the pandemic and online only during it. It was kept in the scoping "
        "map but left out of the quantitative pool; a sensitivity analysis "
        "adds it with its correlations entered as r = 0. The pre-specified "
        "conversions from "
        "Cohen's d, η², t, and F were not needed because no included study "
        "reported only those statistics."
    )
    add_para(doc, p3, indent_first=True)

    p4 = (
        "For primary studies that reported Emotional Stability rather than "
        "Neuroticism, the sign of the correlation was reversed so that all "
        "effect sizes in the Neuroticism pool were oriented toward higher "
        "Neuroticism; outcomes on which lower values indicate better "
        "performance were sign-reversed in the same way. The protocol also "
        "specified a HEXACO-to-Big-Five crosswalk (Ashton & Lee, 2007) with "
        "an accompanying sensitivity analysis; no HEXACO study contributed "
        "to the quantitative pool, so neither was applied."
    )
    add_para(doc, p4, indent_first=True)


def build_methods_part4(doc):
    """Statistical analysis + Moderator analyses + Publication bias."""
    add_h2(doc, "Statistical Analysis and Synthesis Strategy")

    p1 = (
        "A separate random-effects meta-analysis was estimated for each Big "
        "Five trait. Because each trait was pooled separately and each "
        "study contributed a single effect size per trait, the effect sizes "
        "within a pool were independent. The between-study variance (τ²) "
        "was estimated by restricted maximum likelihood (REML), and 95% "
        "confidence intervals for the pooled effect used the "
        "Hartung-Knapp-Sidik-Jonkman (HKSJ) adjustment with a t "
        "distribution on k − 1 degrees of freedom (IntHout et al., 2014). "
        "Heterogeneity was quantified with Cochran's Q, I², and τ², and "
        "95% prediction intervals were computed on k − 2 degrees of freedom "
        "(Higgins et al., 2009). Effect-size magnitudes were interpreted "
        "against the benchmarks of Gignac and Szodorai (2016): r ≈ .10 "
        "small, r ≈ .20 typical, and r ≈ .30 large."
    )
    add_para(doc, p1, indent_first=True)

    add_h2(doc, "Moderator and Sensitivity Analyses")

    p3 = (
        "Nine moderators were pre-specified: six categorical (learning "
        "modality, education level, region, era, outcome type, personality "
        "instrument) and three continuous (publication year, log sample "
        "size, risk-of-bias score). Only three categorical moderators had "
        "at least two contributing studies in each of at least two levels: "
        "region (Asia vs. non-Asia), era (pre-COVID vs. COVID), and outcome "
        "type (objective vs. self-rated achievement). Each of these was "
        "tested within each trait with a mixed-effects meta-regression on "
        "the level indicators (REML residual τ²) using the Knapp–Hartung "
        "adjustment, F(levels − 1, k − levels) (Knapp & Hartung, 2003; "
        "Viechtbauer et al., 2015), over the levels with k ≥ 2. As "
        "pre-registered, p-values were Holm-corrected (Holm, 1979) within each trait; "
        "the family comprised the three moderators that could be tested. "
        "Subgroup pooled estimates are also reported for each level. The "
        "other six moderators are described narratively."
    )
    add_para(doc, p3, indent_first=True)

    p4 = (
        "The pooled estimates were re-estimated (a) excluding studies with "
        "a risk-of-bias score below 5; (b) excluding the author's own prior "
        "primary study (Tokiwa, 2025); (c) excluding the effect sizes "
        "converted from β; (d) excluding studies with an analytic sample "
        "below 50; (e) removing each study in turn (leave-one-out); and "
        "(f) with the DerSimonian–Laird estimator of τ² in place of REML. "
        "All six were pre-registered. A seventh analysis, not "
        "pre-registered, (g) added A-37 Zheng and Zheng (2023), with its "
        "correlations entered as r = 0. A change of |Δr| > .05 was "
        "treated as potentially influential."
    )
    add_para(doc, p4, indent_first=True)

    add_h2(doc, "Publication Bias")

    p5 = (
        "Small-study effects were examined per trait with funnel plots, "
        "Egger's regression test (Egger et al., 1997), and Duval and "
        "Tweedie's (2000) trim-and-fill procedure with the L0 estimator. "
        "The side on which studies were imputed was chosen from the sign "
        "of the slope in a REML meta-regression of the effect sizes on "
        "their standard errors, and the centre was re-estimated on the "
        "trimmed set at each iteration (the procedure of metafor's "
        f"trimfill). These diagnostics have little power at k = {k_range()} "
        "(Sterne et al., 2011), so they are reported descriptively: a "
        "non-significant Egger test does not show that bias is absent, and "
        "trim-and-fill estimates are a sensitivity check, not a correction. "
        "The pre-registered Peters' test and p-curve analysis were not "
        "carried out (see Deviations)."
    )
    add_para(doc, p5, indent_first=True)


def build_methods_part5(doc):
    """GRADE confidence + Deviations from pre-registration + Software."""
    add_h2(doc, "Completeness-of-Evidence Mapping (adapted from GRADE)")

    p1 = (
        "Confidence in each pooled association was summarised with an "
        "adaptation of the GRADE framework (Schünemann et al., 2019), used "
        "here as a completeness-of-evidence map rather than a formal "
        "certainty rating. Each trait started at High and was lowered one "
        "level for each serious and two levels for each very serious "
        "concern in five domains: risk of bias (serious if the mean JBI "
        "score of the contributing studies was below 5), inconsistency "
        "(serious if I² exceeded 50%, very serious if it exceeded 75%), "
        "indirectness (population, exposure, or outcome relative to the "
        "review question), imprecision (serious if the 95% confidence "
        "interval included both zero and a value of |r| ≥ .10, or was "
        "wider than .20), and publication bias (serious if Egger's test "
        "was significant, if trim-and-fill changed the estimate by more "
        "than .05, or if the search could not be comprehensive)."
    )
    add_para(doc, p1, indent_first=True)

    add_h2(doc, "Deviations from the Pre-registered Protocol")

    p2 = (
        "The executed review departed from the pre-registered protocol "
        "(DOI 10.17605/OSF.IO/E5W47) in the following ways. (1) Search: "
        "the six pre-registered databases were not searched; the executed "
        "search is described under Information Sources. (2) Review "
        "process: the pre-registered intra-rater reliability checks "
        "(re-screening 10% of titles and abstracts and 20% of full texts, "
        "re-extracting 10% of studies, and re-rating 20% of risk-of-bias "
        "assessments after a wash-out interval) were not carried out, and "
        "corresponding authors were not contacted for missing statistics "
        "or modality information. (3) Records: the numbers of records "
        "identified, de-duplicated, and excluded at title and abstract "
        "screening were not logged, so the flow diagram reports them as "
        "approximate. (4) Dependent effect sizes: robust variance "
        "estimation (Tipton, 2015) and three-level models were planned on "
        "the assumption that studies would contribute several effect sizes "
        "to one pool; because each trait was pooled separately with one "
        "effect size per study, neither was needed. (5) Moderators: six of "
        "the nine pre-registered moderators could not be tested; the Holm "
        "correction was therefore applied over the three moderators that "
        "were tested. The pre-registered subgroup Q_between test was "
        "replaced by a Knapp–Hartung meta-regression, because a χ² test on "
        "subgroup standard errors is unreliable with two to eight studies "
        "per subgroup (Viechtbauer et al., 2015). (6) Sensitivity: the "
        "HEXACO crosswalk analysis was not applicable, and an analysis "
        "that adds A-37 back was included. (7) Pool: A-37 Zheng and Zheng (2023) was "
        "moved from the quantitative pool to the narrative map during "
        "revision, after the pooled results were known, because most of "
        "its cohorts were taught face to face; the sensitivity analysis "
        "shows the estimates with it included. (8) Publication "
        "bias: Peters' test and p-curve analysis (Simonsohn et al., 2014) "
        "were not carried out; the pool contains too few statistically "
        "significant primary estimates for a meaningful p-curve. "
        "(9) Software: the analyses were run in Python rather than in R "
        "with metafor. (10) Framing: the review is reported as a scoping "
        "review under PRISMA-ScR rather than as a systematic review and "
        "meta-analysis under PRISMA 2020. In addition, a benchmark "
        "meta-analysis published after registration (Chen et al., 2025) "
        "was added to the face-to-face comparison corpus."
    )
    add_para(doc, p2, indent_first=True)

    p3 = (
        "The six moderators that could not be tested quantitatively are "
        "described here for orientation only. (a) Personality instrument: "
        "the Big Five Inventory family predominated among the retained "
        "studies; NEO, IPIP, HEXACO, and TIPI instruments each appeared in "
        "three or fewer studies. Within the quantitative pool, "
        "Conscientiousness correlations from BFI-family instruments ranged "
        "from .068 (Bahçekapılı & Karaman, 2020) to .370 (Abe, 2020), the "
        "NEO-FFI study gave .410 (Elvers et al., 2003, N = 21), and the two "
        "10-item TIPI studies gave .144 (Rivers, 2021) and 0, the value "
        "entered for Zheng and Zheng (2023). (b) Publication "
        "year: the retained studies span "
        f"{V['ret.year_min']}–{V['ret.year_max']}, concentrated in "
        "2020–2023. (c) Sample size: in the Conscientiousness pool the "
        "largest samples (e.g., Yu, 2021, N = 1,152; Bahçekapılı & "
        "Karaman, 2020, N = 525) gave smaller correlations than the "
        "smallest ones (Abe, 2020, N = 92; Elvers et al., 2003, online "
        "subsample N = 21), a pattern compatible with small-study effects "
        + ("that Egger's test also flags" if V["egger.C.p_raw"] < .05
           else "that Egger's test did not detect")
        + f" ({V['egger.C.short']}). "
        "(d) Risk-of-bias score: every study in the quantitative pool "
        f"scored 5 or more (mean = {V['pool.rob_mean']}), so the score "
        "could not be examined. "
        "(e) Modality: one retained study used an engagement scale that "
        "was not specific to online learning (A-13 Dang et al., 2025), one "
        "contributed separate online and blended subsamples (A-02 Alkış & "
        "Taşkaya Temizel, 2018), and A-37 Zheng and Zheng (2023) pooled "
        "graduate cohorts taught face to face before and after the "
        "pandemic and online during it. (f) Education level: the two "
        "retained K-12 studies (A-25 Tokiwa, 2025; A-26 Wang et al., 2023) "
        "did not report trait–achievement correlations, so the "
        "quantitative pool contains no K-12 sample."
    )
    add_para(doc, p3, indent_first=True)

    add_h2(doc, "Software and Reproducibility")

    p4 = (
        "All analyses were run in Python 3.11 with NumPy and SciPy; figures "
        "were drawn with Matplotlib. REML τ² estimates, Knapp–Hartung "
        "moderator tests, and DerSimonian–Laird estimates were checked "
        "against the independent implementations in PyMARE, and agreed to "
        "within 10⁻⁷. The analysis code, the extraction file, the search "
        "log, and the outputs behind every table and figure are openly "
        "available at https://github.com/etoki/paper (folder "
        "metaanalysis/analysis), so every number in this paper can be "
        "reproduced."
    )
    add_para(doc, p4, indent_first=True)


import re as _re


def _set_run_font(run, font_name="Times New Roman", size_pt=12):
    run.font.name = font_name
    run.font.size = Pt(size_pt)
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.append(rFonts)
    rFonts.set(qn("w:eastAsia"), font_name)


def _add_ref_paragraph(doc, ref_text):
    """Render a single APA reference with <i>...</i> segments italicized and
    hanging indent (first line flush, subsequent lines indented 0.5 inch)."""
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.line_spacing = 2.0
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)
    pf.left_indent = Inches(0.5)
    pf.first_line_indent = Inches(-0.5)  # hanging indent

    # Split on <i>...</i> tags
    pattern = _re.compile(r"(<i>.*?</i>)", _re.DOTALL)
    parts = pattern.split(ref_text)
    for part in parts:
        if not part:
            continue
        if part.startswith("<i>") and part.endswith("</i>"):
            run = p.add_run(part[3:-4])
            run.italic = True
            _set_run_font(run)
        else:
            run = p.add_run(part)
            _set_run_font(run)


def build_references(doc):
    """Render the References section in APA 7th format."""
    from references_data import REFERENCES

    doc.add_page_break()
    p = doc.add_paragraph("References", style="Heading 1")
    set_double_space(p)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER  # APA 7: References heading centered
    set_cell_font(p, bold=True)

    for ref in REFERENCES:
        _add_ref_paragraph(doc, ref)


def build_results_part1(doc):
    """Results heading + Study selection (PRISMA) + Study characteristics."""
    p = doc.add_paragraph("Results", style="Heading 1")
    set_double_space(p)

    add_h2(doc, "Study Selection")

    excl = V["excluded_catalogued"]
    p1 = (
        "Figure 1 summarises the selection process. The executed search "
        "returned about 80 records from the eight web queries, alongside "
        "28 primary studies already known from preliminary searching and "
        "five benchmark meta-analyses whose reference lists were screened. "
        "The numbers of duplicates and of records excluded at title and "
        "abstract screening were not logged. From the full-text stage "
        "onward the counts are exact: 37 reports were sought, one could "
        "not be retrieved (A-27), 36 were assessed, and 11 were excluded "
        "(no Big Five measure, 5; face-to-face only, 4: A-09, A-10, A-14, "
        "A-16; sample overlap with A-04, 1: A-05; no extractable effect "
        f"size, 1: A-24), leaving {V['n_retained']} studies. Table 1 "
        f"catalogues {V['n_catalogued']} of the 37 reports, namely the "
        "retained studies and the exclusions with study identifiers; the "
        "five reports without a Big Five measure (Myers-Briggs types, "
        "proactive personality, technology-acceptance or use-intention "
        "constructs, and digital competencies) and A-14 Eilam et al. "
        "(2009) are documented in the reading notes. Of the "
        f"{V['n_retained']} retained studies, {V['n_pool']} contributed "
        "correlations between personality and achievement to the "
        f"quantitative pool ({V['n_direct']} direct, {V['n_beta']} "
        f"converted from β). The other {V['n_retained'] - V['n_pool']} "
        "comprised 13 studies with only secondary outcomes, two without an "
        "extractable trait–achievement correlation (A-25 Tokiwa, 2025; "
        "A-26 Wang et al., 2023), and A-37 Zheng and Zheng (2023), which "
        "was left out of the pool (see Methods)."
    )
    add_para(doc, p1, indent_first=True)

    add_h2(doc, "Characteristics of Included Studies")

    edu = V["ret.education_level"]
    reg = V["ret.region"]
    era = V["ret.era"]
    p3 = (
        f"Table 1 lists the {V['n_catalogued']} catalogued studies. The "
        f"{V['n_retained']} retained studies were published between "
        f"{V['ret.year_min']} and {V['ret.year_max']}, about half of them "
        "during the COVID-19 period (2020–2022). Analytic sample sizes "
        f"ranged from {V['ret.N_min']} (Elvers et al., 2003) to "
        f"{V['ret.N_max']} (Wang et al., 2023), with a median of "
        f"{V['ret.N_median']}. The {V['n_pool']} studies in the "
        f"quantitative pool comprised {V['C.N']} participants (k = "
        f"{V['C.k']} for Conscientiousness and Neuroticism; k = {V['O.k']} "
        "for Openness, Extraversion, and Agreeableness, because A-15 "
        "Elvers et al. reported only Conscientiousness and Neuroticism)."
    )
    add_para(doc, p3, indent_first=True)

    p4 = (
        f"Of the {V['n_retained']} retained studies, "
        f"{V['ret.modality']['fully_online']} were conducted in fully "
        "online settings; the remaining study (A-13 Dang et al., 2025) "
        "used an engagement scale that was not specific to online "
        "learning. Few studies stated whether instruction was synchronous "
        "or asynchronous. By education level, "
        f"{edu['Undergraduate']} studies sampled undergraduates, "
        f"{edu['Mixed_UG_Grad']} mixed undergraduate and graduate students, "
        f"{edu['Graduate']} graduate students, {edu['K-12']} K-12 students, "
        f"and {edu['Mixed_secondary_postsecondary']} mixed secondary and "
        f"post-secondary students. By region, {reg['Asia']} were from Asia "
        f"(including Middle Eastern samples), {reg['Europe']} from Europe, "
        f"and {reg['North_America']} from North America; within the "
        "quantitative pool only two studies were Asian (A-28 Yu, 2021, "
        "China; A-31 Rivers, 2021, Japan). By period, "
        f"{era['pre-COVID']} studies were pre-COVID, {era['COVID']} from "
        f"the COVID period, {era['post-COVID']} post-COVID, and "
        f"{era['Mixed_3era']} (A-37 Zheng & Zheng, 2023) pooled cohorts "
        "from all three periods."
    )
    add_para(doc, p4, indent_first=True)

    p5 = (
        "The Big Five Inventory family was the most common instrument; "
        "the others were NEO-FFI, IPIP and Mini-IPIP, HEXACO-PI-R, the "
        "10-item TIPI and TIPI-J, and other Big Five scales, including "
        "Chinese-language scales and McCrae's scale used by Yu (2021). "
        "In the quantitative pool, achievement was measured as GPA or "
        "course grade (A-01, A-02, A-23, A-29, A-31), an "
        "introductory psychology exam (A-15), a MOOC platform composite "
        "(A-28), or self-rated performance (A-22, A-30). In the other "
        "retained studies, the only outcomes were satisfaction, "
        "engagement, preference, or perception; these were not pooled."
    )
    add_para(doc, p5, indent_first=True)

    add_h2(doc, "Risk of Bias Across Included Studies")

    p6 = (
        "Risk-of-bias scores for each study are given in Table 1. Across "
        f"the {V['n_retained']} retained studies the mean JBI score was "
        f"{V['ret.rob_mean']} (SD = {V['ret.rob_sd']}, range "
        f"{V['ret.rob_min']}–{V['ret.rob_max']}); {V['ret.rob_ge5']} "
        f"studies scored 5 or more and {V['ret.rob_lt5']} scored below 5 "
        f"({', '.join(V['ret.rob_lt5_ids'])}). Every study in the "
        f"quantitative pool scored 5 or more (mean = {V['pool.rob_mean']})."
    )
    add_para(doc, p6, indent_first=True)


def build_results_part2(doc):
    """Primary pooled effects for each Big Five trait + overall heterogeneity."""
    add_h2(doc, "Exploratory Pooled Effect Sizes")

    intro = (
        "Table 2 gives the pooled estimates, and Figures 2 to 6 show the "
        "forest plots."
    )
    add_para(doc, intro, indent_first=True)

    p_c1 = (
        "Conscientiousness. The pooled correlation was positive "
        f"({_trait_sentence('C')}). It was the largest of the five traits, "
        "in line with Mapping Prior 1, but below that prior's expected "
        "range (ρ = .20–.35) and below the face-to-face benchmarks of "
        "ρ = .19–.28 (Poropat, 2009; McAbee & Oswald, 2013; Vedel, 2014; "
        "Mammadov, 2022; Meyer et al., 2023; Chen et al., 2025). "
        + ("Both its confidence interval and its prediction interval "
           "excluded zero, the latter only just."
           if pi_excludes_zero("C") else
           "Its confidence interval excluded zero, but its prediction "
           "interval did not.")
    )
    add_para(doc, p_c1, indent_first=True)

    p_o1 = (
        "Openness to Experience. The pooled correlation was small and its "
        f"confidence interval included zero ({_trait_sentence('O')}). "
        "Mapping Prior 2, which expected Openness to be the second-largest "
        "positive association, was not supported. Heterogeneity was very "
        "high: the converted MOOC estimate of Yu (2021; r = .349) and the "
        "asynchronous Moodle estimate of Rivers (2021; r = −.066) lie at "
        "the two ends of the distribution."
    )
    add_para(doc, p_o1, indent_first=True)

    p_e1 = (
        "Extraversion. The pooled correlation was close to zero "
        f"({_trait_sentence('E')}), consistent with Mapping Prior 5, which "
        "expected a null or weakly negative association. The negative "
        "estimates came from the two Asian studies (Rivers, 2021, "
        "r = −.173; Yu, 2021, r = −.074, converted from β = −.076)."
    )
    add_para(doc, p_e1, indent_first=True)

    p_a1 = (
        "Agreeableness. The pooled correlation was small and its "
        f"confidence interval included zero ({_trait_sentence('A')}). "
        "Mapping Prior 3, which expected a weaker association than in "
        "face-to-face samples (ρ = .05–.10), was not supported, but the "
        "estimate depends heavily on one study: removing Yu (2021; "
        f"r = .483, converted from β = .442) changes the pooled r to "
        f"{V['loo.A.r']} (Δr = {V['loo.A.d']}). Heterogeneity was extreme."
    )
    add_para(doc, p_a1, indent_first=True)

    p_n1 = (
        "Neuroticism. The pooled correlation was close to zero "
        f"({_trait_sentence('N')}), so Mapping Prior 4, which expected a "
        "negative association, was not supported. Study-level signs were "
        "mixed: Rodrigues et al. (2024) reported r = −.142 and Bahçekapılı "
        "and Karaman (2020) r = −.072, whereas the two converted "
        "estimates were positive (Yu, 2021, r = .086; Kaspar et al., "
        "2023, r = .246). Without the converted estimates the pooled r "
        f"was {V['sens.exclude_beta_converted.N.r']} "
        f"(95% CI {V['sens.exclude_beta_converted.N.ci']})."
    )
    add_para(doc, p_n1, indent_first=True)

    add_h2(doc, "Between-Study Heterogeneity")

    i2s = sorted(((float(V[t + '.I2'].rstrip('%')), t) for t in TRAITS))
    p_het = (
        "Heterogeneity was substantial for every trait, with I² from "
        f"{i2s[0][0]:.1f}% ({TRAIT_NAMES[i2s[0][1]]}) to {i2s[-1][0]:.1f}% "
        f"({TRAIT_NAMES[i2s[-1][1]]}). {pi_summary()} For "
        f"Agreeableness the interval spanned {V['A.pi']}: a new online "
        "study could find an association of quite different size, or of "
        "opposite sign."
    )
    add_para(doc, p_het, indent_first=True)


def build_results_part3(doc):
    """Moderator + Sensitivity + Publication bias + GRADE."""
    add_h2(doc, "Moderator Analyses")

    M = V["_moderators"]
    p1 = (
        "Three moderators could be tested: region (Asia vs. non-Asia), "
        "era (pre-COVID vs. COVID), and outcome type (objective vs. "
        "self-rated achievement). Table 3 gives the subgroup estimates and "
        "tests. Every contrast has two studies in its smaller subgroup, so "
        "the tests have little power and the subgroup estimates are "
        "imprecise."
    )
    add_para(doc, p1, indent_first=True)

    others = [t for t in TRAITS if t not in ("E", "A")]
    min_p_other = min(M["region"][t]["_between"]["p"] for t in others)
    p2 = (
        "Region. With two Asian studies (A-28 Yu, China; A-31 Rivers, "
        f"Japan) against {non_asia_k()} non-Asian studies, the Extraversion "
        f"estimate was lower in the Asian subgroup (r = {V['region.E.Asia.r']}, "
        f"95% CI {V['region.E.Asia.ci']}) than in the non-Asian subgroup "
        f"(r = {V['region.E.non-Asia.r']}, {V['region.E.non-Asia.ci']}); "
        f"{V['region.E.F']}, {V['region.E.p']}, {V['region.E.pholm']}. "
        "The Agreeableness estimate was higher in the Asian subgroup "
        f"(r = {V['region.A.Asia.r']}, {V['region.A.Asia.ci']}, vs. "
        f"{V['region.A.non-Asia.r']}, {V['region.A.non-Asia.ci']}; "
        f"{V['region.A.F']}, {V['region.A.p']}, {V['region.A.pholm']}), "
        "but the two Asian studies disagree sharply (r = .483 and .118). "
        "The meta-regression assumes a common residual variance, so this "
        "contrast is not interpretable. The other three traits did not "
        f"differ by region (all p ≥ {apa_floor(min_p_other)})."
    )
    add_para(doc, p2, indent_first=True)

    min_p_era = min(M["era"][t]["_between"]["p"] for t in TRAITS)
    p3 = (
        "Era. No trait differed between pre-COVID and COVID-period "
        f"studies (all p ≥ {apa_floor(min_p_era)}). For Conscientiousness, "
        f"the estimates were {V['era.C.pre-COVID.r']} before and "
        f"{V['era.C.COVID.r']} during the pandemic. The post-COVID period "
        "could not be tested, because no post-COVID study contributed a "
        "trait–achievement correlation."
    )
    add_para(doc, p3, indent_first=True)

    others = [t for t in TRAITS if t != "E"]
    min_p_out = min(M["outcome_type"][t]["_between"]["p"] for t in others)
    p4 = (
        "Outcome type. The Extraversion estimate was close to zero for "
        f"objective outcomes (k = {V['outcome_type.E.objective.k']}, "
        f"r = {V['outcome_type.E.objective.r']}, "
        f"{V['outcome_type.E.objective.ci']}) and positive for self-rated "
        f"performance (k = {V['outcome_type.E.self-report.k']}, "
        f"r = {V['outcome_type.E.self-report.r']}, "
        f"{V['outcome_type.E.self-report.ci']}); "
        f"{V['outcome_type.E.F']}, {V['outcome_type.E.p']}, "
        f"{V['outcome_type.E.pholm']}. The other traits did not differ by "
        f"outcome type (all p ≥ {apa_floor(min_p_out)})."
    )
    add_para(doc, p4, indent_first=True)

    p4b = (
        "These contrasts overlap. Both Asian studies used objective "
        "outcomes and asynchronous delivery, and both self-rated studies "
        "were European (A-22 Quigley et al., 2022; A-30 Kaspar et al., "
        "2023). The region and outcome-type differences for Extraversion "
        "therefore rest on the same few studies and cannot be separated "
        "from each other or from modality."
    )
    add_para(doc, p4b, indent_first=True)

    add_h2(doc, "Sensitivity Analyses")

    def d(key, t):
        return V[f"sens.{key}.{t}.d"]

    big_beta = [t for t in TRAITS
                if abs(V[f"sens.exclude_beta_converted.{t}.d_raw"]) > .05]
    p5 = (
        "Table 4 summarises the sensitivity analyses. Excluding the "
        "author's own study (Tokiwa, 2025) changed nothing, because it "
        "contributed no correlation to the pool. The DerSimonian–Laird "
        f"estimator changed the estimates by at most {max_abs('dl_estimator')}. "
        "Excluding studies with N < 50 (A-15 Elvers et al.) changed the "
        f"estimates by at most {max_abs('exclude_small_n')}. No study in "
        "the pool scored below the risk-of-bias threshold, so that "
        "analysis repeats the primary one. Adding A-37 Zheng and Zheng "
        "(2023) with r = 0 lowered the estimates slightly "
        f"(Conscientiousness {d('include_a37', 'C')}; largest change "
        f"{max_abs('include_a37')}). Excluding the two studies "
        "whose effects were converted from β (A-28 Yu; A-30 Kaspar) "
        "changed the estimates by more than .05 for "
        f"{join_names(big_beta)} ("
        + ", ".join(f"{TRAIT_NAMES[t]} to "
                    f"{V[f'sens.exclude_beta_converted.{t}.r']}"
                    for t in big_beta)
        + "); Conscientiousness "
        f"stayed positive ({V['sens.exclude_beta_converted.C.r']}, "
        f"{V['sens.exclude_beta_converted.C.ci']})."
    )
    add_para(doc, p5, indent_first=True)

    p5b = (
        "In the leave-one-out analysis, the Conscientiousness estimate "
        f"ranged from {V['loo.C.range']}. The most influential single "
        f"study was Yu (2021) for Agreeableness (r = {V['loo.A.r']} "
        f"without it; Δr = {V['loo.A.d']}), Openness ({V['loo.O.r']}; "
        f"Δr = {V['loo.O.d']}), and Extraversion ({V['loo.E.r']}; "
        f"Δr = {V['loo.E.d']}), and Kaspar et al. (2023) for Neuroticism "
        f"({V['loo.N.r']}; Δr = {V['loo.N.d']})."
    )
    add_para(doc, p5b, indent_first=True)

    add_h2(doc, "Publication-Bias Diagnostics (descriptive only)")

    p6 = (
        f"Funnel plots are shown in Figures 7 to 11. {egger_summary()} "
        f"With k = {k_range()} the test has little power, so the "
        "non-significant results do not show that bias is absent."
    )
    add_para(doc, p6, indent_first=True)

    tf_parts = []
    for t in TRAITS:
        k0 = V[f"tf.{t}.k0"]
        if k0:
            tf_parts.append(
                f"{TRAIT_NAMES[t]}, {k0} on the {V[f'tf.{t}.side']} "
                f"(adjusted r = {V[f'tf.{t}.r']}, {V[f'tf.{t}.ci']})")
    none = [TRAIT_NAMES[t] for t in TRAITS if not V[f"tf.{t}.k0"]]
    def _lo(key):
        return float(V[key][1:].split(",")[0].replace(MINUS, "-"))
    assert _lo("tf.C.ci") > 0 and _lo("tf.A.ci") > 0 and V["tf.A.k0"] > 0
    p7 = (
        "Trim-and-fill imputed studies for "
        + "; ".join(tf_parts)
        + f", and none for {join_names(none, names=False)}. The "
        "Conscientiousness estimate remained positive after imputation. "
        "The Agreeableness adjustment moved the estimate upward and its "
        "interval above zero, which reflects the influence of Yu (2021) "
        "on the funnel rather than evidence of a larger true effect. "
        "These adjusted values are a sensitivity check, not corrected "
        "estimates."
    )
    add_para(doc, p7, indent_first=True)

    add_h2(doc, "Completeness-of-Evidence Summary")

    # The prose below describes these ratings; fail loudly if they change.
    assert {t: grade(t)["rating"] for t in TRAITS} == {
        "C": "Low", "E": "Low", "O": "Very Low", "A": "Very Low",
        "N": "Very Low"}, {t: grade(t)["rating"] for t in TRAITS}
    assert V["loo.A.study"].startswith("A-28") and V["loo.O.study"].startswith("A-28") \
        and V["loo.E.study"].startswith("A-28") and V["loo.N.study"].startswith("A-30")
    p8 = (
        "Table 5 summarises the confidence ratings. The "
        "Conscientiousness association was rated Low: its confidence "
        "interval excluded zero and it was robust to the sensitivity "
        "analyses, but heterogeneity was substantial, "
        + ("Egger's test suggested small-study effects, "
           if V["egger.C.p_raw"] < .05 else "")
        + "and the search was not comprehensive. The Extraversion estimate "
        "was also rated Low, because its confidence interval was narrow "
        "around zero. The Openness, Agreeableness, and Neuroticism "
        "estimates were rated Very Low, because of very high "
        "heterogeneity, imprecision, and the incomplete search."
    )
    add_para(doc, p8, indent_first=True)


def build_discussion_part1(doc):
    """Discussion heading + Summary of findings + Mapping-Prior evaluation."""
    p = doc.add_paragraph("Discussion", style="Heading 1")
    set_double_space(p)

    add_h2(doc, "Summary of Findings")

    p1 = (
        "This scoping review mapped the evidence on the Big Five traits "
        "and academic achievement in online learning. Of "
        f"{V['n_catalogued']} catalogued studies, {V['n_retained']} were "
        f"retained and {V['n_pool']} contributed to an exploratory "
        f"quantitative pool (N = {V['C.N']}). Conscientiousness was the "
        f"only trait with a clearly positive association (r = {V['C.r']}, "
        f"95% CI {V['C.ci']}), and it held up in every sensitivity "
        f"analysis. The estimates for Agreeableness (r = {V['A.r']}, "
        f"{V['A.ci']}), Openness (r = {V['O.r']}, {V['O.ci']}), "
        f"Neuroticism (r = {V['N.r']}, {V['N.ci']}), and Extraversion "
        f"(r = {V['E.r']}, {V['E.ci']}) were small, with confidence "
        "intervals that included zero. The Agreeableness and Openness "
        "estimates depended heavily on one converted study (Yu, 2021). "
        "Exploratory subgroup contrasts suggested a lower Extraversion "
        f"estimate in the two Asian studies ({V['region.E.F']}, "
        f"{V['region.E.pholm']}) and a higher one for self-rated outcomes "
        f"({V['outcome_type.E.F']}, {V['outcome_type.E.pholm']}). These "
        "contrasts rest on the same two-study subgroups and cannot be "
        "separated from one another."
    )
    add_para(doc, p1, indent_first=True)

    add_h2(doc, "Evaluation of the Mapping Priors")

    p_h1 = (
        "Mapping Prior 1 (Conscientiousness as the strongest positive "
        f"correlate) was supported in rank: r = {V['C.r']} was the largest "
        "estimate. It fell below the prior's expected range (.20–.35) and "
        "below the face-to-face benchmarks of Poropat (2009; ρ = .22), "
        "McAbee and Oswald (2013; ρ = .26), Vedel (2014; ρ = .26), "
        "Mammadov (2022; ρ = .27), and Chen et al. (2025; r = .206). The "
        "upper limit of its confidence interval reached that range."
    )
    add_para(doc, p_h1, indent_first=True)

    p_h2 = (
        "Mapping Prior 2 (Openness as the second-strongest positive "
        f"correlate) was not supported: r = {V['O.r']} was smaller than "
        "both the Conscientiousness and the Agreeableness estimate, and "
        "its confidence interval included zero. One possible reason is "
        "that the pool is almost entirely undergraduate, and Openness "
        "effects are known to weaken with educational level (Mammadov, "
        "2022; Meyer et al., 2023)."
    )
    add_para(doc, p_h2, indent_first=True)

    p_h3 = (
        "Mapping Prior 3 (Agreeableness weaker than in face-to-face "
        f"samples) was not supported in direction: r = {V['A.r']} was "
        "slightly above the face-to-face range of .05–.10. This estimate "
        "rests largely on Yu (2021); without that study it is "
        f"{V['loo.A.r']}, "
        + ("within" if .05 <= float(V['loo.A.r'].replace(MINUS, '-')) <= .10
           else "close to")
        + " the face-to-face range. Narrative "
        "evidence from studies outside the pool (Wang et al., 2023; "
        "Baruth & Cohen, 2023) points in the same direction for East "
        "Asian and Israeli samples, but this does not offset the "
        "single-study dependence."
    )
    add_para(doc, p_h3, indent_first=True)

    p_h4 = (
        "Mapping Prior 4 (Neuroticism negative, more so in fully online "
        f"courses) was not supported: r = {V['N.r']}. Without the two "
        "converted estimates the pooled r was "
        f"{V['sens.exclude_beta_converted.N.r']} "
        f"(95% CI {V['sens.exclude_beta_converted.N.ci']}), negative in direction "
        "but still compatible with zero. The fully online versus blended "
        "comparison could not be made, because only one study (Alkış & "
        "Taşkaya Temizel, 2018) reported a blended subsample."
    )
    add_para(doc, p_h4, indent_first=True)

    p_h5 = (
        "Mapping Prior 5 (Extraversion null or weakly negative) was "
        f"supported: r = {V['E.r']}, {V['E.ci']}. The proposed mechanism, "
        "in which a negative sociability facet cancels a positive "
        "assertiveness facet, could not be tested, because few studies "
        "report facets."
    )
    add_para(doc, p_h5, indent_first=True)


def build_discussion_part2(doc):
    """Theoretical implications + Hypotheses for future testing."""
    add_h2(doc, "Theoretical Implications")

    p1 = (
        "The Conscientiousness estimate is somewhat below the face-to-face "
        "benchmarks. This is consistent with an extension of the "
        "Personality-Achievement Saturation Hypothesis (PASH; Meyer et "
        "al., 2023) proposed here: if Conscientiousness reaches grades "
        "partly through behaviour that teachers observe, online courses "
        "with automated grading and less observation should weaken that "
        "channel. The extension is the author's proposal, not part of the "
        "original PASH. Trait Activation Theory (Tett & Burnett, 2003) "
        "makes a different prediction: the cues that activate "
        "conscientious study — deadlines, assessments, milestones — are "
        "preserved online, so the association should be preserved too. "
        "The present data cannot tell these accounts apart. The gap from "
        f"the benchmarks (Δr ≈ {apa(.19 - V['C.r_raw'], 2)}–"
        f"{apa(.28 - V['C.r_raw'], 2)}, depending on the benchmark) is "
        "small, the confidence interval reaches the benchmark range, and "
        "the confidence rating is Low."
    )
    add_para(doc, p1, indent_first=True)

    p2 = (
        "For Extraversion, the near-zero pooled estimate matches the "
        "face-to-face literature at the tertiary level (Poropat, 2009; "
        "Chen et al., 2025). The subgroup contrasts suggest that the "
        "estimate may be lower in some contexts, but they rest on two "
        "Asian studies and two self-rated studies, and these subgroups "
        "overlap with one another and with asynchronous delivery. Several "
        "explanations fit equally well: that extraverts are "
        "disadvantaged in collectivistic online contexts with few social "
        "channels, that self-rated performance carries an "
        "extraversion-linked self-enhancement bias (Paulhus, 1998), that "
        "asynchronous delivery removes the social cues that activate "
        "extraverted engagement (Tett & Burnett, 2003), or that the "
        "difference reflects the particular samples and instruments of "
        "four studies. Testing these explanations needs primary studies "
        "that vary one factor at a time, such as studies that collect "
        "objective and self-rated outcomes from the same learners."
    )
    add_para(doc, p2, indent_first=True)

    p3 = (
        "The Agreeableness contrast by region does not support a cultural "
        "interpretation. The two Asian studies give very different "
        "estimates (r = .483 and .118), so the Asian subgroup has no "
        "meaningful pooled value. The higher Agreeableness estimates "
        "reported for Asian face-to-face samples (Mammadov, 2022; Chen et "
        "al., 2025) remain a hypothesis for online learning."
    )
    add_para(doc, p3, indent_first=True)

    p4 = (
        "Finally, no trait differed between pre-COVID and COVID-period "
        "studies. In particular, the expectation that emergency remote "
        "teaching would increase the importance of Conscientiousness was "
        "not borne out. Post-COVID studies did not report correlations, "
        "so later updates should revisit this question."
    )
    add_para(doc, p4, indent_first=True)

    add_h2(doc, "Hypotheses for Future Practical Testing")

    p_caveat = (
        "The following hypotheses follow from the map but are not "
        "recommendations. The pool is small, heterogeneity is "
        "substantial, and the subgroup contrasts rest on two studies per "
        "subgroup. Each hypothesis needs to be tested in adequately "
        "powered primary research before it informs instructional design."
    )
    add_para(doc, p_caveat, indent_first=True)

    p5 = (
        "HPT-1. Conscientiousness is the only trait whose confidence "
        "interval excludes zero. Online features that scaffold "
        "self-regulated learning — scheduling tools, progress tracking, "
        "deadline reminders — may therefore be most useful for learners "
        "low in Conscientiousness."
    )
    add_para(doc, p5, indent_first=True)

    p6 = (
        "HPT-2. If the Extraversion contrasts reflect missing social "
        "channels, adding synchronous elements to asynchronous courses "
        "should help extraverted learners. If they reflect "
        "self-enhancement, self-rated performance should be checked "
        "against objective indicators. A study that varies delivery mode "
        "and collects both outcome types in the same learners could test "
        "both."
    )
    add_para(doc, p6, indent_first=True)

    p7 = (
        "HPT-3. The Openness estimate is small. Exploration-oriented "
        "features of institutional learning platforms (supplementary "
        "multimedia, flexible navigation) may not translate into "
        "achievement for high-Openness learners under standardised online "
        "assessment."
    )
    add_para(doc, p7, indent_first=True)


def build_discussion_part3(doc):
    """Robust vs fragile + Strengths + Limitations + Future research."""
    add_h2(doc, "Distinguishing Robust from Fragile Findings")

    p_robust2 = (
        "The positive Conscientiousness association is the most robust "
        f"result: r = {V['C.r']}, with a confidence interval that excludes "
        "zero. It ranged from "
        f"{V['loo.C.range']} when each study was removed in turn, stayed "
        f"positive without the converted estimates "
        f"({V['sens.exclude_beta_converted.C.r']}) and after trim-and-fill "
        f"({V['tf.C.r']}), and matches the direction of all eight "
        "face-to-face meta-analyses. "
        + (f"Its prediction interval also excludes zero ({V['C.pi']}), "
           "but only just."
           if pi_excludes_zero("C") else
           f"Its prediction interval still includes zero ({V['C.pi']}).")
    )
    add_para(doc, p_robust2, indent_first=True)

    p_robust3 = (
        "Everything else is fragile: (i) the Agreeableness and Openness "
        "estimates, which depend on Yu (2021) and change by more than .05 "
        "without the converted studies; (ii) the Neuroticism estimate, "
        "which changes sign without the converted studies; (iii) the "
        "Extraversion region and outcome-type contrasts, which rest on "
        "two studies per subgroup and overlap with each other and with "
        "modality; and (iv) the Agreeableness region contrast, which is "
        "not interpretable. The hypotheses above that depend on these "
        "results should be weighted accordingly."
    )
    add_para(doc, p_robust3, indent_first=True)

    add_h2(doc, "Strengths of the Present Review")

    p1 = (
        "This is, to the author's knowledge, the first attempt to map the "
        "Big Five–achievement evidence specifically for online learning. "
        "None of the prior face-to-face meta-analyses examined learning "
        "modality (Chen et al., 2025; Mammadov, 2022; McAbee & Oswald, "
        "2013; Meyer et al., 2023; Poropat, 2009; Stajkovic et al., 2018; "
        "Vedel, 2014; Zell & Lesick, 2022), and the closest review of "
        "online studies did not pool effect sizes (Hunter et al., 2025). "
        "The protocol was pre-registered, and every deviation from it is "
        "reported. The extraction file, analysis code, and outputs are "
        "public, the key estimators were checked against an independent "
        "implementation, and the robustness of each estimate was examined "
        "with seven sensitivity analyses."
    )
    add_para(doc, p1, indent_first=True)

    add_h2(doc, "Limitations")

    p2 = (
        f"First, the quantitative pool is small (k = {V['n_pool']}: "
        f"{V['n_direct']} studies with direct correlations and "
        f"{V['n_beta']} converted from β). With so "
        "few studies the pooled estimates are unstable, heterogeneity is "
        "poorly estimated, and every moderator contrast has two studies in "
        "its smaller subgroup. Second, the search did not cover the "
        "pre-registered bibliographic databases. A web search engine, "
        "ERIC's web interface, open-access repositories, and citation "
        "snowballing cannot be assumed to find every eligible study, so "
        "the map is not an exhaustive census. The record counts at the "
        "identification and screening stages were not logged, so the "
        "flow diagram is approximate before the full-text stage."
    )
    add_para(doc, p2, indent_first=True)

    p3 = (
        "Third, one reviewer did all screening, extraction, and "
        "risk-of-bias rating, with assistance from a large language model, "
        "and the pre-registered intra-rater reliability checks were not "
        "carried out. Single-reviewer work is vulnerable to inclusion and "
        "coding errors that a second reviewer would catch. Analysis errors "
        "were also found during revision: an incorrect β-to-r conversion "
        "and an inappropriate subgroup test in earlier versions were "
        "corrected for this version. Fourth, two effect sizes are "
        "converted from β, and one study (A-37 Zheng & Zheng, 2023) was "
        "moved to the narrative map during revision, after the pooled "
        "results were known, because most of its cohorts were taught face "
        "to face and its correlations were reported only as |r| < .10. "
        "Adding it back with r = 0 changes the estimates by at most "
        f"{max_abs('include_a37')} (Table 4)."
    )
    add_para(doc, p3, indent_first=True)

    p4 = (
        "Fifth, the English-language restriction excluded potentially "
        "relevant Chinese-, Japanese-, Korean-, and Spanish-language "
        "studies, which matters most for the regional contrasts. Sixth, "
        "outcomes ranged from objective grades and exam scores to "
        "self-rated performance, which adds heterogeneity. Seventh, the "
        "author's own primary study (Tokiwa, 2025) was among the retained "
        "studies; it contributed no correlation to the pool, and excluding "
        "it changed nothing. Eighth, facet-level analysis was not possible "
        "because few studies report facets."
    )
    add_para(doc, p4, indent_first=True)

    add_h2(doc, "Future Research Directions")

    p6 = (
        "The most useful next step is a replication search of the "
        "pre-registered bibliographic databases (including PsycINFO, "
        "Scopus, and Web of Science) with record-level logging and a "
        "second reviewer, which would allow a systematic review proper. "
        "Primary studies are needed that (a) compare the same learners in "
        "asynchronous and synchronous conditions, (b) collect objective "
        "and self-rated outcomes together, (c) report facet-level "
        "correlations with hierarchical inventories such as the BFI-2 or "
        "NEO-PI-R, (d) sample K-12 and Asian online learners, and (e) "
        "report zero-order correlations for all five traits, including "
        "non-significant ones. Mediators such as self-efficacy, "
        "self-regulated learning, and platform engagement would clarify "
        "the pathways (Stajkovic et al., 2018). Longitudinal designs "
        "across the pre-, during-, and post-COVID periods, such as Zheng "
        "and Zheng (2023), would help separate period effects from "
        "sampling."
    )
    add_para(doc, p6, indent_first=True)

    p6b = (
        "Discipline is a further candidate moderator. The retained "
        "studies span psychology, education, computing, health sciences, "
        "business, and general education, with only one to three studies "
        "per discipline in the quantitative pool. Future syntheses should "
        "record discipline as a categorical extraction field and test "
        "whether the Conscientiousness and Extraversion associations "
        "differ between more and less structured disciplines and between "
        "formats with more or less peer interaction."
    )
    add_para(doc, p6b, indent_first=True)



def build_conclusion(doc):
    """Conclusion as separate Heading 1 section (APA sample convention)."""
    p_h = doc.add_paragraph("Conclusion", style="Heading 1")
    set_double_space(p_h)

    p7 = (
        "This scoping review found a small but robust positive association "
        "between Conscientiousness and academic achievement in online "
        f"learning (r = {V['C.r']}, 95% CI {V['C.ci']}), somewhat below "
        "face-to-face benchmarks. Associations for the other four traits "
        "were small and imprecise, and two of them depended on a single "
        "study. The Extraversion subgroup differences by region and "
        "outcome type are hypotheses for primary research: they rest on "
        "two studies per subgroup and cannot be separated from each other "
        "or from delivery mode. The evidence base is small and the search "
        "was not exhaustive, so the review's main contribution is a map of "
        "what is known and a list of the studies that are missing, not a "
        "settled effect size."
    )
    add_para(doc, p7, indent_first=True)


def _style_table(table):
    """Apply APA-style formatting (Times New Roman 11pt for compact tables)."""
    for row in table.rows:
        for cell in row.cells:
            for para in cell.paragraphs:
                para.paragraph_format.space_before = Pt(0)
                para.paragraph_format.space_after = Pt(0)
                para.paragraph_format.line_spacing = 1.15
                for run in para.runs:
                    run.font.name = "Times New Roman"
                    run.font.size = Pt(10)
                    rPr = run._element.get_or_add_rPr()
                    rFonts = rPr.find(qn("w:rFonts"))
                    if rFonts is None:
                        rFonts = OxmlElement("w:rFonts")
                        rPr.append(rFonts)
                    rFonts.set(qn("w:eastAsia"), "Times New Roman")


def _add_table_title(doc, number, title):
    """APA Table N title format: bold number line + italic title line."""
    p = doc.add_paragraph()
    set_double_space(p)
    run = p.add_run(f"Table {number}")
    run.bold = True
    run.font.name = "Times New Roman"
    run.font.size = Pt(12)
    p2 = doc.add_paragraph()
    set_double_space(p2)
    run2 = p2.add_run(title)
    run2.italic = True
    run2.font.name = "Times New Roman"
    run2.font.size = Pt(12)


def build_table2_pooled(doc):
    """Table 2: Pooled random-effects meta-analytic correlations per trait."""
    doc.add_page_break()
    _add_table_title(
        doc, 2,
        "Random-Effects Pooled Correlations Between Big Five Personality "
        "Traits and Academic Achievement in Online Learning Environments",
    )
    order = sorted(TRAITS, key=lambda t: V[f"{t}.r_raw"], reverse=True)
    data = [["Trait", "k", "N", "r", "95% CI", "95% PI", "I²", "τ²", "Q(df), p"]]
    for t in order:
        data.append([TRAIT_NAMES[t], V[f"{t}.k"], V[f"{t}.N"], V[f"{t}.r"],
                     V[f"{t}.ci"], V[f"{t}.pi"], V[f"{t}.I2"], V[f"{t}.tau2"],
                     V[f"{t}.Qcell"]])
    _fill_table(doc, data)
    _add_table_note(
        doc,
        "k = number of studies contributing a correlation (direct, or "
        "converted from β with Peterson and Brown's rule) to the trait "
        "pool. N = total sample size. r = random-effects pooled "
        "correlation (REML τ², Fisher z scale, back-transformed). 95% CI = "
        "Hartung-Knapp-Sidik-Jonkman confidence interval. 95% PI = "
        "prediction interval for the true correlation in a new study. "
        "Q = Cochran's heterogeneity statistic. Traits are sorted by "
        "pooled r."
    )


def build_table3_moderators(doc):
    """Table 3: Subgroup moderator analyses (region, era, outcome type)."""
    doc.add_page_break()
    _add_table_title(
        doc, 3,
        "Moderator Analyses: Region, Era, and Outcome Type per Big Five Trait",
    )
    M = V["_moderators"]
    labels = {"region": "Region", "era": "Era", "outcome_type": "Outcome"}
    level_names = {"self-report": "self-rated"}
    data = [["Moderator / Trait", "Level", "k", "r [95% CI]",
             "F(df), p; Holm p"]]
    for mod in ("region", "era", "outcome_type"):
        for t in TRAITS:
            res = M[mod][t]
            first = True
            for lv, sub in res.items():
                if lv == "_between" or "r_pooled" not in sub:
                    continue
                data.append([
                    f"{labels[mod]}: {TRAIT_NAMES[t]}" if first else "",
                    level_names.get(lv, lv), str(sub["k"]),
                    f"{apa(sub['r_pooled'])} {ci(sub['r_ci_lo'], sub['r_ci_hi'])}",
                    V[f"{mod}.{t}.cell"] if first else "",
                ])
                first = False
    _fill_table(doc, data)
    _add_table_note(
        doc,
        "k = number of studies in the subgroup. F = Knapp–Hartung test from "
        "a mixed-effects meta-regression on the subgroup indicator (REML "
        "residual τ²). Holm p = Holm-adjusted p within each trait across "
        "the three moderators. Asia = East Asian or Middle Eastern "
        "samples; within the quantitative pool only A-28 (China) and A-31 "
        "(Japan) are Asian. Era: pre-COVID = data collected up to 2019; "
        "COVID = 2020–2022; no post-COVID study contributed a correlation. "
        "Outcome: objective = GPA, course grade, examination, or "
        "MOOC platform composite; self-rated = self-reported performance "
        "(A-22, A-30). Both Asian studies used objective outcomes and "
        "asynchronous delivery, so the region and outcome contrasts "
        "overlap."
    )


def build_table4_sensitivity(doc):
    """Table 4: Sensitivity analyses and small-study diagnostics."""
    doc.add_page_break()
    _add_table_title(
        doc, 4,
        "Sensitivity Analyses and Small-Study Diagnostics for the Pooled "
        "Correlations",
    )
    rows_spec = [
        ("Exclude Tokiwa (2025; COI)", "exclude_tokiwa"),
        ("Exclude β-converted (A-28, A-30)", "exclude_beta_converted"),
        ("Exclude RoB < 5 (none in pool)", "exclude_low_quality"),
        ("Exclude N < 50 (A-15)", "exclude_small_n"),
        ("DerSimonian–Laird τ²", "dl_estimator"),
        ("Include A-37 (r = 0)", "include_a37"),
    ]
    data = [["Trait", "Analysis", "k", "r [95% CI]", "Δr"]]
    for t in TRAITS:
        data.append([TRAIT_NAMES[t], "Primary", V[f"{t}.k"],
                     f"{V[f'{t}.r']} {V[f'{t}.ci']}", "—"])
        for label, key in rows_spec:
            flag = " ‡" if abs(V[f"sens.{key}.{t}.d_raw"]) > .05 else ""
            data.append(["", label, V[f"sens.{key}.{t}.k"],
                         f"{V[f'sens.{key}.{t}.r']} {V[f'sens.{key}.{t}.ci']}",
                         V[f"sens.{key}.{t}.d"] + flag])
        loo_flag = " ‡" if abs(V[f"loo.{t}.d_raw"]) > .05 else ""
        data.append(["", "Leave-one-out (range of r)", "—",
                     V[f"loo.{t}.range"],
                     f"max {V[f'loo.{t}.d']}{loo_flag}"])
        data.append(["", "Egger regression", V[f"{t}.k"],
                     V[f"egger.{t}.short"], "—"])
        data.append(["", f"Trim-and-fill ({V[f'tf.{t}.k0']} imputed, "
                         f"{V[f'tf.{t}.side']})",
                     str(int(V[f"{t}.k"]) + V[f"tf.{t}.k0"]),
                     f"{V[f'tf.{t}.r']} {V[f'tf.{t}.ci']}", V[f"tf.{t}.d"]])
    _fill_table(doc, data)
    _add_table_note(
        doc,
        "Δr = change from the primary pooled correlation; ‡ marks "
        "|Δr| > .05. No study in the pool scored below 5 on risk of bias, "
        "so that analysis repeats the primary one. A-37, which is left out "
        "of the primary pool, is added with its correlations entered as "
        "r = 0. Leave-one-out "
        "gives the range of pooled r when each study is removed in turn "
        "and the largest change. Trim-and-fill uses the L0 estimator; the "
        "side is that on which studies were imputed. Egger and "
        "trim-and-fill results are descriptive: both have little power at "
        f"k = {k_range()}."
    )


def build_table5_grade(doc):
    """Table 5: Completeness-of-Evidence Summary per trait."""
    doc.add_page_break()
    _add_table_title(
        doc, 5,
        "Completeness-of-Evidence Summary for the Pooled Associations "
        "Between Big Five Traits and Academic Achievement in Online Learning",
    )
    data = [["Trait (k, N)", "Risk of bias", "Inconsistency", "Indirectness",
             "Imprecision", "Publication bias", "r", "Rating"]]
    for t in TRAITS:
        g = grade(t)
        data.append([
            f"{TRAIT_NAMES[t]}\n(k = {V[f'{t}.k']}, N = {V[f'{t}.N']})",
            f"{g['rob']}\n(mean JBI = {V[f'{t}.rob_mean']})",
            f"{g['incon']}\n(I² = {V[f'{t}.I2']})",
            g["indirect"],
            f"{g['imprec']}\n(CI {V[f'{t}.ci']})",
            f"{g['pubbias']}\n(search not comprehensive; Egger {V[f'egger.{t}.short'].split(', ')[1]})",
            V[f"{t}.r"],
            g["rating"],
        ])
    _fill_table(doc, data)
    _add_table_note(
        doc,
        "Ratings adapt GRADE (Schünemann et al., 2019) as a "
        "completeness-of-evidence map. Each trait starts at High and is "
        "lowered one level for each serious and two levels for each very "
        "serious concern. Risk of bias: serious if the mean JBI score of "
        "the contributing studies is below 5. Inconsistency: serious if "
        "I² > 50%, very serious if I² > 75%. Imprecision: serious if the "
        "95% CI includes zero and a value of |r| ≥ .10, or is wider than "
        ".20. Publication bias is rated serious for every trait because "
        "the search could not cover the main bibliographic databases. "
        "Indirectness: two pooled studies used self-rated performance; "
        "this was judged not serious and is examined in Table 3."
    )


def _add_figure_title(doc, number, title):
    """APA Figure N title: bold number on one line, italic title on next."""
    p = doc.add_paragraph()
    set_double_space(p)
    run = p.add_run(f"Figure {number}")
    run.bold = True
    run.font.name = "Times New Roman"
    run.font.size = Pt(12)
    p2 = doc.add_paragraph()
    set_double_space(p2)
    run2 = p2.add_run(title)
    run2.italic = True
    run2.font.name = "Times New Roman"
    run2.font.size = Pt(12)


def build_figure1_prisma(doc):
    """Insert PRISMA-ScR flow diagram as Figure 1."""
    from pathlib import Path as _Path
    doc.add_page_break()
    _add_figure_title(
        doc, 1,
        "PRISMA-ScR Flow Diagram of the Study Selection Process",
    )

    img_path = (_Path(__file__).parent.parent / "analysis"
                / "figures" / "prisma_flow.png")
    if img_path.exists():
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run()
        run.add_picture(str(img_path), width=Inches(6.0))

    _add_table_note(
        doc,
        "Flow diagram following PRISMA-ScR (Tricco et al., 2018), with the "
        "layout of Page et al. (2021). Records were identified from eight "
        "web-search-engine queries (about 80 hits), 28 primary studies "
        "already known from preliminary searching, and the reference lists "
        "of five benchmark meta-analyses. The numbers of duplicates and of "
        "records excluded at title and abstract screening were not logged. "
        "From the full-text stage onward the counts are exact: of 37 "
        "reports sought, one could not be retrieved (A-27) and 36 were "
        "assessed; 11 were excluded (no Big Five measure, 5; face-to-face "
        "only, 4: A-09, A-10, A-14, A-16; sample overlap, 1: A-05; no "
        "extractable effect size, 1: A-24), leaving 25 studies. Of these, "
        f"{V['n_pool']} contributed correlations to the quantitative pool "
        f"({V['n_direct']} direct, {V['n_beta']} converted from β) and "
        f"{V['n_retained'] - V['n_pool']} entered the narrative map only, "
        "including A-37, which was left out of the pool. Table 1 "
        "catalogues 31 of the 37 reports."
    )


def _add_figure_image(doc, number, title, note, img_filename, width_inches=6.0):
    """Insert a figure with APA title + image + Note caption."""
    from pathlib import Path as _Path
    doc.add_page_break()
    _add_figure_title(doc, number, title)

    img_path = (_Path(__file__).parent.parent / "analysis"
                / "figures" / img_filename)
    if img_path.exists():
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run()
        run.add_picture(str(img_path), width=Inches(width_inches))

    p_note = doc.add_paragraph()
    set_double_space(p_note)
    run = p_note.add_run("Note. ")
    run.italic = True
    run.font.name = "Times New Roman"
    run.font.size = Pt(11)
    run2 = p_note.add_run(note)
    run2.font.name = "Times New Roman"
    run2.font.size = Pt(11)


def build_forest_plots(doc):
    """Figures 2–6: Forest plots per Big Five trait (ordered by effect size)."""
    base_note = (
        "Forest plot of study-level Pearson correlations between "
        "{trait_name} and academic achievement in online learning "
        "environments. Each square represents a study; square size is "
        "proportional to the inverse-variance weight. Horizontal bars "
        "show 95% confidence intervals. The red diamond represents the "
        "random-effects REML pooled estimate with HKSJ 95% confidence "
        "interval. Studies are sorted by effect size (descending). "
        "See Table 2 for pooled statistics."
    )

    specs = [
        (2, "Conscientiousness", "forest_C.png"),
        (3, "Agreeableness",     "forest_A.png"),
        (4, "Openness",          "forest_O.png"),
        (5, "Neuroticism",       "forest_N.png"),
        (6, "Extraversion",      "forest_E.png"),
    ]
    for fig_num, trait_name, filename in specs:
        _add_figure_image(
            doc,
            fig_num,
            f"Forest Plot: {trait_name} and Academic Achievement in "
            f"Online Learning Environments",
            base_note.format(trait_name=trait_name),
            filename,
        )


def build_funnel_plots(doc):
    """Figures 7–11: Funnel plots per Big Five trait."""
    base_note = (
        "Funnel plot of Pearson correlations between {trait_name} and "
        "academic achievement in online learning environments. Points "
        "represent individual studies. The vertical dashed red line "
        "indicates the random-effects pooled correlation. Diagonal "
        "dashed lines represent pseudo-95% confidence bounds under the "
        "assumption of no heterogeneity. Asymmetric dispersion may "
        "suggest small-study effects or publication bias. Egger's test: "
        "{egger}. See Table 4 for the other diagnostics."
    )
    specs = [(7, "C"), (8, "A"), (9, "O"), (10, "N"), (11, "E")]
    for fig_num, t in specs:
        _add_figure_image(
            doc,
            fig_num,
            f"Funnel Plot: {TRAIT_NAMES[t]} and Academic Achievement in "
            f"Online Learning Environments",
            base_note.format(trait_name=TRAIT_NAMES[t],
                             egger=V[f"egger.{t}.short"]),
            f"funnel_{t}.png",
            width_inches=5.0,
        )


def _load_studies_for_table1():
    """Read data_extraction_populated.csv and return compact summary rows."""
    import csv
    from pathlib import Path as _Path
    csv_path = (_Path(__file__).parent.parent / "analysis"
                / "data_extraction_populated.csv")
    with csv_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    short_modality = {
        "fully_online": "Online",
        "fully_online (emergency remote)": "Online",
        "face-to-face": "FtF",
        "mixed": "Mixed",
        "not_online_specific": "Online*",
    }
    short_edu = {
        "Undergraduate": "UG",
        "Graduate": "Grad",
        "K-12": "K-12",
        "Mixed_UG_Grad": "UG/Grad",
        "Mixed_secondary_postsecondary": "Sec/Post",
    }

    out = []
    for r in rows:
        modality = short_modality.get(r.get("modality", ""), r.get("modality", "") or "—")
        if r.get("study_id") == "A-37":
            modality = "FtF/Online"
        out.append([
            r.get("study_id", ""),
            f"{r.get('first_author', '')} ({r.get('year', '')})",
            str(r.get("n_analyzed", "") or r.get("n_total", "") or "—"),
            r.get("country", "") or "—",
            modality,
            short_edu.get(r.get("education_level", ""), r.get("education_level", "") or "—"),
            r.get("era", "") or "—",
            r.get("region", "") or "—",
            r.get("personality_instrument", "") or "—",
            r.get("outcome_type", "") or "—",
            _table1_status(r),
            str(r.get("risk_of_bias_score", "") or "—"),
        ])
    return out


def build_table1_characteristics(doc):
    """Table 1: Study characteristics of all included studies."""
    doc.add_page_break()
    _add_table_title(
        doc, 1,
        "Characteristics of the 31 Catalogued Studies and Their "
        "Contributions to the Quantitative Synthesis",
    )

    # Use landscape-oriented section for this table? Keep portrait for simplicity.
    header = [
        "ID", "Author (Year)", "N", "Country", "Mode", "Level",
        "Era", "Region", "Instrument", "Outcome", "Inclusion", "RoB",
    ]
    data_rows = _load_studies_for_table1()

    all_rows = [header] + data_rows
    table = doc.add_table(rows=len(all_rows), cols=len(header))
    table.style = "Light Grid Accent 1"
    for i, row in enumerate(all_rows):
        for j, val in enumerate(row):
            cell = table.rows[i].cells[j]
            cell.text = val
            if i == 0:
                for para in cell.paragraphs:
                    for run in para.runs:
                        run.bold = True

    # Use smaller font for this wide table
    for row in table.rows:
        for cell in row.cells:
            for para in cell.paragraphs:
                para.paragraph_format.space_before = Pt(0)
                para.paragraph_format.space_after = Pt(0)
                para.paragraph_format.line_spacing = 1.0
                for run in para.runs:
                    run.font.name = "Times New Roman"
                    run.font.size = Pt(8)
                    rPr = run._element.get_or_add_rPr()
                    rFonts = rPr.find(qn("w:rFonts"))
                    if rFonts is None:
                        rFonts = OxmlElement("w:rFonts")
                        rPr.append(rFonts)
                    rFonts.set(qn("w:eastAsia"), "Times New Roman")

    p_note = doc.add_paragraph()
    set_double_space(p_note)
    run = p_note.add_run("Note. ")
    run.italic = True
    run.font.name = "Times New Roman"
    run.font.size = Pt(10)
    run2 = p_note.add_run(TABLE1_NOTE)
    run2.font.name = "Times New Roman"
    run2.font.size = Pt(10)


def italicize_stats_in_doc(doc):
    """Auto-italicize APA statistical variables in all paragraphs.

    Pattern: match standalone r, p, k, N, M, SD, t, F, df, β, ρ, R², I²,
    τ², Q, z, n when followed by =, <, >, or (.  Does not touch text in
    table cells, headings, or already-italicized runs.
    """
    import re as _re
    # Stat tokens to italicize (most common APA 7th)
    STATS = r"(?:r|p|k|N|n|M|SD|df|F|t|z|β|ρ|R²|I²|τ²|τ|Q|QM|Q_between|Q_b)"
    # Match: whitespace/paren/start + STATS + space/operator/paren/comma/end
    pattern = _re.compile(
        r"(?<=[\s(\[])" + STATS + r"(?=[\s=<>≤≥,.\])]|$)"
    )

    def process_paragraph(para):
        # Skip headings and empty
        if "Heading" in para.style.name:
            return
        if not para.text:
            return

        # Reconstruct full text with italic flags from runs
        # Then rebuild runs based on pattern
        segments = []  # list of (text, italic)
        for run in para.runs:
            segments.append((run.text, run.italic))

        # Concatenate all text for pattern matching
        full_text = "".join(t for t, _ in segments)
        existing_italic = set()  # offsets already italic
        offset = 0
        for t, it in segments:
            if it:
                existing_italic.update(range(offset, offset + len(t)))
            offset += len(t)

        # Find all matches, but skip if overlap with existing italic
        matches = list(pattern.finditer(full_text))
        if not matches:
            return

        # Clear runs
        for run in list(para.runs):
            run._element.getparent().remove(run._element)

        # Rebuild
        cursor = 0
        for m in matches:
            start, end = m.start(), m.end()
            if any(i in existing_italic for i in range(start, end)):
                continue
            if start > cursor:
                r = para.add_run(full_text[cursor:start])
                _set_run_font(r)
            r_italic = para.add_run(full_text[start:end])
            r_italic.italic = True
            _set_run_font(r_italic)
            cursor = end
        if cursor < len(full_text):
            r = para.add_run(full_text[cursor:])
            _set_run_font(r)

    for para in doc.paragraphs:
        process_paragraph(para)
    # Also process table cell paragraphs
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for para in cell.paragraphs:
                    process_paragraph(para)


def _trait_sentence(t):
    return (
        f"k = {V[t + '.k']}, N = {V[t + '.N']}, r = {V[t + '.r']}, "
        f"95% CI {V[t + '.ci']}, 95% PI {V[t + '.pi']}; "
        f"{V[t + '.Q']}, I² = {V[t + '.I2']}, τ² = {V[t + '.tau2']}"
    )

def apa_floor(p):
    """Two-decimal floor of a p-value, for 'all p ≥ .xx' statements."""
    return apa(math.floor(p * 100) / 100, 2)

def max_abs(key):
    return apa(max(abs(V[f"sens.{key}.{t}.d_raw"]) for t in TRAITS))

def join_names(items, names=True):
    words = [TRAIT_NAMES[i] for i in items] if names else list(items)
    if len(words) <= 1:
        return "".join(words)
    if len(words) == 2:
        return f"{words[0]} and {words[1]}"
    return ", ".join(words[:-1]) + f", and {words[-1]}"

NUM_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
             6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}


def join_words(words, conj="and"):
    words = list(words)
    if len(words) <= 1:
        return "".join(words)
    if len(words) == 2:
        return f"{words[0]} {conj} {words[1]}"
    return ", ".join(words[:-1]) + f", {conj} {words[-1]}"


def pi_excludes_zero(t):
    return V[f"{t}.pi_lo_raw"] > 0 or V[f"{t}.pi_hi_raw"] < 0


def pi_summary():
    excl = [t for t in TRAITS if pi_excludes_zero(t)]
    if not excl:
        return "All five 95% prediction intervals included zero."
    plural = "s" if len(excl) > 1 else ""
    return (f"Only the {join_names(excl)} prediction interval{plural} "
            f"excluded zero ({'; '.join(V[f'{t}.pi'] for t in excl)}).")


def non_asia_k():
    ks = sorted({V["_moderators"]["region"][t]["non-Asia"]["k"] for t in TRAITS})
    return join_words([NUM_WORDS[k] for k in ks], "or")


def k_range():
    ks = sorted(int(V[f"{t}.k"]) for t in TRAITS)
    return f"{ks[0]}–{ks[-1]}" if ks[0] != ks[-1] else str(ks[0])


def egger_summary():
    sig = [t for t in TRAITS if V[f"egger.{t}.p_raw"] < .05]
    ns = [t for t in TRAITS if t not in sig]
    short = [f"{TRAIT_NAMES[t]} ({V[f'egger.{t}.short']})" for t in ns]
    if not sig:
        return f"Egger's test was not significant for any trait: {'; '.join(short)}."
    full = [f"{TRAIT_NAMES[t]} ({V[f'egger.{t}']})" for t in sig]
    return (f"Egger's test was significant for {join_words(full)} and not "
            f"for {join_words(short, 'or')}.")


def grade(t):
    """Completeness-of-evidence rating for one trait, from the rules stated
    in Methods: start High; −1 per serious, −2 per very serious concern."""
    i2 = float(V[f"{t}.I2"].rstrip("%"))
    lo, hi = (float(x.replace("−", "-")) for x in V[f"{t}.ci"][1:-1].split(", "))
    out = {"rob": "Not serious", "indirect": "Not serious", "down": 0}
    if i2 > 75:
        out["incon"], out["down"] = "Very serious", out["down"] + 2
    elif i2 > 50:
        out["incon"], out["down"] = "Serious", out["down"] + 1
    else:
        out["incon"] = "Not serious"
    crosses = lo < 0 < hi
    if (crosses and max(abs(lo), abs(hi)) >= .10) or hi - lo > .20:
        out["imprec"], out["down"] = "Serious", out["down"] + 1
    else:
        out["imprec"] = "Not serious"
    out["pubbias"], out["down"] = "Serious", out["down"] + 1
    levels = ["Very Low", "Low", "Moderate", "High"]
    out["rating"] = levels[max(0, 3 - out["down"])]
    return out

def _fill_table(doc, data):
    table = doc.add_table(rows=len(data), cols=len(data[0]))
    table.style = "Light Grid Accent 1"
    for i, row in enumerate(data):
        for j, val in enumerate(row):
            cell = table.rows[i].cells[j]
            cell.text = val
            if i == 0:
                for para in cell.paragraphs:
                    for run in para.runs:
                        run.bold = True
    _style_table(table)
    return table

def _add_table_note(doc, text):
    p_note = doc.add_paragraph()
    set_double_space(p_note)
    run = p_note.add_run("Note. ")
    run.italic = True
    run.font.name = "Times New Roman"
    run.font.size = Pt(11)
    run2 = p_note.add_run(text)
    run2.font.name = "Times New Roman"
    run2.font.size = Pt(11)

def _table1_status(row):
    """Status shown in Table 1, derived from the data rather than labels."""
    import pool as _pool
    sid = row.get("study_id", "")
    sources = {(_pool.extract_effect_for_trait(row, t) or (0, 0, ""))[2]
               for t in TRAITS} - {""}
    if sid in _pool.POOL_EXCLUSIONS:
        return "Map only†"
    if sources:
        return "Pool (β→r)" if sources == {"beta_converted"} else "Pool (r)"
    status = row.get("inclusion_status", "")
    if status in ("include", "include_with_caveat", "include_secondary",
                  "exclude_from_primary"):
        return "Map only"
    if status == "include_COI":
        return "Map only (COI)"
    return {
        "exclude_overlap": "Exc (overlap)",
        "PDF_unavailable": "Not retrieved",
    }.get(status, "Exc (no ES)" if sid == "A-24" else "Exc (FtF)")

TABLE1_NOTE = (
    "ID = study identifier used throughout the review. N = analytic sample "
    "size. Mode: Online = fully online; FtF = face-to-face; Mixed = blended "
    "or mode-comparing design; Online* = engagement scale not specific to "
    "online learning. Level: UG = "
    "undergraduate; Grad = graduate; UG/Grad = mixed; K-12 = elementary "
    "through high school; Sec/Post = mixed secondary and post-secondary. "
    "Era: pre-COVID ≤ 2019; COVID = 2020–2022; post-COVID ≥ 2023. Status: "
    "Pool (r) = contributed direct correlations to the quantitative pool; "
    "Pool (β→r) = contributed correlations converted from β (Peterson & "
    "Brown, 2005); Map only = retained for the narrative map (secondary outcomes or no "
    "extractable trait–achievement correlation); † = A-37, left out of the "
    "pool because its graduate cohorts were taught face to face before "
    "and after the pandemic and online only during it, and its "
    "correlations were reported only as |r| < .10; COI = the author's own "
    "study; Exc = excluded at full text (FtF = face-to-face only; no ES = "
    "no extractable effect size; overlap = sample overlap with A-04); Not "
    "retrieved = full text unavailable. RoB = Joanna Briggs Institute "
    "8-item score (0–8; higher = lower risk of bias). The table lists 31 of "
    "the 37 reports sought; A-14 Eilam et al. (2009; face-to-face) and five "
    "reports without a Big Five measure were excluded at full text and are "
    "documented in the reading notes in the project repository, which "
    "also record corrections to bibliographic details (A-06, A-07, A-08, "
    "A-12, A-13, A-16, A-18, A-20)."
)


def build_manuscript(for_journal: bool, output_path: str):
    """Build full manuscript. When for_journal=True, Preprint Statement is added."""
    doc = Document()
    configure_page(doc)
    configure_styles(doc)
    build_title_page(doc)
    doc.add_page_break()
    build_declarations(doc, for_journal=for_journal)
    doc.add_page_break()
    build_abstract(doc)
    build_intro_part1(doc)
    build_intro_part2(doc)
    build_intro_part3(doc)
    build_methods_part1(doc)
    build_methods_part2(doc)
    build_methods_part3(doc)
    build_methods_part4(doc)
    build_methods_part5(doc)
    build_results_part1(doc)
    build_results_part2(doc)
    build_results_part3(doc)
    build_discussion_part1(doc)
    build_discussion_part2(doc)
    build_discussion_part3(doc)
    build_conclusion(doc)
    build_references(doc)
    # APA 7th back-matter order: Tables first, then Figures
    build_table1_characteristics(doc)
    build_table2_pooled(doc)
    build_table3_moderators(doc)
    build_table4_sensitivity(doc)
    build_table5_grade(doc)
    build_figure1_prisma(doc)
    build_forest_plots(doc)
    build_funnel_plots(doc)
    # APA 7th: italicize statistical variables (r, p, N, k, etc.) auto-post-process
    italicize_stats_in_doc(doc)
    doc.save(output_path)
    print(f"Wrote {output_path}")


def build_split_titledecl(output_path: str, for_journal: bool = True):
    """Split 1/4: Title page + Declarations only."""
    doc = Document()
    configure_page(doc)
    configure_styles(doc)
    build_title_page(doc)
    doc.add_page_break()
    build_declarations(doc, for_journal=for_journal)
    italicize_stats_in_doc(doc)
    doc.save(output_path)
    print(f"Wrote {output_path}")


def build_split_body(output_path: str):
    """Split 2/4: Main text only (Abstract through References, no Tables/Figures)."""
    doc = Document()
    configure_page(doc)
    configure_styles(doc)
    build_abstract(doc)
    build_intro_part1(doc)
    build_intro_part2(doc)
    build_intro_part3(doc)
    build_methods_part1(doc)
    build_methods_part2(doc)
    build_methods_part3(doc)
    build_methods_part4(doc)
    build_methods_part5(doc)
    build_results_part1(doc)
    build_results_part2(doc)
    build_results_part3(doc)
    build_discussion_part1(doc)
    build_discussion_part2(doc)
    build_discussion_part3(doc)
    build_conclusion(doc)
    build_references(doc)
    italicize_stats_in_doc(doc)
    doc.save(output_path)
    print(f"Wrote {output_path}")


def build_split_tables(output_path: str):
    """Split 3/4: Tables 1–5 only."""
    doc = Document()
    configure_page(doc)
    configure_styles(doc)
    build_table1_characteristics(doc)
    build_table2_pooled(doc)
    build_table3_moderators(doc)
    build_table4_sensitivity(doc)
    build_table5_grade(doc)
    italicize_stats_in_doc(doc)
    doc.save(output_path)
    print(f"Wrote {output_path}")


def build_split_figures(output_path: str):
    """Split 4/4: Figures 1–11 only (PRISMA + 5 forest + 5 funnel)."""
    doc = Document()
    configure_page(doc)
    configure_styles(doc)
    build_figure1_prisma(doc)
    build_forest_plots(doc)
    build_funnel_plots(doc)
    italicize_stats_in_doc(doc)
    doc.save(output_path)
    print(f"Wrote {output_path}")


def main():
    import os as _os
    # Combined versions (preprint + journal)
    build_manuscript(for_journal=False, output_path=OUTPUT_PREPRINT)
    build_manuscript(for_journal=True, output_path=OUTPUT_JOURNAL)

    # Split versions (journal version only — preprint typically submitted as combined)
    split_dir = "/home/user/paper/metaanalysis/paper_v4/split"
    _os.makedirs(split_dir, exist_ok=True)
    build_split_titledecl(f"{split_dir}/01_title_declarations.docx", for_journal=True)
    build_split_body(f"{split_dir}/02_body.docx")
    build_split_tables(f"{split_dir}/03_tables.docx")
    build_split_figures(f"{split_dir}/04_figures.docx")


if __name__ == "__main__":
    main()
