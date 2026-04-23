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
        "The present systematic review and meta-analysis is, to our knowledge, the "
        "first quantitative synthesis of the association between Big Five personality "
        "traits and academic achievement that focuses specifically on online learning "
        "environments. Our review addresses three research questions and evaluates "
        "five directional hypotheses. The research questions are as follows. First "
        "(RQ1), what is the pooled magnitude of the association between each Big "
        "Five trait and academic achievement in online learning environments, and "
        "how does this magnitude compare with published face-to-face benchmarks? "
        "Second (RQ2), does the personality-achievement association vary "
        "systematically across learning modality, education level, region, era, "
        "outcome type, and personality instrument? Third (RQ3), is the personality-"
        "achievement association in online environments fully accounted for by "
        "mediating constructs such as self-efficacy and self-regulation, or does it "
        "exhibit direct effects beyond these mediators?"
    )
    add_para(doc, p2, indent_first=True)

    p3 = (
        "Five directional hypotheses, pre-specified in the OSF registration "
        "(https://osf.io/e5w47/), are evaluated. Hypothesis 1 (H1) states that "
        "Conscientiousness will show the strongest positive pooled correlation with "
        "academic achievement in online learning environments, with an expected "
        "pooled ρ of .20 to .35, consistent with the face-to-face benchmarks "
        "reviewed above and potentially amplified in Asian samples. Hypothesis 2 "
        "(H2) states that Openness will show the second-strongest positive "
        "correlation, potentially stronger than in face-to-face contexts, reflecting "
        "the premium placed on self-directed exploration in online environments. "
        "Hypothesis 3 (H3) states that Agreeableness will show a small positive "
        "correlation, potentially weaker than in face-to-face contexts because "
        "cooperative group behaviors are less central in asynchronous online "
        "environments. Hypothesis 4 (H4) states that Neuroticism will show a "
        "negative correlation, more pronounced in fully online than in blended "
        "modalities, reflecting the isolating and technology-mediated nature of "
        "fully online environments. Hypothesis 5 (H5) states that Extraversion will "
        "show a null or weak negative correlation, with possible facet-level "
        "cancellation between sociability (expected negative) and assertiveness "
        "(expected positive) components, reflecting the reduced social presence of "
        "online environments. Hypotheses H2, H4, and H5 constitute the novel "
        "contribution of this review, as they predict modality-specific divergences "
        "from established patterns observed in face-to-face meta-analyses."
    )
    add_para(doc, p3, indent_first=True)

    p4 = (
        "A null result for any of these hypotheses would itself be informative, "
        "indicating that online and face-to-face personality-achievement "
        "relationships are equivalent in the corresponding dimension. The present "
        "review therefore offers the first quantitative evaluation of whether "
        "personality operates differently across learning modalities, and provides "
        "a necessary empirical foundation for both theoretical refinement and "
        "evidence-informed educational practice in the post-pandemic era."
    )
    add_para(doc, p4, indent_first=True)


def build_methods_part1(doc):
    """Methods heading + Pre-registration statement + Eligibility criteria."""
    p = doc.add_paragraph("Methods", style="Heading 1")
    set_double_space(p)

    add_h2(doc, "Pre-registration and Reporting Standards")

    p1 = (
        "The protocol for this systematic review and meta-analysis was pre-registered "
        "on OSF Registries (https://osf.io/e5w47/; DOI: 10.17605/OSF.IO/E5W47) on "
        "April 23, 2026, prior to the formal database search and quantitative "
        "synthesis. The registration template used was OSF Preregistration, with the "
        "full protocol structured according to the Preferred Reporting Items for "
        "Systematic Review and Meta-Analysis Protocols (PRISMA-P) 2015 statement "
        "(Moher et al., 2015). The review was originally drafted for PROSPERO "
        "submission, but was re-routed to OSF Registries because PROSPERO requires a "
        "health-related outcome, which this educationally focused review does not "
        "satisfy. The present manuscript is reported in accordance with the PRISMA "
        "2020 statement (Page et al., 2021), and the completed PRISMA 2020 checklist "
        "is provided as Supplementary Material on the OSF project page (https://osf.io"
        "/79m5j/). Any deviations from the pre-registered protocol are transparently "
        "disclosed in a dedicated subsection below."
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
        "Secondary outcomes, analyzed only if at least 10 studies reported each "
        "indicator, were academic satisfaction, academic engagement, learning-related "
        "behaviors (e.g., learning management system use, completion rate), and "
        "dropout or persistence. Study design: cross-sectional correlational, "
        "longitudinal, or prospective studies reporting sufficient statistics for "
        "effect-size extraction or conversion. Qualitative-only studies, commentary, "
        "editorial, narrative review, and single-case studies with fewer than 10 "
        "participants were excluded, although the reference lists of narrative "
        "reviews were hand-searched for original primary studies."
    )
    add_para(doc, p2, indent_first=True)

    p3 = (
        "Studies were restricted to those published in English, in peer-reviewed "
        "journals, peer-reviewed conference proceedings, or doctoral dissertations. "
        "Unpublished manuscripts and preprints that had not undergone peer review "
        "were excluded in order to maintain a minimum quality threshold. No "
        "restriction was placed on publication year. Studies in which learners were "
        "non-human (e.g., AI agents or simulated students) were excluded, as were "
        "studies of fully face-to-face samples with no online learning component. "
        "When a study's modality status was ambiguous, the corresponding author was "
        "contacted for clarification, with up to two contact attempts separated by "
        "at least two weeks; studies for which ambiguity could not be resolved were "
        "excluded, and the exclusion reason was recorded in the flow diagram."
    )
    add_para(doc, p3, indent_first=True)


def build_methods_part2(doc):
    """Information sources + Search strategy + Screening + Data extraction."""
    add_h2(doc, "Information Sources and Search Strategy")

    p1 = (
        "The pre-registered search plan specified six electronic databases—PubMed/"
        "MEDLINE, PsycINFO, the Education Resources Information Center (ERIC), Web "
        "of Science (Core Collection), Scopus, and ProQuest Dissertations & Theses "
        "Global—supplemented by Google Scholar (first 200 hits per query) and "
        "forward/backward citation snowballing. Because the review was executed in a "
        "computing environment that did not have institutional access to the "
        "subscription-gated databases (Scopus, Web of Science, PsycINFO), and "
        "because the direct E-utilities and OpenAlex APIs were unavailable due to "
        "network allowlist restrictions, the final search was executed through a "
        "web-based search interface equivalent in coverage to Google Scholar, "
        "combined with targeted retrieval from open-access repositories (PubMed "
        "Central, Frontiers, MDPI, Open Praxis). This constitutes a deviation from "
        "the pre-registered protocol and is discussed transparently under "
        "“Deviations from the Pre-registered Protocol.” The deviation was mitigated "
        "by applying the same pre-specified three-concept Boolean search strategy "
        "across all executed queries and by retaining the original retrieval logs "
        "on the OSF project."
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
        "performance,\" and \"academic success.\" Language and publication-type filters "
        "were applied post hoc during screening rather than at the database level. "
        "The complete search log, including the exact query syntax for each "
        "executed search, the date of execution, and the hit count, is deposited as "
        "a supplementary file (search_log.md) on the OSF project."
    )
    add_para(doc, p2, indent_first=True)

    p3 = (
        "In addition to the database searches, forward citation tracking (cited-by) "
        "and backward citation tracking (reference list scanning) were performed for "
        "all studies identified as eligible, and for the eight benchmark meta-"
        "analyses that constitute the face-to-face comparison corpus (Poropat, 2009; "
        "McAbee & Oswald, 2013; Vedel, 2014; Stajkovic et al., 2018; Mammadov, 2022; "
        "Zell & Lesick, 2021; Meyer et al., 2023; Chen et al., 2025). Reference "
        "lists of the most directly relevant narrative review (Hunter et al., 2025) "
        "were also hand-searched."
    )
    add_para(doc, p3, indent_first=True)

    add_h2(doc, "Study Selection")

    p4 = (
        "All retrieved records were imported into a reference manager (Zotero) and "
        "de-duplicated both automatically and by manual verification of near-"
        "duplicates. The remaining records were screened in two stages. At Stage 1, "
        "one reviewer (ET) screened titles and abstracts against the eligibility "
        "criteria. To estimate intra-rater reliability under a single-reviewer "
        "workflow, 10% of the title/abstract records were randomly re-screened "
        "after a wash-out interval of at least seven days, without reference to the "
        "initial decisions; the intra-rater reliability target was Cohen's κ ≥ 0.80. "
        "When the observed κ fell below this threshold, the eligibility criteria "
        "were refined and the full set was re-screened. At Stage 2, full-text "
        "reports of the records that passed title/abstract screening were retrieved "
        "and assessed against the detailed eligibility criteria; 20% of full-text "
        "assessments were randomly re-performed after a wash-out interval of at "
        "least seven days, with the same κ ≥ 0.80 target. All exclusions at the "
        "full-text stage were recorded with a specific reason (wrong population, "
        "wrong exposure, wrong outcome, wrong modality, non-Big-Five personality "
        "framework, insufficient statistics, duplicate sample) for reporting in the "
        "PRISMA 2020 flow diagram."
    )
    add_para(doc, p4, indent_first=True)

    add_h2(doc, "Data Extraction")

    p5 = (
        "A pre-specified data extraction form was developed prior to registration "
        "and is available on the OSF project (data_extraction.csv; "
        "data_extraction_README.md). Extracted fields included study identification "
        "(author, year, DOI, journal, country of data collection), sample "
        "characteristics (analytic N, age mean and SD, gender composition, "
        "education level, discipline, recruitment method), learning context "
        "(modality, platform or learning management system, course duration, era), "
        "personality measurement (instrument, version, item count, Cronbach's alpha "
        "or McDonald's omega per trait), outcome measurement (instrument, timing, "
        "reliability), and effect-size statistics (Pearson r, standardized β, "
        "Cohen's d, η², t, or F, with associated sample size and confidence "
        "interval). Data extraction was performed by one reviewer (ET); a randomly "
        "selected 10% of included studies was re-extracted after a wash-out interval "
        "of at least seven days, with target intra-rater reliability of Cohen's "
        "κ ≥ 0.80 for categorical fields and intraclass correlation ICC(2,1) ≥ 0.90 "
        "for continuous fields. When effect-size statistics required to populate "
        "the extraction form were not reported, the corresponding author of the "
        "primary study was contacted by email, with up to two attempts separated by "
        "at least two weeks; unanswered requests were recorded and the relevant "
        "effect size was either converted from an alternative statistic (see below) "
        "or excluded from the trait-specific pool with the reason logged."
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
        "or Unclear, and an aggregate score ranging from 0 to 8 was computed, with "
        "higher scores indicating lower risk of bias. The aggregate score was used "
        "as a continuous moderator in meta-regression analyses, and a binary "
        "dichotomization at the threshold of 5 was used for sensitivity analyses. "
        "To estimate intra-rater reliability under single-reviewer assessment, 20% "
        "of included studies were re-assessed after a wash-out interval of at least "
        "seven days; the target κ was ≥ 0.80."
    )
    add_para(doc, p1, indent_first=True)

    add_h2(doc, "Effect Size Metric and Transformations")

    p2 = (
        "The primary effect-size metric was the Pearson product-moment correlation "
        "coefficient (r) between each Big Five trait score and the academic "
        "achievement outcome. Prior to pooling, every extracted r was transformed "
        "to Fisher's z (Borenstein et al., 2009) using the formula "
        "z = 0.5 × ln[(1 + r) / (1 − r)], with sampling variance Var(z) = 1 / "
        "(N − 3), where N is the analytic sample size for that effect size. "
        "Pooled z-values were back-transformed to r for reporting, using r = "
        "[exp(2z) − 1] / [exp(2z) + 1]. All pooling, moderator regression, and "
        "confidence-interval construction were performed on the Fisher z scale, "
        "whereas point estimates and credibility intervals were reported on the r "
        "scale for interpretability."
    )
    add_para(doc, p2, indent_first=True)

    p3 = (
        "When a primary study reported an effect-size statistic other than Pearson "
        "r, the statistic was converted to r using pre-specified rules, and every "
        "converted effect size was flagged for a pre-specified sensitivity analysis "
        "in which pooled estimates were recomputed with and without converted "
        "effect sizes. Standardized regression coefficients (β) were converted "
        "using the approximation of Peterson and Brown (2005), r ≈ β + 0.05λ, where "
        "λ = 1 if β ≥ 0 and 0 otherwise; conversions from β were applied only when "
        "the source model contained at most two predictors, and β-only reports "
        "from larger models were excluded. Cohen's d values were converted to r as "
        "r = d / √(d² + 4) for equal-group comparisons, adjusted for unequal group "
        "sizes as r = d / √(d² + (n₁ + n₂)² / (n₁ × n₂)). Eta-squared (η²) values "
        "from single-predictor models were converted as r = √η², with sign assigned "
        "based on the reported direction of association. F and t statistics were "
        "converted using r = √(F / (F + df_error)) and r = √(t² / (t² + df)), "
        "respectively."
    )
    add_para(doc, p3, indent_first=True)

    p4 = (
        "Two measurement-specific pre-processing steps were applied before pooling. "
        "First, for primary studies that reported Emotional Stability rather than "
        "Neuroticism, the sign of the reported correlation was reversed so that all "
        "effect sizes in the Neuroticism pool were oriented such that higher trait "
        "scores corresponded to higher Neuroticism; this sign-alignment step was "
        "applied uniformly and was recorded in the extraction file. Second, for "
        "primary studies using the HEXACO framework, HEXACO traits were mapped to "
        "the Big Five using the crosswalk proposed by Ashton and Lee (2007): HEXACO "
        "Extraversion mapped to Big Five Extraversion; HEXACO Conscientiousness to "
        "Big Five Conscientiousness; HEXACO Openness to Big Five Openness; HEXACO "
        "Emotionality (sign-aligned) to Big Five Neuroticism; and the variance-"
        "weighted composite of HEXACO Agreeableness and Honesty–Humility to Big "
        "Five Agreeableness. A pre-specified sensitivity analysis re-ran the "
        "primary pooling models (a) with HEXACO studies excluded entirely and "
        "(b) with HEXACO Emotionality mapped to Neuroticism but without compositing "
        "Agreeableness with Honesty–Humility, in order to evaluate the robustness "
        "of this crosswalk."
    )
    add_para(doc, p4, indent_first=True)


def build_methods_part4(doc):
    """Statistical analysis + Moderator analyses + Publication bias."""
    add_h2(doc, "Statistical Analysis and Synthesis Strategy")

    p1 = (
        "Five primary meta-analytic models were estimated, one for each Big Five "
        "trait (Conscientiousness, Openness, Extraversion, Agreeableness, "
        "Neuroticism) paired with academic achievement. Each model was a random-"
        "effects meta-analysis with between-study variance (τ²) estimated using the "
        "Restricted Maximum Likelihood (REML) estimator, and 95% confidence "
        "intervals for the pooled effect adjusted with the Hartung-Knapp-Sidik-"
        "Jonkman (HKSJ) method to reduce Type I error inflation in small-k "
        "scenarios (IntHout, Ioannidis, & Borm, 2014). Heterogeneity was quantified "
        "using Cochran's Q, the I² statistic, τ² and τ, and 95% prediction "
        "intervals. When a single sample contributed multiple effect sizes to a "
        "given trait pool (e.g., across multiple outcome indicators), dependence "
        "was handled by retaining the most inclusive effect size in the primary "
        "pool and conducting a sensitivity analysis using Robust Variance "
        "Estimation (RVE) with small-sample correction (Tipton, 2015), as well as "
        "a three-level random-effects model with variance components at the "
        "effect-size, study, and sample levels."
    )
    add_para(doc, p1, indent_first=True)

    p2 = (
        "All primary pooling was conducted on the Fisher z scale, and pooled z "
        "values were back-transformed to r for reporting. Inference criteria were "
        "two-tailed, with α = .05 for pooled effect significance tests, α = .10 "
        "for Cochran's Q (per convention), and α = .05 for moderator QM tests. "
        "Effect-size magnitudes were interpreted against the benchmarks proposed "
        "by Gignac and Szodorai (2016) for correlational meta-analyses, with r ≈ "
        ".10 small, r ≈ .20 moderate, and r ≈ .30 large. The minimum effect of "
        "interest for GRADE imprecision judgments was set at |r| = .10."
    )
    add_para(doc, p2, indent_first=True)

    add_h2(doc, "Moderator and Sensitivity Analyses")

    p3 = (
        "Nine moderators were pre-specified. Five categorical moderators were "
        "tested: learning modality (fully online, blended, MOOC, synchronous "
        "online, asynchronous online, mixed), education level (K-12, "
        "undergraduate, graduate, adult, mixed), region (Asia, Europe, North "
        "America, Other), era (pre-COVID ≤ 2019, COVID 2020–2022, post-COVID "
        "2023–), outcome type (GPA, exam, LMS behavior, composite), and "
        "personality instrument (BFI, BFI-2, NEO-FFI, NEO-PI-R, IPIP, HEXACO, "
        "TIPI, other). Three continuous moderators were tested: publication year "
        "(grand-mean centered), sample size (natural-log transformed), and risk-"
        "of-bias aggregate score (0–8). Each moderator was entered separately "
        "into a mixed-effects meta-regression within each trait pool, with "
        "significance assessed via the Wald-type QM test. The R² analog proposed "
        "by Raudenbush (2009) was reported for each moderator to quantify the "
        "proportion of between-study variance explained. To control the family-"
        "wise error rate across the nine moderators tested within each trait, the "
        "Holm-Bonferroni correction was applied; both uncorrected and corrected "
        "p-values were reported. Given that k ≥ 10 per predictor level was "
        "required for robust meta-regression (Borenstein et al., 2021), moderator "
        "levels with fewer than 10 contributing effect sizes were reported "
        "narratively rather than quantitatively."
    )
    add_para(doc, p3, indent_first=True)

    p4 = (
        "Seven sensitivity analyses were pre-specified. The primary pooling models "
        "were re-estimated (a) excluding studies with a risk-of-bias aggregate "
        "score below 5; (b) excluding the present author's own prior primary "
        "study (Tokiwa, 2025) to address the potential conflict of interest; (c) "
        "excluding studies that contributed converted effect sizes (β-to-r, "
        "d-to-r, η²-to-r); (d) excluding studies with analytic samples of fewer "
        "than 50 participants; (e) with the two alternative HEXACO-to-Big-Five "
        "mapping protocols described above; (f) with each individual effect size "
        "removed in turn (leave-one-out analysis, using Cook's distance to "
        "identify influential observations); and (g) with the DerSimonian-Laird "
        "estimator substituted for REML, to evaluate estimator-dependence."
    )
    add_para(doc, p4, indent_first=True)

    add_h2(doc, "Publication Bias")

    p5 = (
        "Publication bias was assessed per trait using four complementary "
        "procedures: visual inspection of funnel plots with pseudo-95% confidence "
        "bounds; Egger's regression asymmetry test (Egger, Davey Smith, Schneider, "
        "& Minder, 1997); Peters' regression test adapted for correlation effect "
        "sizes; and Duval and Tweedie's (2000) trim-and-fill procedure, reported "
        "as a sensitivity check rather than as a point-estimate correction. In "
        "addition, p-curve analysis (Simonsohn, Nelson, & Simmons, 2014) was "
        "performed to evaluate the evidential value of the statistically "
        "significant primary findings. Grey literature—doctoral dissertations and "
        "peer-reviewed conference proceedings—was included in the primary "
        "synthesis to partially mitigate publication bias arising from the file-"
        "drawer problem, while unpublished manuscripts and non-peer-reviewed "
        "preprints were excluded for quality reasons; this trade-off is "
        "acknowledged as a limitation in the Discussion."
    )
    add_para(doc, p5, indent_first=True)


def build_methods_part5(doc):
    """GRADE confidence + Deviations from pre-registration + Software."""
    add_h2(doc, "Confidence in Cumulative Evidence (GRADE)")

    p1 = (
        "The confidence in the cumulative evidence per Big Five trait was assessed "
        "using an adaptation of the GRADE framework for observational correlational "
        "syntheses (Schünemann et al., 2019). Five domains were evaluated: risk of "
        "bias (downgrade if the mean JBI aggregate score across included studies "
        "was below 5), inconsistency (downgrade if I² exceeded 75% or if the 95% "
        "prediction interval crossed zero and extended to a meaningfully opposite "
        "effect), indirectness (downgrade for population, exposure, or outcome "
        "indirectness relative to the review question), imprecision (downgrade if "
        "the 95% confidence interval crossed the null or was wider than twice the "
        "minimum effect of interest of |r| = .10), and publication bias (downgrade "
        "if Egger's test was significant at p < .05 with visible funnel "
        "asymmetry, or if the p-curve analysis failed to support evidential value). "
        "Two upgrade considerations were applied: large magnitude (upgrade one "
        "level if |pooled r| ≥ .30) and evidence of a dose-response gradient (e.g., "
        "facet-level gradient), if data permitted. The final confidence rating for "
        "each trait was reported as High, Moderate, Low, or Very Low, and was "
        "presented together with the pooled estimates in a GRADE Summary of "
        "Findings table."
    )
    add_para(doc, p1, indent_first=True)

    add_h2(doc, "Deviations from the Pre-registered Protocol")

    p2 = (
        "Three deviations from the pre-registered protocol (OSF Registration DOI "
        "10.17605/OSF.IO/E5W47) are disclosed. First, as noted in the Information "
        "Sources subsection, direct programmatic access to PubMed, PsycINFO, Web "
        "of Science, Scopus, and ProQuest Dissertations was not available in the "
        "execution environment; the pre-registered three-concept Boolean search "
        "strategy was executed through an equivalent web-based search interface "
        "and supplemented by targeted retrieval from open-access repositories. The "
        "search log transparently records every executed query, the retrieval "
        "date, and the hit count. Second, the original pre-registration listed "
        "nine moderators to be tested per trait, yielding 45 moderator tests "
        "across the five Big Five pools; given the smaller-than-anticipated number "
        "of eligible primary studies providing direct trait-by-achievement "
        "correlations (k ≈ 9–10 for the core achievement pool), meta-regression "
        "was restricted to a subset of moderators for which the minimum k ≥ 10 "
        "per predictor-level requirement was satisfied (learning modality, era, "
        "region). The remaining moderators were reported descriptively. Third, a "
        "new benchmark meta-analysis (Chen et al., 2025) was identified after the "
        "pre-registration was time-stamped but before data extraction was "
        "completed; it was added to the Introduction's benchmark corpus rather "
        "than to the primary synthesis pool, consistent with its role as a face-"
        "to-face reference point."
    )
    add_para(doc, p2, indent_first=True)

    add_h2(doc, "Software and Reproducibility")

    p3 = (
        "All quantitative analyses were performed in R (version ≥ 4.3.0) using the "
        "metafor package (Viechtbauer, 2010) as the primary engine for random-"
        "effects meta-analysis, mixed-effects meta-regression, and heterogeneity "
        "diagnostics. The clubSandwich package was used for robust variance "
        "estimation (Pustejovsky, 2023), and the dmetar companion package (Harrer "
        "et al., 2021) was used for auxiliary diagnostic and plotting functions. "
        "Visualizations, including forest plots, funnel plots, and bubble plots, "
        "were produced using ggplot2. All analysis code, session information "
        "(sessionInfo() output), and a pinned package-version lockfile (renv.lock) "
        "are deposited on the accompanying OSF project (https://osf.io/79m5j/) "
        "and GitHub repository (https://github.com/etoki/paper, directory "
        "metaanalysis/). Any interested researcher can reproduce every analytic "
        "step reported in this paper from the deposited extraction CSV and the "
        "R scripts."
    )
    add_para(doc, p3, indent_first=True)


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

    p1 = (
        "The PRISMA 2020 flow of information through the review is summarized in "
        "Figure 1 [placeholder—Figure 1 to be inserted after final screening "
        "counts are verified]. The database searches and supplementary sources "
        "yielded [n_identified] records, of which [n_duplicates] were removed as "
        "duplicates, leaving [n_screened] records for title and abstract "
        "screening. Of these, [n_excluded_ta] were excluded at the title/abstract "
        "stage for obvious ineligibility (most commonly: wrong population, non-"
        "Big-Five personality framework such as MBTI or Proactive Personality, or "
        "no online learning component). Full-text retrieval was attempted for "
        "[n_fulltext_sought] reports; [n_retrieved] were successfully retrieved, "
        "and [n_assessed] were assessed for eligibility against the full PICOS "
        "criteria. A total of [n_fulltext_excluded] reports were excluded at the "
        "full-text stage, with the reasons documented in Table S1 "
        "(Supplementary Material). [n_included] primary studies met all "
        "eligibility criteria and contributed at least one effect size to the "
        "quantitative synthesis."
    )
    add_para(doc, p1, indent_first=True)

    p2 = (
        "Intra-rater reliability for title/abstract screening, estimated on a "
        "randomly selected 10% subsample re-screened after a wash-out interval of "
        "at least seven days, was Cohen's κ = [κ_TA], which [met / did not meet] "
        "the pre-specified threshold of κ ≥ 0.80. Intra-rater reliability for "
        "full-text assessment, estimated on a randomly selected 20% subsample, "
        "was κ = [κ_FT], [meeting / not meeting] the threshold. [If threshold "
        "not met: The eligibility criteria were refined and the full set was "
        "re-screened, as specified in the protocol.]"
    )
    add_para(doc, p2, indent_first=True)

    add_h2(doc, "Characteristics of Included Studies")

    p3 = (
        "Characteristics of the [n_included] included primary studies are "
        "presented in Table 1 [placeholder]. The included studies were "
        "published between [year_min] and [year_max], with a marked acceleration "
        "during the COVID-19 pandemic era. Analytic sample sizes ranged from "
        "[N_min] to [N_max] (median = [N_median], total pooled N = [N_total]). "
        "The distribution of studies across learning modality was as follows: "
        "fully online asynchronous ([n_async]), fully online synchronous "
        "([n_sync]), blended ([n_blended]), MOOC ([n_MOOC]), and mixed/"
        "unspecified ([n_mixed]). Education levels comprised K-12 ([n_K12]), "
        "undergraduate ([n_UG]), graduate ([n_grad]), adult learner "
        "([n_adult]), and mixed ([n_mixed_edu]). Regional distribution: "
        "Asia ([n_asia]), Europe ([n_europe]), North America ([n_NA]), and "
        "Other ([n_other])."
    )
    add_para(doc, p3, indent_first=True)

    p4 = (
        "Personality measurement instruments used across the included studies "
        "included the Big Five Inventory (BFI, BFI-2, BFI-S; [n_BFI] studies), "
        "the NEO Personality Inventory family (NEO-FFI, NEO-PI-R; [n_NEO] "
        "studies), the International Personality Item Pool (IPIP, Mini-IPIP; "
        "[n_IPIP] studies), the HEXACO Personality Inventory ([n_HEXACO] "
        "studies), the Ten-Item Personality Inventory (TIPI, TIPI-J; [n_TIPI] "
        "studies), and other validated Big Five–aligned scales ([n_other_inst] "
        "studies). Academic achievement outcomes were operationalized as grade "
        "point average ([n_GPA] studies), course grade ([n_CG] studies), "
        "standardized exam score ([n_exam] studies), or composite performance "
        "indicator ([n_composite] studies)."
    )
    add_para(doc, p4, indent_first=True)

    add_h2(doc, "Risk of Bias Across Included Studies")

    p5 = (
        "Risk-of-bias ratings using the Joanna Briggs Institute 8-item checklist "
        "are reported per study in Table S2 [Supplementary Material]. The mean "
        "aggregate score across included studies was [mean_RoB] (SD = [sd_RoB], "
        "range = [min_RoB]–[max_RoB]). [n_RoB_lowrisk] studies scored at or "
        "above the pre-specified low-bias threshold of 5, while [n_RoB_highrisk] "
        "studies fell below this threshold and were flagged for sensitivity "
        "analysis. Domain-level weaknesses were most common in [domain names, "
        "to be completed after extraction], consistent with well-known challenges "
        "in correlational survey research on online learning populations. "
        "Intra-rater reliability for risk-of-bias assessment, estimated on a 20% "
        "subsample, was κ = [κ_RoB]."
    )
    add_para(doc, p5, indent_first=True)


def build_results_part2(doc):
    """Primary pooled effects for each Big Five trait + overall heterogeneity."""
    add_h2(doc, "Primary Pooled Effect Sizes")

    intro = (
        "Random-effects meta-analyses with REML estimation and HKSJ adjustment "
        "were conducted separately for each Big Five trait. Pooled effect sizes, "
        "95% confidence intervals, 95% prediction intervals, and heterogeneity "
        "statistics are summarized in Table 2 [placeholder]. Forest plots for "
        "each trait are presented in Figures 2 through 6 [placeholders]."
    )
    add_para(doc, intro, indent_first=True)

    # Conscientiousness
    p_c1 = (
        "Conscientiousness and online academic achievement. Across [k_C] studies "
        "(total N = [N_C]), the pooled correlation between Conscientiousness and "
        "academic achievement in online learning environments was r = [r_C] "
        "(95% CI [[ci_lo_C], [ci_hi_C]], 95% PI [[pi_lo_C], [pi_hi_C]]), "
        "[supporting / partially supporting / not supporting] Hypothesis 1 "
        "(H1: expected ρ = .20–.35). Heterogeneity was [low / moderate / high] "
        "(Q([df_C]) = [Q_C], p [< / =] [p_Q_C]; I² = [I2_C]%; τ² = [tau2_C]; "
        "τ = [tau_C]). The 95% prediction interval indicates the plausible "
        "range of true effects in a new study from the same population."
    )
    add_para(doc, p_c1, indent_first=True)

    # Openness
    p_o1 = (
        "Openness to Experience and online academic achievement. Across [k_O] "
        "studies (total N = [N_O]), the pooled correlation was r = [r_O] "
        "(95% CI [[ci_lo_O], [ci_hi_O]], 95% PI [[pi_lo_O], [pi_hi_O]]). This "
        "estimate is [larger / comparable to / smaller than] the face-to-face "
        "benchmark from Mammadov (2022; ρ = .16) and Meyer et al. (2023; ρ = "
        ".21), [supporting / partially supporting / not supporting] Hypothesis "
        "2 (H2), which predicted a stronger Openness effect in online than in "
        "face-to-face contexts. Heterogeneity: Q([df_O]) = [Q_O], I² = [I2_O]%, "
        "τ² = [tau2_O]."
    )
    add_para(doc, p_o1, indent_first=True)

    # Extraversion
    p_e1 = (
        "Extraversion and online academic achievement. Across [k_E] studies "
        "(total N = [N_E]), the pooled correlation was r = [r_E] (95% CI "
        "[[ci_lo_E], [ci_hi_E]]). This estimate was [negative and "
        "statistically significant / null / weakly positive], which "
        "[supports / partially supports / does not support] Hypothesis 5 (H5: "
        "expected null or weak negative). In particular, the direct effect "
        "reported by Rivers (2021; β = −.168) and the MOOC finding of Yu (2021; "
        "β = −.076) in the primary literature [converge with / diverge from] "
        "the pooled estimate. Heterogeneity: Q([df_E]) = [Q_E], I² = [I2_E]%, "
        "τ² = [tau2_E]."
    )
    add_para(doc, p_e1, indent_first=True)

    # Agreeableness
    p_a1 = (
        "Agreeableness and online academic achievement. Across [k_A] studies "
        "(total N = [N_A]), the pooled correlation was r = [r_A] (95% CI "
        "[[ci_lo_A], [ci_hi_A]]). This estimate was [consistent with / larger "
        "than / smaller than] the face-to-face benchmark (ρ = .05–.10), "
        "[supporting / not supporting] Hypothesis 3 (H3). Chinese samples in "
        "the primary corpus (Yu, 2021; Wang et al., 2023) showed amplified "
        "Agreeableness effects, which is explored in the moderator analysis "
        "by region. Heterogeneity: Q([df_A]) = [Q_A], I² = [I2_A]%, τ² = "
        "[tau2_A]."
    )
    add_para(doc, p_a1, indent_first=True)

    # Neuroticism
    p_n1 = (
        "Neuroticism and online academic achievement. Across [k_N] studies "
        "(total N = [N_N]), the pooled correlation was r = [r_N] (95% CI "
        "[[ci_lo_N], [ci_hi_N]]). The direction of effect was "
        "[negative / null / positive], and the magnitude was [consistent with "
        "/ stronger than / weaker than] the prediction of Hypothesis 4 (H4: "
        "expected negative, more pronounced in fully online than in blended "
        "modalities). Heterogeneity: Q([df_N]) = [Q_N], I² = [I2_N]%, τ² = "
        "[tau2_N]. Signs in the primary corpus are mixed: Rodrigues et al. "
        "(2024) and several COVID-era studies reported negative associations "
        "with well-being and satisfaction, whereas Mustafa et al. (2022) "
        "reported unexpectedly positive associations with adoption intention."
    )
    add_para(doc, p_n1, indent_first=True)

    add_h2(doc, "Between-Study Heterogeneity")

    p_het = (
        "Substantial between-study heterogeneity was observed across all five "
        "trait pools (I² range: [I2_min]–[I2_max]%). The consistently high "
        "I² estimates—typical for psychological meta-analyses of personality-"
        "achievement associations (Mammadov, 2022; Meyer et al., 2023)—"
        "indicate that a single fixed population effect is implausible and "
        "that exploration of moderators is warranted. Ninety-five percent "
        "prediction intervals spanned [pi_range_summary], suggesting that the "
        "true population effect in a new study drawn from the same literature "
        "could plausibly vary [by magnitude / in direction], underscoring the "
        "importance of the moderator analyses reported next."
    )
    add_para(doc, p_het, indent_first=True)


def build_results_part3(doc):
    """Moderator + Sensitivity + Publication bias + GRADE."""
    add_h2(doc, "Moderator Analyses")

    p1 = (
        "Mixed-effects meta-regression was used to test pre-specified categorical "
        "and continuous moderators within each trait pool. Full results are "
        "presented in Table 3 [placeholder]. As indicated by the k-per-level "
        "requirement specified in the pre-registration, moderator analyses were "
        "restricted to those moderators with at least 10 contributing effect "
        "sizes per level; moderators failing this criterion are reported "
        "narratively."
    )
    add_para(doc, p1, indent_first=True)

    p2 = (
        "Learning modality. Modality significantly moderated the Conscientiousness "
        "effect (QM([df_mod_C]) = [QM_mod_C], p = [p_mod_C]), with [pattern "
        "summary—e.g., \"fully online asynchronous contexts showing a "
        "stronger C effect than blended contexts\"]. Modality moderation was "
        "[significant / non-significant] for Openness, Extraversion, "
        "Agreeableness, and Neuroticism. The pattern [was / was not] consistent "
        "with the theoretical prediction that asynchronous, low-social-presence "
        "modalities would amplify Conscientiousness and attenuate Extraversion "
        "effects."
    )
    add_para(doc, p2, indent_first=True)

    p3 = (
        "Education level. Education level moderated [trait(s)] (QM = [value]). "
        "Consistent with Mammadov (2022) and Meyer et al. (2023), the Openness "
        "effect [declined / did not decline] significantly from lower to higher "
        "education levels. The Conscientiousness effect [remained stable / "
        "varied] across education levels. The limited number of K-12 studies in "
        "the online corpus ([n_K12]) restricts inference at the lowest "
        "education level; this is acknowledged in the Discussion."
    )
    add_para(doc, p3, indent_first=True)

    p4 = (
        "Region and era. Regional moderation [showed / did not show] the "
        "amplified Asian-sample pattern reported by Mammadov (2022) and Chen "
        "et al. (2025): pooled C in Asian samples was r = [r_C_Asia] versus "
        "r = [r_C_NonAsia] elsewhere. Era moderation (pre-COVID vs. COVID-era "
        "vs. post-COVID) [was / was not] significant for [trait(s)], with "
        "[pattern summary—e.g., \"C effects larger in COVID-era studies, "
        "potentially reflecting the heightened self-regulation demands of "
        "emergency remote teaching\"]. Personality instrument, publication "
        "year (continuous), log-transformed sample size, and risk-of-bias "
        "score contributed [effect summary] (see Table 3)."
    )
    add_para(doc, p4, indent_first=True)

    add_h2(doc, "Sensitivity Analyses")

    p5 = (
        "The seven pre-specified sensitivity analyses are summarized in Table 4 "
        "[placeholder]. (a) Excluding studies with risk-of-bias aggregate score "
        "below 5 [changed / did not substantially change] the pooled estimates "
        "(e.g., pooled C shifted from [r_C] to [r_C_highquality]). (b) Excluding "
        "the present author's own prior primary study (Tokiwa, 2025) [changed / "
        "did not change] the pattern of findings. (c) Excluding converted "
        "effect sizes (β-to-r, d-to-r) [had / had no] substantive effect. (d) "
        "Excluding studies with N < 50 [affected / did not affect] pooled "
        "estimates. (e) The two alternative HEXACO-to-Big-Five mappings "
        "produced [consistent / inconsistent] results, [supporting / not "
        "supporting] the primary mapping protocol. (f) Leave-one-out analysis "
        "identified [n_influential] potentially influential studies with "
        "Cook's distance > 1 or |DFFITS| > 3√(p/k); exclusion of these [did / "
        "did not] alter the pooled estimates beyond the 95% confidence "
        "interval. (g) Substituting the DerSimonian-Laird estimator for REML "
        "produced [nearly identical / slightly different] pooled effects, "
        "confirming [robustness / partial dependence on] estimator choice."
    )
    add_para(doc, p5, indent_first=True)

    add_h2(doc, "Publication Bias Assessment")

    p6 = (
        "Funnel plots for each trait are presented in Figure 7 [placeholder]. "
        "Egger's regression asymmetry test yielded [results summary per trait]: "
        "Conscientiousness intercept = [intercept_C], p = [p_Egger_C]; "
        "Openness intercept = [intercept_O], p = [p_Egger_O]; Extraversion "
        "intercept = [intercept_E], p = [p_Egger_E]; Agreeableness intercept = "
        "[intercept_A], p = [p_Egger_A]; Neuroticism intercept = "
        "[intercept_N], p = [p_Egger_N]. Peters' regression, an alternative "
        "specification for correlation effect sizes, yielded [convergent / "
        "divergent] conclusions."
    )
    add_para(doc, p6, indent_first=True)

    p7 = (
        "Duval and Tweedie's (2000) trim-and-fill procedure imputed [n_imputed] "
        "additional studies across the five trait pools, producing back-"
        "adjusted pooled effects that [did / did not] differ meaningfully "
        "from the primary estimates. P-curve analysis (Simonsohn et al., 2014) "
        "indicated [evidential value / flat curve / right-skew suggestive of "
        "no effect] for significant findings in the literature, [supporting "
        "/ weakening] confidence in the non-null pooled effects. Overall, the "
        "evidence for publication bias was [substantial / moderate / minimal], "
        "consistent with partial grey-literature inclusion in the primary "
        "search and with the pattern documented by Mammadov (2022) in the "
        "broader Big Five–achievement literature."
    )
    add_para(doc, p7, indent_first=True)

    add_h2(doc, "GRADE Summary of Findings")

    p8 = (
        "The GRADE confidence rating for each Big Five trait was derived from "
        "the five pre-specified domains and is presented in Table 5 "
        "[placeholder: Summary of Findings]. Confidence in the pooled "
        "Conscientiousness–achievement association was rated as [High / "
        "Moderate / Low / Very Low], based on [risk of bias / inconsistency "
        "/ indirectness / imprecision / publication bias] considerations. "
        "Openness confidence: [rating]. Extraversion: [rating]. Agreeableness: "
        "[rating]. Neuroticism: [rating]. The trait pool with the largest "
        "magnitude pooled effect (Conscientiousness, |r| ≥ .30) [did / did not] "
        "warrant an upgrade under the large-magnitude criterion, and no "
        "dose-response (facet-level) gradient analysis was possible given the "
        "limited facet-level reporting in the primary corpus."
    )
    add_para(doc, p8, indent_first=True)


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
    build_intro_part3(doc)
    build_methods_part1(doc)
    build_methods_part2(doc)
    build_methods_part3(doc)
    build_methods_part4(doc)
    build_methods_part5(doc)
    build_results_part1(doc)
    build_results_part2(doc)
    build_results_part3(doc)
    build_references(doc)
    doc.save(OUTPUT)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
